"""Chrono-Slot Pointer Weaver (CS-PW v2): scaled ShortConv+Chunk-GQA hybrid + 4-slot Chrono-CfC +
delexicalized frame/phrase pointer decoder, and the spoiler-safe story graph (SGRF).

Binary contract: ``cspw-v2-spec.md`` (layout, magic, shifts) + ``cspw-v2-kernel-addendum.md`` (exact
integer semantics shared with the C++ kernel).  Little-endian throughout.

* ``cspw.bin`` (CSPW v2, exactly 393,728 B): 512-B header + int8 tensors at fixed offsets
  (``CSPW_LAYOUT``) + NUL-padded frame/phrase strings + zero tail.
* ``story-graph.idx`` (SGRF v1, 512-B sectors): header, ``StoryEntityRecord`` (64 B, sorted by
  (term_hash, book_path_hash)), ``StoryEventRecord`` (256 B, contiguous per entity, ascending cutoff),
  ``StoryAliasRecord`` (24 B, sorted by alias_hash).

``int8_cspw_weave`` is the bit-exact integer reference (numpy int64, all ``>>`` arithmetic/floor,
``//`` floor division); ``verify_c_parity.py`` checks it against an independent C implementation and
``ChronoSlotPointerWeaver`` (torch) reproduces it exactly in ``quant`` mode (STE gradients).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

try:
    from model import fnv1a64
except ImportError:  # Hugging Face layout ships model.py as plhn_model.py
    from plhn_model import fnv1a64

# ----------------------------------------------------------------------------------------------
# cspw.bin layout
# ----------------------------------------------------------------------------------------------

CSPW_MAGIC = b"CSPW"
CSPW_VERSION = 2
D_MODEL = 192
D_SLOT = 128
N_SLOTS = 4
N_FRAMES = 256
N_PHRASES = 512
N_HEADS = 4
N_KV = 2
HEAD_DIM = 48
MAX_SCENES = 4
FRAME_BYTES = 64
PHRASE_BYTES = 100
FRAMES_PER_SLOT = N_FRAMES // N_SLOTS
PHRASES_PER_SLOT = N_PHRASES // N_SLOTS
CSPW_HEADER = struct.Struct("<4sHHIIIIII16f")
CSPW_HEADER_BYTES = 512
CSPW_FILE_BYTES = 393_728

# (name, shape, byte offset) of every int8 tensor
CSPW_LAYOUT = [
    ("up_proj", (D_SLOT, D_MODEL), 512),
    ("gqa_q", (D_MODEL, N_HEADS * HEAD_DIM), 25_088),
    ("gqa_k", (D_MODEL, N_KV * HEAD_DIM), 61_952),
    ("gqa_v", (D_MODEL, N_KV * HEAD_DIM), 80_384),
    ("gqa_o", (N_HEADS * HEAD_DIM, D_MODEL), 98_816),
    ("down_proj", (D_MODEL, D_SLOT), 135_680),
    ("salience_w", (D_SLOT,), 160_256),
    ("slot_gate_w", (N_SLOTS, D_SLOT, D_SLOT), 160_512),
    ("frame_bank", (N_FRAMES, D_SLOT), 226_048),
    ("phrase_bank", (N_PHRASES, D_SLOT), 275_200),
]
SALIENCE_B_OFFSET = 160_384
FRAME_STRINGS_OFFSET = 258_816
PHRASE_STRINGS_OFFSET = 340_736
TAIL_OFFSET = 391_936
assert TAIL_OFFSET + 1_792 == CSPW_FILE_BYTES
assert all(off + int(np.prod(shape)) <= CSPW_FILE_BYTES for _, shape, off in CSPW_LAYOUT)

# addendum §B tables
EXP_LUT = [256, 235, 215, 197, 181, 166, 152, 140, 128, 117, 108, 99, 91, 83, 76, 70, 64, 59, 54, 49, 45, 41,
           38, 35, 32, 29, 27, 25, 23, 21, 19, 17, 16, 15, 13, 12, 11, 10, 10, 9, 8, 7, 7, 6, 6, 5, 5, 4, 4,
           4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0]
FRAC_Q15 = [32768, 34219, 35734, 37316, 38968, 40693, 42495, 44376, 46341, 48393, 50535, 52773, 55109, 57549,
            60097, 62757]
NEG_Q15 = [32768, 31379, 30048, 28774, 27554, 26386, 25268, 24196, 23170, 22188, 21247, 20347, 19484, 18658,
           17867, 17109, 16384]
assert EXP_LUT[:64] == [round(256 * 2 ** (-t / 8)) for t in range(64)] and len(EXP_LUT) == 65
assert FRAC_Q15 == [round(32768 * 2 ** (i / 16)) for i in range(16)]
assert NEG_Q15 == [round(32768 * 2 ** (-i / 16)) for i in range(17)]
TAU_LUT = [max(16, (FRAC_Q15[c & 15] * (1 << (c >> 4)) + 16384) >> 15) for c in range(256)]
DEFAULT_TEMPLATES = ("%E", "%O", "recently %A", "%B")
PLACEHOLDERS = "ERAOLB"
TEXT_MAX_BYTES = 511

# cspw_golden.bin (addendum §D)
GOLDEN_MAGIC = b"CSGV"
GOLDEN_HEADER = struct.Struct("<4sIII")
GOLDEN_CASE = struct.Struct("<BBH4I512b128b4B4H512b4hhh")
assert GOLDEN_CASE.size == 1196


def sat8(a):
    return np.clip(a, -127, 127)


def tau_of(g: int) -> int:
    return TAU_LUT[64 + (int(g) >> 1)]


def decay_q15(dp: int, tau: int) -> int:
    """f = 32768 * 2^(-dp/tau) in Q15 via the 17-entry interpolation LUT (addendum §B.11)."""
    x = (int(dp) * 256) // int(tau)
    if x >= 4096:
        return 0
    ip, xf = x >> 8, x & 255
    seg, t = xf >> 4, xf & 15
    base = (NEG_Q15[seg] * (16 - t) + NEG_Q15[seg + 1] * t + 8) >> 4
    return base if ip == 0 else (base + (1 << (ip - 1))) >> ip


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def utf8_fit(s: str, max_bytes: int) -> bytes:
    """UTF-8 bytes of ``s`` cut to <= max_bytes at a codepoint boundary (never splits a sequence)."""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return b
    b = b[:max_bytes]
    i = len(b) - 1
    while i >= 0 and (b[i] & 0xC0) == 0x80:
        i -= 1
    if i >= 0:
        lead = b[i]
        need = 1 if lead < 0x80 else 4 if lead >= 0xF0 else 3 if lead >= 0xE0 else 2
        if i + need > len(b):
            b = b[:i]
    return b


# ----------------------------------------------------------------------------------------------
# Pack / parse cspw.bin
# ----------------------------------------------------------------------------------------------


def pack_cspw(tensors: dict, salience_b: int, frames: list, phrases: list, scales=None) -> bytes:
    blob = bytearray(CSPW_FILE_BYTES)
    sc = list(scales or [1.0 / 128] * 9 + [0.0] * 7)
    sc += [0.0] * (16 - len(sc))
    CSPW_HEADER.pack_into(blob, 0, CSPW_MAGIC, CSPW_VERSION, 0, D_MODEL, D_SLOT, N_SLOTS, N_FRAMES, N_PHRASES, 0,
                          *sc[:16])
    for name, shape, off in CSPW_LAYOUT:
        a = np.asarray(tensors[name])
        assert a.shape == shape, (name, a.shape, shape)
        assert a.min() >= -127 and a.max() <= 127, name
        raw = np.ascontiguousarray(a, dtype=np.int8).tobytes()
        blob[off:off + len(raw)] = raw
    assert -32768 <= int(salience_b) <= 32767
    struct.pack_into("<h", blob, SALIENCE_B_OFFSET, int(salience_b))
    assert len(frames) <= N_FRAMES and len(phrases) <= N_PHRASES
    for i, s in enumerate(frames):
        b = utf8_fit(s or "", FRAME_BYTES - 1)
        blob[FRAME_STRINGS_OFFSET + i * FRAME_BYTES:FRAME_STRINGS_OFFSET + i * FRAME_BYTES + len(b)] = b
    for i, s in enumerate(phrases):
        b = utf8_fit(s or "", PHRASE_BYTES - 1)
        blob[PHRASE_STRINGS_OFFSET + i * PHRASE_BYTES:PHRASE_STRINGS_OFFSET + i * PHRASE_BYTES + len(b)] = b
    assert len(blob) == CSPW_FILE_BYTES
    return bytes(blob)


class CspwWeights:
    """Parsed cspw.bin (int64 numpy tensors + decoded strings)."""

    def __init__(self, blob: bytes):
        if len(blob) != CSPW_FILE_BYTES:
            raise ValueError(f"cspw.bin must be {CSPW_FILE_BYTES} bytes, got {len(blob)}")
        h = CSPW_HEADER.unpack_from(blob, 0)
        if h[:9] != (CSPW_MAGIC, CSPW_VERSION, 0, D_MODEL, D_SLOT, N_SLOTS, N_FRAMES, N_PHRASES, 0):
            raise ValueError(f"bad CSPW header {h[:9]}")
        self.scales = h[9:]
        self.t = {}
        for name, shape, off in CSPW_LAYOUT:
            n = int(np.prod(shape))
            self.t[name] = np.frombuffer(blob, dtype=np.int8, count=n, offset=off).astype(np.int64).reshape(shape)
        self.salience_b = struct.unpack_from("<h", blob, SALIENCE_B_OFFSET)[0]
        fr = np.frombuffer(blob, np.uint8, N_FRAMES * FRAME_BYTES, FRAME_STRINGS_OFFSET).reshape(N_FRAMES, -1)
        ph = np.frombuffer(blob, np.uint8, N_PHRASES * PHRASE_BYTES, PHRASE_STRINGS_OFFSET).reshape(N_PHRASES, -1)
        self.frames = [_cstr(bytes(r)) for r in fr]
        self.phrases = [_cstr(bytes(r)) for r in ph]
        self.frame_ok = np.array([r[0] != 0 for r in fr])
        self.phrase_ok = np.array([r[0] != 0 for r in ph])
        if any(blob[96:512]) or any(blob[SALIENCE_B_OFFSET + 2:SALIENCE_B_OFFSET + 128]) or any(blob[TAIL_OFFSET:]):
            raise ValueError("CSPW reserved bytes must be zero")


# ----------------------------------------------------------------------------------------------
# Bit-exact integer reference
# ----------------------------------------------------------------------------------------------


def int8_cspw_core(w: CspwWeights, scenes, pages, echo=None) -> dict:
    """Addendum §B steps 1-12 for M (1..4) chronological scene vectors (int8[M][128])."""
    t = w.t
    S = np.asarray(scenes, dtype=np.int64).reshape(-1, D_SLOT)
    M = len(S)
    assert 1 <= M <= MAX_SCENES
    p = [int(v) for v in pages]
    h = sat8((S @ t["up_proj"]) >> 7)
    q = sat8((h @ t["gqa_q"]) >> 7)
    k = sat8((h @ t["gqa_k"]) >> 7)
    v = sat8((h @ t["gqa_v"]) >> 7)
    attn = np.zeros((M, N_HEADS * HEAD_DIM), np.int64)
    for hd in range(N_HEADS):
        kv = hd >> 1
        qh, kh, vh = (q[:, hd * HEAD_DIM:(hd + 1) * HEAD_DIM], k[:, kv * HEAD_DIM:(kv + 1) * HEAD_DIM],
                      v[:, kv * HEAD_DIM:(kv + 1) * HEAD_DIM])
        for a in range(M):
            s = (kh[:a + 1] @ qh[a]) >> 6
            e = np.array([EXP_LUT[min(int(s.max() - x), 64)] for x in s], np.int64)
            tot = int(e.sum())
            wb = (e * 256 + tot // 2) // tot
            attn[a, hd * HEAD_DIM:(hd + 1) * HEAD_DIM] = sat8((wb @ vh[:a + 1]) >> 8)
    h = sat8(h + ((attn @ t["gqa_o"]) >> 7))
    v128 = sat8((h @ t["down_proj"]) >> 7)
    r = sat8((S + v128) >> 1)
    g = np.clip(((r @ t["salience_w"]) >> 7) + w.salience_b + 128, 0, 255)
    tau = [tau_of(x) for x in g]
    u = np.stack([sat8((r @ t["slot_gate_w"][kk].T) >> 7) for kk in range(N_SLOTS)])  # (4, M, 128)
    slots = np.zeros((N_SLOTS, D_SLOT), np.int64)
    wgt = g + 32
    den = int(wgt.sum())
    num = (wgt[:, None] * u[0]).sum(0)
    slots[0] = sat8((2 * num + den) // (2 * den))
    slots[1] = u[1, 0]
    arc = u[2, 0].copy()
    for m in range(1, M):
        f = decay_q15(max(0, p[m] - p[m - 1]), tau[m])
        alpha = min(max((32768 - f) + ((int(g[m]) * f) >> 9), 4096), 32768)
        arc = sat8(((32768 - alpha) * arc + alpha * u[2, m] + 16384) >> 15)
    slots[2] = arc
    if echo is not None:
        slots[3] = sat8((u[3, M - 1] + np.asarray(echo, np.int64)) >> 1)
    else:
        slots[3] = u[3, M - 1]
    return {"M": M, "g": [int(x) for x in g], "tau": tau, "slots": slots, "u": u, "r": r}


def _argmax_first(scores, ok):
    best, bi = None, -1
    for i, (s, o) in enumerate(zip(scores, ok)):
        if o and (best is None or s > best):
            best, bi = int(s), i
    return bi


def int8_cspw_weave(w: CspwWeights, scenes, pages, echo=None) -> dict:
    """Integer Chrono-CfC + pointer selection: slots, frame index per slot group, %R phrase, %A scene."""
    out = int8_cspw_core(w, scenes, pages, echo)
    slots, u, M = out["slots"], out["u"], out["M"]
    fb, pb = w.t["frame_bank"], w.t["phrase_bank"]
    out["frame_idx"] = []
    for kk in range(N_SLOTS):
        lo = kk * FRAMES_PER_SLOT
        bi = _argmax_first(fb[lo:lo + FRAMES_PER_SLOT] @ slots[kk], w.frame_ok[lo:lo + FRAMES_PER_SLOT])
        out["frame_idx"].append(-1 if bi < 0 else lo + bi)
    out["phrase_idx"] = _argmax_first(pb @ slots[0], w.phrase_ok)
    arc_scene = -1
    if M > 1:
        sc = u[2, 1:M] @ slots[2]
        best = None
        for m in range(1, M):
            if best is None or int(sc[m - 1]) >= best:
                best, arc_scene = int(sc[m - 1]), m
    out["arc_scene"] = arc_scene
    return out


# ----------------------------------------------------------------------------------------------
# Golden cases (addendum §D)
# ----------------------------------------------------------------------------------------------


def pack_golden_case(scenes, pages, echo, res) -> bytes:
    M = len(scenes)
    sc = np.zeros((MAX_SCENES, D_SLOT), np.int64)
    sc[:M] = np.asarray(scenes, np.int64)
    pg = list(pages) + [0] * (MAX_SCENES - M)
    g = list(res["g"]) + [0] * (MAX_SCENES - M)
    tau = list(res["tau"]) + [0] * (MAX_SCENES - M)
    ec = np.zeros(D_SLOT, np.int64) if echo is None else np.asarray(echo, np.int64)
    return GOLDEN_CASE.pack(M, int(echo is not None), 0, *pg, *sc.ravel().tolist(), *ec.tolist(), *g, *tau,
                            *np.asarray(res["slots"]).ravel().tolist(), *res["frame_idx"], res["phrase_idx"],
                            res["arc_scene"])


def pack_golden(cases: list[bytes]) -> bytes:
    return GOLDEN_HEADER.pack(GOLDEN_MAGIC, 1, len(cases), 0) + b"".join(cases)


def unpack_golden(blob: bytes):
    magic, ver, n, _ = GOLDEN_HEADER.unpack_from(blob, 0)
    assert magic == GOLDEN_MAGIC and ver == 1 and len(blob) == GOLDEN_HEADER.size + n * GOLDEN_CASE.size
    out = []
    for i in range(n):
        f = GOLDEN_CASE.unpack_from(blob, GOLDEN_HEADER.size + i * GOLDEN_CASE.size)
        M, he = f[0], f[1]
        pages = list(f[3:7])[:M]
        sc = np.array(f[7:519], np.int64).reshape(MAX_SCENES, D_SLOT)[:M]
        echo = np.array(f[519:647], np.int64) if he else None
        g, tau = list(f[647:651])[:M], list(f[651:655])[:M]
        slots = np.array(f[655:1167], np.int64).reshape(N_SLOTS, D_SLOT)
        out.append({"scenes": sc, "pages": pages, "echo": echo, "g": g, "tau": tau, "slots": slots,
                    "frame_idx": list(f[1167:1171]), "phrase_idx": f[1171], "arc_scene": f[1172]})
    return out


# ----------------------------------------------------------------------------------------------
# story-graph.idx (SGRF v1)
# ----------------------------------------------------------------------------------------------

SGRF_MAGIC = b"SGRF"
SGRF_VERSION = 1
SGRF_HEADER = struct.Struct("<4sHHIIIIII")
SGRF_ENTITY = struct.Struct("<QQIIHBB32s4s")
SGRF_EVENT = struct.Struct("<IHBB64s56s128b")
SGRF_ALIAS = struct.Struct("<QQII")
assert SGRF_ENTITY.size == 64 and SGRF_EVENT.size == 256 and SGRF_ALIAS.size == 24
SECTOR = 512
SLOT_HINT_CROSS = 3


def _align(n: int) -> int:
    return (n + SECTOR - 1) // SECTOR * SECTOR


def pack_sgrf(entities: list[dict], aliases: list[dict]) -> bytes:
    """entities: {term_hash, book_path_hash, first_cutoff, entity_kind, flags, en_name_or_role, book_short,
    events: [{cutoff, chapter_or_page, slot_hint, salience_hint, event_gloss, target_or_loc, vec}]};
    aliases: {alias_hash, canonical_hash, reveal_cutoff}.  Events are re-sorted ascending by cutoff
    (stable), entities by (term_hash, book_path_hash), aliases by alias_hash."""
    ents = sorted(entities, key=lambda e: (e["term_hash"], e["book_path_hash"]))
    keys = [(e["term_hash"], e["book_path_hash"]) for e in ents]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (term_hash, book_path_hash) entity")
    ent_raw, ev_raw = [], []
    for e in ents:
        evs = sorted(e["events"], key=lambda x: x["cutoff"])
        if len(evs) > 0xFFFF:
            raise ValueError("too many events")
        ent_raw.append(SGRF_ENTITY.pack(e["term_hash"], e["book_path_hash"], e["first_cutoff"], len(ev_raw), len(evs),
                                        e.get("entity_kind", 1), e.get("flags", 0),
                                        utf8_fit(e.get("en_name_or_role", ""), 31),
                                        utf8_fit(e.get("book_short", ""), 4)))
        for ev in evs:
            vec = [int(x) for x in ev["vec"]]
            assert len(vec) == D_SLOT and min(vec) >= -127 and max(vec) <= 127
            ev_raw.append(SGRF_EVENT.pack(ev["cutoff"], min(0xFFFF, ev.get("chapter_or_page", 0)), ev.get("slot_hint", 0),
                                          ev.get("salience_hint", 0), utf8_fit(ev.get("event_gloss", ""), 63),
                                          utf8_fit(ev.get("target_or_loc", ""), 55), *vec))
    al = sorted(aliases, key=lambda a: a["alias_hash"])
    al_raw = [SGRF_ALIAS.pack(a["alias_hash"], a["canonical_hash"], a["reveal_cutoff"], 0) for a in al]
    ent_off = SECTOR
    ev_off = _align(ent_off + 64 * len(ent_raw))
    al_off = _align(ev_off + 256 * len(ev_raw))
    total = _align(al_off + 24 * len(al_raw))
    blob = bytearray(total)
    SGRF_HEADER.pack_into(blob, 0, SGRF_MAGIC, SGRF_VERSION, 0, len(ent_raw), len(ev_raw), len(al_raw), ent_off,
                          ev_off, al_off)
    for i, r in enumerate(ent_raw):
        blob[ent_off + 64 * i:ent_off + 64 * (i + 1)] = r
    for i, r in enumerate(ev_raw):
        blob[ev_off + 256 * i:ev_off + 256 * (i + 1)] = r
    for i, r in enumerate(al_raw):
        blob[al_off + 24 * i:al_off + 24 * (i + 1)] = r
    return bytes(blob)


class StoryGraph:
    """Parsed story-graph.idx with the device lookup rules (spoiler + alias reveal gates)."""

    def __init__(self, blob: bytes):
        (magic, ver, _flags, ne, nv, na, eo, vo, ao) = SGRF_HEADER.unpack_from(blob, 0)
        if magic != SGRF_MAGIC or ver != SGRF_VERSION or eo != SECTOR or eo % SECTOR or vo % SECTOR or ao % SECTOR:
            raise ValueError("bad SGRF header")
        if len(blob) < ao + 24 * na or vo < eo + 64 * ne or ao < vo + 256 * nv:
            raise ValueError("truncated / overlapping SGRF")
        self.entities = []
        for i in range(ne):
            th, bh, first, start, cnt, kind, flags, name, short = SGRF_ENTITY.unpack_from(blob, eo + 64 * i)
            self.entities.append({"term_hash": th, "book_path_hash": bh, "first_cutoff": first, "event_start": start,
                                  "event_count": cnt, "entity_kind": kind, "flags": flags,
                                  "en_name_or_role": _cstr(name), "book_short": _cstr(short)})
        self.events = []
        for i in range(nv):
            f = SGRF_EVENT.unpack_from(blob, vo + 256 * i)
            self.events.append({"cutoff": f[0], "chapter_or_page": f[1], "slot_hint": f[2], "salience_hint": f[3],
                                "event_gloss": _cstr(f[4]), "target_or_loc": _cstr(f[5]),
                                "vec": np.array(f[6:], np.int64)})
        self.aliases = [SGRF_ALIAS.unpack_from(blob, ao + 24 * i)[:3] for i in range(na)]
        self.ent_keys = [(e["term_hash"], e["book_path_hash"]) for e in self.entities]
        assert self.ent_keys == sorted(self.ent_keys), "entities not sorted"
        assert [a[0] for a in self.aliases] == sorted(a[0] for a in self.aliases), "aliases not sorted"
        self._ent = {k: i for i, k in enumerate(self.ent_keys)}
        self._alias = {}
        for a in self.aliases:
            self._alias.setdefault(a[0], []).append(a)

    def find_entity(self, term_hash: int, book_hash: int):
        i = self._ent.get((term_hash, book_hash))
        if i is None:
            i = self._ent.get((term_hash, 0))
        return None if i is None else self.entities[i]

    def lookup(self, term_hash: int, book_hash: int, cur: int):
        """-> (entity, eligible events) or None.  Alias Reveal Gate (alias first: the first alias row
        whose reveal gate passes rewrites the term to its canonical hash) + Spoiler Gate."""
        for a in self._alias.get(term_hash, ()):
            if cur == 0 or cur >= a[2]:
                term_hash = a[1]
                break
        ent = self.find_entity(term_hash, book_hash)
        if ent is None or (cur > 0 and ent["first_cutoff"] > cur):
            return None
        evs = self.events[ent["event_start"]:ent["event_start"] + ent["event_count"]]
        evs = [e for e in evs if cur == 0 or e["cutoff"] <= cur]
        return (ent, evs) if evs else None


def select_scenes(events: list[dict]):
    """Addendum §C on already-eligible events -> (story scene events (<=3), echo event or None)."""
    cross = [e for e in events if e["slot_hint"] == SLOT_HINT_CROSS]
    story = [e for e in events if e["slot_hint"] != SLOT_HINT_CROSS]
    echo = cross[-1] if cross else None
    if not story:
        story, echo = cross, None
    n = len(story)
    if n == 0:
        return [], None
    picks = [story[0]]
    if n >= 3:
        mid = max(range(1, n - 1), key=lambda i: (story[i]["salience_hint"], -i))
        picks.append(story[mid])
    if n >= 2:
        picks.append(story[n - 1])
    return picks, echo


def fill_frame(template: str, values: dict) -> str | None:
    """Substitute %E/%R/%O/%A/%L/%B (``%%`` -> ``%``, ``%`` + any other char stays literal);
    None when any placeholder used resolves to ''."""
    out, i, n = [], 0, len(template)
    while i < n:
        c = template[i]
        if c == "%" and i + 1 < n and template[i + 1] in PLACEHOLDERS:
            val = values.get(template[i + 1], "")
            if not val:
                return None
            out.append(val)
            i += 2
        elif c == "%" and i + 1 < n and template[i + 1] == "%":
            out.append("%")
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _part(w: CspwWeights | None, idx: int, k: int, values: dict, trace: list | None = None) -> str:
    """One refresher part; ``trace`` gets (k, frame index or -1 for the default, template, text)."""
    if w is not None and idx >= 0:
        s = fill_frame(w.frames[idx], values)
        if s is not None:
            if trace is not None and s:
                trace.append((k, idx, w.frames[idx], s))
            return s
    s = fill_frame(DEFAULT_TEMPLATES[k], values) or ""
    if trace is not None and s:
        trace.append((k, -1, DEFAULT_TEMPLATES[k], s))
    return s


def weave_refresher(w: CspwWeights | None, ent: dict, events: list[dict], word: str = "",
                    trace: list | None = None):
    """Device-side refresher text (addendum §C) for eligible ``events`` -> (text, integer weave or None).
    ``trace`` (optional list) receives the emitted parts, see ``_part``; the echo segment is the text
    after ``" | Echo: "``."""
    scenes, echo = select_scenes(events)
    if not scenes:
        return "", None
    M = len(scenes)
    res = None
    if w is not None:
        res = int8_cspw_weave(w, [e["vec"] for e in scenes], [e["chapter_or_page"] for e in scenes],
                              echo["vec"] if echo is not None else None)
        arc_scene = res["arc_scene"]
    else:
        arc_scene = M - 1 if M >= 2 else -1
    arc = scenes[arc_scene] if arc_scene >= 0 else None
    base = {"E": ent["en_name_or_role"] or word,
            "R": w.phrases[res["phrase_idx"]] if res is not None and res["phrase_idx"] >= 0 else "",
            "O": scenes[0]["event_gloss"],
            "A": arc["event_gloss"] if arc is not None else "",
            "B": (echo["target_or_loc"] or echo["event_gloss"]) if echo is not None else ""}
    fidx = res["frame_idx"] if res is not None else [-1] * N_SLOTS
    role = _part(w, fidx[0], 0, {**base, "L": scenes[M - 1]["target_or_loc"]}, trace)
    origin = _part(w, fidx[1], 1, {**base, "L": scenes[0]["target_or_loc"]}, trace)
    arc_txt = ""
    if arc is not None:
        arc_txt = _part(w, fidx[2], 2, {**base, "L": arc["target_or_loc"] or scenes[0]["target_or_loc"]}, trace)
    echo_txt = _part(w, fidx[3], 3, {**base, "L": echo["target_or_loc"]}, trace) if echo is not None else ""
    text = "Story: " + role
    rest = [p for p in (origin, arc_txt) if p]
    if rest:
        text += " \u2014 " + "; ".join(rest)
    if not text.endswith((".", "!", "?")):
        text += "."
    if echo_txt:
        text += " | Echo: " + echo_txt
    return utf8_fit(text, TEXT_MAX_BYTES).decode("utf-8"), res


def timeline_entities(timelines: list[dict], vec_of) -> list[dict]:
    """``cspw_timelines.jsonl`` rows -> ``pack_sgrf`` entities.  Story events keep their schema fields
    (vec = ``vec_of(scene_text)``, int8[128]); each cross-book echo becomes a slot_hint-3 event visible
    from the entity's first appearance (cutoff = first_cutoff, target_or_loc = other book's title).
    Rows sharing (term_hash, book_hash) are merged (min first_cutoff, union of events)."""
    ents: dict = {}
    for tl in timelines:
        key = (int(tl["term_hash"]), int(tl.get("book_hash") or 0))
        first = int(tl["first_cutoff"])
        evs = [{"cutoff": int(ev["cutoff"]),
                "chapter_or_page": min(0xFFFF, int(ev.get("chapter_or_page", ev.get("page", 0)) or 0)),
                "slot_hint": int(ev.get("slot_hint", 1 if ev.get("m", 0) == 0 else 2)),
                "salience_hint": max(0, min(255, int(ev.get("salience_hint", 0)))),
                "event_gloss": ev.get("event_gloss", ""), "target_or_loc": ev.get("target_or_loc", ""),
                "vec": vec_of(ev["scene_text"])} for ev in tl["events"]]
        for x in tl.get("cross_book_echoes", []) or []:
            evs.append({"cutoff": first, "chapter_or_page": 0, "slot_hint": SLOT_HINT_CROSS, "salience_hint": 0,
                        "event_gloss": x.get("gloss", ""), "target_or_loc": x.get("title", ""),
                        "vec": vec_of(x.get("text") or x.get("gloss", ""))})
        if key in ents:
            e = ents[key]
            e["first_cutoff"] = min(e["first_cutoff"], first)
            e["events"] += evs
        else:
            ents[key] = {"term_hash": key[0], "book_path_hash": key[1], "first_cutoff": first,
                         "entity_kind": int(tl.get("entity_kind", 1)), "flags": 0,
                         "en_name_or_role": tl.get("en_name_or_role", ""),
                         "book_short": (tl.get("book_short") or Path(tl.get("book_path", "")).stem)[:4],
                         "events": evs}
    for e in ents.values():
        e["events"] = e["events"][:0xFFFF]
    return list(ents.values())


def alias_rows(timelines: list[dict], aliases: list[dict]) -> list[dict]:
    """Alias records from ``cspw_aliases.jsonl`` + timeline ``aliases`` (deduplicated)."""
    rows = {}
    for a in aliases:
        rows[(int(a["alias_hash"]), int(a["canonical_hash"]), int(a["reveal_cutoff"]))] = None
    for tl in timelines:
        for a in tl.get("aliases", []) or []:
            rows[(int(a["alias_hash"]), int(tl["term_hash"]), int(a["reveal_cutoff"]))] = None
    return [{"alias_hash": a, "canonical_hash": c, "reveal_cutoff": r} for a, c, r in rows]


# ----------------------------------------------------------------------------------------------
# Torch module (QAT path bit-exact with int8_cspw_core)
# ----------------------------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:

    def _ste(x, xq):
        return xq.detach() + (x - x.detach())

    def _fq(w, scale):
        return _ste(w, torch.clamp(torch.floor(w / scale + 0.5), -127, 127) * scale)

    def _pow2(maxabs: float) -> float:
        return 2.0 ** math.ceil(math.log2(max(maxabs, 1e-8) / 127.0))

    _EXP_T = torch.tensor(EXP_LUT, dtype=torch.float64)
    _TAU_T = torch.tensor(TAU_LUT, dtype=torch.float64)
    _NEG_T = torch.tensor(NEG_Q15, dtype=torch.float64)

    class ChronoSlotPointerWeaver(nn.Module):
        """``d_model=None``: no hybrid core (r = s, Arm 0).  ``pool``: 'cfc' (4-slot Chrono-CfC, exported)
        or 'flat' (salience-free mean over scenes for every slot, Arms 0-3).  Activations are integers
        (int8 range); matrices are trained in units of 1/128 (exported int8 = W * 128, ``>> 7``)."""

        def __init__(self, d_model: int | None = D_MODEL, pool: str = "cfc", n_heads: int = N_HEADS,
                     n_kv: int = N_KV):
            super().__init__()
            self.d_model, self.pool, self.quant = d_model, pool, False
            d = d_model
            if d is not None:
                self.head_dim = d // n_heads
                self.n_heads, self.n_kv = n_heads, n_kv
                s_in, s_d = 128.0 / math.sqrt(D_SLOT), 128.0 / math.sqrt(d)
                self.up_proj = nn.Parameter(torch.randn(D_SLOT, d) * s_in / 128 * 1.5)
                self.gqa_q = nn.Parameter(torch.randn(d, d) * s_d / 128 * 1.2)
                self.gqa_k = nn.Parameter(torch.randn(d, n_kv * self.head_dim) * s_d / 128 * 1.2)
                self.gqa_v = nn.Parameter(torch.randn(d, n_kv * self.head_dim) * s_d / 128)
                self.gqa_o = nn.Parameter(torch.randn(d, d) * s_d / 128 * 0.5)
                self.down_proj = nn.Parameter(torch.randn(d, D_SLOT) * s_d / 128 * 0.25)  # small residual
            self.salience_w = nn.Parameter(torch.randn(D_SLOT) / math.sqrt(D_SLOT) * 0.5)
            self.salience_b = nn.Parameter(torch.zeros(()))
            eye = torch.eye(D_SLOT).expand(N_SLOTS, D_SLOT, D_SLOT).clone()
            self.slot_gate_w = nn.Parameter(eye * 0.9 + torch.randn(N_SLOTS, D_SLOT, D_SLOT) * 0.02)
            self.frame_bank = nn.Parameter(torch.randn(N_FRAMES, D_SLOT) * 0.1)
            self.phrase_bank = nn.Parameter(torch.randn(N_PHRASES, D_SLOT) * 0.1)
            self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / 400.0)))

        # ---- quantisation helpers
        def _w(self, p):
            return _fq(p, 1.0 / 128) if self.quant else p

        def bank_scales(self) -> dict:
            return {n: _pow2(float(getattr(self, n).detach().abs().max())) for n in ("frame_bank", "phrase_bank")}

        def banks(self):
            """Frame / phrase banks; QAT: int8 grid (pow2 scale) in the float units of the bank, so logits keep
            their magnitude across the QAT switch and argmax == the device's integer argmax (pow2 scaling
            is exact)."""
            if not self.quant:
                return self.frame_bank, self.phrase_bank
            sc = self.bank_scales()
            return _fq(self.frame_bank, sc["frame_bank"]), _fq(self.phrase_bank, sc["phrase_bank"])

        def _fl(self, x):
            return _ste(x, torch.floor(x)) if self.quant else x

        def _mm(self, x, W):
            """sat8((x @ W_int) >> 7) with W in units of 1/128."""
            return torch.clamp(self._fl(x @ self._w(W)), -127, 127)

        # ---- forward
        def forward(self, S, pages, M, echo=None, has_echo=None) -> dict:
            """S (B,4,128) integer-valued scene vectors (rows >= M ignored), pages (B,4), M (B,) in 1..4,
            echo (B,128), has_echo (B,) bool.  Returns slots (B,4,128), g (B,4), tau (B,4), u (B,4,4,128)."""
            B, T, _ = S.shape
            dev, dt = S.device, S.dtype
            valid = torch.arange(T, device=dev)[None, :] < M[:, None]
            if self.d_model is not None:
                h = self._mm(S, self.up_proj)
                q, k, v = self._mm(h, self.gqa_q), self._mm(h, self.gqa_k), self._mm(h, self.gqa_v)
                hd, H, KV = self.head_dim, self.n_heads, self.n_kv
                qh = q.view(B, T, H, hd).transpose(1, 2)  # (B,H,T,hd)
                rep = H // KV
                kh = k.view(B, T, KV, hd).transpose(1, 2).repeat_interleave(rep, 1)
                vh = v.view(B, T, KV, hd).transpose(1, 2).repeat_interleave(rep, 1)
                s = self._fl((qh @ kh.transpose(-1, -2)) / 64.0)  # (B,H,T,T)
                causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
                s = s.masked_fill(~causal, -1e9)
                tdiff = (s.max(-1, keepdim=True).values - s).clamp(0, 64)
                e_f = 256.0 * torch.pow(2.0, -tdiff / 8.0)
                if self.quant:
                    e_q = _EXP_T.to(dev)[tdiff.detach().round().long().clamp(0, 64)].to(dt)
                    e = _ste(e_f, e_q)
                else:
                    e = e_f
                e = e * causal.to(dt)
                tot = e.sum(-1, keepdim=True)
                w_f = e * 256.0 / tot
                if self.quant:
                    w = _ste(w_f, torch.floor((e.detach() * 256 + torch.floor(tot.detach() / 2)) / tot.detach()))
                else:
                    w = w_f
                ctx = torch.clamp(self._fl((w @ vh) / 256.0), -127, 127)  # (B,H,T,hd)
                attn = ctx.transpose(1, 2).reshape(B, T, H * hd)
                h = torch.clamp(h + self._fl(attn @ self._w(self.gqa_o)), -127, 127)
                v128 = self._mm(h, self.down_proj)
                r = torch.clamp(self._fl((S + v128) / 2.0), -127, 127)
            else:
                r = S
            sb = _ste(self.salience_b, torch.round(self.salience_b)) if self.quant else self.salience_b
            g = torch.clamp(self._fl(r @ self._w(self.salience_w)) + sb + 128.0, 0, 255)
            code = 64.0 + (torch.floor(g.detach() / 2) if self.quant else g / 2)
            tau_f = torch.clamp(torch.pow(2.0, (64.0 + g / 2) / 16.0), min=16.0)
            tau = _ste(tau_f, _TAU_T.to(dev)[code.long()].to(dt)) if self.quant else tau_f
            Gw = self._w(self.slot_gate_w)  # (4, out d, in j)
            u = torch.clamp(self._fl(torch.einsum("btj,kdj->bktd", r, Gw)), -127, 127)  # (B,4,T,128)
            vf = valid.to(dt)
            last = (M - 1).clamp(min=0)
            ar = torch.arange(B, device=dev)
            if self.pool == "flat":
                mean = (u * vf[:, None, :, None]).sum(2) / M.to(dt)[:, None, None]
                slots = mean
            else:
                wgt = (g + 32.0) * vf
                den = wgt.sum(1)
                num = (wgt[:, None, :, None] * u[:, 0:1]).sum(2)[:, 0]
                s0_f = num / den[:, None]
                if self.quant:
                    s0 = _ste(s0_f, torch.floor((2 * num.detach() + den.detach()[:, None]) / (2 * den.detach()[:, None])))
                else:
                    s0 = s0_f
                s0 = torch.clamp(s0, -127, 127)
                s1 = u[:, 1, 0]
                arc = u[:, 2, 0]
                for m in range(1, T):
                    dp = (pages[:, m] - pages[:, m - 1]).clamp(min=0).to(dt)
                    f_f = 32768.0 * torch.pow(2.0, -dp / tau[:, m])
                    if self.quant:
                        f = _ste(f_f, self._decay_exact(dp, tau[:, m].detach()))
                    else:
                        f = f_f
                    gf = self._fl(g[:, m] * f / 512.0)
                    alpha = torch.clamp((32768.0 - f) + gf, 4096.0, 32768.0)[:, None]
                    new = torch.clamp(self._fl(((32768.0 - alpha) * arc + alpha * u[:, 2, m] + 16384.0) / 32768.0),
                                      -127, 127)
                    arc = torch.where((m < M)[:, None], new, arc)
                u3 = u[ar, 3, last]
                if echo is not None:
                    he = has_echo[:, None]
                    s3 = torch.where(he, torch.clamp(self._fl((u3 + echo) / 2.0), -127, 127), u3)
                else:
                    s3 = u3
                slots = torch.stack([s0, s1, arc, s3], 1)
            return {"slots": slots, "g": g, "tau": tau, "u": u, "r": r, "valid": valid}

        @staticmethod
        def _decay_exact(dp, tau):
            x = torch.floor(dp * 256.0 / tau)
            big = x >= 4096
            xi = x.clamp(max=4095).long()
            ip, xf = xi >> 8, xi & 255
            seg, t = xf >> 4, xf & 15
            neg = _NEG_T.to(dp.device)
            base = torch.floor((neg[seg] * (16 - t) + neg[seg + 1] * t + 8) / 16.0)
            sh = torch.floor((base + torch.where(ip > 0, torch.pow(2.0, (ip - 1).clamp(min=0).double()), 0.0))
                             / torch.pow(2.0, ip.double()))
            f = torch.where(ip == 0, base, sh)
            return torch.where(big, torch.zeros_like(f), f).to(dp.dtype)

        def frame_logits(self, slots):
            """(B,4,64) slot-group frame scores (raw dots * learned scale)."""
            fb, _ = self.banks()
            fbk = fb.view(N_SLOTS, FRAMES_PER_SLOT, D_SLOT)
            return torch.einsum("bkd,kfd->bkf", slots, fbk) * self.log_scale.exp()

        def phrase_logits(self, slots, typed: bool):
            """typed: (B,512) %R scores from slot0 (device rule); untyped: (B,4,128) per-slot phrase range."""
            _, pb = self.banks()
            if typed:
                return slots[:, 0] @ pb.T * self.log_scale.exp()
            pbk = pb.view(N_SLOTS, PHRASES_PER_SLOT, D_SLOT)
            return torch.einsum("bkd,kpd->bkp", slots, pbk) * self.log_scale.exp()

        # ---- export
        @torch.no_grad()
        def export_tensors(self) -> tuple[dict, int]:
            assert self.d_model == D_MODEL and self.pool == "cfc", "only the full d=192 CfC weaver is exported"

            def q128(p):
                return np.clip(np.floor(p.detach().cpu().double().numpy() * 128 + 0.5), -127, 127).astype(np.int8)

            t = {n: q128(getattr(self, n)) for n in ("up_proj", "gqa_q", "gqa_k", "gqa_v", "gqa_o", "down_proj",
                                                       "salience_w", "slot_gate_w")}
            sc = self.bank_scales()
            for n in ("frame_bank", "phrase_bank"):
                t[n] = np.clip(np.floor(getattr(self, n).detach().cpu().double().numpy() / sc[n] + 0.5),
                               -127, 127).astype(np.int8)
            return t, int(torch.round(self.salience_b).item())
