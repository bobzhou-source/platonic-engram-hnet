"""Platonic-Engram-HNet: tokenizer-free byte encoder for the X4 Pro concept engine.

Exported artefacts (little-endian):

* ``platonic-hnet.bin`` (PLHN v1): 32-byte header + int8 payload (``PLHN_LAYOUT`` order).
  Header struct ``<IHHHBBBBffffH``: magic, version, d_byte, d_concept, n_prelude, n_core,
  router_dim, engram_buckets_log2, scale_embed, scale_router, scale_proj, scale_core, reserved.
  (The listed contract fields sum to 34 B; ``reserved`` is uint16 so the header is exactly 32 B,
  agreed with the firmware.)  Payload 94,976 B, file 95,008 B.
* ``engram.bin`` (ENGR v1): 512-byte header sector + 65,536 x 64-byte slots
  ``{u64 span_hash_fnv1a64, i8 concept_delta[24], char gloss[32]}``.
* ``*.idx`` (ICPT v1): 16-byte header + 192-byte records
  ``{u64 path_hash, u32 page_or_chunk, u32 timestamp_or_flags, char label[48], i8 vec[128]}``.

Integer inference (bit-exact on device, identical in ``plhn_int_forward`` (pure Python),
``plhn_int_forward_np`` (numpy), the C reference in ``verify_c_parity.py`` and the QAT torch path
``PlatonicEngramHNet.encode_int``):

* Activations are int8 at scale 2^-5.  Every exported scale is an exact power of two 2^-S.
  ``sh(a,s)`` = round-half-up arithmetic shift, ``rq(acc,s)`` = clamp(sh(acc,s), -127, 127).
* Matrices are stored ``[in][out]``; depthwise taps ``dw[c][0..2]`` = x_{t-2}, x_{t-1}, x_t.
* Gated ShortConv block (LFM2 double gate, hard-sigmoid Q7 gate gs = clamp(g+64, 0, 128))::

      g = rq(x.gate, S); gs = clamp(g+64,0,128); y = (x*gs+64)>>7
      z = rq(causal_dw3(y), S); o = rq(z.pw, S); o2 = (o*gs+64)>>7
      encoder: x' = sat8(x + o2);  core: x' = sat8(x + rq(o2.core_out_proj, S))

* bytes (first 256) -> byte_embed -> 2 encoder blocks -> H-Net cosine router
  (boundary iff t==0 or len>=12 or (lead byte and (len+u8len>12 or q_t.k_{t-1} <= 0)), i.e.
  p_t = (1-cos)/2 >= 0.5 without splitting UTF-8) -> chunk mean (RECIP table) -> chunk_proj ->
  Engram delta (FNV1a64(span) % slots, first 24 dims, always applied) -> Bigram-span Engram
  (chunk c >= 1, len(c-1)+len(c) <= 18: h2 = FNV1a64(bytes[start(c-1):end(c)]), applied only when
  h2 != 0 and the slot's stored span_hash == h2: x += sat8(sh(delta >> 1, sg - 5))) -> Ouro core x2 ->
  sum over chunks -> exact isqrt L2 -> int8 (scale 127).
"""

from __future__ import annotations

import math
import re
import struct
from operator import mul

# ----------------------------------------------------------------------------------------------
# Binary contract
# ----------------------------------------------------------------------------------------------

PLHN_MAGIC = 0x504C484E
PLHN_VERSION = 1
PLHN_HEADER = struct.Struct("<IHHHBBBBffffH")
assert PLHN_HEADER.size == 32

D_BYTE = 64
D_CONCEPT = 128
ROUTER_DIM = 32
N_PRELUDE = 2
N_CORE = 2  # Ouro loops (shared weights)
ENGRAM_LOG2 = 16
ENGRAM_SLOTS = 1 << ENGRAM_LOG2
ENGRAM_DIMS = 24
GLOSS_BYTES = 32
MAX_SPAN = 12
MAX_BYTES = 256
BIGRAM_MAX_BYTES = 18  # bigram-span Engram: len(chunk[c-1]) + len(chunk[c]) <= 18
ACT_SHIFT = 5  # activation scale 2^-5

# (name, shape, scale group) in payload order.
PLHN_LAYOUT = [
    ("byte_embed", (256, D_BYTE), "embed"),
    ("conv1_dw", (D_BYTE, 3), "proj"),
    ("conv1_pw", (D_BYTE, D_BYTE), "proj"),
    ("conv1_gate", (D_BYTE, D_BYTE), "proj"),
    ("conv2_dw", (D_BYTE, 3), "proj"),
    ("conv2_pw", (D_BYTE, D_BYTE), "proj"),
    ("conv2_gate", (D_BYTE, D_BYTE), "proj"),
    ("router_q", (D_BYTE, ROUTER_DIM), "router"),
    ("router_k", (D_BYTE, ROUTER_DIM), "router"),
    ("chunk_proj", (D_BYTE, D_CONCEPT), "proj"),
    ("core_conv_dw", (D_CONCEPT, 3), "core"),
    ("core_conv_pw", (D_CONCEPT, D_CONCEPT), "core"),
    ("core_gate", (D_CONCEPT, D_CONCEPT), "core"),
    ("core_out_proj", (D_CONCEPT, D_CONCEPT), "core"),
]
SCALE_GROUPS = ("embed", "router", "proj", "core")
PLHN_PAYLOAD_BYTES = sum(r * c for _, (r, c), _ in PLHN_LAYOUT)
PLHN_FILE_BYTES = PLHN_HEADER.size + PLHN_PAYLOAD_BYTES
assert PLHN_PAYLOAD_BYTES == 94976 and PLHN_FILE_BYTES == 95008

ENGR_MAGIC = 0x454E4752
ENGR_VERSION = 1
ENGR_HEADER = struct.Struct("<IIIIf")
ENGR_SECTOR = 512
ENGR_SLOT = struct.Struct("<Q24b32s")
assert ENGR_SLOT.size == 64
ENGR_FILE_BYTES = ENGR_SECTOR + ENGRAM_SLOTS * ENGR_SLOT.size

ICPT_MAGIC = 0x49435054
ICPT_VERSION = 1
ICPT_HEADER = struct.Struct("<IIII")
ICPT_RECORD = struct.Struct("<QII48s128b")
assert ICPT_HEADER.size == 16 and ICPT_RECORD.size == 192

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
RECIP = [0] + [(65536 + n // 2) // n for n in range(1, MAX_SPAN + 1)]


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def u8len(b: int) -> int:
    if b < 0x80:
        return 1
    if b >= 0xF0:
        return 4
    if b >= 0xE0:
        return 3
    if b >= 0xC0:
        return 2
    return 1


def is_cont(b: int) -> bool:
    return (b & 0xC0) == 0x80


def text_bytes(text, max_bytes: int = MAX_BYTES) -> bytes:
    data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return data[:max_bytes]


def scale_to_shift(scale: float) -> int:
    s = -math.log2(scale)
    shift = int(round(s))
    if abs(s - shift) > 1e-6:
        raise ValueError(f"scale {scale!r} is not an exact power of two")
    return shift


def sh(a: int, s: int) -> int:
    if s > 0:
        return (a + (1 << (s - 1))) >> s
    if s < 0:
        return a << (-s)
    return a


def sat8(a: int) -> int:
    return -127 if a < -127 else (127 if a > 127 else a)


def rq(acc: int, s: int) -> int:
    return sat8(sh(acc, s))


def rdiv(a: int, n: int) -> int:
    return (a + n // 2) // n if a >= 0 else -((-a + n // 2) // n)


def l2_int8(s):
    """Exact integer L2 normalisation of int sums to int8 at scale 127."""
    n2 = sum(int(v) * int(v) for v in s)
    n = math.isqrt(n2)
    if n == 0:
        return [0] * len(s)
    return [sat8(rdiv(127 * int(v), n)) for v in s]


def boundary_step(t: int, b: int, length: int, dot_le0: bool) -> bool:
    """H-Net router decision for byte t (length = bytes already in the open chunk)."""
    if t == 0 or length >= MAX_SPAN:
        return True
    if is_cont(b):
        return False
    return length + u8len(b) > MAX_SPAN or dot_le0


# ----------------------------------------------------------------------------------------------
# Word-boundary supervision targets for the router (training only; nothing here is exported)
# ----------------------------------------------------------------------------------------------

BND_NONE, BND_START, BND_INSIDE = 0, 1, 2  # per-byte label: continuation byte / word start / inside
_HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
# non-Han text: a word keeps ONE leading space (" bank" is one chunk, like the space-prefixed
# pieces of byte-level BPE), whitespace runs and every other symbol are their own words
_NONHAN_TOK = re.compile(r" ?[^\W_]+(?:['’][^\W_]+)*|\s+|.", re.S)


def word_boundary_labels(text: str, cut=None) -> bytes:
    """One label per UTF-8 byte of ``text``: ``BND_START`` on the lead byte of the first character
    of every word, ``BND_INSIDE`` on the lead byte of every later character of a word and
    ``BND_NONE`` on continuation bytes (the router never decides there).  Han runs are split into
    words by ``cut`` (jieba ``lcut``; ``None`` = the whole run is one word, used for dictionary
    headwords); other text by ``_NONHAN_TOK``."""
    out = bytearray()

    def emit(word: str):
        for i, ch in enumerate(word):
            n = len(ch.encode("utf-8"))
            out.append(BND_START if i == 0 else BND_INSIDE)
            out.extend(b"\0" * (n - 1))

    pos = 0
    for m in _HAN_RUN.finditer(text):
        for w in _NONHAN_TOK.findall(text[pos:m.start()]):
            emit(w)
        for w in (cut(m.group(0)) if cut is not None else [m.group(0)]):
            emit(w)
        pos = m.end()
    for w in _NONHAN_TOK.findall(text[pos:]):
        emit(w)
    return bytes(out)


# ----------------------------------------------------------------------------------------------
# Binary parsing
# ----------------------------------------------------------------------------------------------


def parse_plhn(blob: bytes) -> dict:
    if len(blob) != PLHN_FILE_BYTES:
        raise ValueError(f"platonic-hnet.bin must be {PLHN_FILE_BYTES} bytes, got {len(blob)}")
    (magic, version, d_byte, d_concept, n_prelude, n_core, router_dim, log2, s_embed, s_router,
     s_proj, s_core, _reserved) = PLHN_HEADER.unpack_from(blob, 0)
    if (magic, version, d_byte, d_concept, n_prelude, n_core, router_dim, log2) != (
            PLHN_MAGIC, PLHN_VERSION, D_BYTE, D_CONCEPT, N_PRELUDE, N_CORE, ROUTER_DIM, ENGRAM_LOG2):
        raise ValueError("bad PLHN header")
    shifts = {"embed": scale_to_shift(s_embed), "router": scale_to_shift(s_router),
              "proj": scale_to_shift(s_proj), "core": scale_to_shift(s_core)}
    tensors = {}
    off = PLHN_HEADER.size
    for name, (r, c), _ in PLHN_LAYOUT:
        n = r * c
        tensors[name] = struct.unpack_from(f"<{n}b", blob, off)
        off += n
    return {"shifts": shifts, "tensors": tensors,
            "scales": {"embed": s_embed, "router": s_router, "proj": s_proj, "core": s_core}}


def parse_engr_header(blob: bytes) -> dict:
    magic, version, num_slots, slot_bytes, scale = ENGR_HEADER.unpack_from(blob, 0)
    if magic != ENGR_MAGIC or version != ENGR_VERSION or slot_bytes != 64:
        raise ValueError("bad ENGR header")
    if len(blob) < ENGR_SECTOR + num_slots * slot_bytes:
        raise ValueError("truncated engram.bin")
    return {"num_slots": num_slots, "shift": scale_to_shift(scale), "scale": scale}


def engram_delta(engram_blob: bytes | None, hdr: dict | None, span: bytes):
    if engram_blob is None:
        return None
    slot = fnv1a64(span) % hdr["num_slots"]
    off = ENGR_SECTOR + slot * 64 + 8
    return struct.unpack_from("<24b", engram_blob, off)


def engram_bigram_delta(engram_blob: bytes | None, hdr: dict | None, span: bytes):
    """Bigram-span lookup: the slot's 24-d delta only when its stored span_hash equals FNV1a64(span)."""
    if engram_blob is None:
        return None
    h = fnv1a64(span)
    off = ENGR_SECTOR + (h % hdr["num_slots"]) * 64
    if h == 0 or struct.unpack_from("<Q", engram_blob, off)[0] != h:
        return None
    return struct.unpack_from("<24b", engram_blob, off + 8)


# ----------------------------------------------------------------------------------------------
# Pure-Python integer reference (stdlib only)
# ----------------------------------------------------------------------------------------------


def _cols(flat, rows: int, cols: int):
    return [flat[j::cols] for j in range(cols)]


def _matvec(x, cols, s):
    return [rq(sum(map(mul, x, c)), s) for c in cols]


def _block_py(xs, gate_cols, dw, pw_cols, s, out_cols=None):
    d = len(gate_cols)
    zero = [0] * d
    ys, gss = [], []
    for x in xs:
        gs = [min(max(g + 64, 0), 128) for g in _matvec(x, gate_cols, s)]
        ys.append([(a * b + 64) >> 7 for a, b in zip(x, gs)])
        gss.append(gs)
    out = []
    for t, x in enumerate(xs):
        y2 = ys[t - 2] if t >= 2 else zero
        y1 = ys[t - 1] if t >= 1 else zero
        y0 = ys[t]
        z = [rq(dw[3 * c] * y2[c] + dw[3 * c + 1] * y1[c] + dw[3 * c + 2] * y0[c], s) for c in range(d)]
        o = _matvec(z, pw_cols, s)
        o2 = [(a * b + 64) >> 7 for a, b in zip(o, gss[t])]
        r = _matvec(o2, out_cols, s) if out_cols is not None else o2
        out.append([sat8(a + b) for a, b in zip(x, r)])
    return out


class PlhnIntRef:
    """Parsed PLHN (+ optional ENGR) for repeated pure-Python integer inference."""

    def __init__(self, model_blob: bytes, engram_blob: bytes | None = None, bigram: bool = True):
        m = parse_plhn(model_blob)
        self.sh = m["shifts"]
        t = m["tensors"]
        self.embed = t["byte_embed"]
        self.blocks = []
        for i in (1, 2):
            self.blocks.append((_cols(t[f"conv{i}_gate"], D_BYTE, D_BYTE), t[f"conv{i}_dw"],
                                _cols(t[f"conv{i}_pw"], D_BYTE, D_BYTE)))
        self.q_cols = _cols(t["router_q"], D_BYTE, ROUTER_DIM)
        self.k_cols = _cols(t["router_k"], D_BYTE, ROUTER_DIM)
        self.proj_cols = _cols(t["chunk_proj"], D_BYTE, D_CONCEPT)
        self.core = (_cols(t["core_gate"], D_CONCEPT, D_CONCEPT), t["core_conv_dw"],
                     _cols(t["core_conv_pw"], D_CONCEPT, D_CONCEPT),
                     _cols(t["core_out_proj"], D_CONCEPT, D_CONCEPT))
        self.engram = engram_blob
        self.engram_hdr = parse_engr_header(engram_blob) if engram_blob is not None else None
        self.bigram = bigram

    def spans(self, text):
        data = text_bytes(text)
        return data, self._segment(data, self._encode_bytes(data))

    def _encode_bytes(self, data: bytes):
        se, sp = self.sh["embed"], self.sh["proj"]
        h = [[sat8(sh(self.embed[b * D_BYTE + d], se - ACT_SHIFT)) for d in range(D_BYTE)] for b in data]
        for gate_cols, dw, pw_cols in self.blocks:
            h = _block_py(h, gate_cols, dw, pw_cols, sp)
        return h

    def _segment(self, data: bytes, h):
        sr = self.sh["router"]
        q = [_matvec(x, self.q_cols, sr) for x in h]
        k = [_matvec(x, self.k_cols, sr) for x in h]
        spans, start, length = [], 0, 0
        for t, b in enumerate(data):
            dot_le0 = t > 0 and sum(map(mul, q[t], k[t - 1])) <= 0
            if boundary_step(t, b, length, dot_le0):
                if t > 0:
                    spans.append((start, t))
                start, length = t, 0
            length += 1
        if data:
            spans.append((start, len(data)))
        return spans

    def forward(self, text) -> list:
        data = text_bytes(text)
        if not data:
            return [0] * D_CONCEPT
        h = self._encode_bytes(data)
        spans = self._segment(data, h)
        sp = self.sh["proj"]
        xs = []
        for ci, (a, e) in enumerate(spans):
            n = e - a
            m = [sat8((sum(h[t][d] for t in range(a, e)) * RECIP[n] + 32768) >> 16) for d in range(D_BYTE)]
            x = _matvec(m, self.proj_cols, sp)
            delta = engram_delta(self.engram, self.engram_hdr, data[a:e])
            if delta is not None:
                gsh = self.engram_hdr["shift"] - ACT_SHIFT
                for d in range(ENGRAM_DIMS):
                    x[d] = sat8(x[d] + sat8(sh(delta[d], gsh)))
                if self.bigram and ci >= 1 and e - spans[ci - 1][0] <= BIGRAM_MAX_BYTES:
                    bd = engram_bigram_delta(self.engram, self.engram_hdr, data[spans[ci - 1][0]:e])
                    if bd is not None:
                        for d in range(ENGRAM_DIMS):
                            x[d] = sat8(x[d] + sat8(sh(bd[d] >> 1, gsh)))
            xs.append(x)
        gate_cols, dw, pw_cols, out_cols = self.core
        for _ in range(N_CORE):
            xs = _block_py(xs, gate_cols, dw, pw_cols, self.sh["core"], out_cols)
        s = [sum(x[d] for x in xs) for d in range(D_CONCEPT)]
        return l2_int8(s)


def plhn_int_forward(model, engram, text) -> list:
    """Pure-Python reference. ``model``: PLHN bytes or PlhnIntRef; ``engram``: ENGR bytes or None."""
    ref = model if isinstance(model, PlhnIntRef) else PlhnIntRef(model, engram)
    return ref.forward(text)


# ----------------------------------------------------------------------------------------------
# Numpy integer reference (fast; identical arithmetic)
# ----------------------------------------------------------------------------------------------


class PlhnIntNp:
    def __init__(self, model_blob: bytes, engram_blob: bytes | None = None, bigram: bool = True):
        import numpy as np

        self.np = np
        m = parse_plhn(model_blob)
        self.sh = m["shifts"]
        self.w = {name: np.asarray(m["tensors"][name], dtype=np.int64).reshape(shape)
                  for name, shape, _ in PLHN_LAYOUT}
        self.engram_hdr = parse_engr_header(engram_blob) if engram_blob is not None else None
        if engram_blob is not None:
            raw = np.frombuffer(engram_blob, dtype=np.uint8, count=self.engram_hdr["num_slots"] * 64,
                                offset=ENGR_SECTOR).reshape(-1, 64)
            self.engram_delta = raw[:, 8:32].view(np.int8).astype(np.int64)
            self.engram_hash = raw[:, 0:8].copy().view("<u8").ravel()
        else:
            self.engram_delta = None
            self.engram_hash = None
        self.bigram = bigram

    def _sh(self, a, s):
        if s > 0:
            return (a + (1 << (s - 1))) >> s
        if s < 0:
            return a << (-s)
        return a

    def _rq(self, a, s):
        return self.np.clip(self._sh(a, s), -127, 127)

    def _block(self, x, gate, dw, pw, s, out=None):
        np = self.np
        gs = np.clip(self._rq(x @ gate, s) + 64, 0, 128)
        y = (x * gs + 64) >> 7
        yp = np.concatenate([np.zeros((2, x.shape[1]), np.int64), y], 0)
        z = self._rq(dw[:, 0] * yp[:-2] + dw[:, 1] * yp[1:-1] + dw[:, 2] * yp[2:], s)
        o2 = (self._rq(z @ pw, s) * gs + 64) >> 7
        r = self._rq(o2 @ out, s) if out is not None else o2
        return np.clip(x + r, -127, 127)

    def encode_bytes(self, data: bytes):
        np, w = self.np, self.w
        idx = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        h = np.clip(self._sh(w["byte_embed"][idx], self.sh["embed"] - ACT_SHIFT), -127, 127)
        for i in (1, 2):
            h = self._block(h, w[f"conv{i}_gate"], w[f"conv{i}_dw"], w[f"conv{i}_pw"], self.sh["proj"])
        return h

    def segment(self, data: bytes, h):
        q = self._rq(h @ self.w["router_q"], self.sh["router"])
        k = self._rq(h @ self.w["router_k"], self.sh["router"])
        dots = (q[1:] * k[:-1]).sum(1)
        spans, start, length = [], 0, 0
        for t, b in enumerate(data):
            if boundary_step(t, b, length, t > 0 and dots[t - 1] <= 0):
                if t > 0:
                    spans.append((start, t))
                start, length = t, 0
            length += 1
        if data:
            spans.append((start, len(data)))
        return spans

    def chunk_states(self, text):
        np, w = self.np, self.w
        data = text_bytes(text)
        if not data:
            return data, [], np.zeros((0, D_CONCEPT), np.int64)
        h = self.encode_bytes(data)
        spans = self.segment(data, h)
        m = np.stack([np.clip((h[a:e].sum(0) * RECIP[e - a] + 32768) >> 16, -127, 127) for a, e in spans])
        x = self._rq(m @ w["chunk_proj"], self.sh["proj"])
        if self.engram_delta is not None:
            nslots = self.engram_hdr["num_slots"]
            gsh = self.engram_hdr["shift"] - ACT_SHIFT
            slots = [fnv1a64(data[a:e]) % nslots for a, e in spans]
            delta = np.clip(self._sh(self.engram_delta[slots], gsh), -127, 127)
            x[:, :ENGRAM_DIMS] = np.clip(x[:, :ENGRAM_DIMS] + delta, -127, 127)
            if self.bigram:
                for c in range(1, len(spans)):
                    a, e = spans[c - 1][0], spans[c][1]
                    if e - a > BIGRAM_MAX_BYTES:
                        continue
                    h2 = fnv1a64(data[a:e])
                    slot = h2 % nslots
                    if h2 == 0 or int(self.engram_hash[slot]) != h2:
                        continue
                    bd = np.clip(self._sh(self.engram_delta[slot] >> 1, gsh), -127, 127)
                    x[c, :ENGRAM_DIMS] = np.clip(x[c, :ENGRAM_DIMS] + bd, -127, 127)
        for _ in range(N_CORE):
            x = self._block(x, w["core_gate"], w["core_conv_dw"], w["core_conv_pw"], self.sh["core"],
                            w["core_out_proj"])
        return data, spans, x

    def forward(self, text):
        _, _, x = self.chunk_states(text)
        if len(x) == 0:
            return [0] * D_CONCEPT
        return l2_int8([int(v) for v in x.sum(0)])


# ----------------------------------------------------------------------------------------------
# Torch training model (QAT path is bit-exact with the integer reference)
# ----------------------------------------------------------------------------------------------

try:
    import numpy as _np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - integer references stay usable without torch
    torch = None

if torch is not None:

    ACT = float(1 << ACT_SHIFT)
    QMAX = 127.0

    def ste(x, xq):
        """Forward value is exactly xq (xq + 0), gradient flows to x."""
        return xq.detach() + (x - x.detach())

    def pow2_scale(maxabs: float) -> float:
        """Smallest power-of-two scale with maxabs/scale <= 127."""
        if maxabs <= 0:
            return 2.0 ** -ACT_SHIFT
        return 2.0 ** math.ceil(math.log2(maxabs / QMAX))

    def fq_weight(w, scale: float):
        q = torch.clamp(torch.floor(w / scale + 0.5), -QMAX, QMAX) * scale
        return ste(w, q)

    def fq_act(x):
        q = torch.clamp(torch.floor(x * ACT + 0.5), -QMAX, QMAX) / ACT
        return ste(x, q)

    def hard_gate(g):
        return torch.clamp(g * 0.25 + 0.5, 0.0, 1.0)

    def causal_dw3(y, dw):
        """y (B,T,D), dw (D,3) taps [x_{t-2}, x_{t-1}, x_t]."""
        yp = F.pad(y, (0, 0, 2, 0))
        return dw[:, 0] * yp[:, :-2] + dw[:, 1] * yp[:, 1:-1] + dw[:, 2] * yp[:, 2:]

    class GatedShortConv(nn.Module):
        """LFM2-style double-gated ShortConv(k=3); optional output projection (core)."""

        def __init__(self, d: int, out_proj: bool):
            super().__init__()
            self.dw = nn.Parameter(torch.randn(d, 3) * 0.3)
            self.pw = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
            self.gate = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
            self.out = nn.Parameter(torch.randn(d, d) / math.sqrt(d) * 0.5) if out_proj else None

        def forward(self, x, wq, act):
            out = wq(self.out) if self.out is not None else None
            return _gsc(x, wq(self.gate), wq(self.dw), wq(self.pw), out, act is fq_act)

    def _gsc(x, gate, dw, pw, out, quant: bool):
        """GatedShortConv body on (already fake-quantised) weights; ``quant`` -> int8 activations."""
        act = fq_act if quant else _identity
        g = act(x @ gate)
        gs = hard_gate(g)
        y = act(x * gs)
        z = act(causal_dw3(y, dw))
        o2 = act(act(z @ pw) * gs)
        r = act(o2 @ out) if out is not None else o2
        return act(x + r)

    def _identity(x):
        return x

    class PlatonicEngramHNet(nn.Module):
        """Byte encoder + H-Net router + Engram + Ouro core; training heads are amputated at export."""

        def __init__(self, use_engram: bool = True, target_ratio: float = 6.0, use_bigram: bool = False):
            super().__init__()
            self.use_engram = use_engram
            self.target_ratio = target_ratio
            self.quant = False
            self.byte_embed = nn.Parameter(torch.randn(256, D_BYTE) * 0.5)
            self.conv1 = GatedShortConv(D_BYTE, out_proj=False)
            self.conv2 = GatedShortConv(D_BYTE, out_proj=False)
            self.router_q = nn.Parameter(torch.randn(D_BYTE, ROUTER_DIM) / math.sqrt(D_BYTE))
            self.router_k = nn.Parameter(torch.randn(D_BYTE, ROUTER_DIM) / math.sqrt(D_BYTE))
            self.chunk_proj = nn.Parameter(torch.randn(D_BYTE, D_CONCEPT) / math.sqrt(D_BYTE))
            self.core = GatedShortConv(D_CONCEPT, out_proj=True)
            self.engram = nn.Parameter(torch.zeros(ENGRAM_SLOTS, ENGRAM_DIMS)) if use_engram else None
            if self.engram is not None:
                nn.init.zeros_(self.engram)  # unseen slots contribute exactly 0 (no noise on novel spans)
            self.engram_drop = 0.0  # training-time Engram slot dropout (plain attribute, not exported)
            # Bigram-span Engram: slot -> stored span_hash (int64 bit pattern of the uint64 FNV hash);
            # the bigram delta is applied only on an exact hash match (same table as engram.bin)
            self.use_bigram = bool(use_bigram and use_engram)
            self.register_buffer("engram_hash", torch.zeros(ENGRAM_SLOTS, dtype=torch.long) if use_engram else None)
            # ---- training heads (not exported) ----
            self.dec_w1 = nn.Linear(D_CONCEPT, D_BYTE)
            self.dec_pos = nn.Parameter(torch.randn(MAX_SPAN, D_BYTE) * 0.1)
            self.dec_w2 = nn.Linear(D_BYTE, 256)
            self.sense_log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
            self.nce_log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

        # ---- quantisation helpers ----
        def group_params(self) -> dict:
            return {
                "embed": [self.byte_embed],
                "router": [self.router_q, self.router_k],
                "proj": [self.conv1.dw, self.conv1.pw, self.conv1.gate, self.conv2.dw, self.conv2.pw,
                         self.conv2.gate, self.chunk_proj],
                "core": [self.core.dw, self.core.pw, self.core.gate, self.core.out],
            }

        def group_scales(self) -> dict:
            """Power-of-two scale per group; one device->host sync for all groups."""
            groups = list(self.group_params().items())
            maxes = [torch.stack([p.detach().abs().max() for p in ps]).max() for _, ps in groups]
            names = [g for g, _ in groups]
            if self.engram is not None:
                maxes.append(self.engram.detach().abs().max())
                names.append("engram")
            vals = torch.stack(maxes).double().cpu().tolist()
            return {g: pow2_scale(v) for g, v in zip(names, vals)}

        def _wq_fns(self):
            if not self.quant:
                self._scales = None
                return {g: _identity for g in (*SCALE_GROUPS, "engram")}, _identity
            scales = self.group_scales()
            self._scales = scales
            return {g: (lambda w, s=s: fq_weight(w, s)) for g, s in scales.items()}, fq_act

        def set_engram_hashes(self, hashes) -> None:
            """uint64 numpy array / int list (ENGRAM_SLOTS) -> the int64 ``engram_hash`` buffer."""
            arr = _np.asarray(hashes, dtype=_np.uint64).view(_np.int64)
            self.engram_hash.copy_(torch.from_numpy(arr.copy()).to(self.engram_hash.device))

        # ---- encoder ----
        def encode(self, texts, device=None) -> dict:
            X, lens = pack_texts(texts, device or self.byte_embed.device)
            return self.encode_tensor(X, lens)

        def encode_tensor(self, X, lens) -> dict:
            """X (B,L) int64 bytes (0-padded), lens (B,) int64, both on the model device.
            Segmentation, Engram hashing and pooling all run as device tensor ops (one host sync
            for the chunk count)."""
            device = X.device
            B, L = X.shape
            valid = torch.arange(L, device=device)[None, :] < lens[:, None]
            wq, act = self._wq_fns()

            # F.embedding: same values as table[X], but its backward is a dense scatter instead of
            # the (very slow, heavily duplicated-index) index_put_ accumulate path
            h = act(F.embedding(X, wq["embed"](self.byte_embed)))
            h = self.conv1(h, wq["proj"], act)
            h = self.conv2(h, wq["proj"], act)

            q = act(h @ wq["router"](self.router_q))
            k = act(h @ wq["router"](self.router_k))
            dots = (q[:, 1:] * k[:, :-1]).sum(-1)
            cos = dots / (q[:, 1:].norm(dim=-1) * k[:, :-1].norm(dim=-1) + 1e-6)
            p = torch.cat([torch.ones(B, 1, device=device), 0.5 * (1.0 - cos)], 1)  # (B,L)

            dle = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), dots.detach() <= 0], 1)
            bnd, cid, n_chunks, slots, starts_t, C = segment_tensor(X, lens, dle)

            # chunk pooling (exact RECIP mean in quant mode), STE-weighted by router confidence
            sums = torch.zeros(B, C + 1, D_BYTE, device=device, dtype=h.dtype)
            sums.scatter_add_(1, cid.unsqueeze(-1).expand(-1, -1, D_BYTE), h)
            cnt = torch.zeros(B, C + 1, device=device, dtype=h.dtype)
            cnt.scatter_add_(1, cid, valid.to(h.dtype))
            sums, cnt = sums[:, :C], cnt[:, :C]
            mean = sums / cnt.clamp(min=1).unsqueeze(-1)
            if self.quant:
                recip = torch.tensor(RECIP, dtype=torch.float64, device=device)[cnt.long().clamp(max=MAX_SPAN)]
                exact = torch.floor((sums.double() * ACT) * recip.unsqueeze(-1) / 65536.0 + 0.5)
                exact = torch.clamp(exact, -QMAX, QMAX).to(h.dtype) / ACT
                m = ste(mean, exact)
            else:
                m = mean
            p_start = torch.gather(p, 1, starts_t)
            m = m * (1.0 + (p_start - p_start.detach())).unsqueeze(-1)

            x = act(m @ wq["proj"](self.chunk_proj))
            if self.engram is not None and self.use_engram:
                delta = act(wq["engram"](F.embedding(slots, self.engram)))  # elementwise fq, table-wide scale
                keep = None
                if self.training and self.engram_drop > 0:
                    keep = (torch.rand(B, C, device=device) >= self.engram_drop).unsqueeze(-1).to(delta.dtype)
                    delta = delta * keep
                x24 = act(x[..., :ENGRAM_DIMS] + delta)
                if self.use_bigram and C > 1:
                    h2, ok2 = bigram_hashes(X, lens, starts_t, n_chunks)
                    slot2 = h2 & (ENGRAM_SLOTS - 1)
                    hit = ok2 & (h2 != 0) & (self.engram_hash[slot2] == h2)
                    e2 = wq["engram"](F.embedding(slot2, self.engram))
                    if self.quant:
                        es = self._scales["engram"]
                        half = torch.floor(torch.round(e2.detach() / es) * 0.5) * es  # (k >> 1) * scale
                        e2 = ste(e2 * 0.5, half)
                    else:
                        e2 = e2 * 0.5
                    d2 = act(e2) * hit.unsqueeze(-1).to(e2.dtype)
                    if keep is not None:
                        d2 = d2 * keep
                    x24 = act(x24 + d2)
                x = torch.cat([x24, x[..., ENGRAM_DIMS:]], -1)
            cvalid = torch.arange(C, device=device)[None, :] < n_chunks[:, None]
            for _ in range(N_CORE):
                x = self.core(x, wq["core"], act)
            xm = x * cvalid.unsqueeze(-1).to(x.dtype)
            s = xm.sum(1)
            z = s / n_chunks.clamp(min=1).to(x.dtype).unsqueeze(-1)
            c = F.normalize(s, dim=-1, eps=1e-8)

            vf = valid.to(h.dtype)
            # (boundary count, router-prob mass, byte count): additive, so length-bucketed
            # sub-batches combine into exactly the single-batch ratio loss
            ratio_stats = torch.stack([(bnd.to(h.dtype) * vf).sum(), (p * vf).sum(), vf.sum()])
            bpc = lens.sum().to(torch.float64) / n_chunks.sum().clamp(min=1).to(torch.float64)
            return {"X": X, "valid": valid, "h": h, "chunk_states": x, "chunk_valid": cvalid,
                    "chunk_id": cid, "starts": starts_t, "n_chunks": n_chunks, "sum": s, "z": z, "c": c,
                    "p": p, "cos": torch.cat([torch.zeros(B, 1, device=device, dtype=cos.dtype), cos], 1),
                    "bnd": bnd, "slots": slots,
                    "ratio_stats": ratio_stats, "ratio_loss": ratio_loss(ratio_stats, self.target_ratio),
                    "bytes_per_chunk": bpc, "lens": lens}

        def byte_losses(self, enc, recon: bool):
            """Next-byte LM through the concept bottleneck (+ optional own-chunk reconstruction)."""
            X, valid, h, xs, cid = enc["X"], enc["valid"], enc["h"], enc["chunk_states"], enc["chunk_id"]
            B, L = X.shape
            C = xs.shape[1]
            xs_pad = torch.cat([torch.zeros_like(xs[:, :1]), xs, torch.zeros_like(xs[:, :1])], 1)
            # next-byte: byte t from the last chunk completed before the chunk holding t-1 began
            # (its end <= t-2, fixed by bytes <= t-1, so no leak of b_t via the boundary) + h_{t-1}
            cid_prev = F.pad(cid, (1, 0))[:, :-1].clamp(max=C)
            prev_c = torch.gather(xs_pad, 1, cid_prev.unsqueeze(-1).expand(-1, -1, D_CONCEPT))
            h_prev = F.pad(h, (0, 0, 1, 0))[:, :-1]
            logits = self.dec_w2(F.gelu(self.dec_w1(prev_c) + h_prev))
            m = valid.clone()
            m[:, 0] = False
            lm = masked_ce(logits, X, m)
            out = {"next_byte": lm}
            if recon:
                own = torch.gather(xs_pad, 1, (cid + 1).clamp(max=C).unsqueeze(-1).expand(-1, -1, D_CONCEPT))
                start = torch.gather(enc["starts"], 1, cid.clamp(max=C - 1))
                off = (torch.arange(L, device=X.device)[None, :] - start).clamp(0, MAX_SPAN - 1)
                rl = self.dec_w2(F.gelu(self.dec_w1(own) + F.embedding(off, self.dec_pos)))
                out["recon"] = masked_ce(rl, X, valid)
            return out

        # ---- exact integer output through the QAT graph ----
        @torch.no_grad()
        def encode_int(self, texts) -> list:
            prev = self.quant
            self.quant = True
            try:
                enc = self.encode(texts)
            finally:
                self.quant = prev
            sums = torch.round(enc["sum"].double() * ACT).long().cpu().tolist()
            return [l2_int8(s) if n > 0 else [0] * D_CONCEPT for s, n in zip(sums, enc["lens"].tolist())]

    def masked_ce(logits, target, mask):
        """Mean cross-entropy over ``mask`` positions without boolean indexing (no host sync, no
        index_put_ backward); 0 when the mask is empty."""
        tgt = target.masked_fill(~mask, -100).reshape(-1)
        s = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt, reduction="sum")
        return s / mask.sum().clamp(min=1)

    def ratio_loss(stats, N: float):
        """H-Net ratio loss from ``ratio_stats`` (summed over any number of sub-batches)."""
        F_ratio = stats[0] / stats[2].clamp(min=1)
        G = stats[1] / stats[2].clamp(min=1)
        return N / (N - 1.0) * ((N - 1.0) * F_ratio * G + (1.0 - F_ratio) * (1.0 - G))

    def boundary_stats(enc, labels, margin: float = 0.2):
        """Supervised word-boundary hinge on the router cosine (the device fires a boundary at a
        UTF-8 lead byte iff q_t.k_{t-1} <= 0): word starts want cos <= -margin, lead bytes inside a
        word want cos >= +margin.  ``labels`` (B, >=L) uint8 ``word_boundary_labels`` aligned with
        ``enc["X"]``.  Returns additive stats (hinge sum / count for starts, then for inside bytes,
        then correct router decisions / supervised bytes) so length-bucketed sub-batches combine
        exactly; byte 0 is unsupervised (always a boundary)."""
        cos, valid = enc["cos"], enc["valid"]
        lab = labels[:, :cos.shape[1]].to(torch.long)
        sup = valid.clone()
        sup[:, 0] = False
        start = (lab == BND_START) & sup
        inside = (lab == BND_INSIDE) & sup
        fs, fi = start.to(cos.dtype), inside.to(cos.dtype)
        with torch.no_grad():
            fire = cos <= 0
            correct = ((fire & start) | (~fire & inside)).to(cos.dtype).sum()
        return torch.stack([(F.relu(cos + margin) * fs).sum(), fs.sum(), (F.relu(margin - cos) * fi).sum(), fi.sum(),
                            correct, fs.sum() + fi.sum()])

    def boundary_loss(stats):
        """Class-balanced mean hinge from (summed) ``boundary_stats``."""
        return 0.5 * (stats[0] / stats[1].clamp(min=1) + stats[2] / stats[3].clamp(min=1))

    def pack_texts(texts, device):
        """str/bytes list -> (X (B,L) int64 zero-padded bytes, lens (B,) int64) on ``device``."""
        datas = [text_bytes(t) for t in texts]
        L = max(1, max((len(d) for d in datas), default=1))
        arr = _np.zeros((len(datas), L), dtype=_np.uint8)
        for i, d in enumerate(datas):
            arr[i, :len(d)] = _np.frombuffer(d, dtype=_np.uint8)
        lens = torch.tensor([len(d) for d in datas], dtype=torch.long)
        return torch.from_numpy(arr).to(device).long(), lens.to(device)

    _FNV_OFFSET_I64 = FNV_OFFSET - (1 << 64)  # same 64-bit pattern as int64 (two's-complement wrap)

    def segment_tensor(X, lens, dle):
        """H-Net boundary pass (exactly ``boundary_step``), fully on X.device with no per-byte loop.

        A chunk opened at byte s closes at the first t' in (s, s+12] with
        len>=12 | (lead(t') & (len+u8len(t')>12 | dle[t'])) | t'>=n, len = t'-s.  That defines a
        successor f(s); the chunk starts are the orbit 0, f(0), f(f(0)), ..., computed by pointer
        doubling (log2 L gathers).  Engram slots are FNV-1a 64 of each span in int64 (wrap-around
        multiply == uint64), masked to the 2^16 slots.
        Returns bnd (B,L) bool, chunk_id (B,L) (invalid -> C), n_chunks (B,), slots (B,C),
        starts (B,C) (0 for pad chunks) and C = max(1, max n_chunks) as a Python int."""
        B, L = X.shape
        lens = lens.to(torch.long)
        # doubling depth for L <= MAX_BYTES; extra levels are no-ops because positions < L never
        # set the higher bits
        nbits = max(1, (max(L, MAX_BYTES) - 1).bit_length())
        bnd, n_chunks = _seg_bounds(X, lens, dle, nbits)
        C = max(1, int(n_chunks.max())) if B else 1
        chunk_id, slots, starts = _seg_spans(X, lens, bnd, n_chunks, C)
        return bnd, chunk_id, n_chunks, slots, starts, C

    def _seg_bounds(X, lens, dle, nbits: int):
        B, L = X.shape
        dev = X.device
        K = MAX_SPAN
        pos = torch.arange(L, device=dev)
        lead = ((X & 0xC0) != 0x80).to(torch.uint8)
        u8 = torch.where(X < 0x80, 1, torch.where(X >= 0xF0, 4, torch.where(X >= 0xE0, 3,
                                                                             torch.where(X >= 0xC0, 2, 1))))

        def win(t):  # (B,L) -> (B,L,K): values at t' = s+1 .. s+K (zero beyond L)
            return F.pad(t, (0, K))[:, 1:].unfold(1, K, 1)

        ks = torch.arange(1, K + 1, device=dev)
        close = ((ks >= K)[None, None, :]
                 | (win(lead).bool() & ((ks + win(u8) > K) | win(dle.to(torch.uint8)).bool()))
                 | ((pos[:, None] + ks[None, :])[None] >= lens[:, None, None]))
        e = pos[None, :] + 1 + close.to(torch.uint8).argmax(-1)  # first closing t' (k=K always closes)
        f = torch.where(pos[None, :] < lens[:, None], torch.minimum(e, lens[:, None]),
                        torch.full_like(e, L))
        f = torch.cat([f, torch.full((B, 1), L, dtype=f.dtype, device=dev)], 1)  # node L = sink
        jumps = [f]
        for _ in range(nbits - 1):
            jumps.append(torch.gather(jumps[-1], 1, jumps[-1]))
        P = torch.zeros(B, L, dtype=f.dtype, device=dev)  # P[:, n] = f^n(0)
        for b, J in enumerate(jumps):
            bit = ((pos >> b) & 1).bool()[None, :]
            P = torch.where(bit, torch.gather(J, 1, P), P)
        valid = pos[None, :] < lens[:, None]
        bnd = torch.zeros(B, L + 1, dtype=torch.bool, device=dev).scatter(1, P, True)[:, :L] & valid
        return bnd, bnd.sum(1)

    def _seg_spans(X, lens, bnd, n_chunks, C: int):
        B, L = X.shape
        dev = X.device
        pos = torch.arange(L, device=dev)
        valid = pos[None, :] < lens[:, None]
        chunk_id = torch.where(valid, bnd.long().cumsum(1) - 1, C)
        dump = torch.where(bnd, chunk_id, C)
        starts = torch.zeros(B, C + 1, dtype=torch.long, device=dev).scatter(
            1, dump, pos.expand(B, L).contiguous())[:, :C]
        j = torch.arange(C, device=dev)[None, :]
        nxt = torch.cat([starts[:, 1:], torch.zeros(B, 1, dtype=torch.long, device=dev)], 1)
        ends = torch.where(j + 1 < n_chunks[:, None], nxt, lens[:, None])
        span_len = torch.where(j < n_chunks[:, None], ends - starts, 0)
        h = torch.full((B, C), _FNV_OFFSET_I64, dtype=torch.long, device=dev)
        for kk in range(MAX_SPAN):
            m = kk < span_len
            byte = torch.gather(X, 1, (starts + kk).clamp(max=L - 1))
            h = torch.where(m, (h ^ byte) * FNV_PRIME, h)
        slots = torch.where(span_len > 0, h & (ENGRAM_SLOTS - 1), 0)
        return chunk_id, slots, starts

    def bigram_hashes(X, lens, starts, n_chunks):
        """Bigram-span Engram keys: for chunk c >= 1, FNV1a64 (int64 wrap) of bytes
        [start(c-1), end(c)) when that span is <= BIGRAM_MAX_BYTES.  Returns (h2 (B,C), ok (B,C))."""
        B, L = X.shape
        C = starts.shape[1]
        dev = X.device
        j = torch.arange(C, device=dev)[None, :]
        nxt = torch.cat([starts[:, 1:], torch.zeros(B, 1, dtype=torch.long, device=dev)], 1)
        ends = torch.where(j + 1 < n_chunks[:, None], nxt, lens[:, None])
        a = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=dev), starts[:, :-1]], 1)  # start(c-1)
        n2 = ends - a
        ok = (j >= 1) & (j < n_chunks[:, None]) & (n2 <= BIGRAM_MAX_BYTES)
        h = torch.full((B, C), _FNV_OFFSET_I64, dtype=torch.long, device=dev)
        for kk in range(BIGRAM_MAX_BYTES):
            m = ok & (kk < n2)
            byte = torch.gather(X, 1, (a + kk).clamp(max=L - 1))
            h = torch.where(m, (h ^ byte) * FNV_PRIME, h)
        return torch.where(ok, h, 0), ok

    # ------------------------------------------------------------------------------------------
    # LeJEPA good-view objective + contrastive / sense-ranking heads
    # ------------------------------------------------------------------------------------------

    def sigreg(z, num_slices: int = 256, t_max: float = 3.0, n_t: int = 17, generator=None):
        """SIGReg: sliced Epps-Pulley test against N(0,1) + covariance whitening ||Cov - I||_F^2 / D."""
        n, d = z.shape
        a = torch.randn(d, num_slices, device=z.device, generator=generator)
        a = a / a.norm(dim=0, keepdim=True)
        proj = z @ a  # (n, M)
        t = torch.linspace(0.0, t_max, n_t, device=z.device)
        tx = proj.unsqueeze(-1) * t  # (n, M, T)
        ecf_re = torch.cos(tx).mean(0)
        ecf_im = torch.sin(tx).mean(0)
        target = torch.exp(-0.5 * t * t)
        err = (ecf_re - target) ** 2 + ecf_im ** 2  # (M, T)
        w = torch.exp(-0.5 * t * t)
        ep = torch.trapezoid(err * w, t, dim=-1).mean() * n
        zc = z - z.mean(0, keepdim=True)
        cov = zc.T @ zc / max(1, n - 1)
        white = ((cov - torch.eye(d, device=z.device)) ** 2).sum() / d
        return ep + white

    def lejepa_loss(z1, z2, lam: float = 0.05):
        inv = ((z1 - z2) ** 2).sum(-1).mean()
        reg = sigreg(torch.cat([z1, z2], 0))
        return (1.0 - lam) * inv + lam * reg, inv, reg

    def info_nce(c1, c2, log_tau):
        logits = c1 @ c2.T / log_tau.exp().clamp(min=1e-3)
        y = torch.arange(len(c1), device=c1.device)
        return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))

    def sense_rank_loss(ctx, senses, log_tau, eps: float = 0.1, pos_mask=None, margin: float = 0.0):
        """In-context sense ranking: context vs in-batch senses (incl. same-headword sibling senses
        when the batch is term-grouped). Label-smoothed soft targets over the positives
        (``pos_mask[i, j]`` = sense j is a correct sense for context i, e.g. identical definition),
        plus a hinge margin between the positive and the hardest negative cosine."""
        cos = ctx @ senses.T
        logits = cos / log_tau.exp().clamp(min=1e-3)
        n = len(ctx)
        if pos_mask is None:
            pos_mask = torch.eye(n, dtype=torch.bool, device=ctx.device)
        npos = pos_mask.sum(-1, keepdim=True).clamp(min=1)
        nneg = (n - npos).clamp(min=1)
        has_neg = npos < n
        pos_w = torch.where(has_neg, 1.0 - eps, 1.0) / npos
        target = torch.where(pos_mask, pos_w, eps / nneg)
        loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        if margin > 0 and n > 1:
            pos = cos.diagonal()
            hard = cos.masked_fill(pos_mask, -4.0).max(-1).values
            loss = loss + F.relu(margin - (pos - hard)).mean()
        return loss

    # ------------------------------------------------------------------------------------------
    # arm0: matched-RAM byte-level BPE transformer baseline
    # ------------------------------------------------------------------------------------------

    class ByteBPE:
        """Minimal byte-level BPE (greedy merges learned on whitespace/CJK-bounded pieces)."""

        def __init__(self, merges=None):
            self.merges = [tuple(m) for m in (merges or [])]
            self.ranks = {m: i for i, m in enumerate(self.merges)}
            self.vocab_size = 256 + len(self.merges)

        # a piece = one whitespace byte (space/LF/TAB) + up to 15 other bytes, or up to 16 other
        # bytes (text start / after a 16-byte cut)
        _PIECE = re.compile(rb"[ \n\t][^ \n\t]{0,15}|[^ \n\t]{1,16}")

        @staticmethod
        def _pieces(data: bytes):
            return ByteBPE._PIECE.findall(data)

        @classmethod
        def train(cls, texts, num_merges: int = 768):
            """Greedy byte-pair merges with incremental pair counts (only words containing the
            merged pair are re-counted), so 20k documents train in seconds, not minutes."""
            from collections import Counter, defaultdict

            words = Counter()
            for t in texts:
                for piece in cls._pieces(text_bytes(t, 4096)):
                    words[piece] += 1
            seqs = [list(w) for w in words]
            freqs = list(words.values())
            pairs = Counter()
            where = defaultdict(set)
            for idx, s in enumerate(seqs):
                for p in zip(s, s[1:]):
                    pairs[p] += freqs[idx]
                    where[p].add(idx)
            merges = []
            for k in range(num_merges):
                if not pairs:
                    break
                (a, b), f = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
                if f < 2:
                    break
                new = 256 + k
                merges.append((a, b))
                for idx in sorted(where.pop((a, b), ())):
                    s, f = seqs[idx], freqs[idx]
                    for p in zip(s, s[1:]):
                        pairs[p] -= f
                        if pairs[p] <= 0:
                            del pairs[p]
                    i, out = 0, []
                    while i < len(s):
                        if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                            out.append(new)
                            i += 2
                        else:
                            out.append(s[i])
                            i += 1
                    seqs[idx] = out
                    for p in zip(out, out[1:]):
                        pairs[p] += f
                        where[p].add(idx)
                pairs.pop((a, b), None)
            return cls(merges)

        def encode(self, data: bytes):
            ids = []
            cache = self.__dict__.setdefault("_piece_cache", {})
            for piece in self._pieces(data):
                hit = cache.get(piece)
                if hit is None:
                    s = list(piece)
                    while len(s) > 1:
                        best = min(range(len(s) - 1), key=lambda i: self.ranks.get((s[i], s[i + 1]), 1 << 30))
                        r = self.ranks.get((s[best], s[best + 1]))
                        if r is None:
                            break
                        s = s[:best] + [256 + r] + s[best + 2:]
                    hit = s
                    if len(cache) < 2_000_000:
                        cache[piece] = hit
                ids.extend(hit)
            return ids

        def token_bytes(self):
            """Byte length of every token id (256 raw bytes + merges)."""
            lens = [1] * 256
            for a, b in self.merges:
                lens.append(lens[a] + lens[b])
            return lens

    class BPETransformerBaseline(nn.Module):
        def __init__(self, bpe: ByteBPE, d: int = 64, layers: int = 2, heads: int = 4, max_len: int = 256):
            super().__init__()
            self.bpe = bpe
            self.tok = nn.Embedding(bpe.vocab_size, d)
            self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out = nn.Linear(d, D_CONCEPT)
            self.lm = nn.Linear(d, bpe.vocab_size)
            self.max_len = max_len
            self.nce_log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
            self.sense_log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
            self._cache = {}

        def _ids(self, data: bytes):
            ids = self._cache.get(data)
            if ids is None:
                ids = self.bpe.encode(data)[: self.max_len]
                if len(self._cache) < 200000:
                    self._cache[data] = ids
            return ids

        def encode(self, texts, device=None) -> dict:
            device = device or self.pos.device
            datas = [text_bytes(t) for t in texts]
            toks = [self._ids(d) or [0] for d in datas]
            T = max(len(t) for t in toks)
            arr = _np.zeros((len(toks), T), dtype=_np.int64)
            for i, t in enumerate(toks):
                arr[i, :len(t)] = t
            tl = torch.tensor([len(t) for t in toks])
            ids = torch.from_numpy(arr).to(device)
            valid = torch.arange(T)[None, :] < tl[:, None]
            return self.encode_ids(ids, valid.to(device), torch.tensor(sum(len(d) for d in datas)))

        def encode_ids(self, ids, valid, n_bytes) -> dict:
            """ids (B,T) int64 token ids, valid (B,T) bool, n_bytes: total bytes (for bytes/token)."""
            T = ids.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1)
            hid = self.enc(self.tok(ids) + self.pos[:T], mask=mask, src_key_padding_mask=~valid)
            vf = valid.unsqueeze(-1).to(hid.dtype)
            z = self.out((hid * vf).sum(1) / vf.sum(1).clamp(min=1))
            bpt = n_bytes.to(valid.device, torch.float64) / valid.sum().clamp(min=1).to(torch.float64)
            return {"ids": ids, "valid": valid, "hid": hid, "z": z, "c": F.normalize(z, dim=-1),
                    "bytes_per_chunk": bpt}

        def lm_loss(self, enc):
            logits = self.lm(enc["hid"][:, :-1])
            m = enc["valid"][:, 1:]
            return masked_ce(logits, enc["ids"][:, 1:], m)

        @torch.no_grad()
        def bits_per_byte(self, enc):
            """(sum next-token NLL in bits, predicted bytes) over tokens 1..T-1 (token 0 unscored)."""
            logits = self.lm(enc["hid"][:, :-1])
            m = enc["valid"][:, 1:]
            tgt = enc["ids"][:, 1:][m]
            nll = F.cross_entropy(logits[m].float(), tgt, reduction="sum") / math.log(2)
            if not hasattr(self, "_tok_bytes"):
                self._tok_bytes = torch.tensor(self.bpe.token_bytes(), dtype=torch.float64)
            nb = self._tok_bytes.to(tgt.device)[tgt].sum()
            return float(nll), float(nb)


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())
