"""PE-Writer v1 (``pe-writer.bin`` PEWR + ``writer-clauses.bin`` WCLS): model, references and file formats.

Exact integer semantics: ``ARCHITECTURE.md, PE-Writer section`` (fills the gaps of ``pe-writer-spec.md``).

* ``PeWriterQAT``          -- PyTorch training model (float, optional int8 fake-quant QAT with STE).
* ``PeWriterIntNp``        -- single-example NumPy integer reference (mirrors the C++ kernel line by line).
* ``PeWriterIntTorch``     -- batched torch integer reference over the exported tensors (eval + parity).
* ``render``               -- token ids -> UTF-8 margin riff bytes (addendum E).
* pack/unpack helpers for PEWR, WCLS and PWGD (golden) files.
* v2 (Arm 4, same PEWR v1 / WCLS v1 bytes): ``frame_vocab`` / ``frame_tokens`` / ``frame_valid`` (decoder locked to
  ``<EOS>`` + slot specials + 32 connector pieces) and ``build_partitioned_bank_layout`` (step-role partitioned
  64-bucket WCLS with step-tagged centroids).
"""
from __future__ import annotations

import math
import re
import struct
from collections import Counter

import numpy as np

# ----------------------------------------------------------------------------- constants
D = 128
N_STEPS = 3
K_LOOPS = 2
N_VOCAB = 2048
T_MAX = 36
N_BUCKETS = 64
N_SLOTS = 64
SLOT_BYTES = 256
CLAUSE_TOKS = 16
CLAUSE_TEXT = 94
CLAUSE_TEXT_MAX = 93
VOCAB_STR = 16
TEXT_CAP = 240
REP_WINDOW = 6
REP_PENALTY = 10000
PHASE_LEN = 12

PAD, BOS, ENT_SUBJECT, ENT_ROLE, BOOK_ECHO, CLAUSE_0 = 0, 1, 2, 3, 4, 5
N_SPECIAL = 8
SPECIAL_NAMES = ["<PAD/EOS>", "<BOS>", "<ENT_SUBJECT>", "<ENT_ROLE>", "<BOOK_ECHO>",
                 "<CLAUSE_0>", "<CLAUSE_1>", "<CLAUSE_2>"]
ROLE_FALLBACK = b"this concept"

# ----------------------------------------------------------------------------- Frame-Locked PE-Writer v2 (Arm 4)
# The decoder may only emit <EOS>, the 6 slot specials and these 32 clean connector pieces (ids 8..39); every other
# lm_head row is W = 0, bias = FRAME_MASK_BIAS, so ids 40..2047 can never win the argmax and the decoder never
# improvises a content word. Two "," ids exist on purpose: the device repetition window (6 tokens) would otherwise
# penalise the comma after <BOOK_ECHO> because of the one after <ENT_SUBJECT>.
FRAME_PIECES = [
    ",", ":", "; in", ", while in", ", much as in", "this passage", ",", ".",
    ";", "\u2014", "while", "yet", "because", "echoing", "in", "as", "and", "so", "where", "though", "like",
    "much as", "even as", "as in", "until", "which", "a pattern", "the scene", "the story", "here", "too", "still",
]
N_FRAME_VOCAB = N_SPECIAL + len(FRAME_PIECES)
assert N_FRAME_VOCAB == 40 and all(0 < len(p.encode("utf-8")) <= VOCAB_STR - 1 for p in FRAME_PIECES)
F_COMMA_SUBJ, F_COLON, F_SYNTH, F_CONFLICT, F_PARALLEL, F_ECHO_FALLBACK, F_COMMA_ECHO, F_STOP = range(8, 16)
FRAME_CONNECTOR = {"synthesis": F_SYNTH, "conflict": F_CONFLICT, "parallel": F_PARALLEL}
FRAME_MASK_BIAS = -30000
# WCLS v2 bucket partition by step role: buckets [0, 21) hold clause_0-role slots, [21, 42) clause_1, [42, 64)
# clause_2 (21 + 21 + 22 buckets x 64 slots).
ROLE_BUCKETS = (21, 21, 22)
ROLE_BUCKET_START = (0, 21, 42)
_CONFLICT_RE = re.compile(r"\b(tension|conflict|clash|struggl|versus|against|contrast|oppos|defian|resist|rival|"
                          r"betray|confront|war|battle|threat|dilemma|paradox|undermin|subvert|challeng|fight|"
                          r"violen|refus|defeat|enem|danger|fear)", re.I)
_PARALLEL_RE = re.compile(r"\b(mirror|parallel|echo|recall|like|resembl|reflect|evok|akin|analog|similar|cycl|"
                          r"recur|repeat|reson|both|as if|reprise|imitat)", re.I)


def frame_vocab() -> list[str]:
    return SPECIAL_NAMES + FRAME_PIECES + [""] * (N_VOCAB - N_FRAME_VOCAB)


def frame_relation(clause0: str) -> str:
    """Connector class of the riff from the retrieved observation clause: conflict, parallel or synthesis."""
    if _CONFLICT_RE.search(clause0):
        return "conflict"
    if _PARALLEL_RE.search(clause0):
        return "parallel"
    return "synthesis"


def frame_tokens(has_subject: bool, has_echo: bool, relation: str) -> list[int]:
    """The v2 frame: ``<ENT_SUBJECT>, <ENT_ROLE>: <CLAUSE_0><conn> <BOOK_ECHO>, <CLAUSE_1>.`` + EOS, where <conn> is
    "; in" / ", while in" / ", much as in". A missing subject drops the "<ENT_SUBJECT>, <ENT_ROLE>:" head (the masked
    <ENT_SUBJECT>'s runner-up is <CLAUSE_0>); a missing echo title is replaced by "this passage" (the masked
    <BOOK_ECHO>'s runner-up)."""
    t = [ENT_SUBJECT, F_COMMA_SUBJ, ENT_ROLE, F_COLON] if has_subject else []
    t += [CLAUSE_0, FRAME_CONNECTOR[relation], BOOK_ECHO if has_echo else F_ECHO_FALLBACK, F_COMMA_ECHO,
          CLAUSE_0 + 1, F_STOP, PAD]
    return t


def frame_valid(tokens, has_subject: bool, has_echo: bool) -> bool:
    """True iff a decoded token row is exactly one of the v2 frames for these inputs."""
    seq = [int(v) for v in tokens]
    seq = seq[:seq.index(PAD) + 1] if PAD in seq else seq
    return any(seq == frame_tokens(has_subject, has_echo, r) for r in FRAME_CONNECTOR)


def tag_directions(vecs: np.ndarray) -> np.ndarray:
    """3 orthonormal step-tag directions: the lowest-variance principal axes of the unit clause vectors, so a tag
    added to a plan shifts every slot score in a bucket by nearly the same amount."""
    X = np.asarray(vecs, np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=True)
    return Vt[-3:][::-1].copy()


def build_partitioned_bank_layout(vecs: np.ndarray, texts: list[str], roles, seed: int = 0, iters: int = 30,
                                  tag_gain: float = 1.0) -> dict:
    """WCLS v2: role-partitioned spherical k-means (ROLE_BUCKETS per step role) with capacity-64 balanced
    assignment; ``clause_toks`` all zero (whole-clause emission only). Centroid of bucket b in partition r =
    int8 unit(unit(mean of its slots) + tag_gain * u_r), u_r = ``tag_directions``[r]: the planner learns to add
    u_r at step r, so step r routes into partition r and the three clauses are always distinct slots."""
    vecs = np.asarray(vecs, np.int64)
    roles = np.asarray(roles, np.int64)
    assert len(vecs) == N_BUCKETS * N_SLOTS == len(texts)
    U = tag_directions(vecs)
    Xn = vecs.astype(np.float64)
    Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(seed)
    cent = np.zeros((N_BUCKETS, D), np.int64)
    bank_vec = np.zeros((N_BUCKETS, N_SLOTS, D), np.int8)
    bank_tok = np.zeros((N_BUCKETS, N_SLOTS, CLAUSE_TOKS), np.uint16)
    bank_role = np.zeros((N_BUCKETS, N_SLOTS), np.uint8)
    bank_text = [[b""] * N_SLOTS for _ in range(N_BUCKETS)]
    for r in range(N_STEPS):
        idx = np.flatnonzero(roles == r)
        k, b0 = ROLE_BUCKETS[r], ROLE_BUCKET_START[r]
        assert len(idx) == k * N_SLOTS, (r, len(idx))
        X = Xn[idx]
        C = X[rng.choice(len(idx), k, replace=False)].copy()
        for _ in range(iters):
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
            a = np.argmax(X @ C.T, 1)
            for b in range(k):
                m = a == b
                C[b] = X[m].mean(0) if m.any() else X[rng.integers(len(idx))]
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        sims = X @ C.T
        order = np.argsort(-sims.max(1), kind="stable")
        fill = np.zeros(k, np.int64)
        assign = np.full(len(idx), -1, np.int64)
        for i in order:
            for b in np.argsort(-sims[i], kind="stable"):
                if fill[b] < N_SLOTS:
                    assign[i] = b
                    fill[b] += 1
                    break
        for b in range(k):
            members = idx[assign == b]
            m = Xn[members].mean(0)
            m /= max(np.linalg.norm(m), 1e-9)
            c = m + tag_gain * U[r]
            cent[b0 + b] = normalize_i8(np.rint(127 * c / np.linalg.norm(c)).astype(np.int64))
            for j, src in enumerate(members):
                bank_vec[b0 + b, j] = vecs[src]
                bank_role[b0 + b, j] = r
                bank_text[b0 + b][j] = fit_text_bytes(normalize_text(texts[src]).strip(), CLAUSE_TEXT_MAX)
    return {"centroids": cent.astype(np.int8), "vec": bank_vec, "tok": bank_tok, "role": bank_role,
            "text": bank_text, "tags": U}


def bucket_partition(b: int) -> int:
    return 0 if b < ROLE_BUCKET_START[1] else 1 if b < ROLE_BUCKET_START[2] else 2



PEWR_MAGIC = b"PEWR"
WCLS_MAGIC = b"WCLS"
PWGD_MAGIC = b"PWGD"
PEWR_HEADER = struct.Struct("<4sHHHHHHHH12s")
WCLS_HEADER = struct.Struct("<4sHHHH20s")
assert PEWR_HEADER.size == 32 and WCLS_HEADER.size == 32

# (name, kind, shape): kind "raw" = int8 raw tensor; "lin" = float32 scale + int16 bias[out] + int8 W[out][in]
PEWR_LAYOUT = [
    ("step_embed", "raw", (N_STEPS, D)),
    ("plan_fc1", "lin", (D, D)),
    ("plan_fc2", "lin", (D, D)),
    ("bucket_centroids", "raw", (N_BUCKETS, D)),
    ("tok_embed", "raw", (N_VOCAB, D)),
    ("pos_embed", "raw", (T_MAX, D)),
    ("conv_in_gate", "lin", (D, D)),
    ("conv_dw", "lin", (D, 3)),
    ("conv_out_proj", "lin", (D, D)),
    ("lm_head", "lin", (N_VOCAB, D)),
    ("copy_gate_proj", "lin", (1, D)),
]
LIN_NAMES = [n for n, k, _ in PEWR_LAYOUT if k == "lin"]
RAW_NAMES = [n for n, k, _ in PEWR_LAYOUT if k == "raw"]
SHAPES = {n: s for n, _, s in PEWR_LAYOUT}


def _tensor_bytes(kind, shape):
    n = int(np.prod(shape))
    return n if kind == "raw" else 4 + 2 * shape[0] + n


OFFSETS = {}
_off = 32
for _n, _k, _s in PEWR_LAYOUT:
    OFFSETS[_n] = _off
    _off += _tensor_bytes(_k, _s)
VOCAB_STR_OFFSET = _off
PEWR_FILE_BYTES = _off + N_VOCAB * VOCAB_STR
WCLS_FILE_BYTES = 32 + N_BUCKETS * N_SLOTS * SLOT_BYTES
assert PEWR_FILE_BYTES == 641726 and VOCAB_STR_OFFSET == 608958 and OFFSETS["lm_head"] == 342580
assert WCLS_FILE_BYTES == 1048608

GOLDEN_CASE = struct.Struct("<128s128s32s32s32s384s3s72s240s")
assert GOLDEN_CASE.size == 1051


# ----------------------------------------------------------------------------- integer helpers
def sat8(x):
    return np.clip(x, -127, 127)


def sh(a, s: int):
    """Firmware rounding shift (round half up) for s >= 0; works on python ints and int64 arrays."""
    if s > 0:
        return (a + (1 << (s - 1))) >> s
    return a


def scale_to_shift(scale: float) -> int:
    if not (scale > 0.0) or not math.isfinite(scale):
        raise ValueError(f"bad scale {scale!r}")
    s = int(round(-math.log2(scale)))
    if s < 0 or s > 30 or math.ldexp(1.0, -s) != scale:
        raise ValueError(f"scale {scale!r} is not 2^-S with S in [0, 30]")
    return s


def normalize_i8(r: np.ndarray) -> np.ndarray:
    """Exact L2 -> int8 (firmware ``normalize``): isqrt + rdiv(127 * r, n)."""
    r = np.asarray(r, dtype=np.int64)
    n = math.isqrt(int(np.dot(r, r)))
    if n == 0:
        return np.zeros(r.shape, np.int64)
    a = 127 * r
    half = n // 2
    q = np.where(a >= 0, (a + half) // n, -((-a + half) // n))
    return sat8(q)


def normalize_i8_rows(r: np.ndarray) -> np.ndarray:
    return np.stack([normalize_i8(x) for x in np.asarray(r, np.int64)])


# ----------------------------------------------------------------------------- UTF-8 / render (addendum E)
_GLUE = set(b",.;:!?)'")


def _is_cont(b: int) -> bool:
    return (b & 0xC0) == 0x80


def utf8_decode_at(buf: bytes, i: int):
    """-> (cp, n) or None (addendum E ``decode``)."""
    if i < 0 or i >= len(buf):
        return None
    b = buf[i]
    if b < 0x80:
        return b, 1
    if 0xC0 <= b < 0xE0:
        n, cp = 2, b & 0x1F
    elif 0xE0 <= b < 0xF0:
        n, cp = 3, b & 0x0F
    elif 0xF0 <= b < 0xF8:
        n, cp = 4, b & 0x07
    else:
        return None
    if i + n > len(buf):
        return None
    for k in range(1, n):
        c = buf[i + k]
        if not _is_cont(c):
            return None
        cp = (cp << 6) | (c & 0x3F)
    return cp, n


def is_cjk(cp: int) -> bool:
    return 0x2E80 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF


def _last_cp(out: bytearray):
    n = len(out)
    if n == 0:
        return None
    i, k = n - 1, 0
    while i > 0 and k < 3 and _is_cont(out[i]):
        i -= 1
        k += 1
    d = utf8_decode_at(bytes(out), i)
    if d is None or i + d[1] != n:
        return None
    return d[0]


def cstr(raw: bytes) -> bytes:
    return bytes(raw).split(b"\0", 1)[0]


def render(tokens, vocab_str: list[bytes], subject: bytes, role: bytes, echo: bytes,
           clause_text: list[bytes]) -> bytes:
    out = bytearray()
    for v in tokens:
        v = int(v)
        if v == PAD:
            break
        if v == ENT_SUBJECT:
            p = subject
        elif v == ENT_ROLE:
            p = role if role else ROLE_FALLBACK
        elif v == BOOK_ECHO:
            p = echo
        elif CLAUSE_0 <= v <= CLAUSE_0 + 2:
            p = clause_text[v - CLAUSE_0]
        elif v >= N_SPECIAL:
            p = cstr(vocab_str[v][:VOCAB_STR])
        else:
            p = b""
        if not p:
            continue
        if p[:2] == b"##":
            out += p[2:]
            continue
        glue = (not out) or p[0] in _GLUE or out[-1] == 0x28
        if not glue:
            lc = _last_cp(out)
            fc = utf8_decode_at(p, 0)
            glue = lc is not None and fc is not None and is_cjk(lc) and is_cjk(fc[0])
        if not glue:
            out += b" "
        out += p
    if len(out) > TEXT_CAP - 1:
        cut = TEXT_CAP - 1
        while cut > 0 and _is_cont(out[cut]):
            cut -= 1
        del out[cut:]
    return bytes(out)


# ----------------------------------------------------------------------------- tokenizer + vocab
_CJK_CLASS = "\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef"
_TOKEN_RE = re.compile(rf"[{_CJK_CLASS}]|(?:(?![{_CJK_CLASS}])\w)+(?:'(?:(?![{_CJK_CLASS}])\w)+)*|\S")
_SPECIAL_RE = re.compile(r"<(ENT_SUBJECT|ENT_ROLE|BOOK_ECHO|CLAUSE_[012])>")
_SPECIAL_ID = {"ENT_SUBJECT": ENT_SUBJECT, "ENT_ROLE": ENT_ROLE, "BOOK_ECHO": BOOK_ECHO,
               "CLAUSE_0": 5, "CLAUSE_1": 6, "CLAUSE_2": 7}


def normalize_text(s: str) -> str:
    return (s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u00a0", " "))


def word_tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(s))


def riff_to_template(riff: str, subject: str, role: str, echo: str, clauses: list[str]) -> str:
    """Rendered teacher riff -> template with <SPECIAL> markers (first case-insensitive occurrence each)."""
    t = normalize_text(riff)
    items = [(c, f"<CLAUSE_{k}>") for k, c in enumerate(clauses)] + [
        (echo, "<BOOK_ECHO>"), (role, "<ENT_ROLE>"), (subject, "<ENT_SUBJECT>")]
    items = [(normalize_text(a).strip(), m) for a, m in items if a and a.strip()]
    items.sort(key=lambda x: -len(x[0]))
    for a, m in items:
        t = re.compile(re.escape(a), re.IGNORECASE).sub(m, t, count=1)
    return t


def template_pieces(template: str) -> list[object]:
    """-> list of special ids (int) and word strings."""
    out, pos = [], 0
    for m in _SPECIAL_RE.finditer(template):
        out += word_tokens(template[pos:m.start()])
        out.append(_SPECIAL_ID[m.group(1)])
        pos = m.end()
    out += word_tokens(template[pos:])
    return out


def build_vocab(texts, extra_texts=()) -> list[str]:
    """2,048 entries: 8 specials + 2,040 most frequent word/punct tokens (<= 15 UTF-8 bytes)."""
    cnt = Counter()
    for t in texts:
        for p in template_pieces(t):
            if isinstance(p, str):
                cnt[p] += 2
    for t in extra_texts:
        for p in word_tokens(t):
            cnt[p] += 1
    words = [w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
             if 0 < len(w.encode("utf-8")) <= VOCAB_STR - 1 and not w.startswith("##")]
    words = words[:N_VOCAB - N_SPECIAL]
    vocab = SPECIAL_NAMES + words
    vocab += [""] * (N_VOCAB - len(vocab))
    return vocab


class Tokenizer:
    def __init__(self, vocab: list[str]):
        assert len(vocab) == N_VOCAB
        self.vocab = vocab
        self.ids = {w: i for i, w in enumerate(vocab) if i >= N_SPECIAL and w}

    def encode_template(self, template: str, max_len: int = T_MAX) -> list[int]:
        """Template -> ids (+ EOS), each special at most once, OOV words dropped, <= max_len incl. EOS."""
        ids, used = [], set()
        for p in template_pieces(template):
            if isinstance(p, int):
                if p in used:
                    continue
                used.add(p)
                ids.append(p)
            elif p in self.ids:
                ids.append(self.ids[p])
        ids = ids[:max_len - 1]
        return ids + [PAD]

    def encode_words(self, text: str, max_len: int = CLAUSE_TOKS) -> list[int]:
        ids = [self.ids[p] for p in word_tokens(text) if p in self.ids][:max_len]
        return ids + [0] * (max_len - len(ids))

    def vocab_bytes(self) -> list[bytes]:
        return [w.encode("utf-8")[:VOCAB_STR - 1] for w in self.vocab]


# ----------------------------------------------------------------------------- WCLS bank layout
def fit_text_bytes(s: str, cap: int) -> bytes:
    b = s.encode("utf-8")
    if len(b) <= cap:
        return b
    cut = cap
    while cut > 0 and _is_cont(b[cut]):
        cut -= 1
    return b[:cut]


def build_bank_layout(vecs: np.ndarray, texts: list[str], roles, tok: Tokenizer, seed: int = 0,
                      iters: int = 25) -> dict:
    """Spherical k-means (64) + capacity-64 balanced assignment -> WCLS tensors + int8 unit centroids."""
    vecs = np.asarray(vecs, np.int64)
    roles = np.asarray(roles, np.int64)
    seen, keep = set(), []
    for i, t in enumerate(texts):
        k = normalize_text(t).strip().lower()
        if k and k not in seen and np.any(vecs[i]):
            seen.add(k)
            keep.append(i)
    keep = keep[:N_BUCKETS * N_SLOTS]
    X = vecs[keep].astype(np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(seed)
    n = len(keep)
    C = Xn[rng.choice(n, N_BUCKETS, replace=n < N_BUCKETS)] + 1e-3 * rng.standard_normal((N_BUCKETS, D))
    for _ in range(iters):
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        a = np.argmax(Xn @ C.T, 1)
        for b in range(N_BUCKETS):
            m = a == b
            C[b] = Xn[m].mean(0) if m.any() else Xn[rng.integers(n)]
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    sims = Xn @ C.T
    order = np.argsort(-sims.max(1), kind="stable")
    fill = np.zeros(N_BUCKETS, np.int64)
    assign = np.full(n, -1, np.int64)
    for i in order:
        for b in np.argsort(-sims[i], kind="stable"):
            if fill[b] < N_SLOTS:
                assign[i] = b
                fill[b] += 1
                break
    cent = np.zeros((N_BUCKETS, D), np.int64)
    for b in range(N_BUCKETS):
        m = assign == b
        cent[b] = normalize_i8(np.rint(127 * (Xn[m].mean(0) if m.any() else C[b])).astype(np.int64))
    bank_vec = np.zeros((N_BUCKETS, N_SLOTS, D), np.int8)
    bank_tok = np.zeros((N_BUCKETS, N_SLOTS, CLAUSE_TOKS), np.uint16)
    bank_role = np.zeros((N_BUCKETS, N_SLOTS), np.uint8)
    bank_text = [[b""] * N_SLOTS for _ in range(N_BUCKETS)]
    slot = np.zeros(N_BUCKETS, np.int64)
    for i in range(n):
        b = assign[i]
        j = slot[b]
        slot[b] += 1
        src = keep[i]
        bank_vec[b, j] = vecs[src]
        bank_tok[b, j] = tok.encode_words(texts[src])
        bank_role[b, j] = int(roles[src]) if src < len(roles) else 0
        bank_text[b][j] = fit_text_bytes(normalize_text(texts[src]).strip(), CLAUSE_TEXT_MAX)
    return {"centroids": cent.astype(np.int8), "vec": bank_vec, "tok": bank_tok, "role": bank_role,
            "text": bank_text}


def pack_wcls(bank: dict) -> bytes:
    blob = bytearray(WCLS_FILE_BYTES)
    WCLS_HEADER.pack_into(blob, 0, WCLS_MAGIC, 1, N_BUCKETS, N_SLOTS, SLOT_BYTES, b"\0" * 20)
    for b in range(N_BUCKETS):
        for j in range(N_SLOTS):
            o = 32 + (b * N_SLOTS + j) * SLOT_BYTES
            blob[o:o + D] = np.asarray(bank["vec"][b, j], np.int8).tobytes()
            blob[o + 128:o + 160] = np.asarray(bank["tok"][b, j], "<u2").tobytes()
            blob[o + 160] = int(bank["role"][b, j])
            t = bank["text"][b][j][:CLAUSE_TEXT_MAX]
            blob[o + 161] = len(t)
            blob[o + 162:o + 162 + len(t)] = t
    return bytes(blob)


class Wcls:
    """Parsed ``writer-clauses.bin`` (validates header + size)."""

    def __init__(self, blob: bytes):
        if len(blob) != WCLS_FILE_BYTES:
            raise ValueError(f"writer-clauses.bin must be {WCLS_FILE_BYTES} bytes, got {len(blob)}")
        magic, ver, nb, ns, sb, _ = WCLS_HEADER.unpack_from(blob, 0)
        if magic != WCLS_MAGIC or ver != 1 or nb != N_BUCKETS or ns != N_SLOTS or sb != SLOT_BYTES:
            raise ValueError("bad WCLS header")
        raw = np.frombuffer(blob, np.uint8, offset=32).reshape(N_BUCKETS, N_SLOTS, SLOT_BYTES)
        self.vec = raw[:, :, :128].view(np.int8).astype(np.int64)
        tok = raw[:, :, 128:160].copy().view("<u2").astype(np.int64)
        self.tok = np.where(tok >= N_VOCAB, 0, tok)
        self.role = raw[:, :, 160].astype(np.int64)
        self.text_len = raw[:, :, 161].astype(np.int64)
        self.valid = self.text_len > 0
        self.text = [[cstr(bytes(raw[b, j, 162:162 + min(int(self.text_len[b, j]), CLAUSE_TEXT_MAX)]))
                      for j in range(N_SLOTS)] for b in range(N_BUCKETS)]


# ----------------------------------------------------------------------------- PEWR pack/parse
def pack_pewr(t: dict, vocab_bytes: list[bytes]) -> bytes:
    """``t``: raw names -> int8 arrays; lin names -> (S, bias int16[out], W int8[out][in])."""
    blob = bytearray(PEWR_FILE_BYTES)
    PEWR_HEADER.pack_into(blob, 0, PEWR_MAGIC, 1, D, D, N_STEPS, N_VOCAB, T_MAX, N_BUCKETS, N_SLOTS, b"\0" * 12)
    for name, kind, shape in PEWR_LAYOUT:
        o = OFFSETS[name]
        if kind == "raw":
            a = np.asarray(t[name])
            assert a.shape == shape and a.min() >= -127 and a.max() <= 127, name
            blob[o:o + a.size] = a.astype(np.int8).tobytes()
        else:
            S, b, W = t[name]
            W, b = np.asarray(W), np.asarray(b)
            assert W.shape == shape and b.shape == (shape[0],) and 0 <= S <= 30, name
            assert W.min() >= -127 and W.max() <= 127 and b.min() >= -32768 and b.max() <= 32767, name
            struct.pack_into("<f", blob, o, math.ldexp(1.0, -int(S)))
            blob[o + 4:o + 4 + 2 * shape[0]] = b.astype("<i2").tobytes()
            o2 = o + 4 + 2 * shape[0]
            blob[o2:o2 + W.size] = W.astype(np.int8).tobytes()
    assert len(vocab_bytes) == N_VOCAB
    for v, s in enumerate(vocab_bytes):
        s = s[:VOCAB_STR - 1]
        o = VOCAB_STR_OFFSET + v * VOCAB_STR
        blob[o:o + len(s)] = s
    return bytes(blob)


def parse_pewr(blob: bytes) -> tuple[dict, list[bytes]]:
    if len(blob) != PEWR_FILE_BYTES:
        raise ValueError(f"pe-writer.bin must be {PEWR_FILE_BYTES} bytes, got {len(blob)}")
    h = PEWR_HEADER.unpack_from(blob, 0)
    if h[0] != PEWR_MAGIC or tuple(h[1:9]) != (1, D, D, N_STEPS, N_VOCAB, T_MAX, N_BUCKETS, N_SLOTS):
        raise ValueError(f"bad PEWR header {h[:9]}")
    t = {}
    for name, kind, shape in PEWR_LAYOUT:
        o = OFFSETS[name]
        n = int(np.prod(shape))
        if kind == "raw":
            t[name] = np.frombuffer(blob, np.int8, n, o).reshape(shape).astype(np.int64)
        else:
            S = scale_to_shift(struct.unpack_from("<f", blob, o)[0])
            b = np.frombuffer(blob, "<i2", shape[0], o + 4).astype(np.int64)
            W = np.frombuffer(blob, np.int8, n, o + 4 + 2 * shape[0]).reshape(shape).astype(np.int64)
            t[name] = (S, b, W)
    vocab = [cstr(blob[VOCAB_STR_OFFSET + v * VOCAB_STR:VOCAB_STR_OFFSET + (v + 1) * VOCAB_STR])
             for v in range(N_VOCAB)]
    return t, vocab


# ----------------------------------------------------------------------------- NumPy integer reference
def _lin(L, x):
    S, b, W = L
    return sh(W @ np.asarray(x, np.int64), S) + b


class PeWriterIntNp:
    """Single-example integer reference (addendum B-E). ``use_planner`` / ``use_bank`` only exist for the
    ablation arms (Arm 0: no planner, Arm 0/1: no clause bank); the device kernel is both True."""

    def __init__(self, pewr_blob: bytes, wcls_blob: bytes | None):
        self.t, self.vocab = parse_pewr(pewr_blob)
        self.bank = Wcls(wcls_blob) if wcls_blob is not None else None

    def plan(self, c_page, c_echo):
        t = self.t
        c_prev = np.asarray(c_page, np.int64)
        c_echo = np.asarray(c_echo, np.int64)
        plans = []
        for s in range(N_STEPS):
            h = sat8((c_prev + c_echo + t["step_embed"][s]) >> 1)
            for _ in range(K_LOOPS):
                a = np.clip(_lin(t["plan_fc1"], h), 0, 127)
                h = normalize_i8(h + _lin(t["plan_fc2"], a))
            plans.append(h)
            c_prev = h
        return np.stack(plans)

    def route(self, c_plan):
        cent = self.t["bucket_centroids"]
        return [int(np.argmax(cent @ c_plan[s])) for s in range(N_STEPS)]

    def retrieve(self, c_plan, buckets, use_bank=True):
        r = np.zeros((N_STEPS, D), np.int64)
        toks = np.zeros((N_STEPS, CLAUSE_TOKS), np.int64)
        sim = [0] * N_STEPS
        text = [b""] * N_STEPS
        has = [False] * N_STEPS
        slots = [-1] * N_STEPS
        if not use_bank or self.bank is None:
            return r, toks, sim, text, has, slots
        for s in range(N_STEPS):
            b = buckets[s]
            best, bj = None, -1
            for j in range(N_SLOTS):
                if not self.bank.valid[b, j]:
                    continue
                d = int(self.bank.vec[b, j] @ c_plan[s])
                if best is None or d > best:
                    best, bj = d, j
            if bj >= 0:
                r[s] = self.bank.vec[b, bj]
                toks[s] = self.bank.tok[b, bj]
                sim[s] = int(np.clip(best >> 7, 0, 127))
                text[s] = self.bank.text[b][bj]
                has[s] = True
                slots[s] = bj
        return r, toks, sim, text, has, slots

    def generate(self, c_page, c_echo, subject: bytes, role: bytes, echo: bytes, use_planner=True,
                 use_bank=True) -> dict:
        t = self.t
        c_page = np.asarray(c_page, np.int64)
        c_plan = self.plan(c_page, c_echo) if use_planner else np.stack([c_page] * N_STEPS)
        buckets = self.route(c_plan)
        r, ctoks, sim, ctext, has, slots = self.retrieve(c_plan, buckets, use_bank)
        y1 = np.zeros(D, np.int64)
        y2 = np.zeros(D, np.int64)
        prev = BOS
        cursor = [0] * N_STEPS
        used = [False] * N_SPECIAL
        out = [0] * T_MAX
        S_dw, b_dw, W_dw = t["conv_dw"]
        excl0 = np.zeros(N_VOCAB, bool)
        excl0[BOS] = True
        if not subject:
            excl0[ENT_SUBJECT] = True
        if not echo:
            excl0[BOOK_ECHO] = True
        for k in range(N_STEPS):
            if not has[k]:
                excl0[CLAUSE_0 + k] = True
        for step in range(T_MAX):
            s = min(2, step // PHASE_LEN)
            u = sat8((t["tok_embed"][prev] + ((c_plan[s] + r[s]) >> 1) + t["pos_embed"][step]) >> 1)
            gs = np.clip(_lin(t["conv_in_gate"], u) + 64, 0, 128)
            y = (u * gs + 64) >> 7
            z = sat8(sh(W_dw[:, 0] * y2 + W_dw[:, 1] * y1 + W_dw[:, 2] * y, S_dw) + b_dw)
            y2, y1 = y1, y
            h = sat8(u + _lin(t["conv_out_proj"], z))
            logit = _lin(t["lm_head"], h)
            gate = int(np.clip(_lin(t["copy_gate_proj"], h)[0] + 64, 0, 128))
            tok_c = int(ctoks[s][cursor[s]]) if cursor[s] < CLAUSE_TOKS else 0
            if 0 < tok_c < N_VOCAB:
                logit[tok_c] += (gate * sim[s]) >> 4
            for v in set(out[max(0, step - REP_WINDOW):step]):
                if v >= N_SPECIAL:
                    logit[v] -= REP_PENALTY
            excl = excl0.copy()
            for v in range(2, N_SPECIAL):
                if used[v]:
                    excl[v] = True
            yt = int(np.argmax(np.where(excl, np.iinfo(np.int64).min, logit)))
            if yt == PAD:
                break
            out[step] = yt
            prev = yt
            if 2 <= yt < N_SPECIAL:
                used[yt] = True
            if CLAUSE_0 <= yt <= CLAUSE_0 + 2:
                cursor[yt - CLAUSE_0] = CLAUSE_TOKS
            elif 0 < tok_c < N_VOCAB and yt == tok_c:
                cursor[s] += 1
        text = render(out, self.vocab, subject, role, echo, ctext)
        return {"c_plan": c_plan, "buckets": buckets, "tokens": out, "text": text, "slots": slots,
                "clause_text": ctext}


# ----------------------------------------------------------------------------- batched torch integer reference
class PeWriterIntTorch:
    """Batched integer forward over exported tensors (float64 matmuls are exact: |dot| < 2^31)."""

    def __init__(self, pewr_blob: bytes, wcls_blob: bytes | None, device="cpu"):
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        mm = torch.float64 if self.dev.type != "mps" else torch.float32
        self.mm = mm
        t, self.vocab = parse_pewr(pewr_blob)
        T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=self.dev)  # noqa: E731
        self.raw = {n: T(t[n]) for n in RAW_NAMES}
        self.lin = {n: (t[n][0], T(t[n][1]), torch.as_tensor(np.asarray(t[n][2]), dtype=mm, device=self.dev))
                    for n in LIN_NAMES}
        self.bank = Wcls(wcls_blob) if wcls_blob is not None else None
        if self.bank is not None:
            self.bvec = T(self.bank.vec)
            self.btok = T(self.bank.tok)
            self.bvalid = torch.as_tensor(self.bank.valid, device=self.dev)

    def _sh(self, a, S):
        return (a + (1 << (S - 1))) >> S if S > 0 else a

    def _L(self, name, x):
        S, b, W = self.lin[name]
        acc = (x.to(self.mm) @ W.T).round().to(self.torch.int64)
        return self._sh(acc, S) + b

    def _norm(self, r):
        torch = self.torch
        n2 = (r * r).sum(-1)
        n = torch.sqrt(n2.to(torch.float64)).floor().to(torch.int64)
        n = torch.where(n * n > n2, n - 1, n)
        n = torch.where((n + 1) * (n + 1) <= n2, n + 1, n)
        a = 127 * r
        nn = n.clamp(min=1).unsqueeze(-1)
        half = nn // 2
        q = torch.where(a >= 0, torch.div(a + half, nn, rounding_mode="floor"),
                        -torch.div(-a + half, nn, rounding_mode="floor"))
        return torch.where(n.unsqueeze(-1) == 0, torch.zeros_like(q), q.clamp(-127, 127))

    def generate(self, c_page, c_echo, subjects, roles, echoes, use_planner=True, use_bank=True) -> list[dict]:
        torch = self.torch
        dev = self.dev
        cp = torch.as_tensor(np.asarray(c_page), dtype=torch.int64, device=dev)
        ce = torch.as_tensor(np.asarray(c_echo), dtype=torch.int64, device=dev)
        B = cp.shape[0]
        if use_planner:
            plans, prev = [], cp
            for s in range(N_STEPS):
                h = (prev + ce + self.raw["step_embed"][s]) >> 1
                h = h.clamp(-127, 127)
                for _ in range(K_LOOPS):
                    a = self._L("plan_fc1", h).clamp(0, 127)
                    h = self._norm(h + self._L("plan_fc2", a))
                plans.append(h)
                prev = h
            c_plan = torch.stack(plans, 1)
        else:
            c_plan = cp.unsqueeze(1).expand(B, N_STEPS, D).clone()
        cent = self.raw["bucket_centroids"].to(self.mm)
        buckets = (c_plan.to(self.mm) @ cent.T).argmax(-1)  # [B,3], first max
        r = torch.zeros(B, N_STEPS, D, dtype=torch.int64, device=dev)
        ctoks = torch.zeros(B, N_STEPS, CLAUSE_TOKS, dtype=torch.int64, device=dev)
        sim = torch.zeros(B, N_STEPS, dtype=torch.int64, device=dev)
        has = torch.zeros(B, N_STEPS, dtype=torch.bool, device=dev)
        slots = torch.full((B, N_STEPS), -1, dtype=torch.int64, device=dev)
        if use_bank and self.bank is not None:
            NEG = torch.iinfo(torch.int64).min
            for s in range(N_STEPS):
                bv = self.bvec[buckets[:, s]]  # [B,64,128]
                sc = (bv.to(self.mm) @ c_plan[:, s].to(self.mm).unsqueeze(-1)).squeeze(-1).round().to(torch.int64)
                val = self.bvalid[buckets[:, s]]
                sc = torch.where(val, sc, torch.full_like(sc, NEG))
                best, bj = sc.max(-1)
                # torch.max returns *an* index; force lowest index among ties
                bj = (sc == best.unsqueeze(-1)).to(torch.int64).argmax(-1)
                ok = val.any(-1)
                ar = torch.arange(B, device=dev)
                r[:, s] = torch.where(ok.unsqueeze(-1), bv[ar, bj], 0)
                ctoks[:, s] = torch.where(ok.unsqueeze(-1), self.btok[buckets[:, s], bj], 0)
                sim[:, s] = torch.where(ok, (best >> 7).clamp(0, 127), 0)
                has[:, s] = ok
                slots[:, s] = torch.where(ok, bj, -1)
        S_dw, b_dw, W_dw = self.lin["conv_dw"]
        W_dw = W_dw.to(torch.int64)
        y1 = torch.zeros(B, D, dtype=torch.int64, device=dev)
        y2 = torch.zeros_like(y1)
        prev = torch.full((B,), BOS, dtype=torch.int64, device=dev)
        cursor = torch.zeros(B, N_STEPS, dtype=torch.int64, device=dev)
        used = torch.zeros(B, N_SPECIAL, dtype=torch.bool, device=dev)
        out = torch.zeros(B, T_MAX, dtype=torch.int64, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        excl0 = torch.zeros(B, N_VOCAB, dtype=torch.bool, device=dev)
        excl0[:, BOS] = True
        excl0[:, ENT_SUBJECT] = torch.as_tensor([not s for s in subjects], device=dev)
        excl0[:, BOOK_ECHO] = torch.as_tensor([not e for e in echoes], device=dev)
        excl0[:, CLAUSE_0:CLAUSE_0 + 3] = ~has
        ar = torch.arange(B, device=dev)
        NEG = torch.iinfo(torch.int64).min
        cond = (c_plan + r) >> 1
        for step in range(T_MAX):
            s = min(2, step // PHASE_LEN)
            u = ((self.raw["tok_embed"][prev] + cond[:, s] + self.raw["pos_embed"][step]) >> 1).clamp(-127, 127)
            gs = (self._L("conv_in_gate", u) + 64).clamp(0, 128)
            y = (u * gs + 64) >> 7
            z = (self._sh(W_dw[:, 0] * y2 + W_dw[:, 1] * y1 + W_dw[:, 2] * y, S_dw) + b_dw).clamp(-127, 127)
            y2, y1 = y1, y
            h = (u + self._L("conv_out_proj", z)).clamp(-127, 127)
            logit = self._L("lm_head", h)
            gate = (self._L("copy_gate_proj", h)[:, 0] + 64).clamp(0, 128)
            cur = cursor[:, s]
            tok_c = torch.where(cur < CLAUSE_TOKS, ctoks[:, s].gather(1, cur.clamp(max=CLAUSE_TOKS - 1)[:, None])[:, 0],
                                torch.zeros_like(cur))
            okc = (tok_c > 0) & (tok_c < N_VOCAB)
            boost = torch.where(okc, (gate * sim[:, s]) >> 4, torch.zeros_like(gate))
            logit = logit.scatter_add(1, tok_c.clamp(0, N_VOCAB - 1)[:, None], boost[:, None])
            if step > 0:
                win = out[:, max(0, step - REP_WINDOW):step]
                pen = torch.zeros(B, N_VOCAB, dtype=torch.bool, device=dev)
                pen.scatter_(1, win, True)
                pen[:, :N_SPECIAL] = False
                logit = logit - REP_PENALTY * pen.to(torch.int64)
            excl = excl0.clone()
            excl[:, 2:N_SPECIAL] |= used[:, 2:]
            logit = torch.where(excl, torch.full_like(logit, NEG), logit)
            mx = logit.max(-1, keepdim=True).values
            yt = (logit == mx).to(torch.int64).argmax(-1)
            yt = torch.where(done, torch.zeros_like(yt), yt)
            stop = yt == PAD
            done = done | stop
            out[:, step] = yt
            prev = torch.where(done, prev, yt)
            sp = (yt >= 2) & (yt < N_SPECIAL)
            used[ar[sp], yt[sp]] = True
            isc = (yt >= CLAUSE_0) & (yt <= CLAUSE_0 + 2)
            cursor[ar[isc], (yt[isc] - CLAUSE_0)] = CLAUSE_TOKS
            adv = (~isc) & okc & (yt == tok_c) & (~done)
            cursor[ar[adv], s] += 1
            if bool(done.all()):
                break
        res = []
        out_np, plan_np, b_np, slots_np = (x.cpu().numpy() for x in (out, c_plan, buckets, slots))
        for i in range(B):
            ct = [self.bank.text[int(b_np[i, k])][int(slots_np[i, k])] if slots_np[i, k] >= 0 else b""
                  for k in range(N_STEPS)]
            toks = [int(v) for v in out_np[i]]
            res.append({"c_plan": plan_np[i], "buckets": [int(v) for v in b_np[i]], "tokens": toks,
                        "text": render(toks, self.vocab, subjects[i], roles[i], echoes[i], ct),
                        "slots": [int(v) for v in slots_np[i]], "clause_text": ct})
        return res


# ----------------------------------------------------------------------------- golden file
def _pad(b: bytes, n: int) -> bytes:
    b = b[:n - 1]
    return b + b"\0" * (n - len(b))


def pack_golden(cases: list[dict]) -> bytes:
    out = [PWGD_MAGIC + struct.pack("<I", len(cases))]
    for c in cases:
        out.append(GOLDEN_CASE.pack(
            np.asarray(c["c_page"], np.int8).tobytes(), np.asarray(c["c_echo"], np.int8).tobytes(),
            _pad(c["subject"], 32), _pad(c["role"], 32), _pad(c["echo_title"], 32),
            np.asarray(c["c_plan"], np.int8).tobytes(), bytes(int(b) for b in c["buckets"]),
            np.asarray(c["tokens"], "<u2").tobytes(), _pad(c["text"], TEXT_CAP)))
    return b"".join(out)


def unpack_golden(blob: bytes) -> list[dict]:
    if blob[:4] != PWGD_MAGIC:
        raise ValueError("bad PWGD magic")
    n = struct.unpack_from("<I", blob, 4)[0]
    if len(blob) != 8 + n * GOLDEN_CASE.size:
        raise ValueError("bad PWGD size")
    cases = []
    for i in range(n):
        f = GOLDEN_CASE.unpack_from(blob, 8 + i * GOLDEN_CASE.size)
        cases.append({
            "c_page": np.frombuffer(f[0], np.int8).astype(np.int64),
            "c_echo": np.frombuffer(f[1], np.int8).astype(np.int64),
            "subject": cstr(f[2]), "role": cstr(f[3]), "echo_title": cstr(f[4]),
            "c_plan": np.frombuffer(f[5], np.int8).reshape(N_STEPS, D).astype(np.int64),
            "buckets": list(f[6]), "tokens": [int(v) for v in np.frombuffer(f[7], "<u2")],
            "text": cstr(f[8])})
    return cases


# ----------------------------------------------------------------------------- PyTorch training model (QAT)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - numpy-only consumers
    torch = None
    nn = None

if torch is not None:

    def ste_round(x):
        """round half up (floor(x + 0.5)) with straight-through gradient."""
        return x + (torch.floor(x + 0.5) - x).detach()

    def ste_floor(x):
        return x + (torch.floor(x) - x).detach()

    def weight_shift(W) -> int:
        m = float(W.detach().abs().max())
        if m <= 0:
            return 7
        return int(min(30, max(0, math.floor(math.log2(127.0 / m)))))

    class QLinear(nn.Module):
        def __init__(self, n_in, n_out, std):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_out, n_in) * std)
            self.bias = nn.Parameter(torch.zeros(n_out))

        def qweight(self):
            S = weight_shift(self.weight)
            Wq = torch.clamp(ste_round(self.weight * (2.0 ** S)), -127, 127) * (2.0 ** -S)
            return Wq, S

        def forward(self, x, qat: bool):
            if not qat:
                return x @ self.weight.T + self.bias
            Wq, _ = self.qweight()
            return ste_round(x @ Wq.T) + torch.clamp(ste_round(self.bias), -32768, 32767)

        def export(self):
            S = weight_shift(self.weight)
            W = torch.clamp(torch.floor(self.weight.detach() * (2.0 ** S) + 0.5), -127, 127)
            b = torch.clamp(torch.floor(self.bias.detach() + 0.5), -32768, 32767)
            return S, b.cpu().numpy().astype(np.int64), W.cpu().numpy().astype(np.int64)

    class PeWriterQAT(nn.Module):
        """Float mirror of the integer kernel (activations live on the int8 integer scale).

        ``qat=True`` fake-quantizes every tensor and every rounding/shift (STE); ``qat=False`` is the same
        graph without rounding (Arms 0-2 train in float and are post-training quantized at export)."""

        def __init__(self, centroids: np.ndarray, use_planner=True, use_bank=True, tau: float = 16.0):
            super().__init__()
            self.use_planner = use_planner
            self.use_bank = use_bank
            self.tau = tau
            self.step_embed = nn.Parameter(torch.randn(N_STEPS, D) * 8)
            self.plan_fc1 = QLinear(D, D, 0.25)
            self.plan_fc2 = QLinear(D, D, 0.25)
            self.tok_embed = nn.Parameter(torch.randn(N_VOCAB, D) * 24)
            self.pos_embed = nn.Parameter(torch.randn(T_MAX, D) * 8)
            self.conv_in_gate = QLinear(D, D, 0.15)
            self.conv_dw = QLinear(3, D, 0.4)
            with torch.no_grad():
                self.conv_dw.weight[:, 2] += 1.0
            self.conv_out_proj = QLinear(D, D, 0.25)
            self.lm_head = QLinear(D, N_VOCAB, 0.25)
            self.copy_gate_proj = QLinear(D, 1, 0.05)
            self.register_buffer("centroids", torch.as_tensor(np.asarray(centroids), dtype=torch.float32))

        # -- helpers
        def _raw(self, p, qat):
            return torch.clamp(ste_round(p), -127, 127) if qat else torch.clamp(p, -127, 127)

        def _fl(self, x, qat):
            return ste_floor(x) if qat else x

        def _norm(self, r, qat):
            n = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-6)
            v = 127.0 * r / n
            return torch.clamp(ste_round(v), -127, 127) if qat else v

        def raw_param_names(self):
            return ["step_embed", "tok_embed", "pos_embed"]

        # -- stage 1
        def plan(self, c_page, c_echo, qat):
            if not self.use_planner:
                return c_page.unsqueeze(1).expand(-1, N_STEPS, -1)
            se = self._raw(self.step_embed, qat)
            prev, plans = c_page, []
            for s in range(N_STEPS):
                h = torch.clamp(self._fl((prev + c_echo + se[s]) / 2, qat), -127, 127)
                for _ in range(K_LOOPS):
                    a = torch.clamp(self.plan_fc1(h, qat), 0, 127)
                    h = self._norm(h + self.plan_fc2(a, qat), qat)
                plans.append(h)
                prev = h
            return torch.stack(plans, 1)

        # -- stage 2 (non-differentiable retrieval, mirrors the kernel)
        @torch.no_grad()
        def retrieve(self, c_plan, bank_vec, bank_tok, bank_valid):
            """bank_*: torch [64,64,128] float, [64,64,16] long, [64,64] bool."""
            B = c_plan.shape[0]
            dev = c_plan.device
            r = torch.zeros(B, N_STEPS, D, device=dev)
            toks = torch.zeros(B, N_STEPS, CLAUSE_TOKS, dtype=torch.long, device=dev)
            sim = torch.zeros(B, N_STEPS, device=dev)
            has = torch.zeros(B, N_STEPS, dtype=torch.bool, device=dev)
            buckets = (torch.round(c_plan) @ self.centroids.T).argmax(-1)
            if not self.use_bank or bank_vec is None:
                return r, toks, sim, has, buckets
            ar = torch.arange(B, device=dev)
            for s in range(N_STEPS):
                bv = bank_vec[buckets[:, s]]
                sc = (bv @ torch.round(c_plan[:, s]).unsqueeze(-1)).squeeze(-1)
                val = bank_valid[buckets[:, s]]
                sc = torch.where(val, sc, torch.full_like(sc, -1e18))
                best, bj = sc.max(-1)
                ok = val.any(-1)
                r[:, s] = torch.where(ok[:, None], bv[ar, bj], torch.zeros_like(r[:, s]))
                toks[:, s] = torch.where(ok[:, None], bank_tok[buckets[:, s], bj], torch.zeros_like(toks[:, s]))
                sim[:, s] = torch.where(ok, torch.clamp(torch.floor(best / 128), 0, 127), torch.zeros_like(best))
                has[:, s] = ok
            return r, toks, sim, has, buckets

        # -- stage 3 (teacher forced, parallel over t)
        def decode_logits(self, c_plan, r, prev_tok, qat):
            """prev_tok [B,T] -> (h [B,T,128], logits [B,T,V] float on the integer logit scale)."""
            B, T = prev_tok.shape
            te = self._raw(self.tok_embed, qat)
            pe = self._raw(self.pos_embed, qat)
            phase = torch.clamp(torch.arange(T, device=prev_tok.device) // PHASE_LEN, max=2)
            cond = self._fl((c_plan + r) / 2, qat)[:, phase]  # [B,T,128]
            u = torch.clamp(self._fl((te[prev_tok] + cond + pe[:T].unsqueeze(0)) / 2, qat), -127, 127)
            gs = torch.clamp(self.conv_in_gate(u, qat) + 64, 0, 128)
            y = self._fl((u * gs + 64) / 128, qat)
            y1 = F.pad(y, (0, 0, 1, 0))[:, :T]
            y2 = F.pad(y, (0, 0, 2, 0))[:, :T]
            stack = torch.stack([y2, y1, y], -1)  # [B,T,128,3]
            if qat:
                Wq, S = self.conv_dw.qweight()
                bq = torch.clamp(ste_round(self.conv_dw.bias), -32768, 32767)
                z = ste_round((stack * Wq).sum(-1)) + bq
            else:
                z = (stack * self.conv_dw.weight).sum(-1) + self.conv_dw.bias
            z = torch.clamp(z, -127, 127)
            h = torch.clamp(u + self.conv_out_proj(z, qat), -127, 127)
            return h, self.lm_head(h, qat)

        def copy_gate(self, h, qat):
            return torch.clamp(self.copy_gate_proj(h, qat)[..., 0] + 64, 0, 128)

        def apply_copy_boost(self, logits, h, ctoks, sim, targets, qat):
            """Teacher-forced clause cursor (addendum D) + differentiable boost on tok_c."""
            if not self.use_bank:
                return logits
            B, T, V = logits.shape
            dev = logits.device
            gate = self.copy_gate(h, qat)  # [B,T]
            cursor = torch.zeros(B, N_STEPS, dtype=torch.long, device=dev)
            ar = torch.arange(B, device=dev)
            boost = torch.zeros(B, T, V, device=dev)
            for t in range(T):
                s = min(2, t // PHASE_LEN)
                cur = cursor[:, s]
                tok_c = torch.where(cur < CLAUSE_TOKS, ctoks[ar, s, cur.clamp(max=CLAUSE_TOKS - 1)],
                                    torch.zeros_like(cur))
                ok = (tok_c > 0) & (tok_c < N_VOCAB)
                b = gate[:, t] * sim[:, s] / 16
                b = ste_floor(b) if qat else b
                boost[ar[ok], t, tok_c[ok]] = b[ok]
                yt = targets[:, t]
                isc = (yt >= CLAUSE_0) & (yt <= CLAUSE_0 + 2)
                cursor[ar[isc], (yt[isc] - CLAUSE_0)] = CLAUSE_TOKS
                adv = (~isc) & ok & (yt == tok_c)
                cursor[ar[adv], s] += 1
            return logits + boost

        def exclusion_mask(self, has_subject, has_echo, has_clause):
            B = has_subject.shape[0]
            m = torch.zeros(B, N_VOCAB, dtype=torch.bool, device=has_subject.device)
            m[:, BOS] = True
            m[:, ENT_SUBJECT] = ~has_subject
            m[:, BOOK_ECHO] = ~has_echo
            m[:, CLAUSE_0:CLAUSE_0 + 3] = ~has_clause
            return m

        def forward(self, batch, bank, qat: bool):
            """Teacher-forced pass -> dict(c_plan, logits [B,T,V] (masked, /tau), r, ...)."""
            c_plan = self.plan(batch["c_page"], batch["c_echo"], qat)
            bv, bt, bval = bank if bank is not None else (None, None, None)
            r, ctoks, sim, has, buckets = self.retrieve(c_plan.detach(), bv, bt, bval)
            tgt = batch["tokens"]
            prev = torch.cat([torch.full_like(tgt[:, :1], BOS), tgt[:, :-1].clamp(min=0)], 1)
            h, logits = self.decode_logits(c_plan, r, prev, qat)
            logits = self.apply_copy_boost(logits, h, ctoks, sim, tgt.clamp(min=0), qat)
            excl = self.exclusion_mask(batch["has_subject"], batch["has_echo"], has)
            logits = logits.masked_fill(excl.unsqueeze(1), -1e9)
            return {"c_plan": c_plan, "logits": logits / self.tau, "has_clause": has, "buckets": buckets}

        @torch.no_grad()
        def generate(self, batch, bank, qat: bool, rep_mask: bool = False, temperature: float = 0.0,
                     generator=None):
            """On-policy greedy/sampled float generation (no hardware repetition mask by default:
            DPO must *see* the loops it learns to reject)."""
            c_plan = self.plan(batch["c_page"], batch["c_echo"], qat)
            bv, bt, bval = bank if bank is not None else (None, None, None)
            r, ctoks, sim, has, _ = self.retrieve(c_plan, bv, bt, bval)
            B = c_plan.shape[0]
            dev = c_plan.device
            out = torch.zeros(B, T_MAX, dtype=torch.long, device=dev)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            used = torch.zeros(B, N_SPECIAL, dtype=torch.bool, device=dev)
            excl0 = self.exclusion_mask(batch["has_subject"], batch["has_echo"], has)
            for t in range(T_MAX):
                prev = torch.cat([torch.full((B, 1), BOS, dtype=torch.long, device=dev), out[:, :t]], 1)
                h, logits = self.decode_logits(c_plan, r, prev, qat)
                logits = self.apply_copy_boost(logits, h, ctoks, sim, out[:, :t + 1], qat)[:, t]
                if rep_mask and t > 0:
                    pen = torch.zeros(B, N_VOCAB, dtype=torch.bool, device=dev)
                    pen.scatter_(1, out[:, max(0, t - REP_WINDOW):t], True)
                    pen[:, :N_SPECIAL] = False
                    logits = logits - REP_PENALTY * pen.float()
                excl = excl0.clone()
                excl[:, 2:N_SPECIAL] |= used[:, 2:]
                logits = logits.masked_fill(excl, -1e9)
                if temperature > 0:
                    p = torch.softmax(logits / self.tau / temperature, -1)
                    yt = torch.multinomial(p, 1, generator=generator)[:, 0]
                else:
                    yt = logits.argmax(-1)
                yt = torch.where(done, torch.zeros_like(yt), yt)
                out[:, t] = yt
                done |= yt == PAD
                sp = (yt >= 2) & (yt < N_SPECIAL)
                used[torch.arange(B, device=dev)[sp], yt[sp]] = True
                if bool(done.all()):
                    break
            return out

        def export_tensors(self) -> dict:
            t = {}
            for n in self.raw_param_names():
                t[n] = torch.clamp(torch.floor(getattr(self, n).detach() + 0.5), -127, 127).cpu().numpy().astype(np.int64)
            t["bucket_centroids"] = self.centroids.detach().cpu().numpy().astype(np.int64)
            for n in LIN_NAMES:
                t[n] = getattr(self, n).export()
            return t

    def sigreg_loss(z, num_proj: int = 64, n_t: int = 17, generator=None):
        """LeJEPA SIGReg: Epps-Pulley distance between the empirical characteristic function of random 1-D
        projections of (standardised) embeddings and that of N(0, 1)."""
        z = z / (127.0 / math.sqrt(D))
        A = torch.randn(z.shape[-1], num_proj, generator=generator).to(z.device)
        A = A / A.norm(dim=0, keepdim=True)
        p = z @ A  # [N,P]
        ts = torch.linspace(-3, 3, n_t, device=z.device)
        tp = p.unsqueeze(-1) * ts  # [N,P,T]
        ecf_r = torch.cos(tp).mean(0)
        ecf_i = torch.sin(tp).mean(0)
        target = torch.exp(-0.5 * ts ** 2)
        w = torch.exp(-0.5 * ts ** 2)
        err = (ecf_r - target) ** 2 + ecf_i ** 2
        return torch.trapz(err * w, ts, dim=-1).mean() * z.shape[0]
