"""PE-Writer v1 on public benchmarks (zero training on any of them), device-exact int8 path.

Systems
  Arm 0..3   ``runs/pe_writer/v1_full/arm{k}.pt`` exported to PEWR int8 (``export_tensors`` + ``pack_pewr``) and
             decoded by ``PeWriterIntTorch`` (bit-exact with the device kernel); arms 0/1 run without the WCLS
             bank, arm 0 without the planner, exactly as in ``train_pe_writer.evaluate``. Arm 3's export is
             asserted byte-identical to ``weights/pe-writer.bin``.
  baselines  Random WCLS clause, Top-1 cosine WCLS clause (int8 dot with c_page over all 4,096 slots), Top-1 BM25
             WCLS clause (English tracks), Slot template (subject, role: top-1 clause, a move that recalls echo),
             and a track-specific extractive baseline (lead sentence / top-BM25 summary sentence / copy of the
             Chinese source / concept join).

Concept vectors come from the deployed ``platonic-hnet.bin`` + ``engram.bin`` through ``model.PlhnIntNp`` (the
numpy integer reference of ``ConceptEngine::encodeText``).

Tracks
  1 BookSum test (chapter level, rows with a human analysis)  -> analytical margin note vs human analysis
  2 NarrativeQA test (Wikipedia summaries as the story text)   -> riff vs human answers / summary
  3 Chinese -> English: Tatoeba cmn-eng, UD Chinese-GSDSimp test, SIGHAN-2005 PKU test
  4 CommonGen validation (allenai/common_gen)                  -> constrained 3-concept generation

Parity: 64 public inputs (16 per track) through Python int8 Arm 3 and the rc10 firmware
``ConceptEngine::generateMarginRiff`` (``verify_pe_writer_parity.compile_firmware``), compared on c_plan, buckets,
tokens and riff bytes.

    external_benchmarks/.venv/bin/python external_benchmarks/run_pe_writer_public_bench.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import platform
import re
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

EB = Path(__file__).resolve().parent
ROOT = EB.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import writer_text as BW  # noqa: E402
import export_pe_writer_int8 as X  # noqa: E402
import model as M  # noqa: E402
import pe_writer_model as PW  # noqa: E402
import train_pe_writer as TP  # noqa: E402
import verify_pe_writer_parity as VP  # noqa: E402

DATA = EB / "data"
RES = EB / "results"
CACHE = EB / "cache" / "pe_writer_bench"
RUN = ROOT / "runs" / "pe_writer" / "v1_full"  # Arms 0-3 (v1 vocab + v1 WCLS bank)
RUN4 = ROOT / "runs" / "pe_writer" / "arm4_frame_locked"  # Arm 4 (frame vocab + proper-noun-free WCLS v2 bank)
MODELS = ROOT / "weights"
FIRMWARE = ROOT / "firmware"
SCORER_EN = "sentence-transformers/all-MiniLM-L12-v2"
SCORER_ML = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FIELD = 31  # subject / role / echo_title: device char[32]
CAP = PW.TEXT_CAP - 1  # 239-byte riff cap, applied to baselines too
SEED = 20261108

ARMS = (0, 1, 2, 3, 4)
ARM_NAMES = {0: "Arm 0 flat AR", 1: "Arm 1 +Ouro planner", 2: "Arm 2 +WCLS bank", 3: "Arm 3 full PE-Writer",
             4: "Arm 4 Frame-Locked PE-Writer v2"}
B_RANDOM, B_COS, B_BM25, B_TPL = "Random WCLS clause", "Top-1 cosine WCLS clause", "Top-1 BM25 WCLS clause", \
    "Slot template + top-1 clause"
B_TPL_V1 = "Slot template + top-1 clause (v1 bank)"
SENT = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)]*\s+")
NAME_STOP = set("""The A An And But Or If When While After Before Then There Here This That These Those He She It They
We You I His Her Its Their Our Your My Mr Mrs Miss Ms Sir Lady Lord Dr St Saint King Queen Prince Princess Duke
Captain Colonel General Father Mother Brother Sister Uncle Aunt God Chapter Chapters Act Scene Book Part Section
Volume Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August
September October November December English French German Italian Russian Spanish Christian Christ Heaven Hell
Meanwhile However Although Though Because Since Once Still Even Yet Also Finally Later Soon Now One Two Three
Who What Where Why How Which Whom Whose Does Did Do Is Are Was Were Has Have Had Can Could Will Would Should
Enter Exit Exeunt Oh Yes No Well Let Thou Thee Thy Sir Madam Good Come""".split())
HONOR = r"(?:Mr|Mrs|Miss|Ms|Sir|Lady|Lord|Dr|Captain|Colonel|Count|Countess|Madame|Monsieur|Father|Mother|Uncle|Aunt)"
SPEECH = r"(?:said|asked|replied|cried|answered|exclaimed|whispered|shouted|thought|says|asks|replies)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fit_bytes(s: str, n: int) -> str:
    """<= n UTF-8 bytes at a codepoint boundary (no whitespace rewrite)."""
    b = s.encode("utf-8")
    if len(b) <= n:
        return s
    return b[:n].decode("utf-8", "ignore")


def cap_text(s: str, n: int = CAP) -> str:
    s = " ".join(s.split())
    if len(s.encode()) <= n:
        return s
    cut = fit_bytes(s, n)
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


def field(s: str) -> str:
    return BW.word_fit(s, FIELD) if s else ""


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT.split(" ".join(text.split())) if s.strip()]


def windows(text: str, max_bytes: int = M.MAX_BYTES - 1) -> list[str]:
    """Sentence-packed <= max_bytes windows (long sentences cut at word / codepoint boundaries)."""
    out, cur = [], ""
    for s in sentences(text):
        parts = [s]
        if len(s.encode()) > max_bytes:
            parts, buf = [], ""
            for w in s.split(" "):
                w = fit_bytes(w, max_bytes)
                if buf and len((buf + " " + w).encode()) > max_bytes:
                    parts.append(buf)
                    buf = w
                else:
                    buf = (buf + " " + w).strip()
            if buf:
                parts.append(buf)
        for p in parts:
            if cur and len((cur + " " + p).encode()) <= max_bytes:
                cur += " " + p
            else:
                if cur:
                    out.append(cur)
                cur = p
    if cur:
        out.append(cur)
    return out or [fit_bytes(" ".join(text.split()), max_bytes) or "-"]


def page_ctx(text: str, pos: int) -> str:
    """Device page context: DictionaryWordSelectActivity::contextAround(8) around the held word at char ``pos``."""
    words = BW.page_words(text)
    if not words:
        return fit_bytes(" ".join(text.split()), M.MAX_BYTES - 1)
    return fit_bytes(BW.context_around(words, BW.held_index(words, pos)), M.MAX_BYTES - 1)


# ----------------------------------------------------------------------------------------------- encoder
class DeviceEncoder:
    """platonic-hnet.bin + engram.bin through ``model.PlhnIntNp`` (bigram on), cached."""

    def __init__(self):
        pb, eb = (MODELS / "platonic-hnet.bin").read_bytes(), (MODELS / "engram.bin").read_bytes()
        self.ref = M.PlhnIntNp(pb, eb, bigram=True)
        self.sha = {"platonic-hnet.bin": sha256_bytes(pb), "engram.bin": sha256_bytes(eb)}
        self.bytes = {"platonic-hnet.bin": len(pb), "engram.bin": len(eb)}
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"plhn_{self.sha['platonic-hnet.bin'][:12]}_{self.sha['engram.bin'][:12]}.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.n_new = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        t0 = time.time()
        for i, t in enumerate(todo):
            self.cache[t] = np.asarray(self.ref.forward(fit_bytes(t, M.MAX_BYTES)), np.int8)
            if (i + 1) % 5000 == 0:
                log(f"  encoded {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
                self.save()
        self.n_new += len(todo)
        if todo:
            self.save()
        return np.stack([self.cache[t] for t in texts]).astype(np.int64) if texts else np.zeros((0, 128), np.int64)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self.cache, protocol=4))
        os.replace(tmp, self.path)


def pool_int8(V: np.ndarray) -> np.ndarray:
    return np.asarray(M.l2_int8([int(x) for x in V.astype(np.int64).sum(0)]), np.int64)


# ----------------------------------------------------------------------------------------------- writers
class Writers:
    """Arms 0-3 decode against the v1 bank of their run; Arm 4 (the installed writer) against its WCLS v2 bank.
    The clause baselines use the installed (v2) bank; ``B_TPL_V1`` repeats the slot template on the v1 bank."""

    def __init__(self):
        self.wcls_v1 = (RUN / "writer-clauses.bin").read_bytes()
        self.wcls = (RUN4 / "writer-clauses.bin").read_bytes()
        assert self.wcls == (MODELS / "writer-clauses.bin").read_bytes(), "Arm 4 WCLS != installed copy"
        self.blobs, self.cfg, self.refs, self.arm_wcls = {}, {}, {}, {}
        for a in ARMS:
            run = RUN4 if a == 4 else RUN
            tok = PW.Tokenizer(json.load(open(run / "vocab.json")))
            m = X.load_arm(run, a)
            self.blobs[a] = PW.pack_pewr(m.export_tensors(), tok.vocab_bytes())
            self.cfg[a] = TP.ARMS[a]
            self.arm_wcls[a] = self.wcls if a == 4 else self.wcls_v1
            self.refs[a] = PW.PeWriterIntTorch(self.blobs[a], self.arm_wcls[a], device="cpu")
        self.device_pewr = (MODELS / "pe-writer.bin").read_bytes()
        assert self.blobs[4] == self.device_pewr, "Arm 4 export differs from models/pe-writer.bin"
        self.bank = PW.Wcls(self.wcls)
        self.flat_vec = self.bank.vec.reshape(-1, PW.D)
        self.flat_valid = self.bank.valid.reshape(-1)
        self.flat_text = [t.decode("utf-8", "replace") for row in self.bank.text for t in row]
        self.valid_idx = np.flatnonzero(self.flat_valid)
        b1 = PW.Wcls(self.wcls_v1)
        self.v1_vec, self.v1_valid = b1.vec.reshape(-1, PW.D), b1.valid.reshape(-1)
        self.v1_text = [t.decode("utf-8", "replace") for row in b1.text for t in row]

    def generate(self, arm: int, c_page, c_echo, subj, role, echo) -> list[dict]:
        cfg = self.cfg[arm]
        b = lambda xs: [x.encode("utf-8") for x in xs]  # noqa: E731
        out = []
        for i in range(0, len(subj), 256):
            sl = slice(i, i + 256)
            out += self.refs[arm].generate(c_page[sl], c_echo[sl], b(subj[sl]), b(role[sl]), b(echo[sl]),
                                           use_planner=cfg["use_planner"], use_bank=cfg["use_bank"])
        return out

    def top1_cos(self, c_page: np.ndarray, v1: bool = False) -> list[int]:
        vec, valid = (self.v1_vec, self.v1_valid) if v1 else (self.flat_vec, self.flat_valid)
        S = c_page.astype(np.int64) @ vec.T
        S = np.where(valid[None, :], S, np.iinfo(np.int64).min)
        return [int(np.argmax(r)) for r in S]  # first max = lowest slot index


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
        Xd = self.cv.fit_transform(docs).astype(np.float64).tocsr()
        n = Xd.shape[0]
        df = np.bincount(Xd.indices, minlength=Xd.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = np.asarray(Xd.sum(1)).ravel()
        avg = max(dl.mean(), 1e-9)
        Xd = Xd.tocoo()
        tf = Xd.data
        w = idf[Xd.col] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[Xd.row] / avg))
        from scipy.sparse import csr_matrix
        self.W = csr_matrix((w, (Xd.row, Xd.col)), shape=Xd.shape)

    def scores(self, queries: list[str]) -> np.ndarray:
        Q = self.cv.transform(queries)
        Q.data[:] = 1.0
        return np.asarray((Q @ self.W.T).todense())


# ----------------------------------------------------------------------------------------------- metrics
STOP = TP.STOPWORDS


def words(s: str) -> list[str]:
    return [w.lower() for w in PW.word_tokens(s) if any(ch.isalnum() for ch in w)]


def f1_counts(h, r) -> float:
    if not h or not r:
        return 0.0
    c = sum((Counter(h) & Counter(r)).values())
    if c == 0:
        return 0.0
    p, q = c / len(h), c / len(r)
    return 2 * p * q / (p + q)


def rouge_l(h, r) -> float:
    L = TP.lcs(h, r)
    if not L:
        return 0.0
    p, q = L / len(h), L / len(r)
    return 2 * p * q / (p + q)


def rep3(ws) -> bool:
    g = [tuple(ws[j:j + 3]) for j in range(len(ws) - 2)]
    return len(g) != len(set(g))


def proper_nouns(text: str) -> list[str]:
    """Capitalised Latin words that are not sentence-initial and not template / function words
    (``train_pe_writer.invented_name`` rule without the grounding check)."""
    toks = PW.word_tokens(text)
    out = []
    for j, w in enumerate(toks):
        if j == 0 or not re.match(r"[A-Z]", w):
            continue
        if toks[j - 1] in (".", "!", "?") or w in TP.TEMPLATE_CAPS:
            continue
        out.append(w)
    return out


class Scorer:
    def __init__(self, repo: str, device: str):
        from sentence_transformers import SentenceTransformer
        self.repo = repo
        self.m = SentenceTransformer(repo, device=device)
        self.max_words = 90
        self.cache = {}

    def emb(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            E = self.m.encode(todo, batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True)
            for t, e in zip(todo, E):
                self.cache[t] = e.astype(np.float32)
        return np.stack([self.cache[t] for t in texts])

    def emb_long(self, texts: list[str]) -> np.ndarray:
        """Long references: sentence-packed <= 90-word chunks (the scorer reads 128 word pieces), mean of the
        normalised chunk embeddings, renormalised."""
        chunks, own = [], []
        for i, t in enumerate(texts):
            start, cur = len(own), []
            for s in sentences(t) or [t]:
                ws = s.split()
                while len(ws) > self.max_words:
                    if cur:
                        chunks.append(" ".join(cur)); own.append(i); cur = []  # noqa: E702
                    chunks.append(" ".join(ws[:self.max_words])); own.append(i)  # noqa: E702
                    ws = ws[self.max_words:]
                if cur and len(cur) + len(ws) > self.max_words:
                    chunks.append(" ".join(cur)); own.append(i); cur = []  # noqa: E702
                cur += ws
            if cur or len(own) == start:
                chunks.append(" ".join(cur) or "-"); own.append(i)  # noqa: E702
        E = self.emb(chunks)
        own = np.asarray(own)
        out = np.zeros((len(texts), E.shape[1]), np.float32)
        np.add.at(out, own, E)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def boot_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def score_system(outs: list[str], items: list[dict], refs_short: list[list[str]], sem: np.ndarray,
                 future: list[set] | None = None, extra: dict | None = None) -> tuple[dict, dict]:
    """-> (aggregate metrics, per-sample arrays). ``sem``: per-sample semantic cosine (already maxed over refs)."""
    n = len(outs)
    per = defaultdict(list)
    bigrams_all, bigrams_uniq = 0, set()
    for i, o in enumerate(outs):
        it = items[i]
        hw = words(o)
        rs = [words(r) for r in refs_short[i]] or [[]]
        per["rouge1"].append(max(f1_counts(hw, r) for r in rs))
        per["rougeL"].append(max(rouge_l(hw, r) for r in rs))
        per["content_f1"].append(max(f1_counts([w for w in hw if w not in STOP], [w for w in r if w not in STOP])
                                     for r in rs))
        per["rep3"].append(float(rep3(hw)))
        bg = list(zip(hw, hw[1:]))
        bigrams_all += len(bg)
        bigrams_uniq.update(bg)
        pn = proper_nouns(o)
        inv = [w for w in pn if w not in it["allowed"]]
        per["invented"].append(float(bool(inv)))
        if pn:
            per["entity_prec"].append(1 - len(inv) / len(pn))
        low = o.lower()
        sw = (not it["subject"]) or it["subject"].lower() in low
        ew = (not it["echo"]) or it["echo"].lower() in low
        per["subject_present"].append(float(sw))
        per["weave"].append(float(sw and ew))
        per["grounded"].append(float(sw and ew and not inv))
        per["words"].append(len(hw))
        per["empty"].append(float(not hw))
        per["truncated"].append(float(len(o.encode()) >= PW.TEXT_CAP - 4))
        per["broken_tail"].append(float(bool(hw) and not re.search(r"[.!?][\"'\u201d\u2019)]*\s*$", o)))
        if future is not None:
            per["spoiler"].append(float(any(w in future[i] for w in re.findall(r"[A-Z][a-z]{2,}", o))))
    per["sem"] = [float(v) for v in sem]
    for k, v in (extra or {}).items():
        per[k] = list(v)
    agg = {"n": n,
           "minilm_cos_x100": round(100 * float(np.mean(per["sem"])), 2),
           "minilm_cos_x100_ci95": [round(100 * v, 2) for v in boot_ci(per["sem"])],
           "rouge1_f": round(float(np.mean(per["rouge1"])), 4),
           "rougeL_f": round(float(np.mean(per["rougeL"])), 4),
           "rougeL_f_ci95": [round(v, 4) for v in boot_ci(per["rougeL"])],
           "content_f1": round(float(np.mean(per["content_f1"])), 4),
           "rep3_pct": round(100 * float(np.mean(per["rep3"])), 2),
           "distinct2_pct": round(100 * len(bigrams_uniq) / max(1, bigrams_all), 2),
           "grounding_precision_pct": round(100 * float(np.mean(per["grounded"])), 2),
           "entity_weave_pct": round(100 * float(np.mean(per["weave"])), 2),
           "subject_present_pct": round(100 * float(np.mean(per["subject_present"])), 2),
           "invented_proper_noun_pct": round(100 * float(np.mean(per["invented"])), 2),
           "entity_token_precision_pct": (round(100 * float(np.mean(per["entity_prec"])), 2)
                                          if per["entity_prec"] else None),
           "n_with_proper_nouns": len(per["entity_prec"]),
           "mean_words": round(float(np.mean(per["words"])), 2),
           "empty_pct": round(100 * float(np.mean(per["empty"])), 2),
           "truncated_239B_pct": round(100 * float(np.mean(per["truncated"])), 2),
           "broken_tail_pct": round(100 * float(np.mean(per["broken_tail"])), 2)}
    if future is not None:
        agg["spoiler_leak_pct"] = round(100 * float(np.mean(per["spoiler"])), 2)
    for k, v in (extra or {}).items():
        agg[k] = round(float(np.mean(v)), 4) if "pct" not in k else round(100 * float(np.mean(v)), 2)
    return agg, dict(per)


# ----------------------------------------------------------------------------------------------- entity helpers
def mid_caps(text: str) -> Counter:
    c = Counter()
    for s in SENT.split(" ".join(text.split())):
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
        for j, w in enumerate(toks):
            if j > 0 and re.fullmatch(r"[A-Z][a-z]{2,}", w) and w not in NAME_STOP:
                c[w] += 1
    return c


def role_for(name: str, text: str) -> str:
    """Deterministic role phrase: person evidence (honorific / speech verb / he-she right after a mention)."""
    if not name:
        return ""
    n = re.escape(name)
    person = (re.search(rf"\b{HONOR}\.? {n}\b", text) or re.search(rf"\b{n},? {SPEECH}\b", text)
              or re.search(rf"\b{SPEECH} {n}\b", text)
              or len(re.findall(rf"\b{n}\b[^.!?]{{0,40}}\b(?:he|she|his|her|him)\b", text)) >= 2)
    return "a central character" if person else "a recurring name"


# ----------------------------------------------------------------------------------------------- track builders
def load_booksum():
    import csv as _csv
    _csv.field_size_limit(1 << 30)
    p = DATA / "booksum" / "test.csv"
    rows = list(_csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        m = re.search(r"section_(\d+)_part_(\d+)", r["summary_path"])
        r["sec"], r["part"] = int(m.group(1)), int(m.group(2))
        r["chapter"] = r["chapter"].strip()
        r["summary_analysis"] = r["summary_analysis"].strip()
        r["title"] = r["book_id"].split(".")[0].strip()
    rows = [r for r in rows if r["chapter"]]
    seqs = defaultdict(list)
    for r in rows:
        seqs[(r["bid"], r["source"])].append(r)
    for k in seqs:
        seqs[k].sort(key=lambda r: (r["sec"], r["part"]))
    return seqs, {"file": "data/booksum/test.csv", "sha256": sha256_file(p), "hf_repo": "kmfoda/booksum",
                  "split": "test (chapter level)"}


def build_track1(enc: DeviceEncoder, min_analysis: int):
    seqs, info = load_booksum()
    items = []
    for (bid, src), rs in sorted(seqs.items()):
        chaps = [r["chapter"] for r in rs]
        caps = [mid_caps(c) for c in chaps]
        lower = Counter(w for c in chaps for w in re.findall(r"\b[a-z][a-z']+\b", c))
        tot = Counter()
        for c in caps:
            tot.update(c)
        ents = {w for w, k in tot.items() if k >= 3 and w.lower() not in lower}
        first = {}
        for i, c in enumerate(chaps):
            for w in set(re.findall(r"\b[A-Z][a-z]{2,}\b", c)) & ents:
                first.setdefault(w, i)
        seen_tokens = set()
        for i, r in enumerate(rs):
            seen_tokens |= set(PW.word_tokens(r["chapter"]))
            if len(r["summary_analysis"]) < min_analysis:
                continue
            ch = r["chapter"]
            cand = [(caps[i][w], w) for w in ents if caps[i][w] > 0]
            if cand:
                subj = max(cand, key=lambda z: (z[0], [-ord(c) for c in z[1]]))[1]
            elif caps[i]:
                subj = max(caps[i].items(), key=lambda z: (z[1], [-ord(c) for c in z[0]]))[0]
            else:
                subj = ""
            m = re.search(rf"\b{re.escape(subj)}\b", ch) if subj else None
            pctx = page_ctx(ch, m.start()) if m else windows(ch)[0]
            echo_text, echo_kind = None, None
            for j in range(i - 1, -1, -1):
                mj = re.search(rf"\b{re.escape(subj)}\b", chaps[j]) if subj else None
                if mj:
                    echo_text, echo_kind = page_ctx(chaps[j], mj.start()), f"earlier chapter (pos {j + 1}) mention"
                    break
            if echo_text is None:
                if i > 0:
                    echo_text, echo_kind = windows(chaps[i - 1])[0], f"earlier chapter (pos {i}) opening"
                else:
                    echo_text, echo_kind = windows(ch)[0], "first chapter: own opening window"
            future = {w for w in ents if first.get(w, -1) > i}
            sents = []
            for para in re.split(r"\n\s*\n", ch):
                ws = para.split()
                if len(ws) >= 6 and not para.strip().isupper():
                    sents = sentences(para)
                    break
            lead = cap_text(sents[0] if sents else " ".join(ch.split()[:40]))
            fields = {"subject": field(subj), "role": role_for(subj, ch), "echo": field(r["title"])}
            items.append({"id": f"{bid}/{src}/{r['sec']}.{r['part']}", "book": r["title"], "source": src,
                          "pos": i + 1, "seq_len": len(rs), **fields,
                          "page_text": pctx, "echo_text": echo_text, "echo_kind": echo_kind,
                          "query": pctx, "chapter_path": r["chapter_path"],
                          "refs": [r["summary_analysis"]], "ref_long": r["summary_analysis"],
                          "extractive": lead, "future": future,
                          "allowed": seen_tokens | set(PW.word_tokens(" ".join(fields.values()))),
                          "_chapter": ch})
    info["n_rows_with_analysis"] = len(items)
    info["books"] = len({it["book"] for it in items})
    info["sources"] = dict(Counter(it["source"] for it in items))
    return items, info


def load_nqa():
    base = DATA / "narrativeqa" / "narrativeqa-master"
    z = DATA / "narrativeqa" / "data" / "narrativeqa-master.zip"
    if not (base / "qaps.csv").exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA / "narrativeqa")
    csv.field_size_limit(1 << 30)
    summ = {r["document_id"]: r["summary"] for r in
            csv.DictReader(open(base / "third_party" / "wikipedia" / "summaries.csv", encoding="utf-8"))
            if r["set"] == "test"}
    docs = {r["document_id"]: r for r in csv.DictReader(open(base / "documents.csv", encoding="utf-8"))
            if r["set"] == "test"}
    qs = [r for r in csv.DictReader(open(base / "qaps.csv", encoding="utf-8")) if r["set"] == "test"]
    return summ, docs, qs, {"file": "data/narrativeqa/data/narrativeqa-master.zip", "sha256": sha256_file(z),
                            "hf_repo": "deepmind/narrativeqa", "split": "test",
                            "story_text": "third_party/wikipedia/summaries.csv (summary setting; full scripts/books "
                                          "not downloaded)"}


def question_entity(q: str, summary: str) -> str:
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", q)
    spans, cur = [], []
    for j, w in enumerate(toks):
        w2 = re.sub(r"'s$", "", w)
        if re.match(r"[A-Z]", w2) and not (j == 0 and w2 in NAME_STOP):
            cur.append(w2)
        else:
            if cur:
                spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    for sp in spans:
        while sp and sp[0] in NAME_STOP and len(sp) > 1 and not re.match(HONOR + "$", sp[0]):
            sp = sp[1:]
        name = " ".join(sp)
        if name and name not in NAME_STOP and re.search(rf"\b{re.escape(name)}\b", summary):
            return name
    c = mid_caps(summary)
    return max(c.items(), key=lambda z: (z[1], [-ord(ch) for ch in z[0]]))[0] if c else ""


def build_track2(per_story: int):
    summ, docs, qs, info = load_nqa()
    byq = defaultdict(list)
    for r in qs:
        byq[r["document_id"]].append(r)
    items = []
    for did in sorted(summ):
        if did not in docs or not byq.get(did):
            continue
        s = " ".join(summ[did].split())
        ws = windows(s)
        bm = BM25(ws) if len(ws) > 1 else None
        title = re.sub(r"\s*\((?:film|novel|play|book)[^)]*\)$", "", docs[did]["wiki_title"]).strip()
        for q in byq[did][:per_story]:
            if bm is not None:
                sc = bm.scores([q["question"]])[0]
                order = np.argsort(-sc, kind="stable")
                top, sec = ws[order[0]], ws[order[1]]
            else:
                top = sec = ws[0]
            subj = question_entity(q["question"], s)
            items.append({"id": f"{did[:10]}/{len(items)}", "book": title, "question": q["question"],
                          "subject": field(subj), "role": role_for(subj, s), "echo": field(title),
                          "page_text": top, "echo_text": sec, "query": q["question"] + " " + top,
                          "refs": [q["answer1"], q["answer2"]], "ref_long": s,
                          "extractive": cap_text(sentences(top)[0] if sentences(top) else top),
                          "allowed": set(PW.word_tokens(s + " " + q["question"] + " " + title))})
    info["stories"] = len({it["id"].split("/")[0] for it in items})
    info["questions_per_story"] = per_story
    return items, info


def zh_subject(text: str) -> tuple[str, str]:
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(60)
    toks = [(w, f) for w, f in pseg.lcut(text) if w.strip()]
    for want, role in (("nr", "a person"), ("ns", "a place"), ("nt", "an organization"), ("nz", "a proper name")):
        for w, f in toks:
            if f.startswith(want) and len(w) >= 2:
                return w, role
    for w, f in toks:
        if f.startswith("n") and len(w) >= 2:
            return w, "a concept"
    for w, f in toks:
        if f.startswith("n"):
            return w, "a concept"
    return (toks[0][0] if toks else ""), "a concept"


def build_track3(n_pku: int):
    cjk = re.compile(r"[\u4e00-\u9fff]")
    subs = {}
    tat = [json.loads(l) for l in gzip.open(DATA / "tatoeba" / "test" / "cmn-eng.jsonl.gz", "rt", encoding="utf-8")]
    subs["tatoeba_cmn_eng"] = ([(r["sentence1"].strip(), r["sentence2"].strip()) for r in tat], "Tatoeba",
                               {"file": "data/tatoeba/test/cmn-eng.jsonl.gz",
                                "sha256": sha256_file(DATA / "tatoeba" / "test" / "cmn-eng.jsonl.gz"),
                                "hf_repo": "mteb/tatoeba-bitext-mining (cmn-eng test)"})
    gsd = [l[len("# text = "):].strip() for l in open(DATA / "ud_zh_gsdsimp" / "zh_gsdsimp-ud-test.conllu",
                                                     encoding="utf-8") if l.startswith("# text = ")]
    subs["ud_zh_gsdsimp_test"] = ([(t, None) for t in gsd], "维基百科",
                                  {"file": "data/ud_zh_gsdsimp/zh_gsdsimp-ud-test.conllu",
                                   "sha256": sha256_file(DATA / "ud_zh_gsdsimp" / "zh_gsdsimp-ud-test.conllu"),
                                   "source": "Universal Dependencies UD_Chinese-GSDSimp test (no English side)"})
    pku_all = [l.strip() for l in open(DATA / "sighan2005" / "pku_test.utf8", encoding="utf-8")]
    pku_ok = [l for l in pku_all if len(cjk.findall(l)) >= 15]
    sel = sorted(set(np.linspace(0, len(pku_ok) - 1, min(n_pku, len(pku_ok))).round().astype(int).tolist()))
    subs["sighan2005_pku_test"] = ([(pku_ok[i], None) for i in sel], "人民日报",
                                   {"file": "data/sighan2005/pku_test.utf8",
                                    "sha256": sha256_file(DATA / "sighan2005" / "pku_test.utf8"),
                                    "source": "SIGHAN-2005 PKU test (People's Daily), lines with >= 15 CJK chars, "
                                              f"{len(sel)} evenly spaced of {len(pku_ok)}"})
    out = {}
    for name, (pairs, label, info) in subs.items():
        items = []
        for i, (zh, en) in enumerate(pairs):
            subj, role = zh_subject(zh)
            prev = pairs[i - 1][0] if i > 0 else pairs[-1][0]
            zfit = fit_bytes(zh, M.MAX_BYTES - 1)
            items.append({"id": f"{name}/{i}", "zh": zh, "subject": field(subj), "role": role, "echo": label,
                          "page_text": zfit, "echo_text": fit_bytes(prev, M.MAX_BYTES - 1), "query": zfit,
                          "refs": [en] if en else [], "ref_en": en, "extractive": cap_text(zh),
                          "allowed": set(PW.word_tokens(zh + " " + subj + " " + role + " " + label))})
        info["n"] = len(items)
        out[name] = (items, info)
    return out


IRREGULAR = dict(p.split(":") for p in """stood:stand ran:run sat:sit took:take taken:take threw:throw thrown:throw
went:go gone:go came:come ate:eat eaten:eat drank:drink drunk:drink drove:drive driven:drive rode:ride ridden:ride
wrote:write written:write gave:give given:give saw:see seen:see held:hold hung:hang caught:catch bought:buy
brought:bring built:build fell:fall fallen:fall flew:fly flown:fly grew:grow grown:grow hit:hit kept:keep led:lead
left:leave lay:lie made:make met:meet paid:pay put:put read:read rang:ring sang:sing sung:sing sank:sink slept:sleep
spoke:speak spoken:speak spent:spend swam:swim swum:swim taught:teach told:tell thought:think wore:wear worn:wear
won:win cut:cut dug:dig fed:feed fought:fight found:find froze:freeze frozen:freeze knelt:kneel knew:know
lit:light shot:shoot shook:shake sold:sell sent:send shone:shine slid:slide stuck:stick struck:strike swept:sweep
swung:swing tore:tear torn:tear woke:wake men:man women:woman children:child feet:foot teeth:tooth mice:mouse
geese:goose people:person leaves:leaf knives:knife wolves:wolf""".split())


def stem(w: str) -> str:
    w = IRREGULAR.get(w.lower(), w.lower())
    for suf in ("ing", "ed", "es", "s", "er"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiouls":
        w = w[:-1]
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def build_track4():
    import pyarrow.parquet as pq
    p = DATA / "commongen" / "validation.parquet"
    rows = pq.read_table(p).to_pylist()
    sets = {}
    for r in rows:
        s = sets.setdefault(r["concept_set_idx"], {"concepts": list(r["concepts"]), "refs": []})
        s["refs"].append(r["target"])
    items = []
    for k in sorted(sets):
        c, refs = sets[k]["concepts"], sets[k]["refs"]
        pad = (c + ["", ""])[:3]
        items.append({"id": f"commongen/{k}", "concepts": c, "subject": field(pad[0]), "role": field(pad[1]),
                      "echo": field(pad[2]), "page_text": " ".join(c), "echo_text": pad[2] or " ".join(c),
                      "query": " ".join(c), "refs": refs, "extractive": " ".join(c) + ".",
                      "allowed": set(PW.word_tokens(" ".join(c)))})
    return items, {"file": "data/commongen/validation.parquet", "sha256": sha256_file(p),
                   "hf_repo": "allenai/common_gen", "split": "validation",
                   "url": "https://huggingface.co/api/datasets/allenai/common_gen/parquet/default/validation/0.parquet",
                   "rows": len(rows), "concept_sets": len(items)}


def coverage(o: str, concepts: list[str]) -> float:
    st = {stem(w) for w in words(o)}
    return float(np.mean([stem(c) in st for c in concepts])) if concepts else 0.0


# ----------------------------------------------------------------------------------------------- run one track
def run_systems(W: Writers, enc: DeviceEncoder, items: list[dict], extractive_name: str, use_bm25: bool,
                c_page=None, rng_seed: int = SEED, arms=ARMS) -> tuple[dict, np.ndarray, np.ndarray]:
    """-> ({system: list[str]}, c_page, c_echo)."""
    if c_page is None:
        c_page = enc.encode([it["page_text"] for it in items])
    c_echo = enc.encode([it["echo_text"] for it in items])
    subj = [it["subject"] for it in items]
    role = [it["role"] for it in items]
    echo = [it["echo"] for it in items]
    outs = {}
    for a in arms:
        t0 = time.time()
        res = W.generate(a, c_page, c_echo, subj, role, echo)
        outs[ARM_NAMES[a]] = [r["text"].decode("utf-8", "replace") for r in res]
        log(f"    {ARM_NAMES[a]}: {len(res)} riffs in {time.time() - t0:.1f}s")
    rng = np.random.default_rng(rng_seed)
    outs[B_RANDOM] = [W.flat_text[int(rng.choice(W.valid_idx))] for _ in items]
    top = W.top1_cos(c_page)
    outs[B_COS] = [W.flat_text[j] for j in top]
    if use_bm25:
        bm = BM25([t if t else "-" for t in W.flat_text])
        S = bm.scores([it["query"] for it in items])
        S[:, ~W.flat_valid] = -1
        outs[B_BM25] = [W.flat_text[int(np.argmax(r))] for r in S]
    def template(top, texts):
        out = []
        for it, j in zip(items, top):
            head = f"{it['subject']}, {it['role'] or PW.ROLE_FALLBACK.decode()}: " if it["subject"] else ""
            tail = f", a move that recalls {it['echo']}." if it["echo"] else "."
            out.append(cap_text(head + texts[j] + tail))
        return out

    outs[B_TPL] = template(top, W.flat_text)
    outs[B_TPL_V1] = template(W.top1_cos(c_page, v1=True), W.v1_text)
    outs[extractive_name] = [it["extractive"] for it in items]
    return outs, c_page, c_echo


def eval_track(outs: dict, items: list[dict], scorer: Scorer, ref_mode: str, future=None, extra_fn=None,
               long_ref_scorer: Scorer | None = None) -> dict:
    """ref_mode: 'long' (one long reference, chunk-pooled), 'multi' (max over short refs), 'zh' (Tatoeba / zh)."""
    res = {}
    if ref_mode == "long":
        R = scorer.emb_long([it["ref_long"] for it in items])
    elif ref_mode == "multi":
        flat = [r for it in items for r in it["refs"]]
        RE = scorer.emb(flat)
        own = np.repeat(np.arange(len(items)), [len(it["refs"]) for it in items])
    for name, o in outs.items():
        H = scorer.emb([x if x else "-" for x in o])
        empty = np.array([not x.strip() for x in o])
        extra = {}
        if ref_mode == "long":
            sem = (H * R).sum(1)
        elif ref_mode == "multi":
            s = (H[own] * RE).sum(1)
            sem = np.full(len(items), -1.0)
            np.maximum.at(sem, own, s)
        else:  # zh: semantic vs English ref (if any) and vs the Chinese source
            Z = scorer.emb([it["zh"] for it in items])
            semz = (H * Z).sum(1)
            if items[0]["ref_en"]:
                E = scorer.emb([it["ref_en"] for it in items])
                sem = (H * E).sum(1)
            else:
                sem = semz
            extra["minilm_ml_cos_to_zh_source_x100"] = np.where(empty, 0.0, semz) * 100
        if long_ref_scorer is not None:
            Hl = long_ref_scorer.emb([x if x else "-" for x in o])
            Rl = long_ref_scorer.emb_long([it["ref_long"] for it in items])
            extra["minilm_cos_to_summary_x100"] = np.where(empty, 0.0, (Hl * Rl).sum(1)) * 100
        if extra_fn is not None:
            extra.update(extra_fn(o))
        sem = np.where(empty, 0.0, sem)
        agg, per = score_system(o, items, [it["refs"] for it in items], sem, future=future, extra=extra)
        res[name] = {"metrics": agg, "_per": per}
    return res


def paired(res: dict, a: str, b: str, key: str = "sem", n_boot: int = 1000) -> dict:
    x = np.asarray(res[a]["_per"][key]) - np.asarray(res[b]["_per"][key])
    lo, hi = boot_ci(x, n_boot)
    scale = 100 if key == "sem" else 1
    return {"a": a, "b": b, "metric": key, "mean_diff": round(scale * float(x.mean()), 3),
            "ci95": [round(scale * lo, 3), round(scale * hi, 3)],
            "a_wins_pct": round(100 * float((x > 0).mean()), 1), "ties_pct": round(100 * float((x == 0).mean()), 1)}


def examples(outs: dict, items: list[dict], idx: list[int], show: list[str], ref_key="ref") -> list[dict]:
    ex = []
    for i in idx:
        it = items[i]
        e = {"id": it["id"], "subject": it["subject"], "role": it["role"], "echo_title": it["echo"],
             "page_text": it["page_text"][:240]}
        for k in ("book", "question", "zh", "concepts"):
            if k in it:
                e[k] = it[k]
        if it.get("refs"):
            e["reference"] = [r[:400] for r in it["refs"][:2]]
        e["outputs"] = {k: outs[k][i] for k in show if k in outs}
        ex.append(e)
    return ex


# ----------------------------------------------------------------------------------------------- footprint / latency
def tensor_bytes(names) -> int:
    return sum(PW._tensor_bytes(k, s) for n, k, s in PW.PEWR_LAYOUT if n in names)


def footprint() -> dict:
    all_names = {n for n, _, _ in PW.PEWR_LAYOUT}
    vocab = PW.N_VOCAB * PW.VOCAB_STR
    plan = {"step_embed", "plan_fc1", "plan_fc2"}
    a0 = all_names - plan - {"bucket_centroids"}
    a1 = all_names - {"bucket_centroids"}
    bank_buf = PW.N_SLOTS * PW.SLOT_BYTES
    f = {ARM_NAMES[0]: {"psram_bytes": 32 + tensor_bytes(a0) + vocab, "sd_bytes": 0, "stream_buf_bytes": 0},
         ARM_NAMES[1]: {"psram_bytes": 32 + tensor_bytes(a1) + vocab, "sd_bytes": 0, "stream_buf_bytes": 0},
         ARM_NAMES[2]: {"psram_bytes": PW.PEWR_FILE_BYTES, "sd_bytes": PW.WCLS_FILE_BYTES,
                        "stream_buf_bytes": bank_buf},
         ARM_NAMES[3]: {"psram_bytes": PW.PEWR_FILE_BYTES, "sd_bytes": PW.WCLS_FILE_BYTES,
                        "stream_buf_bytes": bank_buf},
         ARM_NAMES[4]: {"psram_bytes": PW.PEWR_FILE_BYTES, "sd_bytes": PW.WCLS_FILE_BYTES,
                        "stream_buf_bytes": bank_buf},
         B_RANDOM: {"psram_bytes": 0, "sd_bytes": PW.WCLS_FILE_BYTES, "stream_buf_bytes": PW.SLOT_BYTES},
         B_COS: {"psram_bytes": 0, "sd_bytes": PW.WCLS_FILE_BYTES, "stream_buf_bytes": bank_buf,
                 "note": "scans all 4,096 slots (1 MB SD read per query)"},
         B_BM25: {"psram_bytes": None, "sd_bytes": PW.WCLS_FILE_BYTES, "stream_buf_bytes": None,
                  "note": "needs an inverted index over the clause bank (not on device)"},
         B_TPL: {"psram_bytes": 0, "sd_bytes": PW.WCLS_FILE_BYTES, "stream_buf_bytes": bank_buf},
         B_TPL_V1: {"psram_bytes": 0, "sd_bytes": PW.WCLS_FILE_BYTES, "stream_buf_bytes": bank_buf},
         "Concept join (CommonGen extractive)": {"psram_bytes": 0, "sd_bytes": 0, "stream_buf_bytes": 0}}
    for v in f.values():
        if v["psram_bytes"] is not None:
            v["total_bytes"] = v["psram_bytes"] + v["sd_bytes"]
    return f


def latency(W: Writers, enc: DeviceEncoder, items: list[dict], n: int = 48) -> dict:
    """Host single-example latency (Apple-silicon CPU, numpy integer reference = line-by-line device kernel)."""
    it = items[:n]
    cp = enc.encode([x["page_text"] for x in it])
    ce = enc.encode([x["echo_text"] for x in it])
    out = {}
    for a in ARMS:
        ref = PW.PeWriterIntNp(W.blobs[a], W.arm_wcls[a])
        cfg = W.cfg[a]
        t0 = time.perf_counter()
        for i, x in enumerate(it):
            ref.generate(cp[i], ce[i], x["subject"].encode(), x["role"].encode(), x["echo"].encode(),
                         use_planner=cfg["use_planner"], use_bank=cfg["use_bank"])
        out[ARM_NAMES[a]] = round(1000 * (time.perf_counter() - t0) / len(it), 3)
    t0 = time.perf_counter()
    W.top1_cos(cp)
    out[B_COS] = round(1000 * (time.perf_counter() - t0) / len(it), 3)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in it:
        W.flat_text[int(rng.choice(W.valid_idx))]
    out[B_RANDOM] = round(1000 * (time.perf_counter() - t0) / len(it), 4)
    t0 = time.perf_counter()
    top = W.top1_cos(cp)
    for x, j in zip(it, top):
        cap_text(f"{x['subject']}, {x['role']}: {W.flat_text[j]}, a move that recalls {x['echo']}.")
    out[B_TPL] = round(1000 * (time.perf_counter() - t0) / len(it), 3)
    t0 = time.perf_counter()
    for x in it:
        " ".join(x["concepts"]) + "."
    out["Concept join (CommonGen extractive)"] = round(1000 * (time.perf_counter() - t0) / len(it), 5)
    t0 = time.perf_counter()
    for x in it:
        enc.ref.forward(x["page_text"])
    out["encoder_platonic_hnet_ms"] = round(1000 * (time.perf_counter() - t0) / len(it), 3)
    return out


def pareto(points: dict) -> list[str]:
    """Non-dominated systems: minimise total bytes and latency, maximise quality."""
    keep = []
    for a, pa in points.items():
        dom = False
        for b, pb in points.items():
            if a == b:
                continue
            if (pb["bytes"] <= pa["bytes"] and pb["ms"] <= pa["ms"] and pb["q"] >= pa["q"]
                    and (pb["bytes"] < pa["bytes"] or pb["ms"] < pa["ms"] or pb["q"] > pa["q"])):
                dom = True
                break
        if not dom:
            keep.append(a)
    return keep


# ----------------------------------------------------------------------------------------------- parity
def parity(W: Writers, enc: DeviceEncoder, track_items: dict, per_track: int) -> dict:
    cases, labels = [], []
    for name, items in track_items.items():
        for it in items[:per_track]:
            cases.append({"c_page": None, "page_text": it["page_text"], "echo_text": it["echo_text"],
                          "subject": it["subject"].encode(), "role": it["role"].encode(),
                          "echo_title": it["echo"].encode()})
            labels.append(f"{name}:{it['id']}")
    cp = enc.encode([c["page_text"] for c in cases])
    ce = enc.encode([c["echo_text"] for c in cases])
    for i, c in enumerate(cases):
        c["c_page"], c["c_echo"] = cp[i], ce[i]
    ref = PW.PeWriterIntTorch(W.device_pewr, W.wcls, device="cpu")
    py = ref.generate(cp, ce, [c["subject"] for c in cases], [c["role"] for c in cases],
                      [c["echo_title"] for c in cases])
    npref = PW.PeWriterIntNp(W.device_pewr, W.wcls)
    pn = [npref.generate(c["c_page"], c["c_echo"], c["subject"], c["role"], c["echo_title"]) for c in cases]
    td = tempfile.TemporaryDirectory(prefix="pe-writer-public-parity-")
    exe = Path(td.name) / "pe_writer_fw"
    built = VP.compile_firmware(FIRMWARE, exe)
    if not built:
        raise SystemExit(f"firmware generateMarginRiff not found in {FIRMWARE}")
    t0 = time.perf_counter()
    fw = VP.run_cpp([str(exe), str(MODELS)], cases)
    fw_ms = 1000 * (time.perf_counter() - t0) / len(cases)
    mism, mism_np, fields = [], 0, Counter()
    for i in range(len(cases)):
        d = VP.same(py[i], fw[i])
        if d:
            mism.append({"case": labels[i], "fields": d, "py_text": py[i]["text"].decode("utf-8", "replace"),
                         "fw_text": fw[i]["text"].decode("utf-8", "replace")})
            fields.update(d)
        mism_np += bool(VP.same(pn[i], fw[i]))
    return {"cases": len(cases), "per_track": per_track,
            "bit_exact_python_int8_vs_firmware": len(cases) - len(mism),
            "bit_exact_numpy_ref_vs_firmware": len(cases) - mism_np,
            "compared_fields": ["c_plan[3][128]", "buckets[3]", "tokens[36]", "riff bytes"],
            "mismatches": mism, "mismatch_fields": dict(fields),
            "firmware_source": str(FIRMWARE / "ConceptEngine.cpp"),
            "firmware_sha256": sha256_file(FIRMWARE / "ConceptEngine.cpp"),
            "firmware_git_head": os.popen(f"git -C {FIRMWARE} rev-parse --short HEAD 2>/dev/null").read().strip(),
            "models_dir": str(MODELS), "pe_writer_bin_sha256": sha256_bytes(W.device_pewr),
            "writer_clauses_bin_sha256": sha256_bytes(W.wcls),
            "firmware_host_ms_per_riff_incl_process_and_file_io": round(fw_ms, 3),
            "case_labels": labels,
            "sample": [{"case": labels[i], "riff": fw[i]["text"].decode("utf-8", "replace"),
                        "buckets": fw[i]["buckets"]} for i in range(0, len(cases), max(1, len(cases) // 8))]}


# ----------------------------------------------------------------------------------------------- report
def fmt(v, d=2):
    if v is None:
        return "–"
    if isinstance(v, float) and math.isnan(v):
        return "–"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def table(sysres: dict, cols: list[tuple[str, str, int]]) -> list[str]:
    L = ["| system | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, r in sysres.items():
        m = r["metrics"]
        cells = []
        for k, _, d in cols:
            v = m.get(k)
            if k == "minilm_cos_x100":
                lo, hi = m["minilm_cos_x100_ci95"]
                cells.append(f"{v:.2f} [{lo:.1f}, {hi:.1f}]")
            else:
                cells.append(fmt(v, d))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L


def ex_md(exs: list[dict], keys: list[str]) -> list[str]:
    L = []
    for e in exs:
        head = [f"**{e['id']}**"]
        for k in ("book", "question", "zh", "concepts"):
            if k in e:
                head.append(f"{k}: {e[k] if not isinstance(e[k], list) else ', '.join(e[k])}")
        head.append(f"subject=`{e['subject']}` role=`{e['role']}` echo=`{e['echo_title']}`")
        L.append("- " + " · ".join(head))
        if e.get("reference"):
            L.append(f"  - *Human reference*: {e['reference'][0][:300].strip()}"
                     + ("…" if len(e['reference'][0]) > 300 else ""))
        for k in keys:
            if k in e["outputs"]:
                L.append(f"  - *{k}*: {e['outputs'][k]}")
    return L


def best(sysres: dict, key: str, names=None, lower=False):
    names = names or list(sysres)
    vals = [(sysres[n]["metrics"].get(key), n) for n in names if sysres[n]["metrics"].get(key) is not None]
    return (min if lower else max)(vals) if vals else (None, None)


def write_report(R: dict, path: Path):
    L = ["# PE-Writer v1 → v2 (Arm 4 Frame-Locked) on public benchmarks (device-exact int8)", ""]
    m = R["meta"]
    L += [f"Date: {m['date']}. Host: {m['host']}. Wall time {m['wall_seconds']:.0f} s. Every number below is read "
          "from `pe_writer_public_results.json`, written by `run_pe_writer_public_bench.py` in the same run.", "",
          "## What was run", "",
          f"- **Writers.** Arms 0–3 come from `{m['run']}/arm{{0..3}}.pt` and decode against that run's v1 WCLS "
          f"bank (`{m['writer_clauses_v1_sha256'][:16]}…`). Arm 4 comes from `{m['run_arm4']}/arm4.pt` and decodes "
          "against its proper-noun-free, role-partitioned WCLS v2 bank. Each arm is exported to PEWR int8 and "
          "decoded by `PeWriterIntTorch`, the batched integer reference that matches the device kernel bit for bit. "
          "As in `train_pe_writer.evaluate`, Arm 0 runs with no planner and no bank, and Arm 1 with no bank. The "
          f"Arm-4 export is byte-identical to the installed `models/pe-writer.bin` "
          f"(`{m['pe_writer_bin_sha256'][:16]}…`), and its bank is the installed `writer-clauses.bin` "
          f"(`{m['writer_clauses_bin_sha256'][:16]}…`).",
          f"- **Clause baselines.** *{B_RANDOM}*, *{B_COS}*, *{B_BM25}* and *{B_TPL}* use the installed v2 bank. "
          f"*{B_TPL_V1}* is the same no-learning template on the v1 bank, kept for continuity with the v1 report.",
          f"- **Concept vectors.** `c_page` and `c_echo` come from the deployed `platonic-hnet.bin` "
          f"(`{m['encoder_sha256']['platonic-hnet.bin'][:12]}…`) and `engram.bin` "
          f"(`{m['encoder_sha256']['engram.bin'][:12]}…`) through `model.PlhnIntNp`, the integer reference of "
          "`ConceptEngine::encodeText`. Each input is at most 256 bytes.",
          "- **No training or tuning on any benchmark.** Every input field is built by a fixed rule from the "
          "benchmark *input* side, never from a reference. Each field is cut to 31 bytes (device `char[32]`).",
          f"- **Scorers.** Tracks 1, 2 and 4 use `{SCORER_EN}`; Track 3 uses `{SCORER_ML}`. Both give 384-d sentence "
          "vectors, and the score is cosine × 100 with a bootstrap 95% CI (1,000 resamples). Long references (BookSum "
          "analyses, NarrativeQA summaries) are split into chunks of at most 90 words, embedded, and the normalised "
          "chunk embeddings are mean-pooled. Empty outputs score 0. ROUGE-1, ROUGE-L and content-word F1 use "
          "lower-cased word tokens with no stemming; with several references the best one counts.",
          "- **Metric definitions.**",
          "  - *Grounding precision* follows the training definition: the subject and the echo title both appear, "
          "and the output has no invented proper noun.",
          "  - *Invented proper noun*: a capitalised word that is not sentence-initial, not a template or function "
          "word, and not found in the input the reader has seen. For BookSum that input is every chapter up to the "
          "current one, plus the subject, role and echo. Bank clauses come from the training corpus, not from the "
          "book, so a name copied from a clause **counts as invented** here.",
          "  - *Spoiler leakage*: the output names an entity whose first appearance in that book/source sequence "
          "is after the current chapter.",
          "  - *Rep-3*: the share of outputs that repeat any word 3-gram.",
          "  - *Distinct-2*: unique bigrams divided by total bigrams, over the whole corpus.",
          "  - *Broken tail*: a non-empty output that does not end in sentence-final punctuation (cut off at the "
          "239-byte cap, or ending on a dangling fragment). Bare-clause baselines count here by construction.", ""]
    t1 = R["tracks"]["1_booksum"]
    cols1 = [("minilm_cos_x100", "MiniLM cos×100 [95% CI]", 2), ("rouge1_f", "R-1 F", 4), ("rougeL_f", "R-L F", 4),
             ("content_f1", "content F1", 4), ("spoiler_leak_pct", "spoiler %", 2),
             ("grounding_precision_pct", "grounded %", 1), ("entity_token_precision_pct", "entity-tok prec %", 1),
             ("invented_proper_noun_pct", "invented PN %", 1), ("rep3_pct", "Rep-3 %", 2),
             ("broken_tail_pct", "broken tail %", 2), ("distinct2_pct", "Distinct-2 %", 1), ("mean_words", "words", 1)]
    L += ["## Track 1 — BookSum test: literary commentary vs the human analysis", "",
          f"There are n = {t1['data']['n_rows_with_analysis']} chapter rows with a human `summary_analysis`, covering "
          f"{t1['data']['books']} books and the sources {t1['data']['sources']}. The inputs are built like this:",
          "- `subject`: the chapter's most frequent recurring name, mined from that book/source's own chapter text.",
          "- `role`: \"a central character\" if there is person evidence, otherwise \"a recurring name\".",
          "- `echo_title`: the book title.",
          "- `c_page`: the device `contextAround(8)` window around the subject's first mention in the chapter.",
          "- `c_echo`: the window around the subject's most recent mention in an earlier chapter.",
          f"The echo-source breakdown is {t1['echo_kinds']}.", ""]
    L += table(t1["systems"], cols1)
    L += ["", f"The human analyses themselves have a spoiler rate of {t1['human_reference_spoiler_pct']:.2f}% under "
          "the same detector, which gives a calibration point.", ""]
    if "pooled_ablation" in t1:
        L += ["**Ablation: `c_page` from the whole chapter.** Here `c_page` is the `l2_int8` of the summed vectors of "
              "every ≤256-byte chapter window, instead of the device's 17-word window.", ""]
        L += table(t1["pooled_ablation"], [("minilm_cos_x100", "MiniLM cos×100 [95% CI]", 2),
                                           ("rougeL_f", "R-L F", 4), ("content_f1", "content F1", 4),
                                           ("grounding_precision_pct", "grounded %", 1), ("rep3_pct", "Rep-3 %", 2)])
        L += [""]
    L += ["Paired semantic differences, mean Δ cos×100 with a bootstrap 95% CI:", ""]
    L += [f"- {p['a']} − {p['b']}: {p['mean_diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}], a wins "
          f"{p['a_wins_pct']}%" for p in t1["paired"]]
    t2 = R["tracks"]["2_narrativeqa"]
    cols2 = [("minilm_cos_x100", "MiniLM cos×100 vs answers [95% CI]", 2),
             ("minilm_cos_to_summary_x100", "MiniLM cos×100 vs summary", 2), ("rougeL_f", "R-L F (answers)", 4),
             ("content_f1", "content F1", 4), ("grounding_precision_pct", "grounded %", 1),
             ("invented_proper_noun_pct", "invented PN %", 1), ("rep3_pct", "Rep-3 %", 2),
             ("broken_tail_pct", "broken tail %", 2), ("distinct2_pct", "Distinct-2 %", 1), ("mean_words", "words", 1)]
    L += ["", "## Track 2 — NarrativeQA test: story question / theme synthesis", "",
          f"There are n = {t2['metrics_n']} questions, the first {t2['data']['questions_per_story']} of each of "
          f"{t2['data']['stories']} test stories. The story text is the official Wikipedia summary (the summary "
          "setting; full scripts were not downloaded). The inputs are:",
          "- `c_page`: the summary window that ranks top under BM25 for the question.",
          "- `c_echo`: the second-ranked window, standing in for the related scene.",
          "- `subject`: the first named entity in the question that also occurs in the summary.",
          "- `echo_title`: the story title.",
          "The references are the two human answers (best one counts) and the human summary.", ""]
    L += table(t2["systems"], cols2)
    L += ["", "Paired semantic differences against the answers:", ""]
    L += [f"- {p['a']} − {p['b']}: {p['mean_diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
          for p in t2["paired"]]
    t3 = R["tracks"]["3_zh_to_en"]
    L += ["", "## Track 3 — zero-shot Chinese → English concept-to-text", "",
          "The concept vector is built from **the Chinese text only**: `c_page` is `encodeText` of the Chinese "
          "sentence or passage (≤255 bytes), and `c_echo` is the previous item in the corpus. The `subject` is the "
          "first proper noun or noun found by jieba POS tagging, so it stays in Chinese. The `role` is \"a "
          "person\", \"a place\", \"an organization\" or \"a concept\", chosen from the POS tag. The `echo_title` is "
          "the corpus label (Tatoeba / 维基百科 / 人民日报). Scoring uses the multilingual MiniLM-L12:",
          "- Tatoeba: cosine against the English ground truth, with cosine against the Chinese source as a "
          "secondary score.",
          "- UD-GSDSimp and PKU: these have no English side, so the score is cross-lingual cosine against the "
          "Chinese source.",
          "The *copy source* row is the untouched Chinese sentence. It is a translation-equivalence ceiling for "
          "the scorer, not a competitor.", ""]
    cols3 = [("minilm_cos_x100", "ML-MiniLM cos×100 [95% CI]", 2),
             ("minilm_ml_cos_to_zh_source_x100", "cos×100 vs zh source", 2), ("content_f1", "content F1 (en)", 4),
             ("grounding_precision_pct", "grounded %", 1), ("invented_proper_noun_pct", "invented PN %", 1),
             ("rep3_pct", "Rep-3 %", 2), ("broken_tail_pct", "broken tail %", 2), ("distinct2_pct", "Distinct-2 %", 1),
             ("mean_words", "words", 1)]
    for name, sub in t3.items():
        L += [f"### {name} (n = {sub['data']['n']})", ""]
        L += table(sub["systems"], cols3 if name == "tatoeba_cmn_eng" else
                   [c for c in cols3 if c[0] not in ("minilm_ml_cos_to_zh_source_x100", "content_f1")])
        L += [""] + [f"- {p['a']} − {p['b']}: {p['mean_diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]"
                     for p in sub["paired"]] + [""]
    t4 = R["tracks"]["4_commongen"]
    cols4 = [("coverage_pct", "coverage %", 2), ("all_covered_pct", "all-3 covered %", 2),
             ("minilm_cos_x100", "MiniLM cos×100 vs refs [95% CI]", 2), ("rougeL_f", "R-L F", 4),
             ("content_f1", "content F1", 4), ("rep3_pct", "Rep-3 %", 2), ("distinct2_pct", "Distinct-2 %", 1),
             ("invented_proper_noun_pct", "invented PN %", 1), ("broken_tail_pct", "broken tail %", 2),
             ("mean_words", "words", 1)]
    L += ["## Track 4 — CommonGen validation: constrained 3-concept generation", "",
          f"There are n = {t4['data']['concept_sets']} concept sets ({t4['data']['rows']} reference rows). The "
          "inputs map `subject`, `role` and `echo_title` to concepts 1, 2 and 3. `c_page` is `encodeText` of the "
          "three concepts joined by spaces, and `c_echo` is `encodeText` of concept 3. Coverage counts a concept "
          "as present if some output word matches it after light suffix stripping (-ing/-ed/-es/-s/-er, doubled "
          "consonant, final e) plus a table of 96 irregular forms (stood → stand, …). This approximates the "
          "official spaCy-lemma coverage. BLEU/CIDEr/SPICE are not computed here.", ""]
    L += table(t4["systems"], cols4)
    L += ["", "### Latency / memory Pareto (CommonGen quality)", "",
          "Latency is host single-example time on Apple-silicon CPU. For the arms it uses the numpy integer "
          "reference, which mirrors the C++ kernel line by line; it is not an ESP32-S3 measurement. PSRAM is the "
          "set of PEWR tensors each arm actually reads; SD is the WCLS bank.", "",
          "| system | PSRAM bytes | SD bytes | host ms / riff | coverage % | MiniLM cos×100 | Pareto (all) | "
          "Pareto (arms only) |",
          "|---|---:|---:|---:|---:|---:|:---:|:---:|"]
    for s, p in t4["pareto"]["points"].items():
        L.append(f"| {s} | {p['psram']:,} | {p['sd']:,} | {p['ms']:.3f} | {p['coverage']:.2f} | {p['q']:.2f} | "
                 f"{'**yes**' if s in t4['pareto']['frontier'] else ''} | "
                 f"{'**yes**' if s in t4['pareto']['frontier_arms_only'] else ''} |")
    L += ["", f"The Pareto frontier is computed on (total bytes ↓, latency ↓, MiniLM cos ↑), once over every "
          "system and once over the five PE-Writer arms only. The firmware "
          f"`generateMarginRiff` host time is {R['parity']['firmware_host_ms_per_riff_incl_process_and_file_io']:.2f}"
          f" ms per riff over the {R['parity']['cases']} parity cases; that figure includes process start and per-call file I/O. "
          f"The platonic-hnet encoder numpy reference takes {t4['latency_ms']['encoder_platonic_hnet_ms']:.2f} ms "
          "per text.", ""]
    pr = R["parity"]
    L += ["## Device-exact C++ parity on public inputs", "",
          f"{pr['cases']} public benchmark inputs ({pr['per_track']} from each track) went through the Python int8 "
          f"Arm 4 (`PeWriterIntTorch` over `models/pe-writer.bin` + `models/writer-clauses.bin`) and through the compiled rc10 firmware "
          f"`ConceptEngine::generateMarginRiff` (`{pr['firmware_git_head']}`, ConceptEngine.cpp "
          f"`{pr['firmware_sha256'][:12]}…`, models dir `{pr['models_dir']}`). Both sides were compared on c_plan, "
          "buckets, all 36 tokens and the riff bytes.", "",
          f"- **Python int8 vs firmware: {pr['bit_exact_python_int8_vs_firmware']}/{pr['cases']} bit-exact.**",
          f"- numpy line-by-line reference vs firmware: {pr['bit_exact_numpy_ref_vs_firmware']}/{pr['cases']}.", ""]
    if pr["mismatches"]:
        L += ["Mismatches:"] + [f"- {x['case']}: {x['fields']}" for x in pr["mismatches"]] + [""]
    L += ["## Qualitative examples (public inputs, actual int8 outputs)", ""]
    for tk, title in (("1_booksum", "BookSum"), ("2_narrativeqa", "NarrativeQA")):
        L += [f"### {title}", ""] + ex_md(R["tracks"][tk]["examples"], R["example_systems"]) + [""]
    for name, sub in t3.items():
        L += [f"### {name}", ""] + ex_md(sub["examples"], R["example_systems"]) + [""]
    L += ["### CommonGen", ""] + ex_md(t4["examples"], R["example_systems"] + ["Concept join"]) + [""]
    L += ["## Analysis", ""] + R["analysis"] + [""]
    path.write_text("\n".join(L))


def analysis(R: dict) -> list[str]:
    A = []
    arms = [ARM_NAMES[a] for a in ARMS]
    a0, a3, a4 = ARM_NAMES[0], ARM_NAMES[3], ARM_NAMES[4]
    t1, t2, t4 = R["tracks"]["1_booksum"]["systems"], R["tracks"]["2_narrativeqa"]["systems"], \
        R["tracks"]["4_commongen"]["systems"]
    sets = (("BookSum", "1_booksum", None), ("NarrativeQA", "2_narrativeqa", None),
            ("Tatoeba", "3_zh_to_en", "tatoeba_cmn_eng"), ("UD-GSD", "3_zh_to_en", "ud_zh_gsdsimp_test"),
            ("PKU", "3_zh_to_en", "sighan2005_pku_test"), ("CommonGen", "4_commongen", None))

    def T(tk, sub):
        return R["tracks"][tk] if sub is None else R["tracks"][tk][sub]

    def g(s, n, k):
        return s[n]["metrics"].get(k)

    def win_line(track, sysres, key, label, lower=False):
        v, n = best(sysres, key, lower=lower)
        return (f"- {track} {label}: best = **{n}** ({fmt(v)}); Arm 4 = {fmt(g(sysres, a4, key))}; "
                f"Arm 3 = {fmt(g(sysres, a3, key))}.")

    A.append("**Who wins each headline metric.** Every line below is computed from the tables above.")
    A.append(win_line("BookSum", t1, "minilm_cos_x100", "MiniLM cos×100"))
    A.append(win_line("BookSum", t1, "rougeL_f", "ROUGE-L F"))
    A.append(win_line("BookSum", t1, "grounding_precision_pct", "grounding %"))
    A.append(win_line("BookSum", t1, "rep3_pct", "Rep-3 %", lower=True))
    A.append(win_line("NarrativeQA", t2, "minilm_cos_x100", "MiniLM cos×100 vs answers"))
    for name, sub in R["tracks"]["3_zh_to_en"].items():
        s = {k: v for k, v in sub["systems"].items() if not k.startswith("Copy source")}
        A.append(win_line(f"{name} (excluding the copy-source ceiling)", s, "minilm_cos_x100", "ML-MiniLM cos×100"))
    A.append(win_line("CommonGen", t4, "coverage_pct", "coverage %"))
    A.append(win_line("CommonGen", t4, "minilm_cos_x100", "MiniLM cos×100"))
    A.append("")
    rows = [f"{label}: " + ", ".join(f"{n.split(' ')[1]} {g(T(tk, sub)['systems'], n, 'minilm_cos_x100'):.2f}"
                                     for n in arms) for label, tk, sub in sets]
    A.append("**Across the arms** (MiniLM cos×100 for arms 0/1/2/3/4): " + "; ".join(rows) + ".")
    A.append("")

    def paired_summary(a, b, la, lb):
        tl, worse, better = [], [], []
        for label, tk, sub in sets:
            p = next((p for p in T(tk, sub)["paired"] if p["a"] == a and p["b"] == b), None)
            if p:
                tl.append(f"{label} {p['mean_diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]")
                (worse if p["ci95"][1] < 0 else better if p["ci95"][0] > 0 else []).append(label)
        ties = len(tl) - len(worse) - len(better)
        return (f"Paired {la} − {lb} MiniLM differences (×100, 95% bootstrap CI): " + "; ".join(tl) +
                f". {la} is significantly better on {len(better)} of {len(tl)} sets ({', '.join(better) or 'none'}), "
                f"significantly worse on {len(worse)} ({', '.join(worse) or 'none'}), and {ties} are within noise.")

    A.append("**Arm 4 against Arm 3 and the no-learning slot templates.** " + paired_summary(a4, a3, "Arm 4", "Arm 3"))
    A.append("")
    A.append(f"- *{B_TPL}* (same cleaned v2 bank, top-1 int8-cosine clause, no learned weights): "
             + paired_summary(a4, B_TPL, "Arm 4", "template (v2 bank)"))
    A.append(f"- *{B_TPL_V1}* (the v1 bank Arm 3 ships with): " + paired_summary(a4, B_TPL_V1, "Arm 4", "template (v1 bank)"))
    A.append(f"- Continuity: " + paired_summary(a3, B_TPL_V1, "Arm 3", "template (v1 bank)"))
    A.append("")

    def arrow(k, fmt_s="{:.2f}"):
        return "; ".join(f"{label} {fmt_s.format(g(T(tk, sub)['systems'], a3, k))} → "
                         f"{fmt_s.format(g(T(tk, sub)['systems'], a4, k))}" for label, tk, sub in sets)

    A.append("**What the frame lock changes (Arm 3 → Arm 4, all six sets).**")
    A.append(f"- Broken tails %: {arrow('broken_tail_pct')}.")
    A.append(f"- 239-byte truncation %: {arrow('truncated_239B_pct', '{:.1f}')}.")
    A.append(f"- Rep-3 %: {arrow('rep3_pct')}.")
    A.append(f"- Invented proper nouns %: {arrow('invented_proper_noun_pct', '{:.1f}')}.")
    A.append(f"- Grounding %: {arrow('grounding_precision_pct', '{:.1f}')}.")
    A.append(f"- Mean words: {arrow('mean_words', '{:.1f}')}.")
    A.append("Arm 4 can only emit `<EOS>`, the subject/role/echo slots, two bank clauses and 32 connector pieces "
             "(ids 40..2047 carry W = 0 and bias −30000), so a riff always closes with '.' + `<EOS>` and every "
             "content word comes from a clause. The v2 bank keeps only lowercase train-split teacher clauses of "
             "4–10 words and ≤ 64 bytes with no name-like tokens, so any remaining 'invented' capitalised token "
             "comes from the given subject/echo fields or the detector, not from the writer.")
    A.append("")
    A.append(f"- *Spoilers are structurally rare.* BookSum spoiler leakage is {g(t1, a4, 'spoiler_leak_pct'):.2f}% "
             f"for Arm 4 and {g(t1, a3, 'spoiler_leak_pct'):.2f}% for Arm 3, against "
             f"{R['tracks']['1_booksum']['human_reference_spoiler_pct']:.2f}% for the human analyses under the "
             "same detector.")
    A.append("")
    A.append("**Where a <1 MB frame/clause-engram writer hits its domain limits.**")
    A.append(f"- *Content comes from a fixed 4,096-clause bank, not from the page.* On BookSum, Arm 4 reaches "
             f"MiniLM {g(t1, a4, 'minilm_cos_x100'):.2f} and ROUGE-L {g(t1, a4, 'rougeL_f'):.4f}; the extractive "
             f"lead sentence reaches {g(t1, 'Extractive lead sentence', 'minilm_cos_x100'):.2f} and "
             f"{g(t1, 'Extractive lead sentence', 'rougeL_f'):.4f}. On internal validation the retrieved clauses are "
             "only slightly closer (MiniLM) to the teacher clause than a random clause of the same role, so the "
             "128-d concept vector carries topic, not clause-level meaning.")
    A.append(f"- *Fact-seeking questions are out of scope.* NarrativeQA: Arm 4 {g(t2, a4, 'minilm_cos_x100'):.2f} "
             f"against {g(t2, 'Extractive top-BM25 summary sentence', 'minilm_cos_x100'):.2f} for the top-BM25 "
             "summary sentence.")
    tat = R["tracks"]["3_zh_to_en"]["tatoeba_cmn_eng"]["systems"]
    A.append(f"- *Cross-lingual meaning is weak.* Tatoeba: Arm 4 {g(tat, a4, 'minilm_cos_x100'):.2f}, random-clause "
             f"floor {g(tat, B_RANDOM, 'minilm_cos_x100'):.2f}, copy-source ceiling "
             f"{g(tat, 'Copy source (zh)', 'minilm_cos_x100'):.2f}.")
    A.append(f"- *CommonGen asks for an everyday sentence.* Arm 4 coverage {g(t4, a4, 'coverage_pct'):.1f}% "
             f"(slot template {g(t4, B_TPL, 'coverage_pct'):.1f}%, concept join "
             f"{g(t4, 'Concept join', 'coverage_pct'):.1f}% at zero bytes); coverage comes from the slots.")
    A.append("")
    A.append("**Caveats.** No benchmark data was used for training, bank construction or selection: the v2 bank "
             "is mined from the internal train split, the planner checkpoint is selected on internal-val retrieval "
             "cosine, and the frame was fixed before this run. All inputs are built by rule, without an LLM: "
             "subject from frequency or POS, a two-way role phrase, and echo = book title. MiniLM cosine is an "
             "independent proxy, not a human judgment. NarrativeQA uses the summary setting. CommonGen coverage "
             "uses suffix stripping plus a 96-entry irregular table, not spaCy lemmas. Latency is measured on the "
             "host, not the ESP32-S3.")
    return A


# ----------------------------------------------------------------------------------------------- main
def strip_private(x):
    if isinstance(x, dict):
        return {k: strip_private(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [strip_private(v) for v in x]
    if isinstance(x, set):
        return sorted(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-analysis", type=int, default=50, help="BookSum: min chars of human analysis")
    ap.add_argument("--nqa-per-story", type=int, default=2)
    ap.add_argument("--n-pku", type=int, default=300)
    ap.add_argument("--parity-per-track", type=int, default=16)
    ap.add_argument("--scorer-device", default="cpu")
    ap.add_argument("--no-pooled", action="store_true", help="skip the BookSum whole-chapter c_page ablation")
    ap.add_argument("--report-only", action="store_true",
                    help="re-render analysis + report from the saved pe_writer_public_results.json (no re-run)")
    args = ap.parse_args(argv)
    if args.report_only:
        R = json.loads((RES / "pe_writer_public_results.json").read_text())
        R["analysis"] = analysis(R)
        (RES / "pe_writer_public_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
        write_report(R, RES / "pe_writer_public_report.md")
        log(f"re-rendered {RES / 'pe_writer_public_report.md'}")
        return
    t_start = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    RES.mkdir(parents=True, exist_ok=True)

    log("loading writers + encoder")
    W = Writers()
    enc = DeviceEncoder()
    log("loading scorers")
    S_en = Scorer(SCORER_EN, args.scorer_device)
    S_ml = Scorer(SCORER_ML, args.scorer_device)
    R = {"meta": {}, "tracks": {}}
    samples_out = []
    show = [ARM_NAMES[a] for a in ARMS] + [B_RANDOM, B_COS, B_TPL, B_TPL_V1]
    R["example_systems"] = show + ["Extractive lead sentence", "Extractive top-BM25 summary sentence",
                                   "Copy source (zh)"]
    a3, a4 = ARM_NAMES[3], ARM_NAMES[4]

    def pairs(res, extra=()):
        """Arm 4 (installed) vs every arm/baseline; Arm 3 vs its v1-bank comparisons for continuity."""
        ps = [paired(res, a4, b) for b in [a3, B_TPL, B_TPL_V1, ARM_NAMES[0], ARM_NAMES[2], B_RANDOM, B_COS,
                                           B_BM25, *extra] if b in res]
        return ps + [paired(res, a3, b) for b in [ARM_NAMES[0], ARM_NAMES[2], B_TPL_V1]]

    def dump_samples(track, items, outs, keep=("subject", "role", "echo", "page_text", "echo_text")):
        for i, it in enumerate(items):
            samples_out.append({"track": track, "id": it["id"], **{k: it[k] for k in keep},
                                "refs": [r[:600] for r in it.get("refs", [])],
                                "outputs": {k: v[i] for k, v in outs.items()}})

    # ---- Track 1
    log("track 1: BookSum")
    items1, info1 = build_track1(enc, args.min_analysis)
    log(f"  {len(items1)} rows")
    outs1, cp1, _ = run_systems(W, enc, items1, "Extractive lead sentence", use_bm25=True)
    fut = [it["future"] for it in items1]
    res1 = eval_track(outs1, items1, S_en, "long", future=fut)
    hum = [it["ref_long"] for it in items1]
    hum_spoil = float(np.mean([any(w in fut[i] for w in re.findall(r"[A-Z][a-z]{2,}", h)) for i, h in enumerate(hum)]))
    T1 = {"data": info1, "systems": res1, "human_reference_spoiler_pct": round(100 * hum_spoil, 2),
          "echo_kinds": dict(Counter(it["echo_kind"].split(" (")[0] if "(" in it["echo_kind"] else it["echo_kind"]
                                     for it in items1)),
          "role_counts": dict(Counter(it["role"] for it in items1)),
          "paired": pairs(res1, ["Extractive lead sentence"])}
    if not args.no_pooled:
        log("  pooled-chapter c_page ablation")
        chap_vec = {}
        for it in items1:
            k = it["chapter_path"] + "|" + it["id"].split("/")[0]
            if k not in chap_vec:
                chap_vec[k] = pool_int8(enc.encode(windows(it["_chapter"])))
        cpp = np.stack([chap_vec[it["chapter_path"] + "|" + it["id"].split("/")[0]] for it in items1])
        outs_p, _, _ = run_systems(W, enc, items1, "Extractive lead sentence", use_bm25=False, c_page=cpp,
                                   arms=(2, 3, 4))
        outs_p = {k + " (pooled c_page)": v for k, v in outs_p.items()
                  if k in (ARM_NAMES[2], a3, a4, B_COS, B_TPL)}
        rp = eval_track(outs_p, items1, S_en, "long")
        T1["pooled_ablation"] = rp
        T1["pooled_paired"] = [paired({**res1, **rp}, a + " (pooled c_page)", a) for a in (a3, a4)]
    n1 = len(items1)
    T1["examples"] = examples(outs1, items1, sorted({int(x) for x in np.linspace(0, n1 - 1, 5)}), R["example_systems"])
    dump_samples("1_booksum", items1, outs1)
    R["tracks"]["1_booksum"] = T1

    # ---- Track 2
    log("track 2: NarrativeQA")
    items2, info2 = build_track2(args.nqa_per_story)
    log(f"  {len(items2)} questions")
    outs2, _, _ = run_systems(W, enc, items2, "Extractive top-BM25 summary sentence", use_bm25=True)
    res2 = eval_track(outs2, items2, S_en, "multi", long_ref_scorer=S_en)
    n2 = len(items2)
    R["tracks"]["2_narrativeqa"] = {
        "data": info2, "metrics_n": n2, "systems": res2,
        "paired": pairs(res2, ["Extractive top-BM25 summary sentence"]),
        "examples": examples(outs2, items2, sorted({int(x) for x in np.linspace(0, n2 - 1, 5)}),
                             R["example_systems"])}
    dump_samples("2_narrativeqa", items2, outs2, keep=("subject", "role", "echo", "question", "page_text"))

    # ---- Track 3
    log("track 3: zh -> en")
    T3 = {}
    tr3 = build_track3(args.n_pku)
    for name, (items3, info3) in tr3.items():
        log(f"  {name}: {len(items3)}")
        outs3, _, _ = run_systems(W, enc, items3, "Copy source (zh)", use_bm25=False)
        res3 = eval_track(outs3, items3, S_ml, "zh")
        n3 = len(items3)
        T3[name] = {"data": info3, "systems": res3,
                    "paired": pairs(res3),
                    "examples": examples(outs3, items3, sorted({int(x) for x in np.linspace(0, n3 - 1, 4)}),
                                         R["example_systems"])}
        dump_samples(f"3_{name}", items3, outs3, keep=("subject", "role", "echo", "zh", "page_text"))
    R["tracks"]["3_zh_to_en"] = T3

    # ---- Track 4
    log("track 4: CommonGen")
    items4, info4 = build_track4()
    outs4, _, _ = run_systems(W, enc, items4, "Concept join", use_bm25=True)

    def cov_fn(o):
        c = [coverage(x, it["concepts"]) for x, it in zip(o, items4)]
        return {"coverage_pct": c, "all_covered_pct": [float(v == 1.0) for v in c]}

    res4 = eval_track(outs4, items4, S_en, "multi", extra_fn=cov_fn)
    lat = latency(W, enc, items4)
    fp = footprint()
    fp["Concept join"] = fp.pop("Concept join (CommonGen extractive)")
    lat["Concept join"] = lat.pop("Concept join (CommonGen extractive)")
    pts = {}
    for s in [ARM_NAMES[a] for a in ARMS] + [B_RANDOM, B_COS, B_TPL, "Concept join"]:
        pts[s] = {"psram": fp[s]["psram_bytes"], "sd": fp[s]["sd_bytes"], "bytes": fp[s]["total_bytes"],
                  "ms": lat[s], "q": res4[s]["metrics"]["minilm_cos_x100"],
                  "coverage": res4[s]["metrics"]["coverage_pct"]}
    n4 = len(items4)
    R["tracks"]["4_commongen"] = {
        "data": info4, "systems": res4, "latency_ms": lat, "footprint": fp,
        "pareto": {"points": pts, "frontier": pareto(pts),
                   "frontier_arms_only": pareto({a: pts[a] for a in [ARM_NAMES[i] for i in ARMS]}),
                   "axes": "total bytes (min), host ms (min), MiniLM (max)"},
        "paired": pairs(res4, ["Concept join"]),
        "examples": examples(outs4, items4, sorted({int(x) for x in np.linspace(0, n4 - 1, 5)}),
                             R["example_systems"] + ["Concept join"])}
    dump_samples("4_commongen", items4, outs4, keep=("subject", "role", "echo", "concepts", "page_text"))

    # ---- parity
    log("C++ firmware parity")
    R["parity"] = parity(W, enc, {"booksum": items1, "narrativeqa": items2,
                                  "zh": tr3["tatoeba_cmn_eng"][0][:args.parity_per_track // 2]
                                  + tr3["sighan2005_pku_test"][0][:args.parity_per_track - args.parity_per_track // 2],
                                  "commongen": items4}, args.parity_per_track)
    log(f"  parity {R['parity']['bit_exact_python_int8_vs_firmware']}/{R['parity']['cases']}")

    R["meta"] = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"), "host": f"{platform.machine()} {platform.system()} "
                                                           f"{platform.release()}",
        "args": vars(args), "run": str(RUN.relative_to(ROOT)), "run_arm4": str(RUN4.relative_to(ROOT)),
        "arm_pewr_sha256": {ARM_NAMES[a]: sha256_bytes(W.blobs[a]) for a in ARMS},
        "pe_writer_bin_sha256": sha256_bytes(W.device_pewr), "writer_clauses_bin_sha256": sha256_bytes(W.wcls),
        "writer_clauses_v1_sha256": sha256_bytes(W.wcls_v1),
        "arm4_export_equals_models_pe_writer_bin": W.blobs[4] == W.device_pewr,
        "encoder_sha256": enc.sha, "encoder_bytes": enc.bytes, "encoder_ref": "src/model.py PlhnIntNp (bigram on)",
        "scorers": {"en": SCORER_EN, "multilingual": SCORER_ML, "device": args.scorer_device},
        "arm_configs": {ARM_NAMES[a]: W.cfg[a] for a in ARMS},
        "torch": torch.__version__, "seed": SEED, "new_encodings_this_run": enc.n_new,
        "wall_seconds": time.time() - t_start}
    R["analysis"] = analysis(R)
    out = strip_private(R)
    (RES / "pe_writer_public_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with gzip.open(RES / "pe_writer_public_samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for s in samples_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    write_report(out, RES / "pe_writer_public_report.md")
    log(f"wrote {RES / 'pe_writer_public_results.json'}, pe_writer_public_report.md, pe_writer_public_samples.jsonl.gz")


if __name__ == "__main__":
    main()
