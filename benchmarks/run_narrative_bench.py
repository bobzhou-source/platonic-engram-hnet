"""External narrative benchmarks for Platonic-Engram-HNet encoders + Chrono-Slot Pointer Weaver (CS-PW).

    external_benchmarks/.venv/bin/python external_benchmarks/run_narrative_bench.py

Public datasets (downloaded into ``external_benchmarks/data/``; sha256 recorded in the results):

* BookSum chapter-level test split (``kmfoda/booksum`` ``test.csv``; Kryscinski et al. 2021): 21 books,
  1,431 (chapter, study-guide summary) rows from 6 study-guide sources, with the original chapter text.
* NarrativeQA (``deepmind/narrativeqa`` ``data/narrativeqa-master.zip``; Kocisky et al. 2018): test-set
  questions + Wikipedia story summaries (``qaps.csv``, ``third_party/wikipedia/summaries.csv``).
* LiSCU is NOT used: it is not distributed; it must be re-scraped from the Wayback Machine through a
  PostgreSQL crawler (github.com/huangmeng123/lit_char_data_wayback), see the report.

Models (all loaded from the trained checkpoints; int8 encoders run through ``encode_int`` -- the QAT graph
that is bit-exact with the device kernel, spot-checked here against the exported ``models/platonic-hnet.bin``):

* ``bpe_arm0``  matched-RAM BPE Transformer baseline (runs/v1-final/arm0_bpe_baseline.pt, fp32)
* ``E0``        v1 PLHN (unigram Engram)            -- CS-PW arm0_v1_base encoder
* ``E1``        v2 + Bigram-Span Engram              -- arms 1/2 encoder
* ``E3``        v2 + Bigram + Qwen3 relational KD    -- arms 3/4 encoder (arm4 = full CS-PW)
* CS-PW arms 0-3 (flat mean pooling, torch QAT path) and arm4 (4-slot Chrono-CfC, exact int8 path
  ``int8_cspw_core`` / ``int8_cspw_weave`` over the weights packed from the checkpoint == ``models/cspw.bin``)
* lexical baselines: BM25 (k1=1.5, b=0.75) and char 3-5-gram TF-IDF

Tasks
1. Matching (R@1 / R@5 / MRR): BookSum cross-source summary->summary (within book and global),
   BookSum summary->chapter text (within book, same chapterisation), NarrativeQA question->story summary
   (all 355 test stories; "Who ..." character-question slice reported separately).
2. Temporal state tracking on BookSum chapter sequences (per (book, study-guide source), chronological
   summary order): CS-PW Slot 1 (origin) vs Slot 2 (recent arc) against probes from a *different view*
   (the original chapter text; and other-source summaries where available).
3. Spoiler horizon on BookSum sequences: character entities, reading cutoff k < M; Spoiler Leakage (SL)
   and token Fact Precision (FP, emitted content tokens attested in the book text of chapters <= k).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EB = Path(__file__).resolve().parent
ROOT = EB.parent
sys.path.insert(0, str(ROOT / "src"))
import cspw_model as CM  # noqa: E402
import model as M  # noqa: E402

DATA = EB / "data"
RES = EB / "results"
CACHE = EB / "cache" / "narrative"
RUN = ROOT / "runs" / "cspw-v2"
BPE_CKPT = ROOT / "runs" / "v1-final" / "arm0_bpe_baseline.pt"
MODELS = ROOT / "weights"
ARMS = ["arm0_v1_base", "arm1_arch_bigram_gqa", "arm2_scale_d192", "arm3_qwen_kd", "arm4_full_cspw"]
ENC_OF_ARM = {"arm0_v1_base": "E0", "arm1_arch_bigram_gqa": "E1", "arm2_scale_d192": "E1",
              "arm3_qwen_kd": "E3", "arm4_full_cspw": "E3"}
ENC_CKPT = {"E0": "arm0_v1_base", "E1": "arm1_arch_bigram_gqa", "E3": "arm4_full_cspw"}
ENCODERS = ["bpe_arm0", "E0", "E1", "E3"]

csv.field_size_limit(1 << 30)
WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
CONTENT = re.compile(r"[a-z]{3,}")
SENT = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)]*\s+")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------
# Text windows (the PLHN encoder reads <= 256 bytes per call)
# ----------------------------------------------------------------------------------------------


def _split_long(s: str, max_bytes: int):
    out, cur = [], b""
    for w in s.split():
        b = w.encode("utf-8")
        while len(b) > max_bytes:
            head = CM.utf8_fit(b.decode("utf-8", "ignore"), max_bytes)
            if cur:
                out.append(cur)
                cur = b""
            out.append(head)
            b = b[len(head):]
        if cur and len(cur) + 1 + len(b) <= max_bytes:
            cur += b" " + b
        else:
            if cur:
                out.append(cur)
            cur = b
    if cur:
        out.append(cur)
    return out


def windows(text: str, cap: int, max_bytes: int = M.MAX_BYTES) -> list[str]:
    """Sentence-packed windows of <= max_bytes UTF-8 bytes; > cap windows -> evenly spaced subset."""
    out, cur = [], b""
    for s in SENT.split(" ".join(text.split())):
        s = s.strip()
        if not s:
            continue
        b = s.encode("utf-8")
        for p in ([b] if len(b) <= max_bytes else _split_long(s, max_bytes)):
            if cur and len(cur) + 1 + len(p) <= max_bytes:
                cur += b" " + p
            else:
                if cur:
                    out.append(cur)
                cur = p
    if cur:
        out.append(cur)
    out = [w.decode("utf-8") for w in out] or [text[:64] or "-"]
    if len(out) > cap:
        idx = sorted(set(np.linspace(0, len(out) - 1, cap).round().astype(int).tolist()))
        out = [out[i] for i in idx]
    return out


def content_tokens(text: str, drop=frozenset()) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    return [t for t in CONTENT.findall(text.lower()) if t not in SW and t not in drop]


# ----------------------------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------------------------


def state_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class Encoder:
    """Cached text -> vector.  int8 encoders return the exact int8 concept vector (L2 = 127)."""

    def __init__(self, name: str):
        self.name = name
        self.int8 = name != "bpe_arm0"
        if name == "bpe_arm0":
            ck = torch.load(BPE_CKPT, map_location="cpu", weights_only=False)
            self.net = M.BPETransformerBaseline(M.ByteBPE(ck["bpe_merges"]))
            self.net.load_state_dict(ck["state_dict"], strict=True)
            self.src = str(BPE_CKPT.relative_to(ROOT))
        else:
            ck = torch.load(RUN / f"{ENC_CKPT[name]}.pt", map_location="cpu", weights_only=False)
            assert ck["encoder_name"] == name
            self.net = M.PlatonicEngramHNet(use_engram=True, use_bigram=ck["encoder_cfg"]["use_bigram"])
            self.net.load_state_dict(ck["encoder"], strict=True)
            self.net.quant = True
            self.engram_hash_source = "checkpoint engram_hash buffer"
            if name == "E3" and (MODELS / "engram.bin").exists():
                # E3 is the exported device encoder: use the deployed engram.bin span-hash table so the
                # torch QAT path is bit-exact with models/platonic-hnet.bin + engram.bin (verify_c_parity.py)
                ref = M.PlhnIntNp((MODELS / "platonic-hnet.bin").read_bytes(), (MODELS / "engram.bin").read_bytes())
                self.net.set_engram_hashes(ref.engram_hash)
                self.engram_hash_source = "models/engram.bin (deployed)"
            self.src = str((RUN / f"{ENC_CKPT[name]}.pt").relative_to(ROOT))
        self.net.eval()
        self.state_sha = state_hash(self.net.state_dict())
        self.path = CACHE / f"{name}.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.dirty = False

    @torch.no_grad()
    def encode(self, texts: list[str], bs: int = 256) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            if self.int8:
                vs = np.asarray(self.net.encode_int(chunk), dtype=np.int8)
            else:
                vs = self.net.encode(chunk)["c"].float().numpy().astype(np.float32)
            for t, v in zip(chunk, vs):
                self.cache[t] = v
            if (i // bs) % 40 == 0 and len(todo) > 4 * bs:
                log(f"  {self.name}: {i + len(chunk)}/{len(todo)} texts")
        if todo:
            self.dirty = True
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, M.D_CONCEPT))

    def save(self):
        if self.dirty:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(self.cache, protocol=4))
            os.replace(tmp, self.path)
            self.dirty = False


def norm(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)


class DocEmb:
    """Window matrix (normalised), mean-pooled doc vector (normalised) and the int8 doc vector
    (``l2_int8`` of the summed int8 window vectors, the CS-PW scene input)."""

    def __init__(self, enc: Encoder, docs: list[str], cap: int):
        self.wins = [windows(d, cap) for d in docs]
        flat = [w for ws in self.wins for w in ws]
        V = enc.encode(flat)
        self.W, self.doc, self.int8 = [], [], []
        o = 0
        for ws in self.wins:
            v = V[o:o + len(ws)]
            o += len(ws)
            if enc.int8:
                s = v.astype(np.int64).sum(0)
                q = np.asarray(M.l2_int8([int(x) for x in s]), dtype=np.int64)
                self.int8.append(q)
                self.doc.append(norm(q))
            else:
                self.doc.append(norm(norm(v).mean(0)))
            self.W.append(norm(v))
        self.doc = np.stack(self.doc)

    def maxsim(self, qvec: np.ndarray, idx: list[int]) -> np.ndarray:
        """(nq, len(idx)) max over candidate windows of cos(query doc vector, window)."""
        return np.stack([(qvec @ self.W[j].T).max(1) for j in idx], 1)


# ----------------------------------------------------------------------------------------------
# Lexical baselines
# ----------------------------------------------------------------------------------------------


def bm25_scores(queries: list[str], cands: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
    X = cv.fit_transform(cands).tocsc().astype(np.float64)
    N = X.shape[0]
    df = np.asarray((X > 0).sum(0)).ravel()
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    dl = np.asarray(X.sum(1)).ravel()
    avg = dl.mean() if N else 1.0
    X = X.tocoo()
    denom = X.data + k1 * (1 - b + b * dl[X.row] / avg)
    import scipy.sparse as sp
    Wm = sp.csr_matrix((X.data * (k1 + 1) / denom * idf[X.col], (X.row, X.col)), shape=X.shape)
    Q = cv.transform(queries)
    Q.data[:] = 1.0
    return np.asarray((Q @ Wm.T).todense())


def charngram_scores(queries: list[str], cands: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    C = tv.fit_transform(cands)
    Q = tv.transform(queries)
    return np.asarray((Q @ C.T).todense())


# ----------------------------------------------------------------------------------------------
# CS-PW
# ----------------------------------------------------------------------------------------------


class Cspw:
    def __init__(self, arm: str):
        ck = torch.load(RUN / f"{arm}.pt", map_location="cpu", weights_only=False)
        self.arm, self.cfg = arm, ck["cspw_cfg"]
        self.mod = CM.ChronoSlotPointerWeaver(d_model=self.cfg["d_model"], pool=self.cfg["pool"])
        self.mod.load_state_dict(ck["cspw"], strict=True)
        self.mod.quant = True
        self.mod.eval()
        self.frames, self.phrases = ck["frames"], ck["phrases"]
        self.w = None
        self.bin_info = None
        if self.cfg["pool"] == "cfc":
            blob = CM.pack_cspw(*self.mod.export_tensors(), self.frames, self.phrases)
            dev = MODELS / "cspw.bin"
            self.bin_info = {"packed_sha256": hashlib.sha256(blob).hexdigest(),
                             "models_cspw_bin_sha256": sha256(dev) if dev.exists() else None}
            self.bin_info["identical_to_models_cspw_bin"] = (self.bin_info["packed_sha256"]
                                                            == self.bin_info["models_cspw_bin_sha256"])
            self.w = CM.CspwWeights(dev.read_bytes() if self.bin_info["identical_to_models_cspw_bin"] else blob)

    @torch.no_grad()
    def slots_torch(self, scenes: list[np.ndarray], pages: list[int]) -> np.ndarray:
        S = torch.zeros(1, 4, M.D_CONCEPT, dtype=torch.float64)
        P = torch.zeros(1, 4, dtype=torch.long)
        for m, (s, p) in enumerate(zip(scenes, pages)):
            S[0, m] = torch.as_tensor(s, dtype=torch.float64)
            P[0, m] = int(p)
        Mv = torch.tensor([len(scenes)])
        self.mod.double()
        out = self.mod(S, P, Mv, torch.zeros(1, M.D_CONCEPT, dtype=torch.float64), torch.zeros(1, dtype=torch.bool))
        return out["slots"][0].numpy()

    def slots_int8(self, scenes, pages) -> np.ndarray:
        return CM.int8_cspw_core(self.w, scenes, pages)["slots"]

    def slots(self, scenes, pages) -> np.ndarray:
        return self.slots_int8(scenes, pages) if self.w is not None else self.slots_torch(scenes, pages)


# ----------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------


def load_booksum():
    p = DATA / "booksum" / "test.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["sec"] = int(re.search(r"section_(\d+)_part_(\d+)", r["summary_path"]).group(1))
        r["part"] = int(re.search(r"section_(\d+)_part_(\d+)", r["summary_path"]).group(2))
        r["summary_text"] = r["summary_text"].strip()
        r["chapter"] = r["chapter"].strip()
    rows = [r for r in rows if r["summary_text"] and r["chapter"]]
    seqs = defaultdict(list)
    for r in rows:
        seqs[(r["bid"], r["source"])].append(r)
    for k in seqs:
        seqs[k].sort(key=lambda r: (r["sec"], r["part"]))
        for i, r in enumerate(seqs[k]):
            r["pos"] = i + 1  # 1-based chronological position in this source's sequence
    return rows, dict(seqs), {"file": str(p.relative_to(EB)), "sha256": sha256(p), "hf_repo": "kmfoda/booksum",
                              "split": "test (chapter level)"}


def load_narrativeqa():
    z = DATA / "narrativeqa" / "data" / "narrativeqa-master.zip"
    base = DATA / "narrativeqa" / "narrativeqa-master"
    if not (base / "qaps.csv").exists():
        with zipfile.ZipFile(z) as f:
            f.extractall(DATA / "narrativeqa")
    summ = {r["document_id"]: r for r in csv.DictReader(open(base / "third_party" / "wikipedia" / "summaries.csv",
                                                                  encoding="utf-8")) if r["set"] == "test"}
    docs = {r["document_id"]: r for r in csv.DictReader(open(base / "documents.csv", encoding="utf-8"))
            if r["set"] == "test"}
    qs = [r for r in csv.DictReader(open(base / "qaps.csv", encoding="utf-8")) if r["set"] == "test"]
    return summ, docs, qs, {"file": str(z.relative_to(EB)), "sha256": sha256(z), "hf_repo": "deepmind/narrativeqa",
                            "split": "test", "summaries": "third_party/wikipedia/summaries.csv"}


# ----------------------------------------------------------------------------------------------
# Retrieval evaluation
# ----------------------------------------------------------------------------------------------


def rank_metrics(ranks: list[int]) -> dict:
    r = np.asarray(ranks, dtype=np.float64)
    if not len(r):
        return {"n": 0}
    return {"n": int(len(r)), "R@1": float((r == 1).mean()), "R@5": float((r <= 5).mean()),
            "MRR": float((1.0 / r).mean())}


def gold_rank(scores: np.ndarray, gold: int) -> int:
    """Pessimistic rank: ties with the gold candidate count against it."""
    return int((scores > scores[gold]).sum() + (scores == scores[gold]).sum())


def eval_groups(groups, systems):
    """groups: list of dict(queries=[texts], cands=[texts], gold=[idx], tags=[set]) -> per system ranks.
    systems: name -> fn(group) -> (nq, nc) scores."""
    ranks = {s: [] for s in systems}
    tags = []
    for g in groups:
        tags += g.get("tags", [set()] * len(g["queries"]))
        for s, fn in systems.items():
            sc = fn(g)
            ranks[s] += [gold_rank(sc[i], gi) for i, gi in enumerate(g["gold"])]
    return ranks, tags


def neural_systems(encs: dict, cap_q: int, cap_c: int, cspw4: Cspw | None):
    """Dense systems over all encoders (mean-pooled cos and MaxSim) + arm4 CS-PW slot0 pooling."""
    systems = {}

    def make(enc, mode):
        def fn(g):
            Dq = DocEmb(enc, g["queries"], cap_q)
            Dc = DocEmb(enc, g["cands"], cap_c)
            if mode == "mean":
                return Dq.doc @ Dc.doc.T
            return Dc.maxsim(Dq.doc, list(range(len(g["cands"]))))
        return fn

    for name, enc in encs.items():
        systems[f"{name}/mean"] = make(enc, "mean")
        systems[f"{name}/maxsim"] = make(enc, "maxsim")
    if cspw4 is not None:
        enc = encs["E3"]

        def slot0(D: DocEmb):
            out = []
            for W in D.wins:
                V = enc.encode(W).astype(np.int64)
                idx = sorted(set(np.linspace(0, len(V) - 1, min(4, len(V))).round().astype(int).tolist()))
                out.append(norm(cspw4.slots_int8([V[i] for i in idx], [i + 1 for i in idx])[0]))
            return np.stack(out)

        def fn(g):
            return slot0(DocEmb(enc, g["queries"], cap_q)) @ slot0(DocEmb(enc, g["cands"], cap_c)).T

        systems["arm4_cspw/slot0_role_pool"] = fn
    systems["BM25"] = lambda g: bm25_scores(g["queries"], g["cands"])
    systems["char3-5gram_tfidf"] = lambda g: charngram_scores(g["queries"], g["cands"])
    return systems


def task1(rows, seqs, nqa, encs, cspw4, args):
    res = {}
    # ---- 1a BookSum cross-source summary -> summary
    by_bid = defaultdict(list)
    for r in rows:
        by_bid[r["bid"]].append(r)
    within, n_pairs = [], 0
    for bid, rs in by_bid.items():
        srcs = sorted({r["source"] for r in rs})
        for s in srcs:
            for t in srcs:
                if s == t:
                    continue
                cand = [r for r in rs if r["source"] == t]
                cidx = {}
                for i, r in enumerate(cand):
                    cidx.setdefault(r["chapter_path"], i)
                qs = [r for r in rs if r["source"] == s and r["chapter_path"] in cidx]
                # one query per chapter_path (a source may split a chapter into several parts)
                seen, qq = set(), []
                for r in qs:
                    if r["chapter_path"] not in seen:
                        seen.add(r["chapter_path"])
                        qq.append(r)
                if len(cand) < 2 or not qq:
                    continue
                # a target chapter_path may hold several parts in t: gold = its first part; the other parts of
                # the same chapter are removed from the pool so they cannot pre-empt the gold
                for r in qq:
                    keep = [i for i, c in enumerate(cand) if c["chapter_path"] != r["chapter_path"]
                            or i == cidx[r["chapter_path"]]]
                    within.append({"queries": [r["summary_text"]], "cands": [cand[i]["summary_text"] for i in keep],
                                   "gold": [keep.index(cidx[r["chapter_path"]])]})
                n_pairs += 1
    # collapse identical candidate pools to one group (speed): key by tuple of candidate texts
    def collapse(groups):
        pool = {}
        for g in groups:
            k = tuple(g["cands"])
            if k not in pool:
                pool[k] = {"queries": [], "cands": g["cands"], "gold": []}
            pool[k]["queries"] += g["queries"]
            pool[k]["gold"] += g["gold"]
        return list(pool.values())

    systems = neural_systems(encs, args.cap_summary, args.cap_summary, cspw4)
    g_within = collapse(within)
    log(f"task1a within-book: {sum(len(g['queries']) for g in g_within)} queries, {len(g_within)} pools")
    ranks, _ = eval_groups(g_within, systems)
    res["booksum_xsource_summary_within_book"] = {
        "desc": "query = study-guide summary of a chapter (source s); candidates = all summaries of the same "
                "book from another source t; gold = t's summary of the same chapter (chance R@1 ~ 1/|pool|)",
        "n_queries": len(within), "n_source_pairs": n_pairs,
        "mean_pool_size": float(np.mean([len(g["cands"]) for g in within])),
        "systems": {s: rank_metrics(r) for s, r in ranks.items()}}
    # global: candidates = every test summary of source t (all books)
    glob = []
    for t in sorted({r["source"] for r in rows}):
        cand = [r for r in rows if r["source"] == t]
        first = {}
        for i, r in enumerate(cand):
            first.setdefault((r["bid"], r["chapter_path"]), i)
        seen = set()
        qs, gold = [], []
        for r in rows:
            key = (r["bid"], r["chapter_path"])
            if r["source"] == t or key not in first or (r["source"], key) in seen:
                continue
            seen.add((r["source"], key))
            qs.append(r["summary_text"])
            gold.append(first[key])
        if qs:
            # drop other parts of multi-part target chapters from the pool
            glob.append((t, cand, qs, gold))
    g_glob = []
    for t, cand, qs, gold in glob:
        dup = Counter((c["bid"], c["chapter_path"]) for c in cand)
        keep = [i for i, c in enumerate(cand) if dup[(c["bid"], c["chapter_path"])] == 1 or i in set(gold)]
        pos = {i: j for j, i in enumerate(keep)}
        g_glob.append({"queries": qs, "cands": [cand[i]["summary_text"] for i in keep], "gold": [pos[x] for x in gold]})
    log(f"task1a global: {sum(len(g['queries']) for g in g_glob)} queries")
    ranks, _ = eval_groups(g_glob, systems)
    res["booksum_xsource_summary_global"] = {
        "desc": "same queries, candidates = every test-split summary of source t across all 21 books",
        "n_queries": sum(len(g["queries"]) for g in g_glob),
        "pool_sizes": {t: len(g["cands"]) for (t, *_), g in zip(glob, g_glob)},
        "systems": {s: rank_metrics(r) for s, r in ranks.items()}}
    # ---- 1b BookSum summary -> chapter text (same source chapterisation)
    systems_c = neural_systems(encs, args.cap_summary, args.cap_chapter, cspw4)
    g_ch = []
    for (bid, src), rs in seqs.items():
        uniq = {}
        for r in rs:
            uniq.setdefault(r["chapter_path"], r["chapter"])
        paths = list(uniq)
        if len(paths) < 2:
            continue
        seen, qs, gold = set(), [], []
        for r in rs:
            if r["chapter_path"] in seen:
                continue
            seen.add(r["chapter_path"])
            qs.append(r["summary_text"])
            gold.append(paths.index(r["chapter_path"]))
        g_ch.append({"queries": qs, "cands": [uniq[p] for p in paths], "gold": gold})
    log(f"task1b summary->chapter: {sum(len(g['queries']) for g in g_ch)} queries in {len(g_ch)} sequences")
    ranks, _ = eval_groups(g_ch, systems_c)
    res["booksum_summary_to_chapter_within_book"] = {
        "desc": "query = study-guide summary; candidates = original chapter texts of the same (book, source) "
                f"chapterisation; neural chapter encodings use <= {args.cap_chapter} evenly spaced 256-byte windows",
        "n_queries": sum(len(g["queries"]) for g in g_ch), "n_sequences": len(g_ch),
        "mean_pool_size": float(np.mean([len(g["cands"]) for g in g_ch for _ in g["queries"]])),
        "systems": {s: rank_metrics(r) for s, r in ranks.items()}}
    # ---- 1c NarrativeQA question -> story summary
    summ, docs, qs, _ = nqa
    ids = sorted(summ)
    pos = {d: i for i, d in enumerate(ids)}
    qq = [q for q in qs if q["document_id"] in pos]
    g = {"queries": [q["question"] for q in qq], "cands": [summ[d]["summary"] for d in ids],
         "gold": [pos[q["document_id"]] for q in qq]}
    log(f"task1c narrativeqa: {len(qq)} questions x {len(ids)} summaries")
    systems_n = neural_systems(encs, 4, args.cap_summary_nqa, cspw4)
    ranks, _ = eval_groups([g], systems_n)
    who = [i for i, q in enumerate(qq) if re.match(r"\s*(who|whom|whose)\b", q["question"], re.I)]
    kinds = Counter(docs[d]["kind"] for d in ids if d in docs)
    res["narrativeqa_question_to_summary"] = {
        "desc": "query = NarrativeQA test question; candidates = all test-set Wikipedia story summaries; "
                "gold = the summary of the question's story",
        "n_queries": len(qq), "n_candidates": len(ids), "story_kinds": dict(kinds),
        "systems": {s: rank_metrics(r) for s, r in ranks.items()},
        "who_character_slice": {"n_queries": len(who),
                                "systems": {s: rank_metrics([r[i] for i in who]) for s, r in ranks.items()}}}
    return res


# ----------------------------------------------------------------------------------------------
# Task 2: temporal state tracking
# ----------------------------------------------------------------------------------------------


def pick_scenes(k: int) -> list[int]:
    """Chronological scene positions (0-based) of the first k chapters fed to CS-PW (M <= 4)."""
    return sorted(set(np.linspace(0, k - 1, min(k, CM.MAX_SCENES)).round().astype(int).tolist()))


def task2(rows, seqs, encs, args):
    by_ch = defaultdict(dict)  # (bid, chapter_path) -> source -> summary (first part)
    for r in rows:
        by_ch[(r["bid"], r["chapter_path"])].setdefault(r["source"], r["summary_text"])
    items = []
    for (bid, src), rs in sorted(seqs.items()):
        # one scene per chapter_path in this source's order (multi-part summaries of one chapter are joined)
        chs, seen = [], {}
        for r in rs:
            if r["chapter_path"] in seen:
                chs[seen[r["chapter_path"]]]["summary"] += " " + r["summary_text"]
            else:
                seen[r["chapter_path"]] = len(chs)
                chs.append({"path": r["chapter_path"], "summary": r["summary_text"], "chapter": r["chapter"],
                            "page": r["sec"] + 1})
        for k in range(2, len(chs) + 1):
            pick = pick_scenes(k)
            sc = [chs[i] for i in pick]
            alt = [next((v for s2, v in by_ch[(bid, c["path"])].items() if s2 != src), None) for c in sc]
            items.append({"seq": f"{bid}/{src}", "k": k, "n_chapters": len(chs), "final": k == len(chs),
                          "scene_text": [c["summary"] for c in sc], "probe_chapter": [c["chapter"] for c in sc],
                          "probe_alt": alt, "pages": [c["page"] for c in sc]})
    log(f"task2: {len(items)} items over {len(seqs)} sequences")
    arms = {a: Cspw(a) for a in ARMS}
    out = {"n_items": len(items), "n_sequences": len({it['seq'] for it in items}),
           "n_final_items": sum(it["final"] for it in items),
           "n_items_with_alt_probe": sum(all(a is not None for a in (it["probe_alt"][0], it["probe_alt"][-1]))
                                         for it in items),
           "scene_M_hist": dict(Counter(len(it["pages"]) for it in items)), "results": {}}
    parity = {"checked": 0, "mismatch": 0}
    for ename, enc in encs.items():
        texts = sorted({t for it in items for t in it["scene_text"]})
        D = DocEmb(enc, texts, args.cap_summary)
        idx = {t: i for i, t in enumerate(texts)}
        ptexts = sorted({t for it in items for t in it["probe_chapter"]})
        Dp = DocEmb(enc, ptexts, args.cap_chapter)
        pidx = {t: i for i, t in enumerate(ptexts)}
        atexts = sorted({t for it in items for t in it["probe_alt"] if t})
        Da = DocEmb(enc, atexts, args.cap_summary)
        aidx = {t: i for i, t in enumerate(atexts)}
        systems = {"baseline/flat_mean": None, "baseline/recency_only": None, "baseline/positional_first_last": None}
        if enc.int8:
            for a in ARMS:
                if ENC_OF_ARM[a] == ename:
                    systems[f"cspw/{a}"] = a
                    if a == "arm4_full_cspw":
                        systems["cspw/arm4_full_cspw[no_dpage: pages all equal]"] = a
                        systems["cspw/arm4_full_cspw[unit_dpage: pages 1..M]"] = a
        acc = {s: defaultdict(list) for s in systems}
        for it in items:
            Mx = len(it["pages"])
            if enc.int8:
                S = [D.int8[idx[t]] for t in it["scene_text"]]
            P = np.stack([Dp.doc[pidx[t]] for t in it["probe_chapter"]])
            A = None
            if it["probe_alt"][0] is not None and it["probe_alt"][-1] is not None:
                A = np.stack([Da.doc[aidx[it["probe_alt"][0]]], Da.doc[aidx[it["probe_alt"][-1]]]])
            Sv = np.stack([D.doc[idx[t]] for t in it["scene_text"]])
            for s, a in systems.items():
                def readout(order):
                    """(origin, arc) vectors for scenes permuted by ``order`` (pages stay with positions)."""
                    if a is None:
                        X = Sv[order]
                        if s.endswith("flat_mean"):
                            v = norm(X.mean(0))
                            return v, v
                        if s.endswith("recency_only"):
                            return X[-1], X[-1]
                        return X[0], X[-1]
                    pages = list(it["pages"])
                    if "no_dpage" in s:
                        pages = [1] * Mx
                    elif "unit_dpage" in s:
                        pages = list(range(1, Mx + 1))
                    sl = arms[a].slots([S[i] for i in order], pages)
                    if a == "arm4_full_cspw" and parity["checked"] < 300 and order == list(range(Mx)):
                        st = arms[a].slots_torch([S[i] for i in order], pages)
                        parity["checked"] += 1
                        parity["mismatch"] += int(not np.array_equal(np.asarray(st).round().astype(np.int64), sl))
                    return norm(sl[1]), norm(sl[2])
                ident = list(range(Mx))
                o, r = readout(ident)
                so, sr = o @ P.T, r @ P.T
                ok_o, ok_r = bool(so[0] > so[-1]), bool(sr[-1] > sr[0])
                rec = acc[s]
                rec["origin_pair"].append(ok_o)
                rec["latest_pair"].append(ok_r)
                rec["joint_pair"].append(ok_o and ok_r)
                rec["origin_R1"].append(gold_rank(so, 0) == 1)
                rec["latest_R1"].append(gold_rank(sr, Mx - 1) == 1)
                rec["final_joint"].append(ok_o and ok_r) if it["final"] else None
                rec["final_balanced"].append((ok_o + ok_r) / 2) if it["final"] else None
                if A is not None:
                    ao, ar = o @ A.T, r @ A.T
                    rec["alt_origin_pair"].append(bool(ao[0] > ao[1]))
                    rec["alt_latest_pair"].append(bool(ar[1] > ar[0]))
                    rec["alt_joint_pair"].append(bool(ao[0] > ao[1]) and bool(ar[1] > ar[0]))
                # temporal swap: first and last scene trade places, pages stay -> slots must follow content
                sw = list(range(Mx))
                sw[0], sw[-1] = sw[-1], sw[0]
                o2, r2 = readout(sw)
                s2o, s2r = o2 @ P.T, r2 @ P.T
                rec["swap_joint"].append(ok_o and ok_r and bool(s2o[-1] > s2o[0]) and bool(s2r[0] > s2r[-1]))
        res = {}
        for s, rec in acc.items():
            d = {k: float(np.mean(v)) for k, v in rec.items() if v}
            d["balanced_pair_acc"] = (d["origin_pair"] + d["latest_pair"]) / 2
            if "alt_origin_pair" in d:
                d["alt_balanced_pair_acc"] = (d["alt_origin_pair"] + d["alt_latest_pair"]) / 2
            d["n"] = len(rec["origin_pair"])
            d["n_alt"] = len(rec["alt_origin_pair"])
            d["n_final"] = len(rec["final_joint"])
            res[s] = d
        out["results"][ename] = res
        enc.save()
    out["arm4_int8_vs_torch_quant_parity"] = parity
    return out


# ----------------------------------------------------------------------------------------------
# Task 3: spoiler horizon
# ----------------------------------------------------------------------------------------------

NAME_STOP = set("""The A An And But Or If When While After Before Then There Here This That These Those He She It They
We You I His Her Its Their Our Your My Mr Mrs Miss Ms Sir Lady Lord Dr St Saint King Queen Prince Princess Duke
Captain Colonel General Father Mother Brother Sister Uncle Aunt God Chapter Chapters Act Scene Book Part Section
Volume Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August
September October November December English French German Italian Russian Spanish Christian Christ Heaven Hell
Meanwhile However Although Though Because Since Once Still Even Yet Also Finally Later Soon Now One Two Three
Shmoop Sparknotes SparkNotes Cliffnotes CliffsNotes Gradesaver GradeSaver Novel Summary Analysis Notes
Narrator Author Reader Act""".split())


def extract_entities(rs: list[dict], max_ents: int):
    """Character / place names of one (book, source) sequence: capitalised tokens seen mid-sentence >= 3
    times in >= 2 sections, never as a lower-case word in the sequence's summaries."""
    texts = [r["summary_text"] for r in rs]
    lower = Counter(w for t in texts for w in re.findall(r"\b[a-z][a-z']+\b", t))
    mid = Counter()
    secs = defaultdict(set)
    for i, t in enumerate(texts):
        for s in SENT.split(t):
            toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", s)
            for j, w in enumerate(toks):
                if j > 0 and re.fullmatch(r"[A-Z][a-z]{2,}", w) and w not in NAME_STOP:
                    mid[w] += 1
        for w in set(re.findall(r"\b[A-Z][a-z]{2,}\b", t)):
            secs[w].add(i)
    cands = [w for w, c in mid.items() if c >= 3 and len(secs[w]) >= 2 and w.lower() not in lower]
    cands.sort(key=lambda w: (-len(secs[w]), -mid[w], w))
    return cands[:max_ents]


def fit_words(s: str, n: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= n:
        return s
    cut = CM.utf8_fit(s, n).decode("utf-8")
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


def task3(rows, seqs, encs, args):
    chap_tok = {}
    timelines = []  # one per (sequence, entity)
    for (bid, src), rs in sorted(seqs.items()):
        if len(rs) < 2:
            continue
        ents = extract_entities(rs, args.max_entities)
        # section token sets of the ORIGINAL chapter text (reference for fact precision)
        sec_tok = [set(content_tokens(r["chapter"])) for r in rs]
        prefix, acc = [set()], set()
        for st in sec_tok:
            acc = acc | st
            prefix.append(acc)
        suffix, acc = [set()], set()
        for st in reversed(sec_tok):
            acc = acc | st
            suffix.append(acc)
        suffix = suffix[::-1]  # suffix[i] = union of sections i+1..n (0-based i..n-1)
        chap_tok[(bid, src)] = (sec_tok, prefix, suffix)
        n = len(rs)
        for name in ents:
            evs = []
            for i, r in enumerate(rs):
                sents = [s.strip() for s in SENT.split(" ".join(r["summary_text"].split()))
                         if re.search(rf"\b{re.escape(name)}\b", s)]
                if not sents:
                    continue
                others = [e for e in ents if e != name and re.search(rf"\b{re.escape(e)}\b", sents[0])]
                evs.append({"cutoff": i + 1, "chapter_or_page": i + 1, "salience_hint": min(255, 32 * len(sents)),
                            "event_gloss": fit_words(sents[0], 63), "target_or_loc": others[0] if others else "",
                            "scene_text": " ".join(sents), "sentences": sents})
            if len(evs) < 2:
                continue
            for j, e in enumerate(evs):
                e["slot_hint"] = 1 if j == 0 else 2
            timelines.append({"bid": bid, "src": src, "seq": f"{bid}/{src}", "term": name, "n": n,
                              "term_hash": M.fnv1a64(name.encode("utf-8")),
                              "book_hash": M.fnv1a64(f"booksum/{bid}/{src}".encode("utf-8")),
                              "first_cutoff": evs[0]["cutoff"], "events": evs})
    queries = []
    for ti, tl in enumerate(timelines):
        first = tl["first_cutoff"]
        if first > 1:
            queries.append({"tl": ti, "cur": first - 1, "pre_intro": True})
        for cur in range(first, tl["n"]):  # k < M: later chapters exist
            if any(e["cutoff"] > cur for e in tl["events"]):
                queries.append({"tl": ti, "cur": cur, "pre_intro": False})
    log(f"task3: {len(timelines)} entity timelines, {len(queries)} cutoff queries "
        f"({sum(q['pre_intro'] for q in queries)} pre-introduction probes)")
    out = {"n_timelines": len(timelines), "n_queries": len(queries),
           "n_pre_intro_probes": sum(q["pre_intro"] for q in queries),
           "n_sequences": len({tl["seq"] for tl in timelines}), "systems": {},
           "entity_examples": {}}
    ex_ents = defaultdict(list)
    for tl in timelines:
        if len(ex_ents[tl["seq"]]) < 8:
            ex_ents[tl["seq"]].append(tl["term"])
    out["entity_examples"] = dict(list(ex_ents.items())[:6])

    def score(tl, cur, text, used_cutoffs, produced):
        """-> dict(leak, tokens, supported, spoil_tokens)."""
        sec_tok, prefix, suffix = chap_tok[(tl["bid"], tl["src"])]
        allowed = cur >= tl["first_cutoff"]
        leak = produced and ((not allowed) or any(c > cur for c in used_cutoffs))
        if not produced:
            return {"leak": False, "tokens": 0, "supported": 0, "spoil": 0, "produced": False}
        drop = {t.lower() for t in tl["term"].split()}
        toks = content_tokens(text, drop)
        vis = prefix[min(cur, len(prefix) - 1)]
        fut = suffix[cur] if cur < len(sec_tok) else set()
        sup = sum(t in vis for t in toks)
        spo = sum((t not in vis) and (t in fut) for t in toks)
        return {"leak": bool(leak), "tokens": len(toks), "supported": sup, "spoil": spo, "produced": True}

    def agg(name, recs, examples):
        n = len(recs)
        prod = [r for r in recs if r["produced"]]
        tok = sum(r["tokens"] for r in prod)
        pre = [r for r, q in zip(recs, queries) if q["pre_intro"]]
        post = [r for r, q in zip(recs, queries) if not q["pre_intro"]]
        out["systems"][name] = {
            "spoiler_leakage_SL": sum(r["leak"] for r in recs) / max(1, n),
            "SL_post_introduction": sum(r["leak"] for r in post) / max(1, len(post)),
            "SL_pre_introduction_probes": sum(r["leak"] for r in pre) / max(1, len(pre)),
            "fact_precision_FP_token": sum(r["supported"] for r in prod) / max(1, tok),
            "spoiler_token_rate": sum(r["spoil"] for r in prod) / max(1, tok),
            "coverage_produced": len(prod) / max(1, n), "n_queries": n, "n_produced": len(prod),
            "n_content_tokens": tok, "examples": examples[:4]}

    # ---- CS-PW arm4: exact device path (SGRF gate + int8 weave)
    enc = encs["E3"]
    vec_cache = {}

    def vec_of(text):
        if text not in vec_cache:
            V = enc.encode(windows(text, args.cap_summary)).astype(np.int64)
            vec_cache[text] = M.l2_int8([int(x) for x in V.sum(0)])
        return vec_cache[text]

    a4 = Cspw("arm4_full_cspw")
    ents = []
    for tl in timelines:
        ents.append({"term_hash": tl["term_hash"], "book_path_hash": tl["book_hash"], "first_cutoff": tl["first_cutoff"],
                     "entity_kind": 1, "flags": 0, "en_name_or_role": tl["term"], "book_short": tl["bid"][:4],
                     "events": [{**{k: e[k] for k in ("cutoff", "chapter_or_page", "slot_hint", "salience_hint",
                                                     "event_gloss", "target_or_loc")}, "vec": vec_of(e["scene_text"])}
                                for e in tl["events"]]})
    blob = CM.pack_sgrf(ents, [])
    graph = CM.StoryGraph(blob)
    out["story_graph_idx_bytes"] = len(blob)
    if args.export_sgrf:
        Path(args.export_sgrf).write_bytes(blob)
        out["story_graph_idx_sha256"] = hashlib.sha256(blob).hexdigest()
    for gated in (True, False):
        recs, exs = [], []
        for q in queries:
            tl = timelines[q["tl"]]
            if gated:
                hit = graph.lookup(tl["term_hash"], tl["book_hash"], q["cur"])
            else:
                ent = graph.find_entity(tl["term_hash"], tl["book_hash"])
                hit = (ent, graph.events[ent["event_start"]:ent["event_start"] + ent["event_count"]])
            if hit is None:
                recs.append(score(tl, q["cur"], "", [], False))
                continue
            text, _ = CM.weave_refresher(a4.w, hit[0], hit[1], tl["term"])
            scenes, echo = CM.select_scenes(hit[1])
            used = [e["cutoff"] for e in scenes + ([echo] if echo is not None else [])]
            body = text[len("Story: "):] if text.startswith("Story: ") else text
            recs.append(score(tl, q["cur"], body, used, bool(text)))
            if len(exs) < 6 and text:
                exs.append({"seq": tl["seq"], "term": tl["term"], "cur": q["cur"], "text": text})
        agg("cspw/arm4_full_cspw[SGRF gate + int8 weave]" if gated else "cspw/arm4_full_cspw[ungated weave]", recs, exs)
    # ---- CS-PW arms 0-3 (flat pooling, ungated by design: phrase-bank pointer decoder as in train_cspw.py)
    for a in ARMS[:4]:
        cs = Cspw(a)
        e = encs[ENC_OF_ARM[a]]
        vc = {}

        def v_of(text):
            if text not in vc:
                V = e.encode(windows(text, args.cap_summary)).astype(np.int64)
                vc[text] = M.l2_int8([int(x) for x in V.sum(0)])
            return vc[text]

        recs, exs, per_tl = [], [], {}
        for q in queries:
            tl = timelines[q["tl"]]
            if q["tl"] not in per_tl:  # ungated: the output does not depend on the reading position
                evs = [{"slot_hint": ev["slot_hint"], "salience_hint": ev["salience_hint"], "cutoff": ev["cutoff"],
                        "chapter_or_page": ev["chapter_or_page"], "text": ev["scene_text"]} for ev in tl["events"]]
                scenes, _ = CM.select_scenes(evs)
                sl = cs.slots_torch([v_of(x["text"]) for x in scenes], [x["chapter_or_page"] for x in scenes])
                with torch.no_grad():
                    pl = cs.mod.phrase_logits(torch.as_tensor(sl, dtype=torch.float64)[None], typed=False)[0]
                pidx = pl.argmax(-1).tolist()
                ph = [cs.phrases[128 * k + int(pidx[k])] for k in range(4)]
                per_tl[q["tl"]] = (f"{ph[0]} \u2014 {ph[1]}" + (f"; {ph[2]}" if len(scenes) >= 2 else "") + ".",
                                   [x["cutoff"] for x in scenes])
            text, used = per_tl[q["tl"]]
            recs.append(score(tl, q["cur"], text, used, True))
            if len(exs) < 4:
                exs.append({"seq": tl["seq"], "term": tl["term"], "cur": q["cur"], "text": f"Story: {tl['term']}: {text}"})
        agg(f"cspw/{a}[flat pool, ungated]", recs, exs)
    # ---- retrieval baselines: top-1 mention sentence for "Who is <name>?" (ungated / cutoff-filtered)
    sent_pool = []
    for tl in timelines:
        sent_pool.append([(e["cutoff"], s) for e in tl["events"] for s in e["sentences"]])
    rets = {f"{n}/dense_top1": n for n in encs} | {"BM25/top1": "bm25"}
    for rname, kind in rets.items():
        scores_by_tl = []
        for ti, tl in enumerate(timelines):
            sents = [s for _, s in sent_pool[ti]]
            qtext = f"Who is {tl['term']}?"
            if kind == "bm25":
                sc = bm25_scores([qtext + " " + tl["term"]], sents)[0]
            else:
                en = encs[kind]
                Q = norm(en.encode([qtext]).astype(np.float32))
                Sx = DocEmb(en, sents, 2).doc
                sc = (Q @ Sx.T)[0]
            scores_by_tl.append(sc)
        for gated in (False, True):
            recs, exs = [], []
            for q in queries:
                tl = timelines[q["tl"]]
                pool = sent_pool[q["tl"]]
                sc = scores_by_tl[q["tl"]]
                ok = [i for i, (c, _) in enumerate(pool) if (not gated) or c <= q["cur"]]
                if gated and q["cur"] < tl["first_cutoff"]:
                    ok = []
                if not ok:
                    recs.append(score(tl, q["cur"], "", [], False))
                    continue
                i = max(ok, key=lambda j: (sc[j], -j))
                recs.append(score(tl, q["cur"], pool[i][1], [pool[i][0]], True))
                if len(exs) < 4:
                    exs.append({"seq": tl["seq"], "term": tl["term"], "cur": q["cur"], "text": pool[i][1],
                                "evidence_chapter": pool[i][0]})
            agg(f"retrieval/{rname}[{'cutoff-filtered' if gated else 'ungated'}]", recs, exs)
    for en in encs.values():
        en.save()
    return out


# ----------------------------------------------------------------------------------------------
# Encoder parity vs exported device files
# ----------------------------------------------------------------------------------------------


def device_parity(enc: Encoder, texts: list[str]) -> dict:
    pb, eb = MODELS / "platonic-hnet.bin", MODELS / "engram.bin"
    if not (pb.exists() and eb.exists()):
        return {"available": False}
    ref = M.PlhnIntNp(pb.read_bytes(), eb.read_bytes(), bigram=True)
    got = enc.encode(texts)
    same = sum(int(np.array_equal(np.asarray(ref.forward(t), np.int64), got[i].astype(np.int64)))
               for i, t in enumerate(texts))
    return {"available": True, "encoder": enc.name, "n_texts": len(texts), "bit_exact": same,
            "platonic_hnet_bin_sha256": sha256(pb), "engram_bin_sha256": sha256(eb)}


def render_tables(res: dict) -> str:
    """Markdown tables straight from narrative_results.json (no hand-copied numbers)."""
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    out = []
    t1 = res.get("task1_matching", {})
    for key, b in t1.items():
        out.append(f"### Task 1 - `{key}`\n\n{b['desc']}  \nqueries: **{b['n_queries']}**"
                   + (f", mean pool size {b['mean_pool_size']:.1f}" if "mean_pool_size" in b else "")
                   + (f", candidates {b['n_candidates']}" if "n_candidates" in b else "") + "\n")
        who = b.get("who_character_slice")
        hdr = "| system | n | R@1 | R@5 | MRR |" + (" Who-slice R@1 | Who-slice MRR |" if who else "")
        out.append(hdr + "\n|" + "---|" * (hdr.count("|") - 1))
        for s, m in sorted(b["systems"].items(), key=lambda kv: -kv[1].get("MRR", 0)):
            row = f"| {s} | {m['n']} | {f3(m['R@1'])} | {f3(m['R@5'])} | {f3(m['MRR'])} |"
            if who:
                w = who["systems"][s]
                row += f" {f3(w['R@1'])} | {f3(w['MRR'])} |"
            out.append(row)
        if who:
            out.append(f"\nWho-slice queries: {who['n_queries']}")
        out.append("")
    t2 = res.get("task2_temporal")
    if t2:
        out.append(f"### Task 2 - temporal state tracking\n\nitems **{t2['n_items']}** (sequences {t2['n_sequences']}, "
                   f"final full-book items {t2['n_final_items']}, items with other-source probes "
                   f"{t2['n_items_with_alt_probe']}); scene count hist {t2['scene_M_hist']}; arm4 int8 vs torch-QAT "
                   f"slot parity {t2['arm4_int8_vs_torch_quant_parity']}\n")
        hdr = ("| encoder | system | origin pair | latest pair | balanced | joint | origin R@1 | latest R@1 | "
               "swap joint | final balanced | alt balanced (n) |")
        out.append(hdr + "\n|" + "---|" * (hdr.count("|") - 1))
        for e, sysd in t2["results"].items():
            for s, d in sysd.items():
                out.append(f"| {e} | {s} | {f3(d['origin_pair'])} | {f3(d['latest_pair'])} | "
                           f"{f3(d['balanced_pair_acc'])} | {f3(d['joint_pair'])} | {f3(d['origin_R1'])} | "
                           f"{f3(d['latest_R1'])} | {f3(d['swap_joint'])} | {f3(d.get('final_balanced'))} | "
                           f"{f3(d.get('alt_balanced_pair_acc'))} ({d['n_alt']}) |")
        out.append("")
    t3 = res.get("task3_spoiler")
    if t3:
        out.append(f"### Task 3 - spoiler horizon\n\nentity timelines **{t3['n_timelines']}**, cutoff queries "
                   f"**{t3['n_queries']}** (pre-introduction probes {t3['n_pre_intro_probes']}), sequences "
                   f"{t3['n_sequences']}\n")
        hdr = ("| system | SL | SL post-intro | SL pre-intro | FP (token) | spoiler-token rate | coverage | "
               "content tokens |")
        out.append(hdr + "\n|" + "---|" * (hdr.count("|") - 1))
        for s, d in t3["systems"].items():
            out.append(f"| {s} | {f3(d['spoiler_leakage_SL'])} | {f3(d['SL_post_introduction'])} | "
                       f"{f3(d['SL_pre_introduction_probes'])} | {f3(d['fact_precision_FP_token'])} | "
                       f"{f3(d['spoiler_token_rate'])} | {f3(d['coverage_produced'])} | {d['n_content_tokens']} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-summary", type=int, default=16, help="max 256-byte windows per summary")
    ap.add_argument("--cap-summary-nqa", type=int, default=32, help="max windows per NarrativeQA summary")
    ap.add_argument("--cap-chapter", type=int, default=48, help="max windows per chapter text")
    ap.add_argument("--max-entities", type=int, default=8, help="entities per (book, source) sequence")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--tasks", default="1,2,3")
    ap.add_argument("--tables-only", action="store_true", help="render results/narrative_tables.md and exit")
    ap.add_argument("--out", default="narrative_results.json", help="results file name under results/")
    ap.add_argument("--export-sgrf", default=None, help="also write the task-3 BookSum SGRF story graph to this path")
    args = ap.parse_args(argv)
    if args.tables_only:
        (RES / "narrative_tables.md").write_text(render_tables(json.loads((RES / "narrative_results.json").read_text())))
        return 0
    torch.set_num_threads(args.threads)
    CACHE.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, seqs, bs_info = load_booksum()
    nqa = load_narrativeqa()
    log(f"booksum: {len(rows)} rows, {len(seqs)} (book, source) sequences; narrativeqa: {len(nqa[2])} test "
        f"questions, {len(nqa[0])} summaries")
    encs = {n: Encoder(n) for n in ENCODERS}
    enc_info = {n: {"checkpoint": e.src, "state_sha256_16": e.state_sha, "int8": e.int8,
                    "engram_hash_table": getattr(e, "engram_hash_source", None)} for n, e in encs.items()}
    shared = {}
    for a in ARMS:
        ck = torch.load(RUN / f"{a}.pt", map_location="cpu", weights_only=False)
        shared[a] = {"encoder": ck["encoder_name"], "encoder_state_sha256_16": state_hash(ck["encoder"]),
                     "cspw_cfg": ck["cspw_cfg"]}
    results = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args),
               "datasets": {"booksum": {**bs_info, "n_rows": len(rows), "n_books": len({r['bid'] for r in rows}),
                                        "n_sequences": len(seqs),
                                        "sources": dict(Counter(r["source"] for r in rows))},
                            "narrativeqa": {**nqa[3], "n_test_questions": len(nqa[2]),
                                            "n_test_summaries": len(nqa[0])},
                            "liscu": {"used": False, "reason": "not distributed; recreation requires crawling the "
                                      "Wayback Machine into PostgreSQL (github.com/huangmeng123/lit_char_data_wayback)"}},
               "encoders": enc_info, "arms": shared}
    sample = [r["summary_text"][:240] for r in rows[:40]] + [q["question"] for q in nqa[2][:40]]
    results["device_parity_E3_vs_models_bin"] = device_parity(encs["E3"], sample)
    log("device parity", results["device_parity_E3_vs_models_bin"])
    tasks = set(args.tasks.split(","))
    cspw4 = Cspw("arm4_full_cspw")
    results["arm4_cspw_bin"] = cspw4.bin_info
    out_path = RES / args.out

    def save():
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str))
        os.replace(tmp, out_path)

    try:
        if "1" in tasks:
            results["task1_matching"] = task1(rows, seqs, nqa, encs, cspw4, args)
            save()
        if "2" in tasks:
            results["task2_temporal"] = task2(rows, seqs, encs, args)
            save()
        if "3" in tasks:
            results["task3_spoiler"] = task3(rows, seqs, encs, args)
            save()
    finally:
        for e in encs.values():
            e.save()
    results["wall_seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {results['wall_seconds']} s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
