"""Public WSD / Word-in-Context, cross-lingual zh<->en retrieval and STS / paraphrase benchmarks for the
Platonic-Engram-HNet encoders (E0 v1, E1 +Bigram, E3 +QwenKD = Arm 4 encoder), the matched-RAM BPE Arm 0,
lexical baselines and two 100M+ reference embedding models.

    external_benchmarks/.venv/bin/python external_benchmarks/run_wsd_xling_bench.py [--no-reference]

Datasets (downloaded into ``data/wsd_xling/``, see ``DATASETS`` for exact sources):

* WiC (SuperGLUE / Pilehvar & Camacho-Collados 2019) en: dev 638 (threshold), test 1,400 (gold released)
* XL-WiC zh (WordNet split, Raganato et al. 2020): valid 3,046 (threshold), test 5,538
* MCL-WiC (SemEval-2021 Task 2): en-en dev 1,000 / test 1,000, zh-zh dev 1,000 / test 1,000,
  en-zh (cross-lingual) test 1,000 (threshold from en-en dev, the task's zero-shot protocol)
* Tatoeba cmn-eng (LASER / XTREME, 1,000 pairs): zh->en and en->zh R@1/R@5/R@10/MRR over all 1,000
* STS-B en (mteb/stsbenchmark-sts, test 1,379) and STS-B zh (C-MTEB/STSB, test 1,361): Spearman
* PAWS-X en / zh (test 2,000 each), BQ (test 10,000), ATEC (test 20,000): Spearman (C-MTEB protocol),
  ROC-AUC, cosine separation, accuracy at the dev-tuned threshold

Protocol: every model is a frozen encoder; similarity = cosine.  WiC-style tasks score the pair by the
cosine of the target-word representations (``span``: mean of the contextual chunk / token states that
overlap the target's characters) and, separately, of the whole-context embeddings (``sent``); the decision
threshold is tuned for accuracy on the dev split only and applied unchanged to test.  ROC-AUC is
threshold-free.  Sentences are cut to a <= 256 UTF-8-byte window centred on the target for WiC tasks
(the device encoder reads at most 256 bytes); all models see the same window.  Our H-Net encoders run the
exact int8 QAT graph (bit-exact with the device, checked against the exported ``platonic-hnet.bin`` +
``engram.bin`` with the numpy integer reference).  Arm 0 (BPE transformer) has no int8 export and runs
FP32.

Raw metrics -> ``results/wsd_xling_results.json``; summary -> ``results/wsd_xling_report.md``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import model as M  # noqa: E402

DATA = HERE / "data" / "wsd_xling"
HF = DATA / "hf"
CACHE = HERE / "cache" / "wsd_xling"
RESULTS = HERE / "results"
MODELS_DIR = ROOT / "weights"
CSPW_RUN = ROOT / "runs" / "cspw-v2"
V1_RUN = ROOT / "runs" / "v1-final"
WINDOW_BYTES = M.MAX_BYTES
REF_MODELS = {
    "MiniLM-L12-multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
}
DEV_CAP = 5000  # BQ / ATEC dev pairs used for threshold tuning (seeded sample)
# Qwen3-Embedding-0.6B measured 44 sentences/s and 18 span texts/s on this Mac's MPS (fp16), so the full
# ~170k-text suite would take > 1.5 h.  It therefore runs on seeded subsamples (sent feature only); STS-B and
# Tatoeba stay complete.  Every subsample size is written into the JSON ("caps") and the report.
REF_CAPS = {"Qwen3-Embedding-0.6B": {"wic_dev": 250, "wic_test": 500, "pair_dev": 250, "pair_test": 500, "span": False}}

DATASETS = {
    "wic_en": "https://pilehvar.github.io/wic/package/WiC_dataset.zip (dev + test with test.gold.txt)",
    "xlwic_zh": "https://pilehvar.github.io/xlwic/data/xlwic_datasets.zip (xlwic_wn/chinese_zh)",
    "mclwic": "https://github.com/SapienzaNLP/mcl-wic (SemEval-2021_MCL-WiC_all-datasets.zip + test-gold-data.zip)",
    "tatoeba_cmn_eng": "https://raw.githubusercontent.com/facebookresearch/LASER/main/data/tatoeba/v1/tatoeba.cmn-eng.{cmn,eng}"
                       " (== hf mteb/tatoeba-bitext-mining test/cmn-eng.jsonl.gz)",
    "stsb_en": "hf dataset mteb/stsbenchmark-sts (test.jsonl.gz, validation.jsonl.gz)",
    "stsb_zh": "hf dataset C-MTEB/STSB (test, validation parquet)",
    "pawsx_en": "hf dataset google-research-datasets/paws-x en (test, validation)",
    "pawsx_zh": "hf dataset google-research-datasets/paws-x zh (test, validation)",
    "bq_zh": "hf dataset C-MTEB/BQ (test, validation)",
    "atec_zh": "hf dataset C-MTEB/ATEC (test, validation)",
}


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------

def _wic_item(s1, r1, s2, r2, label):
    return {"s1": s1, "r1": r1, "s2": s2, "r2": r2, "label": int(label)}


def _tok_char_range(sent: str, idx: int):
    toks = sent.split(" ")
    start = sum(len(t) + 1 for t in toks[:idx])
    return [(start, start + len(toks[idx]))]


def load_wic_en():
    out = {}
    for split in ("dev", "test"):
        rows = (DATA / "wic" / split / f"{split}.data.txt").read_text().splitlines()
        gold = (DATA / "wic" / split / f"{split}.gold.txt").read_text().split()
        items = []
        for row, g in zip(rows, gold):
            word, pos, idx, s1, s2 = row.split("\t")
            i1, i2 = map(int, idx.split("-"))
            items.append(_wic_item(s1, _tok_char_range(s1, i1), s2, _tok_char_range(s2, i2), g == "T"))
        out[split] = items
    return out


def load_xlwic_zh():
    d = DATA / "xlwic" / "xlwic_datasets" / "xlwic_wn" / "chinese_zh"

    def parse(rows, labels):
        items = []
        for row, lab in zip(rows, labels):
            f = row.split("\t")
            items.append(_wic_item(f[6], [(int(f[2]), int(f[3]))], f[7], [(int(f[4]), int(f[5]))], int(lab)))
        return items

    valid = [r.split("\t") for r in (d / "zh_valid.txt").read_text().splitlines() if r.strip()]
    test = [r for r in (d / "zh_test_data.txt").read_text().splitlines() if r.strip()]
    gold = (d / "zh_test_gold.txt").read_text().split()
    return {"dev": parse(["\t".join(v[:8]) for v in valid], [v[8] for v in valid]), "test": parse(test, gold)}


def _mcl_ranges(d, k):
    if f"ranges{k}" in d:
        return [tuple(map(int, r.split("-"))) for r in str(d[f"ranges{k}"]).split(",") if r.strip()]
    return [(int(d[f"start{k}"]), int(d[f"end{k}"]))]


def _load_mcl(data_path: Path, gold_path: Path):
    data = json.loads(data_path.read_text())
    gold = {g["id"]: g["tag"] for g in json.loads(gold_path.read_text())}
    return [_wic_item(d["sentence1"], _mcl_ranges(d, 1), d["sentence2"], _mcl_ranges(d, 2), gold[d["id"]] == "T")
            for d in data]


def load_mclwic():
    b = DATA / "mclwic"
    m = b / "MCL-WiC"
    en_dev = _load_mcl(m / "dev/multilingual/dev.en-en.data", m / "dev/multilingual/dev.en-en.gold")
    return {
        "mclwic_en_en": {"dev": en_dev, "test": _load_mcl(m / "test/multilingual/test.en-en.data", b / "test.en-en.gold")},
        "mclwic_zh_zh": {"dev": _load_mcl(m / "dev/multilingual/dev.zh-zh.data", m / "dev/multilingual/dev.zh-zh.gold"),
                         "test": _load_mcl(m / "test/multilingual/test.zh-zh.data", b / "test.zh-zh.gold")},
        "mclwic_en_zh": {"dev": en_dev, "dev_note": "no en-zh dev split exists; threshold from en-en dev",
                         "test": _load_mcl(m / "test/crosslingual/test.en-zh.data", b / "test.en-zh.gold")},
    }


def load_tatoeba():
    zh = [l.strip() for l in (DATA / "tatoeba.cmn-eng.cmn").read_text(encoding="utf-8").splitlines()]
    en = [l.strip() for l in (DATA / "tatoeba.cmn-eng.eng").read_text(encoding="utf-8").splitlines()]
    with gzip.open(HF / "mteb__tatoeba-bitext-mining" / "test__cmn-eng.jsonl.gz", "rt", encoding="utf-8") as f:
        mt = [json.loads(l) for l in f]
    same = sum(a == r["sentence1"].strip() and b == r["sentence2"].strip() for a, b, r in zip(zh, en, mt))
    return {"zh": zh, "en": en, "laser_vs_mteb_identical_pairs": same, "mteb_pairs": len(mt)}


def _pairs(df, score_col):
    s1, s2, sc = df["sentence1"].tolist(), df["sentence2"].tolist(), df[score_col].tolist()
    keep = [(a, b, float(c)) for a, b, c in zip(s1, s2, sc)
            if isinstance(a, str) and isinstance(b, str) and a.strip() and b.strip()]
    return keep, len(s1) - len(keep)


def load_pair_tasks():
    def pq(repo, name):
        return pd.read_parquet(HF / repo / name)

    def jl(repo, name):
        return pd.read_json(HF / repo / name, lines=True)

    spec = {
        "stsb_en": ("sts", jl("mteb__stsbenchmark-sts", "validation.jsonl.gz"), jl("mteb__stsbenchmark-sts", "test.jsonl.gz"), "score"),
        "stsb_zh": ("sts", pq("C-MTEB__STSB", "data__validation-00000-of-00001-f26577b4869b4653.parquet"),
                    pq("C-MTEB__STSB", "data__test-00000-of-00001-8599d9428e96fe39.parquet"), "score"),
        "pawsx_en": ("pair", pq("google-research-datasets__paws-x", "en__validation-00000-of-00001.parquet"),
                     pq("google-research-datasets__paws-x", "en__test-00000-of-00001.parquet"), "label"),
        "pawsx_zh": ("pair", pq("google-research-datasets__paws-x", "zh__validation-00000-of-00001.parquet"),
                     pq("google-research-datasets__paws-x", "zh__test-00000-of-00001.parquet"), "label"),
        "bq_zh": ("pair", pq("C-MTEB__BQ", "data__validation-00000-of-00001-7f96017a1c50f024.parquet"),
                  pq("C-MTEB__BQ", "data__test-00000-of-00001-27441f0c8fdc9664.parquet"), "score"),
        "atec_zh": ("pair", pq("C-MTEB__ATEC", "data__validation-00000-of-00001-807a1bfe59452c6f.parquet"),
                    pq("C-MTEB__ATEC", "data__test-00000-of-00001-4628426553e5fece.parquet"), "score"),
    }
    tasks = {}
    for name, (kind, dev, test, col) in spec.items():
        d, dd = _pairs(dev, col)
        t, td = _pairs(test, col)
        dev_total = len(d)
        if len(d) > DEV_CAP:
            d = random.Random(0).sample(d, DEV_CAP)
        tasks[name] = {"kind": kind, "dev": d, "test": t, "dropped_empty": {"dev": dd, "test": td},
                       "dev_total": dev_total}
    return tasks


def window(text: str, ranges, limit: int = WINDOW_BYTES):
    """Largest char window of ``text`` around the target ranges with <= ``limit`` UTF-8 bytes.
    Returns (window_text, shifted_ranges)."""
    if len(text.encode("utf-8")) <= limit:
        return text, list(ranges)
    a = max(0, min(r[0] for r in ranges))
    b = min(len(text), max(r[1] for r in ranges))
    used = len(text[a:b].encode("utf-8"))
    if used > limit:  # target alone overflows: keep its head
        b = a
        used = 0
        while b < len(text) and used + len(text[b].encode()) <= limit:
            used += len(text[b].encode())
            b += 1
        return text[a:b], [(0, b - a)]
    left = True
    while True:
        moved = False
        for side in ((0, 1) if left else (1, 0)):
            if side == 0 and a > 0:
                n = len(text[a - 1].encode())
                if used + n <= limit:
                    a -= 1
                    used += n
                    moved = True
            elif side == 1 and b < len(text):
                n = len(text[b].encode())
                if used + n <= limit:
                    b += 1
                    used += n
                    moved = True
        left = not left
        if not moved:
            break
    return text[a:b], [(max(0, r[0] - a), min(b, r[1]) - a) for r in ranges]


def wic_views(items):
    t1, r1, t2, r2, y = [], [], [], [], []
    for it in items:
        a, ra = window(it["s1"], it["r1"])
        b, rb = window(it["s2"], it["r2"])
        t1.append(a), r1.append(ra), t2.append(b), r2.append(rb), y.append(it["label"])
    return t1, r1, t2, r2, np.array(y)


def byte_ranges(text: str, ranges):
    return [(len(text[:a].encode("utf-8")), len(text[:b].encode("utf-8"))) for a, b in ranges]


# ----------------------------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------------------------

def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def _batches(texts, bs):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i].encode("utf-8")))
    for k in range(0, len(order), bs):
        yield order[k:k + bs]


class HNetEncoder:
    """Platonic-Engram-HNet encoder from a CS-PW run checkpoint; int8 QAT graph (device-exact)."""

    kind = "hnet"

    def __init__(self, name: str, ckpt: Path, deployed_engram: Path | None = None):
        """``deployed_engram``: use the exported ``engram.bin`` span_hash column as the bigram gate table
        (exactly what the device does; the training graph uses only the mined bigram table)."""
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.name, self.ckpt = name, str(ckpt)
        self.use_bigram = bool(ck["encoder_cfg"]["use_bigram"])
        net = M.PlatonicEngramHNet(use_engram=True, use_bigram=self.use_bigram)
        net.load_state_dict(ck["encoder"], strict=True)
        if deployed_engram is not None:
            blob = deployed_engram.read_bytes()
            raw = np.frombuffer(blob, dtype=np.uint8, count=M.ENGRAM_SLOTS * 64, offset=M.ENGR_SECTOR).reshape(-1, 64)
            net.set_engram_hashes(raw[:, 0:8].copy().view("<u8").ravel())
        net.eval()
        net.quant = True
        self.net = net
        plhn = M.PLHN_FILE_BYTES
        self.footprint = {"params_deploy": M.PLHN_PAYLOAD_BYTES + M.ENGRAM_SLOTS * M.ENGRAM_DIMS,
                          "plhn_bytes": plhn, "engram_bytes": M.ENGR_FILE_BYTES,
                          "deploy_bytes": plhn + M.ENGR_FILE_BYTES, "precision": "int8 (QAT graph)",
                          "bigram_engram": self.use_bigram,
                          "bigram_gate_table": ("deployed engram.bin span_hash column" if deployed_engram
                                                else "training-time mined bigram table (checkpoint engram_hash)"),
                          "bigram_table_nonzero": int((net.engram_hash != 0).sum())}

    @torch.no_grad()
    def sent(self, texts):
        out = np.zeros((len(texts), M.D_CONCEPT), np.float32)
        for idx in _batches(texts, 256):
            v = self.net.encode_int([texts[i] for i in idx])
            out[idx] = np.asarray(v, np.float32) / 127.0
        return _norm(out)

    @torch.no_grad()
    def span(self, texts, ranges):
        out = np.zeros((len(texts), M.D_CONCEPT), np.float32)
        for idx in _batches(texts, 256):
            enc = self.net.encode([texts[i] for i in idx])
            st = enc["starts"].cpu().numpy()
            nc = enc["n_chunks"].cpu().numpy()
            ln = enc["lens"].cpu().numpy()
            xs = enc["chunk_states"].float().cpu().numpy()
            for row, i in enumerate(idx):
                n = int(nc[row])
                if n == 0:
                    continue
                starts = st[row, :n]
                ends = np.append(starts[1:], ln[row])
                sel = np.zeros(n, bool)
                for a, b in byte_ranges(texts[i], ranges[i]):
                    b = min(b, int(ln[row]))
                    if b > a:
                        sel |= (starts < b) & (ends > a)
                out[i] = xs[row, :n][sel].mean(0) if sel.any() else xs[row, :n].sum(0)
        return _norm(out)


class BPEEncoder:
    """Arm 0 matched-RAM BPE transformer baseline (FP32; no int8 export)."""

    kind = "bpe"

    def __init__(self, ckpt: Path):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.name, self.ckpt = "Arm0-BPE", str(ckpt)
        net = M.BPETransformerBaseline(M.ByteBPE(ck["bpe_merges"]))
        net.load_state_dict(ck["state_dict"], strict=True)
        net.eval()
        self.net = net
        self.tok_bytes = np.array(net.bpe.token_bytes())
        self.W = net.out.weight.detach().numpy()
        self.b = net.out.bias.detach().numpy()
        deploy = sum(p.numel() for n, p in net.named_parameters() if not n.startswith(("lm.", "nce_", "sense_")))
        self.footprint = {"params_total": M.count_params(net), "params_deploy": deploy,
                          "deploy_bytes_int8_equiv": deploy, "deploy_bytes_fp32": 4 * deploy,
                          "bpe_merges": len(ck["bpe_merges"]), "precision": "fp32 (no int8 export)"}

    @torch.no_grad()
    def sent(self, texts):
        out = np.zeros((len(texts), M.D_CONCEPT), np.float32)
        for idx in _batches(texts, 256):
            out[idx] = self.net.encode([texts[i] for i in idx])["c"].float().numpy()
        return _norm(out)

    @torch.no_grad()
    def span(self, texts, ranges):
        out = np.zeros((len(texts), M.D_CONCEPT), np.float32)
        for idx in _batches(texts, 256):
            enc = self.net.encode([texts[i] for i in idx])
            hid = enc["hid"].float().numpy()
            ids = enc["ids"].numpy()
            valid = enc["valid"].numpy()
            for row, i in enumerate(idx):
                T = int(valid[row].sum())
                tb = self.tok_bytes[ids[row, :T]]
                ends = np.cumsum(tb)
                starts = ends - tb
                sel = np.zeros(T, bool)
                for a, b in byte_ranges(texts[i], ranges[i]):
                    sel |= (starts < b) & (ends > a)
                h = hid[row, :T][sel].mean(0) if sel.any() else hid[row, :T].mean(0)
                out[i] = self.W @ h + self.b  # the model's own output projection (sentence head)
        return _norm(out)


class DeviceExportEncoder:
    """Arm 4 deployed artefacts (``weights/platonic-hnet.bin`` + ``engram.bin``) through the
    numpy integer reference ``model.PlhnIntNp`` (sentence vectors only)."""

    kind = "device"

    def __init__(self):
        mb = (MODELS_DIR / "platonic-hnet.bin").read_bytes()
        eb = (MODELS_DIR / "engram.bin").read_bytes()
        self.name = "Arm4-device-export"
        self.ref = M.PlhnIntNp(mb, eb, bigram=True)
        self.sha = {"platonic-hnet.bin": hashlib.sha256(mb).hexdigest(), "engram.bin": hashlib.sha256(eb).hexdigest()}
        self.footprint = {"plhn_bytes": len(mb), "engram_bytes": len(eb), "deploy_bytes": len(mb) + len(eb),
                          "precision": "int8 (exported device binary)", "sha256": self.sha}

    def sent_int(self, texts):
        return np.array([self.ref.forward(t) for t in texts], np.int64)


class RefEncoder:
    """100M+ reference embedding model: sentence vectors via sentence-transformers (model's own pooling),
    target-word vectors from the transformer's last hidden states over the target's tokens."""

    kind = "ref"

    def __init__(self, name: str, repo: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.name, self.repo, self.device = name, repo, device
        kw = {"dtype": torch.float16} if device == "mps" and "Qwen" in repo else {}
        self.st = SentenceTransformer(repo, device=device, model_kwargs=kw)
        self.st.max_seq_length = 256
        self.tok = self.st.tokenizer
        self.hf = self.st[0].auto_model
        n = sum(p.numel() for p in self.hf.parameters())
        self.footprint = {"params": n, "deploy_bytes_fp16": 2 * n, "deploy_bytes_fp32": 4 * n,
                          "precision": str(next(self.hf.parameters()).dtype)}

    def sent(self, texts):
        bs = 32 if "Qwen" in self.repo else 128
        v = self.st.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
        return _norm(v)

    @torch.no_grad()
    def span(self, texts, ranges):
        out = np.zeros((len(texts), self.hf.config.hidden_size), np.float32)
        old_side = self.tok.padding_side
        self.tok.padding_side = "right"
        try:
            for idx in _batches(texts, 32):
                batch = [texts[i] for i in idx]
                enc = self.tok(batch, return_offsets_mapping=True, padding=True, truncation=True, max_length=256,
                               return_tensors="pt")
                offs = enc.pop("offset_mapping").numpy()
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hid = self.hf(**enc).last_hidden_state.float().cpu().numpy()
                am = enc["attention_mask"].cpu().numpy().astype(bool)
                for row, i in enumerate(idx):
                    s, e = offs[row, :, 0], offs[row, :, 1]
                    real = am[row] & (e > s)
                    sel = np.zeros(len(s), bool)
                    for a, b in ranges[i]:
                        sel |= real & (s < b) & (e > a)
                    out[i] = hid[row][sel].mean(0) if sel.any() else hid[row][am[row]].mean(0)
        finally:
            self.tok.padding_side = old_side
        return _norm(out)


# ----------------------------------------------------------------------------------------------
# Lexical baselines
# ----------------------------------------------------------------------------------------------

TOKEN = re.compile(r"[a-z0-9']+|[\u3400-\u9fff]")


def toks(t: str):
    return TOKEN.findall(t.lower())


def char3(t: str):
    t = re.sub(r"\s+", " ", t.lower())
    return {t[i:i + 3] for i in range(max(1, len(t) - 2))}


def jaccard(a: str, b: str) -> float:
    A, B = char3(a), char3(b)
    return len(A & B) / max(1, len(A | B))


class BM25:
    def __init__(self, docs, k1: float = 1.2, b: float = 0.75):
        self.docs = [Counter(toks(d)) for d in docs]
        self.len = np.array([sum(d.values()) for d in self.docs], np.float64)
        self.avg = max(1e-9, float(self.len.mean()))
        df = Counter(w for d in self.docs for w in d)
        N = len(self.docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.k1, self.b = k1, b

    def score(self, q: str, j: int) -> float:
        d, s = self.docs[j], 0.0
        norm = self.k1 * (1 - self.b + self.b * self.len[j] / self.avg)
        for w in set(toks(q)):
            f = d.get(w, 0)
            if f:
                s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (f + norm)
        return s

    def matrix(self, queries):
        inv = {}
        for j, d in enumerate(self.docs):
            for w in d:
                inv.setdefault(w, []).append(j)
        S = np.zeros((len(queries), len(self.docs)), np.float32)
        for i, q in enumerate(queries):
            for w in set(toks(q)):
                for j in inv.get(w, ()):
                    f = self.docs[j][w]
                    norm = self.k1 * (1 - self.b + self.b * self.len[j] / self.avg)
                    S[i, j] += self.idf[w] * f * (self.k1 + 1) / (f + norm)
        return S


def pair_bm25(a_list, b_list):
    """Symmetric BM25: mean of BM25(a -> b) over the b corpus and BM25(b -> a) over the a corpus."""
    ba, bb = BM25(a_list), BM25(b_list)
    return np.array([0.5 * (bb.score(a, i) + ba.score(b, i)) for i, (a, b) in enumerate(zip(a_list, b_list))])


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------

def auc(scores, y) -> float:
    y = np.asarray(y).astype(bool)
    r = rankdata(scores)
    npos, nneg = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))


def best_threshold(scores, y):
    s = np.asarray(scores, np.float64)
    y = np.asarray(y).astype(bool)
    order = np.argsort(s)
    ss, yy = s[order], y[order]
    # threshold between ss[k-1] and ss[k]: predict positive for index >= k
    pos_right = np.concatenate([[0], np.cumsum(yy[::-1])])[::-1]  # positives at >= k
    neg_left = np.concatenate([[0], np.cumsum(~yy)])  # negatives at < k
    acc = (pos_right + neg_left) / len(s)
    k = int(np.argmax(acc))
    if k == 0:
        thr = ss[0] - 1e-6
    elif k == len(s):
        thr = ss[-1] + 1e-6
    else:
        thr = 0.5 * (ss[k - 1] + ss[k])
    return float(thr), float(acc[k])


def binary_metrics(dev_s, dev_y, test_s, test_y):
    thr, dev_acc = best_threshold(dev_s, dev_y)
    ty = np.asarray(test_y).astype(bool)
    ts = np.asarray(test_s, np.float64)
    pred = ts > thr
    rho = spearmanr(ts, ty.astype(float)).statistic
    return {"acc": float((pred == ty).mean()), "dev_acc": dev_acc, "threshold": thr, "auc": auc(ts, ty),
            "spearman": float(rho) if np.isfinite(rho) else 0.0,
            "cos_sep": float(ts[ty].mean() - ts[~ty].mean()), "n_test": int(len(ty)), "n_dev": int(len(dev_y)),
            "test_pos_rate": float(ty.mean())}


def sts_metrics(test_s, test_y):
    ts = np.asarray(test_s, np.float64)
    rho = spearmanr(ts, test_y).statistic
    r = pearsonr(ts, test_y).statistic
    return {"spearman": float(rho) if np.isfinite(rho) else 0.0, "pearson": float(r) if np.isfinite(r) else 0.0,
            "n_test": int(len(ts))}


def retrieval_metrics(S, k_list=(1, 5, 10)):
    """Rank of the gold candidate; ties with the gold score are counted **pessimistically** (a tied
    candidate ranks ahead of gold), so an all-zero score row (e.g. no shared n-gram) cannot score R@1."""
    gold = np.diag(S)
    rank = (S >= gold[:, None]).sum(1)
    out = {f"R@{k}": float((rank <= k).mean()) for k in k_list}
    out["MRR"] = float((1.0 / rank).mean())
    out["n_queries"], out["n_candidates"] = int(S.shape[0]), int(S.shape[1])
    out["queries_with_gold_tied"] = int(((S == gold[:, None]).sum(1) > 1).sum())
    return out


def cos_rows(A, B):
    return (A * B).sum(1)


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------

class Cached:
    """Embedding cache for one encoder (keyed by text list hash) so reruns skip slow reference models."""

    def __init__(self, enc, use_cache: bool):
        self.enc, self.use = enc, use_cache
        CACHE.mkdir(parents=True, exist_ok=True)

    def _key(self, kind, texts, ranges=None):
        h = hashlib.sha1(json.dumps([kind, texts, ranges], ensure_ascii=False).encode()).hexdigest()[:20]
        return CACHE / f"{self.enc.name}.{kind}.{h}.npy"

    def sent(self, texts):
        p = self._key("sent", texts)
        if self.use and p.exists():
            return np.load(p)
        v = self.enc.sent(texts)
        np.save(p, v)
        return v

    def span(self, texts, ranges):
        r = [[list(x) for x in rr] for rr in ranges]
        p = self._key("span", texts, r)
        if self.use and p.exists():
            return np.load(p)
        v = self.enc.span(texts, ranges)
        np.save(p, v)
        return v


def _sub(n: int, cap: int | None, seed: int = 0):
    """Seeded subsample indices (all when ``cap`` is None or >= n)."""
    if cap is None or cap >= n:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), cap))


def _take_views(v, idx):
    return tuple([x[i] for i in idx] for x in v[:4]) + (v[4][idx],)


def run_model(enc, tasks, log, caps: dict | None = None):
    """All tasks for one embedding model -> {task: metrics}.  ``caps`` (reference models only, wall-clock
    bound): seeded subsamples {wic_dev, wic_test, pair_dev, pair_test} and ``span`` (False = sent only)."""
    caps = caps or {}
    res = {}
    ce = Cached(enc, use_cache=True)
    feats = ("span", "sent") if caps.get("span", True) else ("sent",)
    for name, t in tasks["wic"].items():
        r = {}
        dv = _take_views(t["dev_views"], _sub(len(t["dev_views"][4]), caps.get("wic_dev")))
        tv = _take_views(t["test_views"], _sub(len(t["test_views"][4]), caps.get("wic_test")))
        for feat in feats:
            if feat == "span":
                d = cos_rows(ce.span(dv[0], dv[1]), ce.span(dv[2], dv[3]))
                s = cos_rows(ce.span(tv[0], tv[1]), ce.span(tv[2], tv[3]))
            else:
                d = cos_rows(ce.sent(dv[0]), ce.sent(dv[2]))
                s = cos_rows(ce.sent(tv[0]), ce.sent(tv[2]))
            r[feat] = binary_metrics(d, dv[4], s, tv[4])
        sel = max(feats, key=lambda f: r[f]["dev_acc"])
        r["dev_selected"] = {"feature": sel, **{k: r[sel][k] for k in ("acc", "auc")}}
        res[name] = r
        log(f"  {enc.name} {name}: " + " | ".join(f"{f} acc {r[f]['acc']:.3f} auc {r[f]['auc']:.3f}" for f in feats))
    tat = tasks["tatoeba"]
    Z, E = ce.sent(tat["zh"]), ce.sent(tat["en"])
    S = Z @ E.T
    res["tatoeba_cmn_eng"] = {"zh2en": retrieval_metrics(S), "en2zh": retrieval_metrics(S.T)}
    log(f"  {enc.name} tatoeba: zh->en R@1 {res['tatoeba_cmn_eng']['zh2en']['R@1']:.3f} "
        f"en->zh R@1 {res['tatoeba_cmn_eng']['en2zh']['R@1']:.3f}")
    for name, t in tasks["pairs"].items():
        test = t["test"] if t["kind"] == "sts" else [t["test"][i] for i in _sub(len(t["test"]), caps.get("pair_test"))]
        ts = cos_rows(ce.sent([a for a, _, _ in test]), ce.sent([b for _, b, _ in test]))
        ty = np.array([c for _, _, c in test])
        if t["kind"] == "sts":
            res[name] = sts_metrics(ts, ty)
        else:
            dev = [t["dev"][i] for i in _sub(len(t["dev"]), caps.get("pair_dev"))]
            ds = cos_rows(ce.sent([a for a, _, _ in dev]), ce.sent([b for _, b, _ in dev]))
            res[name] = binary_metrics(ds, np.array([c for _, _, c in dev]), ts, ty)
        log(f"  {enc.name} {name}: spearman {res[name]['spearman']:.3f}")
    if caps:
        res["caps"] = caps
    return res


def run_lexical(tasks, log):
    out = {"random": {}, "char3_jaccard": {}, "bm25": {}}
    rng = np.random.default_rng(0)
    for name, t in tasks["wic"].items():
        dv, tv = t["dev_views"], t["test_views"]
        maj = int(np.mean(dv[4]) >= 0.5)
        out["random"][name] = {"random": binary_metrics(rng.random(len(dv[4])), dv[4], rng.random(len(tv[4])), tv[4]),
                               "majority_acc": float((tv[4] == maj).mean()), "majority_class": maj}
        out["char3_jaccard"][name] = {"sent": binary_metrics(
            [jaccard(a, b) for a, b in zip(dv[0], dv[2])], dv[4], [jaccard(a, b) for a, b in zip(tv[0], tv[2])], tv[4])}
        out["bm25"][name] = {"sent": binary_metrics(pair_bm25(dv[0], dv[2]), dv[4], pair_bm25(tv[0], tv[2]), tv[4])}
    tat = tasks["tatoeba"]
    n = len(tat["zh"])
    R = rng.random((n, n)).astype(np.float32)
    out["random"]["tatoeba_cmn_eng"] = {"zh2en": retrieval_metrics(R), "en2zh": retrieval_metrics(R.T)}
    zs, es = [char3(a) for a in tat["zh"]], [char3(b) for b in tat["en"]]
    J = np.array([[len(a & b) / max(1, len(a | b)) for b in es] for a in zs], np.float32)
    out["char3_jaccard"]["tatoeba_cmn_eng"] = {"zh2en": retrieval_metrics(J), "en2zh": retrieval_metrics(J.T)}
    out["bm25"]["tatoeba_cmn_eng"] = {"zh2en": retrieval_metrics(BM25(tat["en"]).matrix(tat["zh"])),
                                      "en2zh": retrieval_metrics(BM25(tat["zh"]).matrix(tat["en"]))}
    for name, t in tasks["pairs"].items():
        ta, tb, ty = [a for a, _, _ in t["test"]], [b for _, b, _ in t["test"]], np.array([c for _, _, c in t["test"]])
        da, db, dy = [a for a, _, _ in t["dev"]], [b for _, b, _ in t["dev"]], np.array([c for _, _, c in t["dev"]])
        for base, fn in (("char3_jaccard", lambda A, B: np.array([jaccard(a, b) for a, b in zip(A, B)])),
                         ("bm25", pair_bm25),
                         ("random", lambda A, B: rng.random(len(A)))):
            if t["kind"] == "sts":
                out[base][name] = sts_metrics(fn(ta, tb), ty)
            else:
                out[base][name] = binary_metrics(fn(da, db), dy, fn(ta, tb), ty)
    log("  lexical baselines done")
    return out


def device_parity(enc: HNetEncoder, dev_vecs: np.ndarray, texts):
    """Bit-exact check of an encoder's torch int8 sentence vectors against the exported device binary."""
    a = np.array(enc.net.encode_int(texts), np.int64)
    same = (a == dev_vecs).all(1)
    cos = (_norm(a) * _norm(dev_vecs)).sum(1)
    return {"texts": len(texts), "bit_exact": int(same.sum()), "min_cosine_to_device": float(cos.min()),
            "mean_cosine_to_device": float(cos.mean()),
            "mismatch_examples": [texts[i] for i in np.where(~same)[0][:5]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reference", action="store_true", help="skip the 100M+ reference models")
    ap.add_argument("--ref-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(msg):
        print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)

    # ---- data
    wic = {"wic_en": load_wic_en(), "xlwic_zh": load_xlwic_zh(), **load_mclwic()}
    for t in wic.values():
        t["dev_views"], t["test_views"] = wic_views(t["dev"]), wic_views(t["test"])
        t["windowed"] = {s: int(sum(len(x["s1"].encode()) > WINDOW_BYTES or len(x["s2"].encode()) > WINDOW_BYTES
                                    for x in t[s])) for s in ("dev", "test")}
    tasks = {"wic": wic, "tatoeba": load_tatoeba(), "pairs": load_pair_tasks()}
    data_info = {
        "sources": DATASETS,
        "wic": {k: {"n_dev": len(v["dev"]), "n_test": len(v["test"]), "test_pos_rate": float(np.mean(v["test_views"][4])),
                    "pairs_windowed_to_256B": v["windowed"], **({"dev_note": v["dev_note"]} if "dev_note" in v else {})}
                for k, v in wic.items()},
        "tatoeba_cmn_eng": {"pairs": len(tasks["tatoeba"]["zh"]),
                            "laser_vs_mteb_identical_pairs": tasks["tatoeba"]["laser_vs_mteb_identical_pairs"]},
        "pairs": {k: {"kind": v["kind"], "n_dev_used": len(v["dev"]), "n_dev_total": v["dev_total"],
                      "n_test": len(v["test"]), "dropped_empty": v["dropped_empty"]} for k, v in tasks["pairs"].items()},
    }
    log(f"data: {json.dumps(data_info['wic'])}")

    # ---- models
    encoders = [
        BPEEncoder(V1_RUN / "arm0_bpe_baseline.pt"),
        HNetEncoder("E0-v1", CSPW_RUN / "arm0_v1_base.pt"),
        HNetEncoder("E1-bigram", CSPW_RUN / "arm1_arch_bigram_gqa.pt"),
        HNetEncoder("E3-qwenKD", CSPW_RUN / "arm3_qwen_kd.pt"),
        HNetEncoder("Arm4-device", CSPW_RUN / "arm4_full_cspw.pt", deployed_engram=MODELS_DIR / "engram.bin"),
    ]
    e4 = torch.load(CSPW_RUN / "arm4_full_cspw.pt", map_location="cpu", weights_only=False)["encoder"]
    e3sd = torch.load(CSPW_RUN / "arm3_qwen_kd.pt", map_location="cpu", weights_only=False)["encoder"]
    arm4_same = all(torch.equal(e4[k], e3sd[k]) for k in e4 if k != "engram_hash")

    results = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "script": str(Path(__file__).resolve()),
                        "torch": torch.__version__, "window_bytes": WINDOW_BYTES, "dev_cap_bq_atec": DEV_CAP},
               "data": data_info, "models": {}, "metrics": {}}
    for enc in encoders:
        results["models"][enc.name] = {"ckpt": enc.ckpt, **enc.footprint}

    # ---- Arm 4: the deployed binaries vs the torch graphs, on every Tatoeba + STS-B test text
    dev = DeviceExportEncoder()
    par_texts = tasks["tatoeba"]["zh"] + tasks["tatoeba"]["en"]
    for k in ("stsb_en", "stsb_zh"):
        par_texts += [a for a, _, _ in tasks["pairs"][k]["test"]] + [b for _, b, _ in tasks["pairs"][k]["test"]]
    log(f"device parity on {len(par_texts)} texts ...")
    dpath = CACHE / f"device_parity_{hashlib.sha256(json.dumps(par_texts, ensure_ascii=False).encode()).hexdigest()[:16]}.npy"
    if dpath.exists():
        dvec = np.load(dpath)
    else:
        dvec = dev.sent_int(par_texts)
        np.save(dpath, dvec)
    by_name = {e.name: e for e in encoders}
    results["models"]["Arm4-device"].update({
        "device_files": dev.footprint, "encoder_weights_identical_to_E3_checkpoint": arm4_same,
        "parity_Arm4-device_torch_vs_device_binary": device_parity(by_name["Arm4-device"], dvec, par_texts),
        "parity_E3-training-graph_vs_device_binary": device_parity(by_name["E3-qwenKD"], dvec, par_texts),
        "note": "Arm 4 = E3 encoder weights + the deployed engram.bin, whose span_hash column (unigram gloss "
                "spans + bigrams) also gates the bigram-span Engram on the device; the CS-PW story head "
                "(cspw.bin, story-graph.idx) consumes scene vectors and has no sentence-pair / WiC interface."})
    log(f"parity: {json.dumps({k: v for k, v in results['models']['Arm4-device'].items() if k.startswith('parity')}, ensure_ascii=False)}")

    out = RESULTS / "wsd_xling_results.json"

    def save():
        results["meta"]["wall_sec"] = round(time.time() - t_start, 1)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        os.replace(tmp, out)
        write_report(results)

    log("lexical baselines ...")
    results["metrics"].update(run_lexical(tasks, log))
    for enc in encoders:
        log(f"model {enc.name} ...")
        results["metrics"][enc.name] = run_model(enc, tasks, log)
    save()

    # ---- reference models
    if not args.no_reference:
        for name, repo in REF_MODELS.items():
            try:
                log(f"reference {name} on {args.ref_device} ...")
                enc = RefEncoder(name, repo, args.ref_device)
                results["models"][name] = {"repo": repo, **enc.footprint}
                results["metrics"][name] = run_model(enc, tasks, log, REF_CAPS.get(name))
                del enc
                if args.ref_device == "mps":
                    torch.mps.empty_cache()
            except Exception as exc:  # recorded, never silently dropped
                results["models"][name] = {"repo": repo, "error": repr(exc)}
                log(f"reference {name} FAILED: {exc!r}")
            save()
    log(f"wrote {out} and {RESULTS / 'wsd_xling_report.md'}")


# ----------------------------------------------------------------------------------------------
# Markdown report (every number is read from the results JSON)
# ----------------------------------------------------------------------------------------------

WIC_TASKS = ["wic_en", "xlwic_zh", "mclwic_en_en", "mclwic_zh_zh", "mclwic_en_zh"]
PAIR_TASKS = ["pawsx_en", "pawsx_zh", "bq_zh", "atec_zh"]
STS_TASKS = ["stsb_en", "stsb_zh"]
OURS = ["E0-v1", "E1-bigram", "E3-qwenKD"]
OURS_ALL = OURS + ["Arm4-device"]
LEXICAL = ["random", "char3_jaccard", "bm25"]


def _ci_acc(p, n):
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / max(1, n))


def _ci_rho(r, n):
    """95% Fisher-z interval half-width (approx., symmetric in rho space)."""
    if n < 4:
        return float("nan")
    z = math.atanh(max(-0.999999, min(0.999999, r)))
    se = 1 / math.sqrt(n - 3)
    return 0.5 * (math.tanh(z + 1.96 * se) - math.tanh(z - 1.96 * se))


def _fmt_bytes(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.1f} KB"


def write_report(R: dict) -> None:
    met, mods, data = R["metrics"], R["models"], R["data"]
    refs = [m for m in REF_MODELS if m in met]
    rows_bin = LEXICAL + ["Arm0-BPE"] + OURS_ALL + refs
    L = []
    L.append("# Public WSD / WiC, cross-lingual zh<->en and STS benchmarks — Platonic-Engram-HNet\n")
    L.append(f"Generated {R['meta']['date']} by `external_benchmarks/run_wsd_xling_bench.py` "
             f"(wall {R['meta']['wall_sec']} s). Raw numbers: `results/wsd_xling_results.json`. "
             "Every number below is read from that file.\n")
    L.append("## Protocol\n")
    L.append("- All models are **frozen**; similarity is cosine. No model was trained or tuned on any of these sets.\n"
             "- **WiC-style tasks** (WiC, XL-WiC, MCL-WiC): binary same-sense / different-sense. `span` = cosine of the "
             "target-word vectors (mean of the contextual chunk/token states overlapping the target characters); "
             "`sent` = cosine of the whole-context vectors. One accuracy threshold per model/feature is tuned on the "
             "**dev** split only and applied unchanged to test. ROC-AUC is threshold-free (0.5 = chance). Contexts are "
             f"cut to a <= {R['meta']['window_bytes']}-byte UTF-8 window centred on the target (device input limit); "
             "every model sees the same window.\n"
             "- **Tatoeba cmn-eng**: 1,000 aligned pairs; each side retrieves its translation among all 1,000 "
             "sentences of the other language (R@k, MRR; XTREME reports R@1 as accuracy). Candidates tied with "
             "the gold score rank ahead of it (pessimistic), so lexical scorers get no credit for all-zero rows.\n"
             "- **STS-B en / zh**: Spearman rho (x100) of cosine vs. gold 0-5 score. **PAWS-X en/zh, BQ, ATEC**: "
             "Spearman of cosine vs. binary label (C-MTEB protocol), ROC-AUC, mean cosine separation (pos - neg), "
             f"and accuracy at the dev-tuned threshold (BQ/ATEC dev = {R['meta']['dev_cap_bq_atec']}-pair seeded "
             "sample of the validation split).\n"
             "- Our encoders run the exact int8 QAT graph (`encode_int` / int8 chunk states). Arm 0 BPE has no int8 "
             "export and runs FP32. Reference models run FP32 (MiniLM) / FP16 (Qwen3) on Apple MPS, sentence vectors "
             "with their own sentence-transformers pooling, no instruction prompt.\n"
             "- 95% CIs: binomial for accuracy / R@1, Fisher-z for Spearman.\n")

    L.append("## Datasets (exact counts used)\n")
    L.append("| task | source | dev (threshold) | test | metric |")
    L.append("|---|---|---:|---:|---|")
    src = data["sources"]
    for k in WIC_TASKS:
        d = data["wic"][k]
        s = src["mclwic"] if k.startswith("mclwic") else src[k]
        note = f" ({d['dev_note']})" if "dev_note" in d else ""
        L.append(f"| {k} | {s} | {d['n_dev']}{note} | {d['n_test']} (pos rate {d['test_pos_rate']:.3f}; "
                 f"{d['pairs_windowed_to_256B']['test']} pairs windowed) | acc / AUC |")
    t = data["tatoeba_cmn_eng"]
    L.append(f"| tatoeba_cmn_eng | {src['tatoeba_cmn_eng']} | - | {t['pairs']} pairs "
             f"(LASER == MTEB copy on {t['laser_vs_mteb_identical_pairs']}/{t['pairs']}) | R@1/5/10, MRR |")
    for k in STS_TASKS + PAIR_TASKS:
        d = data["pairs"][k]
        L.append(f"| {k} | {src[k]} | {d['n_dev_used']} of {d['n_dev_total']} | {d['n_test']} | "
                 f"{'Spearman' if d['kind'] == 'sts' else 'Spearman / AUC / acc'} |")
    L.append("")

    L.append("## Models and deployed footprint\n")
    L.append("| model | what | params | deployed bytes | precision |")
    L.append("|---|---|---:|---:|---|")
    m = mods["Arm0-BPE"]
    L.append(f"| Arm0-BPE | matched-RAM byte-BPE transformer (`{Path(m['ckpt']).parent.name}/{Path(m['ckpt']).name}`) | "
             f"{m['params_deploy']:,} | {_fmt_bytes(m['deploy_bytes_int8_equiv'])} int8-equiv / "
             f"{_fmt_bytes(m['deploy_bytes_fp32'])} fp32 | {m['precision']} |")
    desc = {"E0-v1": "PLHN v1 + unigram Engram (CS-PW arm0 encoder)", "E1-bigram": "+ Bigram-Span Engram (arm1/arm2)",
            "E3-qwenKD": "E1 + Qwen3 relational KD (arm3), training-graph bigram table",
            "Arm4-device": "Arm 4 Full CS-PW encoder (= E3 weights) with the deployed engram.bin bigram gate"}
    for k in OURS_ALL:
        m = mods[k]
        L.append(f"| {k} | {desc[k]} | {m['params_deploy']:,} | {_fmt_bytes(m['plhn_bytes'])} PLHN + "
                 f"{_fmt_bytes(m['engram_bytes'])} Engram | {m['precision']}; bigram gate: {m['bigram_gate_table']} "
                 f"({m['bigram_table_nonzero']:,} hashes) |" if m["bigram_engram"] else
                 f"| {k} | {desc[k]} | {m['params_deploy']:,} | {_fmt_bytes(m['plhn_bytes'])} PLHN + "
                 f"{_fmt_bytes(m['engram_bytes'])} Engram | {m['precision']}; no bigram |")
    for k in refs:
        m = mods[k]
        if "error" in m:
            L.append(f"| {k} | `{m['repo']}` | - | - | FAILED: {m['error']} |")
        else:
            L.append(f"| {k} | `{m['repo']}` | {m['params']:,} | {_fmt_bytes(m['deploy_bytes_fp16'])} fp16 | {m['precision']} |")
    a4 = mods["Arm4-device"]
    pa, pe = a4["parity_Arm4-device_torch_vs_device_binary"], a4["parity_E3-training-graph_vs_device_binary"]
    L.append(f"\n**Arm 4 / device check** (exported `weights/platonic-hnet.bin` + `engram.bin`, "
             f"sha256 {a4['device_files']['sha256']['platonic-hnet.bin'][:12]}… / "
             f"{a4['device_files']['sha256']['engram.bin'][:12]}…, run through the numpy integer reference "
             f"`model.PlhnIntNp`): encoder weights identical to the E3 checkpoint = "
             f"**{a4['encoder_weights_identical_to_E3_checkpoint']}**. On all {pa['texts']:,} Tatoeba + STS-B test "
             f"texts the `Arm4-device` torch path is bit-exact with the device binary on **{pa['bit_exact']:,}/"
             f"{pa['texts']:,}**; the E3 training graph is bit-exact on only **{pe['bit_exact']:,}/{pe['texts']:,}** "
             f"(mean cosine to device {pe['mean_cosine_to_device']:.4f}, min {pe['min_cosine_to_device']:.4f}). "
             f"{a4['note']}\n")
    for k in refs:
        c = met[k].get("caps")
        if c:
            L.append(f"**{k} ran on seeded subsamples (wall-clock bound; ±CIs use these n):** WiC dev/test ≤ "
                     f"{c['wic_dev']}/{c['wic_test']} pairs per task, PAWS-X/BQ/ATEC dev/test ≤ {c['pair_dev']}/"
                     f"{c['pair_test']} pairs, span feature {'on' if c['span'] else 'off (sent only)'}; STS-B and "
                     f"Tatoeba complete. All other rows use the full test splits.\n")

    # ---- WiC tables
    for feat, title in (("span", "target-word (span) vectors"), ("sent", "whole-context vectors")):
        L.append(f"## Word-in-Context — {title}: test accuracy (dev-tuned threshold) / ROC-AUC\n")
        L.append("| model | " + " | ".join(WIC_TASKS) + " |")
        L.append("|---|" + "---:|" * len(WIC_TASKS))
        for r in rows_bin:
            if r not in met:
                continue
            cells = []
            for k in WIC_TASKS:
                x = met[r].get(k)
                if x is None:
                    cells.append("-")
                    continue
                if r == "random":
                    y = x["random"]
                    cells.append(f"{y['acc']:.3f} / {y['auc']:.3f} (maj {x['majority_acc']:.3f})")
                    continue
                y = x.get(feat) or (x.get("sent") if r in LEXICAL else None)
                if y is None:
                    cells.append("-")
                    continue
                tag = "" if feat in x else " (sent)"
                cells.append(f"{y['acc']:.3f}±{_ci_acc(y['acc'], y['n_test']):.3f} / {y['auc']:.3f}{tag}")
            L.append(f"| {r} | " + " | ".join(cells) + " |")
        L.append("")

    # ---- Tatoeba
    L.append("## Tatoeba cmn<->eng (1,000 pairs, 1,000 candidates)\n")
    L.append("| model | zh->en R@1 | R@5 | R@10 | MRR | en->zh R@1 | R@5 | R@10 | MRR |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_bin:
        x = met.get(r, {}).get("tatoeba_cmn_eng")
        if not x:
            continue
        a, b = x["zh2en"], x["en2zh"]
        L.append(f"| {r} | {a['R@1']:.3f} | {a['R@5']:.3f} | {a['R@10']:.3f} | {a['MRR']:.3f} | "
                 f"{b['R@1']:.3f} | {b['R@5']:.3f} | {b['R@10']:.3f} | {b['MRR']:.3f} |")
    L.append("")

    # ---- STS / pairs
    L.append("## STS-B / PAWS-X / BQ / ATEC — Spearman rho x100 (cosine vs gold)\n")
    cols = STS_TASKS + PAIR_TASKS
    L.append("| model | " + " | ".join(cols) + " |")
    L.append("|---|" + "---:|" * len(cols))
    for r in rows_bin:
        if r not in met:
            continue
        cells = []
        for k in cols:
            y = met[r].get(k)
            cells.append("-" if y is None else f"{100 * y['spearman']:.1f}±{100 * _ci_rho(y['spearman'], y['n_test']):.1f}")
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Binary paraphrase tasks — ROC-AUC / test accuracy at dev threshold / cosine separation\n")
    L.append("| model | " + " | ".join(PAIR_TASKS) + " |")
    L.append("|---|" + "---:|" * len(PAIR_TASKS))
    for r in rows_bin:
        if r not in met:
            continue
        cells = []
        for k in PAIR_TASKS:
            y = met[r][k]
            cells.append(f"{y['auc']:.3f} / {y['acc']:.3f} / {y['cos_sep']:+.3f}")
        L.append(f"| {r} | " + " | ".join(cells) + " |")
    L.append("\nTest positive rates: " + ", ".join(f"{k} {met['char3_jaccard'][k]['test_pos_rate']:.3f}" for k in PAIR_TASKS)
             + " (majority-class accuracy = max(rate, 1-rate)).\n")

    # ---- automatic win / loss
    L.append("## Automatic win / loss vs baselines (primary metric per task)\n")
    L.append("Primary metric: WiC tasks = test accuracy of the dev-selected feature (span vs sent picked on dev); "
             "Tatoeba = mean R@1 of both directions; STS / paraphrase = Spearman. Subject = the **deployed** "
             "`Arm4-device` encoder; `best ours` (picked on test, i.e. optimistic) is shown for reference. "
             "`best lexical` = best of random / char-3gram Jaccard / BM25 on test (strongest baseline). A difference "
             "smaller than the larger 95% CI half-width is marked `tie`.\n")
    L.append("| task | Arm4-device | best ours (test-picked) | vs best lexical | vs Arm0-BPE | vs MiniLM-L12 | vs Qwen3-0.6B |")
    L.append("|---|---:|---|---|---|---|---|")

    def primary(r, k):
        x = met.get(r, {}).get(k)
        if x is None:
            return None, None
        if k in WIC_TASKS:
            if r == "random":
                y = x["random"]
            elif r in LEXICAL:
                y = x["sent"]
            else:
                y = x[x["dev_selected"]["feature"]]
            return y["acc"], _ci_acc(y["acc"], y["n_test"])
        if k == "tatoeba_cmn_eng":
            v = 0.5 * (x["zh2en"]["R@1"] + x["en2zh"]["R@1"])
            return v, _ci_acc(v, 2 * x["zh2en"]["n_queries"])
        return x["spearman"], _ci_rho(x["spearman"], x["n_test"])

    def cmp(a, ca, b, cb):
        if b is None:
            return "-"
        d = a - b
        word = "tie" if abs(d) < max(ca, cb) else ("WIN" if d > 0 else "loss")
        return f"{word} ({a:.3f} vs {b:.3f})"

    for k in WIC_TASKS + ["tatoeba_cmn_eng"] + STS_TASKS + PAIR_TASKS:
        ao, ac = primary("Arm4-device", k)
        ours = [(primary(r, k), r) for r in OURS_ALL]
        (bo, _), br = max(ours, key=lambda z: z[0][0])
        lex = [(primary(r, k), r) for r in LEXICAL]
        (bl, blc), blr = max(lex, key=lambda z: z[0][0])
        bpe, bpec = primary("Arm0-BPE", k)
        mini, minic = primary("MiniLM-L12-multilingual", k)
        qw, qwc = primary("Qwen3-Embedding-0.6B", k)
        L.append(f"| {k} | {ao:.3f} | {br} {bo:.3f} | {cmp(ao, ac, bl, blc)} [{blr}] | {cmp(ao, ac, bpe, bpec)} | "
                 f"{cmp(ao, ac, mini, minic)} | {cmp(ao, ac, qw, qwc)} |")
    L.append("")
    L.append("### Ablation deltas between our encoders (primary metric)\n")
    L.append("| task | E0-v1 | E1-bigram | E3-qwenKD | Arm4-device | E1-E0 | E3-E1 | Arm4-E3 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in WIC_TASKS + ["tatoeba_cmn_eng"] + STS_TASKS + PAIR_TASKS:
        v = [primary(r, k)[0] for r in OURS_ALL]
        L.append(f"| {k} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[3]:.3f} | {v[1] - v[0]:+.3f} | "
                 f"{v[2] - v[1]:+.3f} | {v[3] - v[2]:+.3f} |")
    L.append("")
    (RESULTS / "wsd_xling_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        write_report(json.loads((RESULTS / "wsd_xling_results.json").read_text()))
    else:
        main()
