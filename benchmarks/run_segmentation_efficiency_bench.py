#!/usr/bin/env python3
"""Public word-segmentation, byte-modelling and hardware-efficiency benchmark for Platonic-Engram-HNet.

Every number written by this script is computed here from downloaded public data and the trained
checkpoints; nothing is copied from a paper or extrapolated.

Stages (``--stages``, default all; each stage caches its output under ``results/_segeff_parts/``):

* ``seg``  SIGHAN Bakeoff 2005 PKU + MSR test sets (gold) and UD_Chinese-GSDSimp test (gold tokens).
           Every int8 H-Net router (exported PLHN bytes, segmented by the compiled C reference kernel,
           checked byte-for-byte against the numpy integer reference) vs heuristic chunkers, the
           matched-RAM byte-BPE Arm 0, the tokenizers of the edge/mobile reference models and two
           supervised references (SIGHAN maximum-matching baseline with the training lexicon, jieba).
           Word P/R/F1 (+ the official SIGHAN ``score`` perl script on PKU/MSR) and boundary P/R/F1.
* ``bpb``  Bits per byte on 256-byte windows of WikiText-2 (test), the SIGHAN PKU/MSR test text,
           UD GSDSimp test text and Tatoeba cmn-eng (bilingual) for Arm 0 BPE vs every H-Net
           encoder (FP32 and the int8 QAT encoder), LFM2-350M as a large-LM reference, and
           gzip/bz2/xz.
* ``eff``  C int8 kernel timing (segment-only, full encode(), CS-PW weave), exact MAC counts, flash
           and RAM accounting for our models; torch FP32 single-thread latency + FlopCounter MACs +
           process RSS for Arm 0 and the external models (MiniLM-L6-v2, BGE-M3, LFM2-350M,
           Qwen3-Embedding-0.6B) on the same 200 queries.
* ``report`` merge -> ``results/segmentation_efficiency_results.json`` + ``..._report.md``.

Run with the shared venv:  external_benchmarks/.venv/bin/python external_benchmarks/run_segmentation_efficiency_bench.py
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import lzma
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

EB = Path(__file__).resolve().parent
ROOT = EB.parent
DATA = EB / "data"
RES = EB / "results"
CACHE = EB / "cache"
PARTS = RES / "_segeff_parts"
MODELS = ROOT / "weights"
sys.path.insert(0, str(ROOT / "src"))

import model as M  # noqa: E402

# ----------------------------------------------------------------------------------------------
# Models under test
# ----------------------------------------------------------------------------------------------

# name -> (checkpoint, router supervision, int8 path, description)
ROUTERS = {
    "v1u_arm1_hnet": ("runs/v1-10k/arm1_hnet_shortconv_ouro.pt", "unsupervised (H-Net ratio loss only)",
                      "PTQ", "v1 H-Net+ShortConv+Ouro, no Engram (10k run, before boundary supervision)"),
    "v1u_arm3_engram": ("runs/v1-10k/arm3_platonic_engram_full.pt", "unsupervised (H-Net ratio loss only)",
                        "QAT", "v1 Platonic-Engram full (10k run, before boundary supervision)"),
    "v2s_arm1_hnet": ("runs/v1-final/arm1_hnet_shortconv_ouro.pt", "jieba word-boundary hinge",
                      "PTQ", "H-Net+ShortConv+Ouro, no Engram (final run)"),
    "v2s_E0": ("runs/v1-final/arm3_platonic_engram_full.pt", "jieba word-boundary hinge", "QAT",
               "E0 = final arm3 Platonic-Engram (CS-PW arm0_v1_base encoder)"),
    "v2s_E1": ("runs/cspw-v2/arm1_arch_bigram_gqa.pt", "jieba word-boundary hinge", "QAT",
               "E1 = E0 + Bigram-Span Engram fine-tune (CS-PW arm1/arm2 encoder)"),
    "v2s_E3": ("runs/cspw-v2/arm4_full_cspw.pt", "jieba word-boundary hinge", "QAT",
               "E3 = E1 + Qwen relational KD (CS-PW arm3/arm4 encoder; deployed platonic-hnet.bin)"),
}
SHARED_ENCODER_CHECK = {"v2s_E1": "runs/cspw-v2/arm2_scale_d192.pt", "v2s_E3": "runs/cspw-v2/arm3_qwen_kd.pt"}
BPE_CKPT = "runs/v1-final/arm0_bpe_baseline.pt"

EXTERNAL = {
    "MiniLM-L6-v2": dict(repo="sentence-transformers/all-MiniLM-L6-v2", pool="mean", lm=False),
    "BGE-M3": dict(repo="BAAI/bge-m3", pool="cls", lm=False),
    "LFM2-350M": dict(repo="LiquidAI/LFM2-350M", pool="mean", lm=True),
    "Qwen3-Embedding-0.6B": dict(repo="Qwen/Qwen3-Embedding-0.6B", pool="last", lm=False),
}

N_QUERIES_PER_SOURCE = 100  # efficiency queries: 100 PKU test sentences + 100 WikiText-2 test windows
C_REPS = 7


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_part(name: str, obj) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = PARTS / f".{name}.json.tmp"
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, PARTS / f"{name}.json")


def load_part(name: str):
    p = PARTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------------------------
# Checkpoints -> torch nets and exported PLHN bytes
# ----------------------------------------------------------------------------------------------


def load_hnet(name: str):
    import torch

    ck = torch.load(ROOT / ROUTERS[name][0], map_location="cpu", weights_only=False)
    state = ck["encoder"] if "encoder" in ck else ck["state_dict"]
    has_engram = "engram" in state
    bigram = bool(ck.get("encoder_cfg", {}).get("use_bigram", False))
    net = M.PlatonicEngramHNet(use_engram=has_engram, use_bigram=bigram)
    missing, unexpected = net.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) <= {"engram_hash"}, (name, missing, unexpected)
    if net.use_bigram:
        # the bigram-span gate reads the exported hash table; use the deployed engram.bin (same corpus table
        # train_cspw installs for E1 and E3 via set_engram_hashes)
        ref = M.PlhnIntNp((MODELS / "platonic-hnet.bin").read_bytes(), (MODELS / "engram.bin").read_bytes())
        net.set_engram_hashes(ref.engram_hash)
    net.eval()
    return net, state


def plhn_blob(net) -> bytes:
    import export_int8 as X

    tensors, scales = X.quantize_model(net)
    return X.pack_plhn(tensors, scales)


def load_bpe():
    import torch

    ck = torch.load(ROOT / BPE_CKPT, map_location="cpu", weights_only=False)
    bpe = M.ByteBPE(ck["bpe_merges"])
    net = M.BPETransformerBaseline(bpe)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


# ----------------------------------------------------------------------------------------------
# C kernel
# ----------------------------------------------------------------------------------------------


def build_kernel(opt: str = "-O2") -> Path:
    import verify_c_parity as V

    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "plhn_cspw_ref.c").write_text(V.C_SOURCE)
    exe = CACHE / f"bench_kernel{opt.replace('-', '_')}"
    subprocess.run(["cc", opt, "-std=c11", "-Wall", "-Wno-unused-function", "-ffp-contract=off", "-I", str(CACHE),
                    "-o", str(exe), str(EB / "bench_kernel.c"), "-lm"], check=True)
    return exe


def c_segment(exe: Path, blob_path: Path, datas: list[bytes]) -> list[list[int]]:
    assert all(b"\n" not in d for d in datas)
    feed = b"".join(d + b"\n" for d in datas)
    out = subprocess.run([str(exe), "seg", str(blob_path)], input=feed, capture_output=True, check=True).stdout
    lines = out.decode().split("\n")[:len(datas)]
    return [[int(x) for x in l.split()] if l else [] for l in lines]


# ----------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------


def load_sighan(corpus: str):
    d = DATA / "sighan2005"
    gold = (d / f"{corpus}_test_gold.utf8").read_text(encoding="utf-8").splitlines()
    raw = (d / f"{corpus}_test.utf8").read_text(encoding="utf-8").splitlines()
    items = []
    mismatch = 0
    for g, r in zip(gold, raw):
        words = g.split()
        text = "".join(words)
        mismatch += text != "".join(r.split())
        items.append({"text": text, "words": words})
    train_words = set((d / f"{corpus}_training_words.utf8").read_text(encoding="utf-8").split())
    train_raw = ["".join(l.split()) for l in (d / f"{corpus}_training.utf8").read_text(encoding="utf-8").splitlines()]
    meta = {"source": "SIGHAN Bakeoff 2005 icwb2-data (github.com/yuikns/icwb2-data @873a51fa)",
            "test_lines": len(gold), "raw_vs_gold_text_mismatch_lines": mismatch,
            "input_text": "gold words joined (identical to testing/*.utf8 except the mismatch lines)",
            "train_lexicon_words": len(train_words)}
    return items, train_words, train_raw, meta


def parse_conllu(path: Path):
    sents, forms, spaces = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if forms:
                sents.append((forms, spaces))
            forms, spaces = [], []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if "-" in cols[0] or "." in cols[0]:
            continue
        forms.append(cols[1])
        spaces.append("SpaceAfter=No" not in cols[9])
    if forms:
        sents.append((forms, spaces))
    return sents


def load_ud():
    d = DATA / "ud_zh_gsdsimp"
    items = []
    for forms, spaces in parse_conllu(d / "zh_gsdsimp-ud-test.conllu"):
        text, starts = "", []
        for f, sp in zip(forms, spaces):
            starts.append(len(text))
            text += f + (" " if sp else "")
        items.append({"text": text.rstrip(" "), "words": forms, "starts": starts})
    train = parse_conllu(d / "zh_gsdsimp-ud-train.conllu")
    train_words = {f for forms, _ in train for f in forms}
    train_raw = ["".join(forms) for forms, _ in train]
    meta = {"source": "UD_Chinese-GSDSimp (github.com/UniversalDependencies/UD_Chinese-GSDSimp master) test split",
            "test_sentences": len(items), "train_lexicon_words": len(train_words)}
    return items, train_words, train_raw, meta


def gold_starts(item) -> list[int]:
    if "starts" in item:
        return item["starts"]
    out, p = [], 0
    for w in item["words"]:
        out.append(p)
        p += len(w)
    return out


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------


def nonspace_words(text: str, starts) -> list[tuple[int, int]]:
    """Predicted/gold segmentation -> word spans over the non-whitespace characters (whitespace is
    never part of a word and always separates words)."""
    S = set(starts)
    spans, k, cur = [], 0, None
    for i, ch in enumerate(text):
        if ch.isspace():
            if cur is not None:
                spans.append((cur, k))
            cur = None
            continue
        if cur is None:
            cur = k
        elif i in S:
            spans.append((cur, k))
            cur = k
        k += 1
    if cur is not None:
        spans.append((cur, k))
    return spans


def score(items, preds, train_words) -> dict:
    tp_w = n_pred = n_gold = 0
    tp_b = n_pb = n_gb = 0
    iv = iv_hit = oov = oov_hit = 0
    pred_chars = 0
    for it, st in zip(items, preds):
        text = it["text"]
        g = nonspace_words(text, gold_starts(it))
        p = nonspace_words(text, st)
        gs, ps = set(g), set(p)
        tp_w += len(gs & ps)
        n_pred += len(p)
        n_gold += len(g)
        pred_chars += sum(e - s for s, e in p)
        gb, pb = {s for s, _ in g[1:]}, {s for s, _ in p[1:]}
        tp_b += len(gb & pb)
        n_pb += len(pb)
        n_gb += len(gb)
        flat = re.sub(r"\s", "", text)
        for s, e in g:
            w = flat[s:e]
            if w in train_words:
                iv += 1
                iv_hit += (s, e) in ps
            else:
                oov += 1
                oov_hit += (s, e) in ps

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    wp, wr = tp_w / max(1, n_pred), tp_w / max(1, n_gold)
    bp, br = tp_b / max(1, n_pb), tp_b / max(1, n_gb)
    return {"word_P": wp, "word_R": wr, "word_F1": f1(wp, wr), "boundary_P": bp, "boundary_R": br,
            "boundary_F1": f1(bp, br), "IV_recall": iv_hit / max(1, iv), "OOV_recall": oov_hit / max(1, oov),
            "OOV_rate": oov / max(1, iv + oov), "gold_words": n_gold, "pred_words": n_pred,
            "chars_per_pred_word": pred_chars / max(1, n_pred)}


@__import__("functools").lru_cache(maxsize=None)
def sighan_scorer() -> Path:
    """Copy of the official SIGHAN ``score`` whose ``$diff`` points at a wrapper around /usr/bin/diff.
    Apple's (FreeBSD) ``diff -y`` prints nothing for identical files, so the unmodified script skips
    the IV/OOV word counting for every perfectly segmented line (P/R/F are unaffected, OOV rate and
    OOV/IV recall are not). The wrapper emits GNU-style common lines for identical inputs."""
    wrap = CACHE / "sideby_diff.sh"
    wrap.write_text('#!/bin/sh\n# $1=-y $2=file1 $3=file2\nif cmp -s "$2" "$3"; then\n'
                    '  awk \'{print $0 "\\t\\t\\t\\t\\t\\t\\t\\t" $0}\' "$2"\nelse\n  /usr/bin/diff "$@"\nfi\n')
    wrap.chmod(0o755)
    src = (DATA / "sighan2005" / "score").read_text(encoding="latin-1")
    assert '$diff = "/usr/bin/diff";' in src
    p = CACHE / "sighan_score_macos"
    p.write_text(src.replace('$diff = "/usr/bin/diff";', f'$diff = "{wrap}";'), encoding="latin-1")
    return p


def official_sighan_score(corpus: str, items, preds, tag: str) -> dict | None:
    d = DATA / "sighan2005"
    out = CACHE / "sighan_out" / f"{corpus}_{tag}.utf8"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for it, st in zip(items, preds):
        text = it["text"]
        words = [re.sub(r"\s", "", text)[s:e] for s, e in nonspace_words(text, st)]
        lines.append("  ".join(words))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        r = subprocess.run(["perl", str(sighan_scorer()), str(d / f"{corpus}_training_words.utf8"),
                            str(d / f"{corpus}_test_gold.utf8"), str(out)], capture_output=True, encoding="utf-8",
                           errors="replace", timeout=3600)  # diff -y columns cut multibyte chars
    except subprocess.TimeoutExpired:
        return None
    vals = {}
    for key, pat in (("R", r"=== TOTAL TRUE WORDS RECALL:\s*([\d.]+)"), ("P", r"=== TOTAL TEST WORDS PRECISION:\s*([\d.]+)"),
                     ("F", r"=== F MEASURE:\s*([\d.]+)"), ("OOV_recall", r"=== OOV Recall Rate:\s*([\d.]+)"),
                     ("IV_recall", r"=== IV Recall Rate:\s*([\d.]+)"), ("OOV_rate", r"=== OOV Rate:\s*([\d.]+)")):
        m = re.search(pat, r.stdout)
        if m:
            vals[key] = float(m.group(1))
    return vals or {"error": (r.stderr or r.stdout)[-400:]}


# ----------------------------------------------------------------------------------------------
# Segmenters
# ----------------------------------------------------------------------------------------------


def byte_to_char_starts(text: str, byte_starts) -> list[int]:
    data = text.encode("utf-8")
    b2c, c = {}, 0
    for i, b in enumerate(data):
        if not M.is_cont(b):
            b2c[i] = c
            c += 1
    return sorted({b2c[b] for b in byte_starts if b in b2c})


def seg_char(text):
    return list(range(len(text)))


def seg_fixed2(text):
    out, run = [], 0
    for i, ch in enumerate(text):
        if ch.isspace():
            run = 0
            continue
        if run % 2 == 0:
            out.append(i)
        run += 1
    return out


class PMISeg:
    """Unsupervised heuristic: split between adjacent characters whose pointwise mutual information
    on the raw (boundary-free) training text is < 0 or that never co-occur."""

    def __init__(self, raw_lines):
        uni, bi = Counter(), Counter()
        for l in raw_lines:
            uni.update(l)
            bi.update(l[i:i + 2] for i in range(len(l) - 1))
        self.uni, self.bi = uni, bi
        self.N1, self.N2 = sum(uni.values()), max(1, sum(bi.values()))

    def __call__(self, text):
        out = [0] if text else []
        for i in range(1, len(text)):
            a, b = text[i - 1], text[i]
            c = self.bi.get(a + b, 0)
            if c == 0 or self.uni[a] == 0 or self.uni[b] == 0:
                out.append(i)
                continue
            pmi = math.log((c / self.N2) / ((self.uni[a] / self.N1) * (self.uni[b] / self.N1)))
            if pmi < 0:
                out.append(i)
        return out


class FMMSeg:
    """SIGHAN bakeoff-style baseline: forward maximum matching with the training-set word list."""

    def __init__(self, words):
        self.words = words
        self.maxlen = min(16, max(len(w) for w in words))

    def __call__(self, text):
        out, i, n = [], 0, len(text)
        while i < n:
            out.append(i)
            for L in range(min(self.maxlen, n - i), 0, -1):
                if L == 1 or text[i:i + L] in self.words:
                    i += L
                    break
        return out


def seg_jieba(text):
    import jieba

    out, p = [], 0
    for w in jieba.lcut(text, HMM=True):
        out.append(p)
        p += len(w)
    return out


def cap_oracle(text, gold):
    """Gold boundaries pushed through the router's exact ``boundary_step`` (12-byte span cap, UTF-8
    safe): the best segmentation any PLHN router can emit."""
    data = text.encode("utf-8")
    gb = set()
    c = 0
    gs = set(gold)
    for i, b in enumerate(data):
        if not M.is_cont(b):
            if c in gs:
                gb.add(i)
            c += 1
    starts, length = [], 0
    for t, b in enumerate(data):
        if M.boundary_step(t, b, length, t in gb):
            starts.append(t)
            length = 0
        length += 1
    return byte_to_char_starts(text, starts)


def tokenizer_segmenter(repo: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)

    def seg(text):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        out = sorted({s for s, e in enc["offset_mapping"] if e > s})
        return out

    return seg, tok


# ----------------------------------------------------------------------------------------------
# Stage: segmentation
# ----------------------------------------------------------------------------------------------


def stage_seg(args):
    import jieba

    jieba.setLogLevel(60)
    exe = build_kernel("-O2")
    datasets = {}
    for c in ("pku", "msr"):
        datasets[f"SIGHAN05-{c.upper()}"] = (c,) + load_sighan(c)
    datasets["UD-GSDSimp"] = ("ud",) + load_ud()

    blobs, router_meta = {}, {}
    (CACHE / "plhn").mkdir(parents=True, exist_ok=True)
    for name, (ck, sup, q, desc) in ROUTERS.items():
        net, _ = load_hnet(name)
        blob = plhn_blob(net)
        p = CACHE / "plhn" / f"{name}.bin"
        p.write_bytes(blob)
        blobs[name] = (p, blob)
        router_meta[name] = {"checkpoint": ck, "router_supervision": sup, "int8": q, "description": desc,
                             "plhn_sha256": sha256(blob), "plhn_bytes": len(blob)}
        if name in SHARED_ENCODER_CHECK:
            import torch

            ck2 = torch.load(ROOT / SHARED_ENCODER_CHECK[name], map_location="cpu", weights_only=False)
            n2 = M.PlatonicEngramHNet(use_engram=True, use_bigram=True)
            n2.load_state_dict(ck2["encoder"], strict=False)
            router_meta[name]["same_plhn_as"] = {SHARED_ENCODER_CHECK[name]: sha256(plhn_blob(n2)) == sha256(blob)}
        log("router", name, router_meta[name]["plhn_sha256"][:12])
    deployed = (MODELS / "platonic-hnet.bin").read_bytes()
    router_meta["v2s_E3"]["equals_deployed_models_platonic_hnet_bin"] = sha256(deployed) == router_meta["v2s_E3"]["plhn_sha256"]

    bpe_net = load_bpe()
    tokenizers = {k: tokenizer_segmenter(v["repo"])[0] for k, v in EXTERNAL.items()}

    out = {"routers": router_meta, "datasets": {}, "results": {}, "c_vs_numpy_parity": {}}
    for dname, (code, items, train_words, train_raw, meta) in datasets.items():
        log("dataset", dname, len(items))
        meta = dict(meta)
        texts = [it["text"] for it in items]
        meta.update({"scored_items": sum(bool(t) for t in texts), "chars": sum(len(t) for t in texts),
                     "utf8_bytes": sum(len(t.encode()) for t in texts),
                     "gold_words_longer_than_4_han": sum(1 for it in items for w in it["words"] if len(w.encode()) > 12)})
        out["datasets"][dname] = meta
        datas = [t.encode("utf-8") for t in texts]
        systems = {}
        # --- int8 H-Net routers through the compiled C kernel; numpy integer reference parity on every line
        for name, (p, blob) in blobs.items():
            t0 = time.time()
            starts_b = c_segment(exe, p, datas)
            c_sec = time.time() - t0
            ref = M.PlhnIntNp(blob)
            mism = 0
            for d, sb in zip(datas, starts_b):
                if d:
                    spans = ref.segment(d, ref.encode_bytes(d))
                    mism += [a for a, _ in spans] != sb
            out["c_vs_numpy_parity"][f"{dname}/{name}"] = {"lines": len(datas), "mismatched_lines": mism}
            systems[f"router:{name}"] = [byte_to_char_starts(t, sb) for t, sb in zip(texts, starts_b)]
            log("  ", name, "C seg", round(c_sec, 2), "s", "numpy mismatches", mism)
        # --- references & baselines
        systems["oracle:gold+12B_cap"] = [cap_oracle(t, gold_starts(it)) for t, it in zip(texts, items)]
        systems["heuristic:char_unigram"] = [seg_char(t) for t in texts]
        systems["heuristic:fixed_2char"] = [seg_fixed2(t) for t in texts]
        pmi = PMISeg(train_raw)
        systems["heuristic:pmi_bigram_unsup"] = [pmi(t) for t in texts]
        bpe_starts = []
        tb = bpe_net.bpe.token_bytes()
        for t, d in zip(texts, datas):
            ids, pos, st = bpe_net.bpe.encode(d), 0, []
            for i in ids:
                st.append(pos)
                pos += tb[i]
            bpe_starts.append(byte_to_char_starts(t, st))
        systems["bpe:arm0_matched_ram"] = bpe_starts
        for k, seg in tokenizers.items():
            systems[f"tokenizer:{k}"] = [seg(t) for t in texts]
        fmm = FMMSeg(train_words)
        systems["supervised:fmm_train_lexicon"] = [fmm(t) for t in texts]
        systems["supervised:jieba_0.42.1"] = [seg_jieba(t) for t in texts]
        res = {}
        official = {}
        if code in ("pku", "msr"):
            sighan_scorer()
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(6) as ex:
                futs = {s: ex.submit(official_sighan_score, code, items, p, s.replace(":", "_")) for s, p in systems.items()}
                official = {s: f.result() for s, f in futs.items()}
        for sname, preds in systems.items():
            r = score(items, preds, train_words)
            if sname in official:
                r["official_sighan_score"] = official[sname]
            res[sname] = r
            log("  ", f"{sname:42s}", f"wordF1 {r['word_F1']:.4f}", f"bndF1 {r['boundary_F1']:.4f}",
                "official", r.get("official_sighan_score"))
        out["results"][dname] = res
        # examples
        ex = []
        for i in [j for j, t in enumerate(texts) if 12 <= len(t) <= 40][:3]:
            t = texts[i]

            def show(st):
                return " | ".join(re.sub(r"\s", "", t)[s:e] for s, e in nonspace_words(t, st))

            ex.append({"gold": show(gold_starts(items[i])), "router:v2s_E3": show(systems["router:v2s_E3"][i]),
                       "router:v1u_arm3_engram": show(systems["router:v1u_arm3_engram"][i]),
                       "bpe:arm0_matched_ram": show(systems["bpe:arm0_matched_ram"][i])})
        out["datasets"][dname]["examples"] = ex
        save_part("seg", out)


# ----------------------------------------------------------------------------------------------
# Stage: bits per byte
# ----------------------------------------------------------------------------------------------


def windows(data: bytes, size: int = M.MAX_BYTES) -> list[bytes]:
    out, i, n = [], 0, len(data)
    while i < n:
        e = min(n, i + size)
        while e < n and M.is_cont(data[e]):
            e -= 1
        out.append(data[i:e])
        i = e
    return out


def bpb_corpora() -> dict:
    import gzip as gz

    import pyarrow.parquet as pq

    wt = pq.read_table(DATA / "wikitext2" / "wikitext-2-raw-v1" / "test-00000-of-00001.parquet").column("text").to_pylist()
    corp = {"WikiText-2-raw test (en)": "".join(wt).encode("utf-8")}
    for c in ("pku", "msr"):
        items = load_sighan(c)[0]
        corp[f"SIGHAN05-{c.upper()} test text (zh news)"] = "".join(it["text"] + "\n" for it in items if it["text"]).encode()
    corp["UD-GSDSimp test text (zh wiki)"] = "".join(it["text"] + "\n" for it in load_ud()[0]).encode()
    rows = [json.loads(l) for l in gz.open(DATA / "tatoeba" / "test" / "cmn-eng.jsonl.gz", "rt", encoding="utf-8")]
    corp["Tatoeba cmn-eng test (bilingual zh/en)"] = "".join(
        r["sentence1"].strip() + "\t" + r["sentence2"].strip() + "\n" for r in rows).encode()
    return corp


def stage_bpb(args):
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    sys.path.insert(0, str(ROOT / "src"))
    import train as TA

    corp = bpb_corpora()
    wins = {k: windows(v) for k, v in corp.items()}
    out = {"corpora": {k: {"bytes": len(v), "windows_256B": len(wins[k]), "sha256": sha256(v)} for k, v in corp.items()},
           "protocol": "independent <=256-byte windows cut on UTF-8 lead bytes (both model kinds see at most 256 bytes); "
                       "H-Net: next byte t>=1 of each window from the concept-bottleneck head (train_ablation.eval_bpc); "
                       "Arm 0 BPE: next-token NLL of tokens 1..T-1 / their bytes (token 0 unscored, as in eval_bpc); "
                       "LFM2-350M reference: BOS + window, all text tokens scored / all window bytes; "
                       "compressors: per-window = each window compressed alone, whole = whole corpus",
           "results": {}}
    texts = {k: [w.decode("utf-8") for w in v] for k, v in wins.items()}
    # compressors
    for cname, fn in (("gzip-9", lambda b: gzip.compress(b, 9)), ("bz2-9", lambda b: bz2.compress(b, 9)),
                      ("xz-9e", lambda b: lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))):
        for k, v in corp.items():
            per = sum(len(fn(w)) for w in wins[k]) * 8 / len(v)
            out["results"].setdefault(f"compressor:{cname} per-window", {})[k] = per
            out["results"].setdefault(f"compressor:{cname} whole-corpus", {})[k] = len(fn(v)) * 8 / len(v)
    # Arm 0 BPE
    bpe = load_bpe()
    out["results"]["bpe:arm0_matched_ram FP32"] = {k: TA.eval_bpc(bpe, t, "bpe") for k, t in texts.items()}
    log("bpb arm0", out["results"]["bpe:arm0_matched_ram FP32"])
    for name in ROUTERS:
        net, _ = load_hnet(name)
        for q in (False, True):
            net.quant = q
            key = f"hnet:{name} {'INT8 encoder (QAT graph)' if q else 'FP32'}"
            out["results"][key] = {k: TA.eval_bpc(net, t, "hnet") for k, t in texts.items()}
            log("bpb", key, out["results"][key])
        net.quant = False
    save_part("bpb", out)
    if not args.skip_lfm2_bpb:
        out["results"]["reference:LFM2-350M FP32 (350M params)"] = lfm2_bpb(texts)
        log("bpb lfm2", out["results"]["reference:LFM2-350M FP32 (350M params)"])
    save_part("bpb", out)


def lfm2_bpb(texts: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = EXTERNAL["LFM2-350M"]["repo"]
    tok = AutoTokenizer.from_pretrained(repo)
    lm = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    res = {}
    for k, ts in texts.items():
        bits = nb = 0.0
        for i in range(0, len(ts), 16):
            batch = ts[i:i + 16]
            # BOS-conditioned (without BOS the model is far off-distribution: measured 2.3 vs 1.3 bpb on a
            # WikiText window); every text token is scored, so the byte count is the whole window
            enc = [[tok.bos_token_id] + tok(t, add_special_tokens=False)["input_ids"] for t in batch]
            T = max(len(e) for e in enc)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.full((len(batch), T), pad, dtype=torch.long)
            am = torch.zeros((len(batch), T), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, :len(e)] = torch.tensor(e)
                am[j, :len(e)] = 1
            with torch.no_grad():
                logits = lm(input_ids=ids, attention_mask=am).logits.float()
            for j, (e, t) in enumerate(zip(enc, batch)):
                n = len(e)
                if n < 2:
                    continue
                nll = F.cross_entropy(logits[j, :n - 1], ids[j, 1:n], reduction="sum") / math.log(2)
                bits += float(nll)
                nb += len(t.encode("utf-8"))
        res[k] = bits / max(1.0, nb)
    return res


# ----------------------------------------------------------------------------------------------
# Stage: efficiency
# ----------------------------------------------------------------------------------------------


def efficiency_queries() -> list[str]:
    import pyarrow.parquet as pq

    pku = [it["text"] for it in load_sighan("pku")[0] if it["text"]][:N_QUERIES_PER_SOURCE]
    pku = [M.text_bytes(t).decode("utf-8", "ignore") for t in pku]
    wt = pq.read_table(DATA / "wikitext2" / "wikitext-2-raw-v1" / "test-00000-of-00001.parquet").column("text").to_pylist()
    wt = [l.strip() for l in wt if len(l.strip()) > 40 and not l.strip().startswith("=")]
    wt = [M.text_bytes(t).decode("utf-8", "ignore") for t in wt][:N_QUERIES_PER_SOURCE]
    return pku + wt


def plhn_macs(n: int, c: int) -> int:
    """Multiply-accumulates of the reference encode() loops for n bytes and c chunks: matvecs, depthwise
    taps, RECIP mean and final sum of squares exact; the 32-MAC router dot is counted on every byte (upper
    bound: the kernel skips it on UTF-8 continuation bytes / capped spans); elementwise gate products excluded."""
    enc_block = 64 * 64 + 64 * 3 + 64 * 64  # gate, depthwise, pointwise
    per_byte = 2 * enc_block + 2 * 64 * 32 + 32  # 2 prelude blocks + router q,k + q.k
    core = 128 * 128 + 128 * 3 + 128 * 128 + 128 * 128  # gate, dw, pw, out
    per_chunk = 64 + 64 * 128 + 2 * core  # RECIP mean, chunk_proj, 2 Ouro loops
    return per_byte * n + per_chunk * c + 128  # final sum of squares


def cspw_macs(M_: int, frames_nonempty: int, phrases_nonempty: int) -> int:
    up, q, kv = 128 * 192, 192 * 192, 2 * 192 * 96
    attn = 4 * sum((a + 1) * 48 * 2 for a in range(M_))
    per = up + q + kv + 192 * 192 + 192 * 128 + 128 + 4 * 128 * 128
    tail = 128 * M_ + 2 * 128 * (M_ - 1) + (frames_nonempty + phrases_nonempty) * 128 + (M_ - 1) * 128
    return M_ * per + attn + tail


def torch_bench(fn, queries, reps: int = 3) -> dict:
    lat = []
    for q in queries:
        best = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(q)
            best.append(time.perf_counter() - t0)
        lat.append(sorted(best)[len(best) // 2])
    return {"median_us": 1e6 * float(np.median(lat)), "mean_us": 1e6 * float(np.mean(lat)),
            "p90_us": 1e6 * float(np.percentile(lat, 90))}


def flop_macs(fn, queries) -> float:
    from torch.utils.flop_counter import FlopCounterMode

    tot = 0
    for q in queries:
        fc = FlopCounterMode(display=False)
        with fc:
            fn(q)
        tot += fc.get_total_flops()
    return tot / 2 / len(queries)


def ext_worker(name: str, qpath: Path, outp: Path):
    """Runs in a fresh process: RSS before/after load and after inference, latency, MACs."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(1)
    queries = json.loads(qpath.read_text())
    cfg = EXTERNAL[name]
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tok = AutoTokenizer.from_pretrained(cfg["repo"])
    model = AutoModel.from_pretrained(cfg["repo"], dtype=torch.float32).eval()
    rss_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    params = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()

    @torch.no_grad()
    def run(q):
        enc = tok(q, return_tensors="pt", truncation=True, max_length=512)
        h = model(**enc).last_hidden_state
        if cfg["pool"] == "cls":
            v = h[:, 0]
        elif cfg["pool"] == "last":
            v = h[:, -1]
        else:
            v = h.mean(1)
        return torch.nn.functional.normalize(v, dim=-1)

    run(queries[0])
    ntok = [len(tok(q)["input_ids"]) for q in queries]
    lat = torch_bench(run, queries[::4], reps=1 if params > 1e8 else 3)  # 50 of the 200 queries (time cap)
    lat["queries_timed"] = len(queries[::4])
    rss_inf = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    macs = flop_macs(run, queries[::20])
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(cfg["repo"], local_files_only=True))
    wfiles = [p for p in snap.rglob("*") if p.suffix in (".safetensors", ".bin") and p.is_file()]
    flash = sum(p.resolve().stat().st_size for p in wfiles)
    outp.write_text(json.dumps({
        "repo": cfg["repo"], "revision": snap.name, "params": params, "embedding_params": emb,
        "weight_files": [p.name for p in wfiles], "flash_bytes_fp_checkpoint": flash,
        "fp32_weight_bytes": 4 * params, "int8_weight_bytes_if_quantized": params,
        "rss_bytes_before_load": rss0, "rss_peak_bytes_after_load": rss_load, "rss_peak_bytes_after_inference": rss_inf,
        "tokens_per_query_mean": float(np.mean(ntok)), "torch_fp32_1thread": lat, "macs_per_query_flopcounter": macs,
        "pooling": cfg["pool"]}))


def stage_eff(args):
    import torch

    torch.set_num_threads(1)
    queries = efficiency_queries()
    qpath = CACHE / "efficiency_queries.json"
    qpath.write_text(json.dumps(queries, ensure_ascii=False))
    out = {"host": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                          text=True).stdout.strip(), "platform": platform.platform(),
                    "python": sys.version.split()[0], "torch": torch.__version__,
                    "cc": subprocess.run(["cc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]},
           "queries": {"n": len(queries), "source": f"{N_QUERIES_PER_SOURCE} SIGHAN05 PKU test sentences + "
                       f"{N_QUERIES_PER_SOURCE} WikiText-2 test paragraphs, each cut to <=256 UTF-8 bytes",
                       "mean_bytes": float(np.mean([len(q.encode()) for q in queries]))},
           "c_kernel": {}, "torch": {}, "external": {}}
    feed = b"".join(q.encode("utf-8") + b"\n" for q in queries)
    deployed = {k: MODELS / k for k in ("platonic-hnet.bin", "engram.bin", "cspw.bin", "booksum-story-graph.idx")}
    out["deployed_files"] = {k: {"bytes": p.stat().st_size, "sha256": sha256(p.read_bytes())} for k, p in deployed.items()}
    import cspw_model as CM

    w = CM.CspwWeights((MODELS / "cspw.bin").read_bytes())
    blob = (MODELS / "cspw.bin").read_bytes()
    fr_ne = sum(blob[CM.FRAME_STRINGS_OFFSET + f * CM.FRAME_BYTES] != 0 for f in range(CM.N_FRAMES))
    ph_ne = sum(blob[CM.PHRASE_STRINGS_OFFSET + f * CM.PHRASE_BYTES] != 0 for f in range(CM.N_PHRASES))
    del w
    for opt in ("-O2",):
        exe = build_kernel(opt)
        configs = {"Arm4 deployed (E3 PLHN + engram.bin + cspw.bin)": (deployed["platonic-hnet.bin"],
                                                                         deployed["engram.bin"], deployed["cspw.bin"])}
        if opt == "-O2":
            for name in ROUTERS:
                configs[f"{name} PLHN only (no Engram)"] = (CACHE / "plhn" / f"{name}.bin", "-", "-")
        for cname, (mp, ep, cp) in configs.items():
            r = subprocess.run([str(exe), "time", str(mp), str(ep), str(cp), str(C_REPS)], input=feed,
                               capture_output=True, check=True)
            rows, weave = [], None
            for line in r.stdout.decode().splitlines():
                if line.startswith("WEAVE"):
                    _, med, mean, calls = line.split()
                    weave = {"median_us": float(med) / 1e3, "mean_us": float(mean) / 1e3, "calls": int(calls)}
                else:
                    n, c, ts, te = map(float, line.split())
                    rows.append((n, c, ts, te))
            a = np.array(rows)
            macs = [plhn_macs(int(n), int(c)) for n, c, _, _ in rows]
            rec = {"flags": f"cc {opt} -std=c11 -ffp-contract=off", "reps_per_query": C_REPS,
                   "mean_bytes": float(a[:, 0].mean()), "mean_chunks": float(a[:, 1].mean()),
                   "bytes_per_chunk": float(a[:, 0].sum() / a[:, 1].sum()),
                   "segment_only_us_median": float(np.median(a[:, 2])) / 1e3,
                   "encode_us_median": float(np.median(a[:, 3])) / 1e3, "encode_us_mean": float(a[:, 3].mean()) / 1e3,
                   "encode_us_p90": float(np.percentile(a[:, 3], 90)) / 1e3,
                   "encode_us_per_byte": float(a[:, 3].sum() / a[:, 0].sum()) / 1e3,
                   "int8_macs_per_query_mean": float(np.mean(macs)),
                   "int8_macs_segment_only_mean": float(np.mean([(2 * (64 * 64 * 2 + 64 * 3) + 2 * 64 * 32 + 32) * int(n)
                                                                  for n, _, _, _ in rows]))}
            if weave:
                rec["cspw_weave_M4"] = weave | {"int8_macs": cspw_macs(4, fr_ne, ph_ne),
                                               "frames_nonempty": int(fr_ne), "phrases_nonempty": int(ph_ne)}
            out["c_kernel"][f"{cname} [{opt}]"] = rec
            log("C", cname, opt, {k: rec[k] for k in ("encode_us_median", "segment_only_us_median", "mean_chunks")},
                rec.get("cspw_weave_M4"))
    # torch FP32 single-thread: our nets + Arm 0
    for name in ROUTERS:
        net, _ = load_hnet(name)

        def run(q, net=net):
            with torch.no_grad():
                return net.encode([q])["c"]

        run(queries[0])
        out["torch"][f"hnet:{name}"] = {"torch_fp32_1thread": torch_bench(run, queries),
                                        "macs_per_query_flopcounter": flop_macs(run, queries[::4]),
                                        "params_total_incl_train_heads": M.count_params(net)}
        log("torch", name, out["torch"][f"hnet:{name}"])
    bpe = load_bpe()

    def run_bpe(q):
        with torch.no_grad():
            return bpe.encode([q])["c"]

    run_bpe(queries[0])
    heads = sum(p.numel() for p in bpe.lm.parameters()) + 2
    out["torch"]["bpe:arm0_matched_ram"] = {
        "torch_fp32_1thread": torch_bench(run_bpe, queries), "macs_per_query_flopcounter": flop_macs(run_bpe, queries[::4]),
        "params_total": M.count_params(bpe), "params_deploy": M.count_params(bpe) - heads,
        "bpe_merges": len(bpe.bpe.merges),
        "tokens_per_query_mean": float(np.mean([len(bpe.bpe.encode(q.encode())) for q in queries]))}
    log("torch bpe", out["torch"]["bpe:arm0_matched_ram"])
    save_part("eff", out)
    # external models, one fresh process each (clean RSS)
    for name in EXTERNAL:
        op = CACHE / f"ext_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.json"
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "--ext-worker", name, str(qpath), str(op)], capture_output=True,
                           text=True)
        if r.returncode != 0:
            out["external"][name] = {"error": r.stderr[-2000:]}
        else:
            out["external"][name] = json.loads(op.read_text())
        log("ext", name, round(time.time() - t0, 1), "s", str(out["external"][name])[:300])
        save_part("eff", out)
    # empty-interpreter baseline RSS for the external process numbers
    r = subprocess.run([sys.executable, "-c", "import resource,torch,transformers;"
                        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)"], capture_output=True, text=True)
    out["external_rss_baseline_bytes_python_torch_transformers_imported"] = int(r.stdout.strip().splitlines()[-1])
    save_part("eff", out)


# ----------------------------------------------------------------------------------------------
# RAM accounting for the on-device streaming firmware (firmware/ConceptEngine.cpp)
# ----------------------------------------------------------------------------------------------

FIRMWARE_SRC = str(ROOT / "firmware" / "ConceptEngine.cpp")


def firmware_ram() -> dict:
    """Activation/state bytes of the firmware streaming encoder and runChronoSlots, summed from the
    arrays declared in ConceptEngine.cpp (lower bound: excludes call frames, spills, alignment)."""
    fwd_state = 2 * (64 + 64) + 2 * (128 + 128) + 4 * 128  # conv1/conv2 BlockState, 2 core BlockState, sums
    run_frame = 32 + 4 * 64 + 64 + 4 * 32 * 2 + 32  # kPrev, pooled, h, qAcc, kAcc, k
    chunk_frame = 64 + 4 * 128 + 128 + 64  # m, acc, x, Engram slot
    block128 = 4 * 128 + 2 * 128 + 128 + 128 + 128  # acc, gs, y, z, o2
    input_bytes = 256
    enc_peak = fwd_state + run_frame + chunk_frame + block128 + input_bytes
    cspw_state = 4 * 192 * 2 + 4 * 96 * 2 + 4 * 128 + 4 * 4 * 128  # h, q, k, v, r, u (heap)
    cspw_frame = 192 + 4 * 192 + 128 + 4 * 4 * 2  # attn, acc, v128, score/weight
    chrono_out = 4 * 128 + 4 + 8 + 8 + 4
    return {"source": FIRMWARE_SRC, "encoder_persistent_state_bytes": fwd_state,
            "encoder_peak_activation_bytes": enc_peak, "cspw_state_heap_bytes": cspw_state,
            "cspw_peak_activation_bytes": cspw_state + cspw_frame + chrono_out,
            "weights_psram_bytes": {"platonic-hnet.bin": 95008, "cspw.bin": 393728},
            "engram_access": "engram.bin stays on SD: 64-byte slot read per chunk (+1 per bigram probe)"}


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------


def pareto(points: dict, cost_keys, q_key) -> list[str]:
    """Names not dominated (<= on every cost, >= quality, strictly better somewhere)."""
    names = [n for n, p in points.items() if p.get(q_key) is not None and all(p.get(c) is not None for c in cost_keys)]
    out = []
    for a in names:
        pa = points[a]
        dom = False
        for b in names:
            if a == b:
                continue
            pb = points[b]
            if all(pb[c] <= pa[c] for c in cost_keys) and pb[q_key] >= pa[q_key] and (
                    any(pb[c] < pa[c] for c in cost_keys) or pb[q_key] > pa[q_key]):
                dom = True
                break
        if not dom:
            out.append(a)
    return out


def stage_report(args):
    seg, bpb, eff = load_part("seg"), load_part("bpb"), load_part("eff")
    fw = firmware_ram()
    summary = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": str(Path(__file__).relative_to(ROOT)),
               "segmentation": seg, "bits_per_byte": bpb, "efficiency": eff, "firmware_ram": fw}
    # ---------------- Pareto table rows
    rows = {}
    if eff and seg:
        dsets = list(seg["results"])

        def segq(sysname):
            vals = [seg["results"][d][sysname]["boundary_F1"] for d in dsets if sysname in seg["results"][d]]
            return float(np.mean(vals)) if len(vals) == len(dsets) else None

        def wordq(sysname):
            vals = [seg["results"][d][sysname]["word_F1"] for d in dsets if sysname in seg["results"][d]]
            return float(np.mean(vals)) if len(vals) == len(dsets) else None

        ck = eff["c_kernel"]
        dep = ck.get("Arm4 deployed (E3 PLHN + engram.bin + cspw.bin) [-O2]", {})
        for name in ROUTERS:
            c = ck.get(f"{name} PLHN only (no Engram) [-O2]", {})
            has_eng = name not in ("v1u_arm1_hnet", "v2s_arm1_hnet")
            rows[f"ours:{name}"] = {
                "params_deploy": M.PLHN_PAYLOAD_BYTES + (M.ENGRAM_SLOTS * M.ENGRAM_DIMS if has_eng else 0),
                "flash_KB": (95008 + (4194816 if has_eng else 0)) / 1024,
                "ram_KB": (95008 + fw["encoder_peak_activation_bytes"]) / 1024,
                "macs": c.get("int8_macs_per_query_mean"), "us_c_int8": c.get("encode_us_median"),
                "us_torch_fp32": eff["torch"].get(f"hnet:{name}", {}).get("torch_fp32_1thread", {}).get("median_us"),
                "boundary_F1_mean": segq(f"router:{name}"), "word_F1_mean": wordq(f"router:{name}")}
        rows["ours:Arm4 full CS-PW (E3 + engram + cspw)"] = {
            "params_deploy": M.PLHN_PAYLOAD_BYTES + M.ENGRAM_SLOTS * M.ENGRAM_DIMS + 393728,
            "flash_KB": (95008 + 4194816 + 393728) / 1024,
            "ram_KB": (95008 + 393728 + fw["encoder_peak_activation_bytes"] + fw["cspw_peak_activation_bytes"]) / 1024,
            "macs": (dep.get("int8_macs_per_query_mean") or 0) + (dep.get("cspw_weave_M4", {}).get("int8_macs") or 0),
            "us_c_int8": (dep.get("encode_us_median") or 0) + (dep.get("cspw_weave_M4", {}).get("median_us") or 0),
            "us_torch_fp32": None, "boundary_F1_mean": segq("router:v2s_E3"), "word_F1_mean": wordq("router:v2s_E3")}
        b = eff["torch"].get("bpe:arm0_matched_ram", {})
        if b:
            rows["baseline:Arm0 matched-RAM BPE"] = {
                "params_deploy": b["params_deploy"], "flash_KB": b["params_deploy"] / 1024,
                "ram_KB": 4 * b["params_deploy"] / 1024, "macs": b["macs_per_query_flopcounter"], "us_c_int8": None,
                "us_torch_fp32": b["torch_fp32_1thread"]["median_us"], "boundary_F1_mean": segq("bpe:arm0_matched_ram"),
                "word_F1_mean": wordq("bpe:arm0_matched_ram")}
        for name, e in eff.get("external", {}).items():
            if "error" in e:
                continue
            rows[f"external:{name}"] = {
                "params_deploy": e["params"], "flash_KB": e["flash_bytes_fp_checkpoint"] / 1024,
                "ram_KB": e["fp32_weight_bytes"] / 1024, "macs": e["macs_per_query_flopcounter"], "us_c_int8": None,
                "us_torch_fp32": e["torch_fp32_1thread"]["median_us"], "boundary_F1_mean": segq(f"tokenizer:{name}"),
                "word_F1_mean": wordq(f"tokenizer:{name}"),
                "rss_peak_MB": e["rss_peak_bytes_after_inference"] / 2**20}
        summary["pareto_rows"] = rows
        summary["pareto_frontier"] = {
            "ram_KB vs boundary_F1_mean": pareto(rows, ["ram_KB"], "boundary_F1_mean"),
            "macs vs boundary_F1_mean": pareto(rows, ["macs"], "boundary_F1_mean"),
            "flash_KB+ram_KB+macs vs boundary_F1_mean": pareto(rows, ["flash_KB", "ram_KB", "macs"], "boundary_F1_mean")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "segmentation_efficiency_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (RES / "segmentation_efficiency_report.md").write_text(render_md(summary))
    log("wrote", RES / "segmentation_efficiency_results.json", RES / "segmentation_efficiency_report.md")


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def render_md(s: dict) -> str:
    seg, bpb, eff, fw = s["segmentation"], s["bits_per_byte"], s["efficiency"], s["firmware_ram"]
    L = ["# Platonic-Engram-HNet — public segmentation, bits-per-byte and efficiency benchmark", "",
         f"Generated {s['generated']} by `{s['script']}`. Raw numbers: `segmentation_efficiency_results.json`. "
         "Every value below was computed by that script on the downloaded public data; nothing is copied or extrapolated.", ""]
    if seg:
        L += ["## 1. Chinese word segmentation (gold human annotation)", ""]
        for d, meta in seg["datasets"].items():
            L.append(f"- **{d}**: {meta.get('source')}; {meta.get('scored_items', '?')} items, {meta.get('chars')} chars, "
                     f"{meta.get('gold_words_longer_than_4_han')} gold words longer than 12 UTF-8 bytes (unreachable by the "
                     f"router's 12-byte span cap).")
        L += ["", "Routers run as exported int8 PLHN bytes through the compiled C reference kernel; C output matched the "
              "numpy integer reference on "
              f"{sum(v['lines'] - v['mismatched_lines'] for v in seg['c_vs_numpy_parity'].values())}/"
              f"{sum(v['lines'] for v in seg['c_vs_numpy_parity'].values())} lines.", ""]
        dsets = list(seg["results"])
        hdr = "| system | " + " | ".join(f"{d} word P / R / F1 | {d} boundary F1 | {d} OOV-R" for d in dsets) + " |"
        L += [hdr, "|---|" + "---|---|---|" * len(dsets)]
        for sname in seg["results"][dsets[0]]:
            cells = []
            for d in dsets:
                r = seg["results"][d][sname]
                cells.append(f"{r['word_P']:.3f} / {r['word_R']:.3f} / **{r['word_F1']:.3f}** | {r['boundary_F1']:.3f} | "
                             f"{r['OOV_recall']:.3f}")
            L.append(f"| {sname} | " + " | ".join(cells) + " |")
        L += ["", "Official SIGHAN `score` script (word level, F / OOV-R) on the same outputs:", "",
              "| system | " + " | ".join(f"{d} F | {d} OOV-R" for d in dsets if d.startswith("SIGHAN")) + " |",
              "|---|" + "---|---|" * sum(d.startswith("SIGHAN") for d in dsets)]
        for sname in seg["results"][dsets[0]]:
            cells = []
            for d in dsets:
                if not d.startswith("SIGHAN"):
                    continue
                o = seg["results"][d][sname].get("official_sighan_score") or {}
                cells.append(f"{o.get('F', '—')} | {o.get('OOV_recall', '—')}")
            L.append(f"| {sname} | " + " | ".join(cells) + " |")
        L += ["", "Router checkpoints:", ""]
        for n, r in seg["routers"].items():
            extra = ""
            if "equals_deployed_models_platonic_hnet_bin" in r:
                extra += f"; byte-identical to deployed `models/platonic-hnet.bin`: {r['equals_deployed_models_platonic_hnet_bin']}"
            if "same_plhn_as" in r:
                extra += f"; same PLHN bytes as {r['same_plhn_as']}"
            L.append(f"- `{n}`: `{r['checkpoint']}` — {r['description']}; router supervision: {r['router_supervision']}; "
                     f"int8 {r['int8']}; sha256 {r['plhn_sha256'][:16]}…{extra}")
        L += [""]
        for d, meta in seg["datasets"].items():
            for ex in meta.get("examples", [])[:2]:
                L.append(f"- {d} example — gold: `{ex['gold']}`; E3: `{ex['router:v2s_E3']}`; v1-unsup arm3: "
                         f"`{ex['router:v1u_arm3_engram']}`; Arm0 BPE: `{ex['bpe:arm0_matched_ram']}`")
        L += [""]
    if bpb:
        L += ["## 2. Bits per byte (lower is better)", "", bpb["protocol"], ""]
        cs = list(bpb["corpora"])
        L += ["| model | " + " | ".join(f"{c} ({bpb['corpora'][c]['windows_256B']} win)" for c in cs) + " |",
              "|---|" + "---|" * len(cs)]
        for m, r in bpb["results"].items():
            L.append(f"| {m} | " + " | ".join(f4(r.get(c)) for c in cs) + " |")
        L += [""]
    if eff:
        L += ["## 3. Efficiency", "",
              f"Host: {eff['host']['cpu']}, {eff['host']['platform']}, {eff['host']['cc']}, torch {eff['host']['torch']}. "
              f"Queries: {eff['queries']['n']} ({eff['queries']['source']}; mean {eff['queries']['mean_bytes']:.0f} B). "
              "C timings: median over queries of the per-query median of "
              f"{C_REPS} repetitions, single thread. Torch timings: FP32, `torch.set_num_threads(1)`, batch 1. "
              "These are Apple-M1 host numbers, not ESP32-S3 device numbers.", "",
              "### 3a. C int8 kernel (bit-exact reference of the firmware arithmetic)", "",
              "| config | encode µs (median) | p90 µs | segment-only µs | µs/byte | int8 MACs/query | bytes/chunk | CS-PW weave(M=4) µs | weave MACs |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, r in eff["c_kernel"].items():
            wv = r.get("cspw_weave_M4", {})
            L.append(f"| {k} | {r['encode_us_median']:.1f} | {r['encode_us_p90']:.1f} | {r['segment_only_us_median']:.1f} | "
                     f"{r['encode_us_per_byte']:.3f} | {r['int8_macs_per_query_mean']:,.0f} | {r['bytes_per_chunk']:.2f} | "
                     f"{wv.get('median_us', float('nan')):.1f} | {wv.get('int8_macs', 0):,} |")
        L += ["", "MACs per PLHN query = 20,896·bytes + 107,328·chunks + 128, counted from the loops of the reference "
              "`encode()` (router q·k dot counted on every byte, so a slight upper bound; elementwise gate multiplies "
              "excluded). CS-PW weave MACs are the exact loop counts of `weave()` for M=4 scenes with the deployed "
              "non-empty frame/phrase rows.", "",
              f"Firmware RAM (from declared buffers in `{fw['source']}`, lower bound): encoder persistent state "
              f"{fw['encoder_persistent_state_bytes']} B, encoder peak activations {fw['encoder_peak_activation_bytes']} B; "
              f"CS-PW state {fw['cspw_state_heap_bytes']} B, CS-PW peak {fw['cspw_peak_activation_bytes']} B; weights in PSRAM "
              f"{fw['weights_psram_bytes']}. {fw['engram_access']}.", "",
              "### 3b. Pareto table (all models, same queries)", "",
              "RAM = resident weights + activations (ours: PSRAM weights + firmware activation bytes; Arm 0: FP32 deploy "
              "params; external: FP32 weights, measured peak process RSS in the last column). Flash = on-disk weight files "
              "(ours: int8 binaries; external: the HF checkpoint as downloaded). Quality = mean boundary / word F1 over the "
              "three segmentation sets (external models: their tokenizer's boundaries).", "",
              "| model | deploy params | flash KB | RAM KB | MACs/query | C int8 µs | torch FP32 1-thr µs | boundary F1 (mean) | word F1 (mean) | peak RSS MB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for n, r in s.get("pareto_rows", {}).items():
            def g(k, fmt):
                v = r.get(k)
                return "—" if v is None else format(v, fmt)
            L.append(f"| {n} | {g('params_deploy', ',')} | {g('flash_KB', ',.1f')} | {g('ram_KB', ',.1f')} | {g('macs', ',.0f')} | "
                     f"{g('us_c_int8', ',.1f')} | {g('us_torch_fp32', ',.1f')} | {g('boundary_F1_mean', '.4f')} | "
                     f"{g('word_F1_mean', '.4f')} | {g('rss_peak_MB', ',.0f')} |")
        L += ["", "Pareto-optimal (non-dominated) rows:", ""]
        for k, v in s.get("pareto_frontier", {}).items():
            L.append(f"- {k}: {', '.join(v)}")
        L += [""]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="seg,bpb,eff,report")
    ap.add_argument("--skip-lfm2-bpb", action="store_true")
    ap.add_argument("--ext-worker", nargs=3, metavar=("NAME", "QUERIES", "OUT"))
    args = ap.parse_args()
    if args.ext_worker:
        ext_worker(args.ext_worker[0], Path(args.ext_worker[1]), Path(args.ext_worker[2]))
        return
    for st in args.stages.split(","):
        log("stage", st)
        {"seg": stage_seg, "bpb": stage_bpb, "eff": stage_eff, "report": stage_report}[st](args)


if __name__ == "__main__":
    main()
