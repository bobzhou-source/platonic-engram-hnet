"""Train + evaluate the 4 Platonic-Engram-HNet ablation arms.

    python src/train.py --arms all --data-dir data --out-dir runs/<name> --steps 6000
    python src/train.py --smoke            # tiny synthetic data, few steps, CPU-friendly

Arms
  arm0_bpe_baseline          matched-RAM byte-BPE transformer (next-token + contrastive + sense)
  arm1_hnet_shortconv_ouro   H-Net router + ShortConv + Ouro core (next-byte + contrastive + sense)
  arm2_platonic_lejepa       arm1 + LeJEPA good views (identity + SIGReg) + byte reconstruction
  arm3_platonic_engram_full  arm2 + 65,536-slot morphological-span Engram + int8 QAT

Benchmarks (eval/): WSD top-1 (wsd_eval.json), cross-lingual R@1/R@5/MRR
(crosslingual_recall_eval.json), bytes-per-chunk + CPU latency, stroke recognition
(stroke_eval.npz: glyph CNN probe into the frozen concept space). arm3 is additionally scored on
the exact int8 device path (encode_int == PLHN integer reference).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["arm0_bpe_baseline", "arm1_hnet_shortconv_ouro", "arm2_platonic_lejepa", "arm3_platonic_engram_full"]
SENSE_KINDS = {"en_example_def", "zh_heteronym_wic", "zh_context_en_def"}
# headword <-> definition kinds: arm3's Engram lexicon (``Data.lex_*``)
LEX_KINDS = {"cedict_zh_en", "zh_en_bridge", "en_word_def"}
LEX_HEAD_BYTES = 64

torch.backends.cuda.matmul.allow_tf32 = False  # QAT bit-exactness needs exact fp32 integer sums
torch.backends.cudnn.allow_tf32 = False


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------


def read_jsonl(path: Path, limit: int | None = None):
    """Skips undecodable/partial lines (e.g. a file caught mid-rewrite) instead of crashing."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(out) >= limit:
                break
    return out


def view_text(v):
    return v["text"] if isinstance(v, dict) else str(v)


def load_corpus_boundaries(data_dir: Path):
    """``tools/build_boundary_views.py`` labels -> ``fn(row_id, n_bytes) -> uint8 array``.  The file
    must match the current ``train_corpus.jsonl`` (sha256) and every row's byte length."""
    p = data_dir / "corpus_boundaries.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing: run tools/build_boundary_views.py")
    z = np.load(p)
    sha = hashlib.sha256((data_dir / "train_corpus.jsonl").read_bytes()).hexdigest()
    if str(z["corpus_sha256"]) != sha:
        raise SystemExit(f"{p} was built for another train_corpus.jsonl")
    labels, offs, lens = z["labels"], z["offsets"], z["lengths"]
    where = {str(i): k for k, i in enumerate(z["ids"])}

    def get(row_id, n):
        k = where[str(row_id)]
        assert int(lens[k]) == n, (row_id, int(lens[k]), n)
        return labels[offs[k]:offs[k] + n]
    return get


class Data:
    """All training text lives on the training device: the corpus as one byte tensor (LM windows
    are sampled with device RNG), every view / glyph-class text as a row of a padded byte table
    (batches are index gathers).  Only arm0's BPE windows are tokenised on the host."""
    def __init__(self, data_dir: Path, seed: int, corpus_limit: int | None, device, heldout_docs: int = 1000):
        self.rng = random.Random(seed)
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        excl_p = data_dir / "train_exclusions.json"
        self.excluded = set(json.loads(excl_p.read_text())["ids"]) if excl_p.exists() else set()
        rows = [r for r in read_jsonl(data_dir / "train_corpus.jsonl", corpus_limit) if r.get("text")]
        self.n_excluded_docs = sum(r.get("id") in self.excluded for r in rows)
        bl = load_corpus_boundaries(data_dir)
        kept = [r for r in rows if r.get("id") not in self.excluded]
        corpus = [r["text"].encode("utf-8") for r in kept]
        corpus_lab = [bl(r["id"], len(d)) for r, d in zip(kept, corpus)]
        # held-out documents for next-byte BPC / efficiency eval (never sampled for training)
        order = list(range(len(corpus)))
        random.Random(seed + 1).shuffle(order)
        n_ho = min(heldout_docs, len(corpus) // 10)
        ho = set(order[:n_ho])
        self.heldout = [corpus[i] for i in sorted(ho)]
        self.heldout_lab = [corpus_lab[i] for i in sorted(ho)]
        self.corpus = [d for i, d in enumerate(corpus) if i not in ho]
        corpus_lab = [d for i, d in enumerate(corpus_lab) if i not in ho]
        all_views = read_jsonl(data_dir / "good_views.jsonl")
        views = [r for r in all_views if r.get("id") not in self.excluded]
        self.n_excluded_views = len(all_views) - len(views)
        bridge_p = data_dir / "lexical_bridge.jsonl"
        bridge = read_jsonl(bridge_p) if bridge_p.exists() else []
        self.n_bridge = len(bridge)
        ctx_p = data_dir / "context_views.jsonl"
        ctx_views = [r for r in (read_jsonl(ctx_p) if ctx_p.exists() else []) if r.get("meta", {}).get("doc") not in self.excluded]
        self.n_context = len(ctx_views)
        views += bridge + ctx_views
        # Engram lexicon (arm3): every dictionary (headword, definition) pair -- all senses of a
        # polysemous headword, so its slot is pulled to the sense centroid, not to one sense -- plus
        # the mined context terms; non-Han headwords also as " word" (the chunk a space-prefixed
        # word forms inside a sentence)
        lex = {}
        for r in views:
            kind = r.get("kind")
            if kind in LEX_KINDS:
                head, defn = view_text(r["a"]), view_text(r["b"])
            elif kind == "zh_context_en_def_mined":
                head, defn = r["meta"]["term"], view_text(r["b"])
            else:
                continue
            head = head.strip()
            if not head or len(head.encode("utf-8")) > LEX_HEAD_BYTES - 1:
                continue
            lex[(head, defn)] = None
            if not M._HAN_RUN.search(head):
                lex[(" " + head, defn)] = None
        self.lex = list(lex)
        # headwords that can stand inside running text as-is (Han, or space-prefixed)
        self.lex_ctx_rows = [i for i, (h, _) in enumerate(self.lex) if h.startswith(" ") or M._HAN_RUN.match(h)]
        self.sense, self.sense_term, self.pairs = [], [], []
        for r in views:
            a, b = view_text(r["a"]), view_text(r["b"])
            if r.get("kind") in SENSE_KINDS:
                self.sense.append((a, b))
                self.sense_term.append(f'{r.get("kind")}:{(r.get("meta") or {}).get("term", a)}')
            else:
                self.pairs.append((a, b))
        if not self.pairs:
            self.pairs = self.sense
        # headwords with >= 2 distinct sense definitions: batched together as hard negatives
        groups = {}
        for i, t in enumerate(self.sense_term):
            groups.setdefault(t, []).append(i)
        self.sense_groups = [ix for ix in groups.values() if len({self.sense[i][1] for i in ix}) >= 2]
        self.kinds = {}
        for r in views:
            self.kinds[r.get("kind")] = self.kinds.get(r.get("kind"), 0) + 1
        ev = data_dir / "eval"
        self.wsd = json.loads((ev / "wsd_eval.json").read_text())["items"]
        self.xling = json.loads((ev / "crosslingual_recall_eval.json").read_text())
        # zero sentence leakage: no held-out eval sentence/passage is a training view text
        eval_sents = {it["sentence"] for it in self.wsd} | {it["context"] for it in self.wsd if it.get("context")}
        eval_sents |= {q["text"] for q in self.xling["queries"]} | {q.get("sentence", q["text"]) for q in self.xling["queries"]}
        self.view_sentence_leaks = sum(t in eval_sents for ab in self.pairs + self.sense for t in ab)
        if self.view_sentence_leaks:
            raise SystemExit(f"{self.view_sentence_leaks} eval sentences are training view texts")
        st = np.load(ev / "stroke_eval.npz")
        self.stroke_images = st["images"]
        self.stroke_chars = [bytes(st["utf8"][i, : int(st["utf8_len"][i])]) for i in range(len(st["images"]))]
        self.stroke_classes = sorted(set(self.stroke_chars))
        cidx = {c: i for i, c in enumerate(self.stroke_classes)}
        self.stroke_y = torch.tensor([cidx[c] for c in self.stroke_chars])
        # fixed render split shared by the training glyph view (arm2+) and the eval probe
        perm = torch.randperm(len(self.stroke_images), generator=torch.Generator().manual_seed(seed))
        ntr = int(0.8 * len(perm))
        self.stroke_train, self.stroke_test = perm[:ntr], perm[ntr:]
        self._stroke_x = None

        # ---- device tensors
        ids = {}

        def tid(t):
            return ids.setdefault(M.text_bytes(t), len(ids))

        dev = self.device
        self.pair_idx = torch.tensor([[tid(a), tid(b)] for a, b in self.pairs], dtype=torch.long, device=dev)
        self.sense_idx = torch.tensor([[tid(a), tid(b)] for a, b in self.sense], dtype=torch.long,
                                      device=dev).reshape(-1, 2)
        self.glyph_tid = torch.tensor([tid(c) for c in self.stroke_classes], dtype=torch.long, device=dev)
        self.lex_def = torch.tensor([tid(d) for _, d in self.lex], dtype=torch.long, device=dev)
        self.text_list = list(ids)
        X, lens = M.pack_texts(self.text_list, "cpu")
        self.text_X = X.to(torch.uint8).to(dev)
        self.text_len = lens.to(dev)
        # headword table: a headword is one word (whole Han run = one word)
        hX, hlen = M.pack_texts([h for h, _ in self.lex] or [""], "cpu")
        hlab = np.zeros(tuple(hX.shape), np.uint8)
        for i, (h, _) in enumerate(self.lex):
            lab = M.word_boundary_labels(h)
            hlab[i, :len(lab)] = np.frombuffer(lab, np.uint8)
        self.lex_X, self.lex_len = hX.to(torch.uint8).to(dev), hlen.to(dev)
        self.lex_lab = torch.from_numpy(hlab).to(dev)
        self.lex_ctx_t = torch.tensor(self.lex_ctx_rows, dtype=torch.long, device=dev)
        self.doc_len = torch.tensor([len(d) for d in self.corpus], dtype=torch.long, device=dev)
        self.doc_off = torch.cumsum(self.doc_len, 0) - self.doc_len
        self.corpus_t = torch.from_numpy(np.frombuffer(b"".join(self.corpus), dtype=np.uint8).copy()).to(dev)
        self.corpus_lab = torch.from_numpy(np.concatenate(corpus_lab) if corpus_lab else np.zeros(0, np.uint8)).to(dev)
        assert self.corpus_lab.shape == self.corpus_t.shape
        self.tok = None

    def stroke_x(self, device):
        if self._stroke_x is None or self._stroke_x.device != torch.device(device):
            self._stroke_x = (torch.from_numpy(self.stroke_images).to(device).float() / 255.0).unsqueeze(1)
        return self._stroke_x

    def lex_in_context(self, li, max_pre: int = 160, max_suf: int = 64):
        """Lexicon rows ``li`` set inside running text: a random training-corpus prefix (0..max_pre
        bytes) + headword + random corpus suffix (0..max_suf bytes), both cut on UTF-8 lead bytes,
        total <= MAX_BYTES.  The router then splits the headword the way it splits it mid-sentence,
        so its Engram slots are trained under in-context segmentation.  Returns (X, lens, labels)
        with the corpus word labels and the headword as one word."""
        dev, g, b = self.device, self.gen, len(li)
        H, hl, Hlab = self.lex_X[li].long(), self.lex_len[li], self.lex_lab[li]
        Wp, n_p, Lp = self.lm_batch_t(b, max_pre)
        Ws, n_s, Ls = self.lm_batch_t(b, max_suf)

        def cut(W, n, limit):
            """largest UTF-8 lead position (or the window end) <= a uniform draw in [0, limit]"""
            mx = W.shape[1]
            r = torch.minimum((torch.rand(b, device=dev, generator=g) * (limit + 1).float()).long(), n)
            ar = torch.arange(mx + 1, device=dev)[None, :]
            Wx = torch.cat([W, torch.zeros_like(W[:, :1])], 1)
            ok = (((Wx & 0xC0) != 0x80) | (ar >= n[:, None])) & (ar <= r[:, None])
            return (ok.long() * ar).max(1).values

        pl = cut(Wp, n_p, torch.full((b,), max_pre, device=dev))
        sl = cut(Ws, n_s, (M.MAX_BYTES - pl - hl).clamp(min=0, max=max_suf))
        tot = pl + hl + sl
        L = int(tot.max())
        j = torch.arange(L, device=dev)[None, :]
        in_h, in_s = (j >= pl[:, None]) & (j < (pl + hl)[:, None]), (j >= (pl + hl)[:, None]) & (j < tot[:, None])
        jp = j.clamp(max=max_pre - 1).expand(b, -1)
        jh = (j - pl[:, None]).clamp(0, H.shape[1] - 1)
        js = (j - (pl + hl)[:, None]).clamp(0, max_suf - 1)
        X = torch.where(j < pl[:, None], torch.gather(Wp, 1, jp), 0)
        X = torch.where(in_h, torch.gather(H, 1, jh), X)
        X = torch.where(in_s, torch.gather(Ws, 1, js), X)
        lab = torch.where(j < pl[:, None], torch.gather(Lp, 1, jp), 0)
        lab = torch.where(in_h, torch.gather(Hlab, 1, jh), lab)
        lab = torch.where(in_s, torch.gather(Ls, 1, js), lab)
        lab = torch.where(in_s & (j == (pl + hl)[:, None]), M.BND_START, lab)  # a new word after the headword
        return X, tot, lab.to(torch.uint8)

    # ---- host-side windows (arm0 BPE tokenisation only)
    def window(self, n_bytes: int = M.MAX_BYTES) -> bytes:
        data = self.rng.choice(self.corpus)
        if len(data) > n_bytes:
            start = self.rng.randrange(0, len(data) - n_bytes + 1)
            while start < len(data) and M.is_cont(data[start]):
                start += 1
            data = data[start:start + n_bytes]
        return data

    def lm_batch(self, b: int):
        return [self.window() for _ in range(b)]

    # ---- device-side batches
    def lm_batch_t(self, b: int, n_bytes: int = M.MAX_BYTES):
        """Same sampling rule as ``window`` (uniform doc, uniform start, skip UTF-8 continuation
        bytes), drawn with the device generator: returns (X (b, n_bytes) int64, lens (b,),
        word-boundary labels (b, n_bytes) uint8)."""
        dev, g = self.device, self.gen
        d = torch.randint(0, len(self.doc_len), (b,), device=dev, generator=g)
        dl, off = self.doc_len[d], self.doc_off[d]
        span = (dl - n_bytes).clamp(min=0) + 1
        start = torch.minimum((torch.rand(b, device=dev, generator=g) * span).long(), span - 1)
        last = len(self.corpus_t) - 1
        for _ in range(3):
            cb = self.corpus_t[(off + start).clamp(max=last)]
            start = start + (((cb & 0xC0) == 0x80) & (start < dl) & (dl > n_bytes)).long()
        n = (dl - start).clamp(min=0, max=n_bytes)
        ar = torch.arange(n_bytes, device=dev)
        pos = ((off + start)[:, None] + ar[None, :]).clamp(max=last)
        keep = ar[None, :] < n[:, None]
        X = self.corpus_t[pos].long() * keep
        return X, n, self.corpus_lab[pos] * keep

    def gather_bytes(self, idx):
        """Rows of the text table -> (X (B, L) int64 trimmed to the longest row, lens (B,))."""
        lens = self.text_len[idx]
        L = max(1, int(lens.max()))
        return self.text_X[idx, :L].long(), lens

    def tokenize_table(self, net):
        """arm0: BPE ids of every table text, once (int16 table on the device)."""
        toks = [net._ids(t) or [0] for t in self.text_list]
        T = max(len(t) for t in toks)
        arr = np.zeros((len(toks), T), np.int16)
        for i, t in enumerate(toks):
            arr[i, :len(t)] = t
        self.tok = torch.from_numpy(arr).to(self.device)
        self.tok_len = torch.tensor([len(t) for t in toks], dtype=torch.long, device=self.device)

    def gather_tokens(self, idx):
        tl = self.tok_len[idx]
        T = max(1, int(tl.max()))
        ids = self.tok[idx, :T].long()
        valid = torch.arange(T, device=self.device)[None, :] < tl[:, None]
        return ids, valid, self.text_len[idx].sum()

    def pair_batch_t(self, b: int):
        return self.pair_idx[torch.randint(0, len(self.pair_idx), (b,), device=self.device, generator=self.gen)]

    def sense_batch_t(self, b: int, grouped_frac: float = 0.5):
        """Half the batch is whole sense groups of polysemous headwords (sibling senses = hard
        negatives); pos[i, j] marks identical definitions (shared sense, not a negative)."""
        if not self.sense:
            return None, None
        idx = []
        while self.sense_groups and len(idx) < int(b * grouped_frac):
            g = self.rng.choice(self.sense_groups)
            idx += self.rng.sample(g, min(len(g), 4))
        idx = idx[:b]
        idx += [self.rng.randrange(len(self.sense)) for _ in range(b - len(idx))]
        si = self.sense_idx[torch.tensor(idx, dtype=torch.long).to(self.device, non_blocking=True)]
        return si, si[:, 1][:, None] == si[:, 1][None, :]

    def eval_texts(self, n: int = 200, labels: bool = False):
        """Held-out 256-byte windows (documents excluded from training) [+ their word labels]."""
        rng = random.Random(7)
        out, labs = [], []
        for _ in range(n):
            k = rng.randrange(len(self.heldout))
            d = self.heldout[k]
            s = rng.randrange(0, max(1, len(d) - M.MAX_BYTES + 1))
            while s < len(d) and M.is_cont(d[s]):
                s += 1
            if not d[s:s + M.MAX_BYTES]:
                s = 0
            out.append(d[s:s + M.MAX_BYTES])
            labs.append(self.heldout_lab[k][s:s + M.MAX_BYTES])
        return (out, labs) if labels else out


def make_smoke_data(root: Path, seed: int = 0) -> Path:
    """Synthetic data with the exact training-data schemas (shape/syntax smoke only)."""
    rng = random.Random(seed)
    en = "the bank river money water light stone house unhappiness running kicked bucket concept word reader".split()
    zh = list("银行河岸水光石家人心书读字天地山风雨云火木金")
    def en_s(n): return " ".join(rng.choice(en) for _ in range(n)).capitalize() + "."
    def zh_s(n): return "".join(rng.choice(zh) for _ in range(n)) + "。"
    (root / "eval").mkdir(parents=True, exist_ok=True)
    texts = []
    with open(root / "train_corpus.jsonl", "w") as f:
        for i in range(400):
            t = en_s(rng.randint(20, 80)) if i % 2 else zh_s(rng.randint(20, 90))
            texts.append(t)
            f.write(json.dumps({"id": i, "lang": "en" if i % 2 else "zh", "source": "book", "source_id": "s",
                                "text": t}, ensure_ascii=False) + "\n")
    labs = [np.frombuffer(M.word_boundary_labels(t, list), np.uint8) for t in texts]  # 1 Han char = 1 word
    lens = np.array([len(x) for x in labs], np.int64)
    np.savez_compressed(root / "corpus_boundaries.npz", labels=np.concatenate(labs), offsets=np.cumsum(lens) - lens,
                        lengths=lens, ids=np.array([str(i) for i in range(400)]),
                        corpus_sha256=np.array(hashlib.sha256((root / "train_corpus.jsonl").read_bytes()).hexdigest()))
    kinds = ["cedict_zh_en", "zh_context_en_def", "zh_en_bridge", "en_word_def", "en_example_def",
             "zh_heteronym_wic", "zh_en_thematic", "book_substack_thematic"]
    with open(root / "good_views.jsonl", "w") as f:
        for i in range(300):
            f.write(json.dumps({"id": i, "kind": kinds[i % len(kinds)], "a": {"lang": "zh", "text": zh_s(8)},
                                "b": {"lang": "en", "text": en_s(6)}, "meta": {}}, ensure_ascii=False) + "\n")
    items = []
    for i in range(40):
        s = en_s(10)
        items.append({"id": i, "lang": "en", "sentence": s, "target": s.split()[0], "span": [0, 3],
                      "candidates": [en_s(5) for _ in range(5)], "answer": rng.randrange(5)})
    (root / "eval" / "wsd_eval.json").write_text(json.dumps({"items": items}, ensure_ascii=False))
    cands = [{"id": f"c{i}", "text": en_s(8)} for i in range(60)]
    qs = [{"id": f"q{i}", "text": zh_s(12), "anchor": "x", "gold": f"c{i}"} for i in range(30)]
    (root / "eval" / "crosslingual_recall_eval.json").write_text(
        json.dumps({"candidates": cands, "queries": qs, "metrics": ["R@1", "R@5", "MRR"]}, ensure_ascii=False))
    chars = [c.encode() for c in zh[:10]]
    n = 120
    utf8 = np.zeros((n, 4), np.uint8)
    lens = np.zeros(n, np.uint8)
    imgs = np.zeros((n, 64, 64), np.uint8)
    for i in range(n):
        c = chars[i % len(chars)]
        utf8[i, : len(c)] = list(c)
        lens[i] = len(c)
        r = np.random.RandomState(i % len(chars))
        imgs[i] = (r.rand(64, 64) > 0.8).astype(np.uint8) * 255
    np.savez(root / "eval" / "stroke_eval.npz", images=imgs, utf8=utf8, utf8_len=lens,
             codepoint=np.array([ord(chars[i % 10].decode()) for i in range(n)], np.int32),
             font_id=np.zeros(n, np.int32), jitter=np.zeros((n, 5), np.float32))
    return root


# ----------------------------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------------------------


ARM_CFG = {
    "arm0_bpe_baseline": dict(kind="bpe", engram=False, lejepa=False, recon=False, glyph=False, qat=False),
    "arm1_hnet_shortconv_ouro": dict(kind="hnet", engram=False, lejepa=False, recon=False, glyph=False, qat=False),
    "arm2_platonic_lejepa": dict(kind="hnet", engram=False, lejepa=True, recon=True, glyph=True, qat=False),
    "arm3_platonic_engram_full": dict(kind="hnet", engram=True, lejepa=True, recon=True, glyph=True, qat=True),
}


def build_model(arm: str, data: Data, args, bpe_merges=None):
    cfg = ARM_CFG[arm]
    if cfg["kind"] == "bpe":
        if bpe_merges is None:
            sample = [d.decode("utf-8", "ignore") for d in data.corpus[: args.bpe_train_texts]]
            bpe = M.ByteBPE.train(sample, args.bpe_merges)
        else:
            bpe = M.ByteBPE(bpe_merges)
        return M.BPETransformerBaseline(bpe)
    return M.PlatonicEngramHNet(use_engram=cfg["engram"], target_ratio=args.target_ratio)


def branch_path(out: Path) -> Path:
    return out / f"{ARMS[2]}.branch.pt"


def encode_views(net, data: Data, idx, buckets: int, bpe: bool):
    """Encode text-table rows ``idx`` as ``buckets`` length-sorted sub-batches, each padded only to
    its own longest row (view texts run 1..256 bytes, median 45, so one padded batch is ~3.5x the
    real bytes).  Returns c, z in the original row order and the H-Net ratio loss over all rows,
    identical to a single-batch encode (ratio stats are additive); the ratio loss is None for BPE."""
    order = torch.argsort(data.tok_len[idx] if bpe else data.text_len[idx])
    cs, zs, stats = [], [], []
    for o in order.tensor_split(buckets):
        if not len(o):
            continue
        ev = net.encode_ids(*data.gather_tokens(idx[o])) if bpe else net.encode_tensor(*data.gather_bytes(idx[o]))
        cs.append(ev["c"])
        zs.append(ev["z"])
        if not bpe:
            stats.append(ev["ratio_stats"])
    inv = torch.argsort(order)
    ratio = None if bpe else M.ratio_loss(torch.stack(stats).sum(0), net.target_ratio)
    return torch.cat(cs)[inv], torch.cat(zs)[inv], ratio


def train_arm(arm: str, data: Data, args, device, log, out: Path):
    """One step = one LM encode + a length-bucketed encode of every view text (pair a/b, sense a/b)
    + one encode of all glyph-class characters (arm2+).  Batches are device-side index gathers; the
    only host syncs per step are the chunk/row counts used to size tensors.  ``--tf32`` lets the
    training matmuls use TF32 tensor cores; it is switched back off before any evaluation, so every
    reported number (and the int8 path, which needs exact fp32 integer sums) runs in full fp32.

    arm3 warm start (``--arm3-branch-frac f > 0``): arm2 snapshots model + glyph CNN + AdamW state
    at step f*steps; arm3 loads it and trains steps f*steps..steps on the same LR schedule with the
    zero-initialised Engram (10 % slot dropout on views, 5x LR, no weight decay) and int8 QAT, so arm3's
    backbone sees exactly as many optimiser steps as every other arm."""
    cfg = ARM_CFG[arm]
    bpe = cfg["kind"] == "bpe"
    torch.manual_seed(args.seed)
    net = build_model(arm, data, args).to(device)
    if bpe:
        t_tok = time.time()
        data.tokenize_table(net)
        log({"event": "bpe_table", "texts": len(data.text_list), "sec": round(time.time() - t_tok, 1)})
    glyph = GlyphCNN().to(device) if cfg["glyph"] else None
    engram_params = [net.engram] if getattr(net, "engram", None) is not None else []
    eid = {id(p) for p in engram_params}
    other = [p for p in net.parameters() if id(p) not in eid]
    groups = [{"params": other, "weight_decay": 0.01}]
    if glyph is not None:
        groups.append({"params": list(glyph.parameters()), "weight_decay": 0.01})
    if engram_params:
        groups.append({"params": engram_params, "weight_decay": 0.0, "lr": args.lr * args.engram_lr_mult})
        net.engram_drop = args.engram_drop
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), fused=device.type == "cuda")
    base_lrs = [g.get("lr", args.lr) for g in opt.param_groups]
    qat_start = int(args.steps * args.qat_start) if cfg["qat"] else None
    branch_step = int(args.steps * args.arm3_branch_frac) if args.arm3_branch_frac > 0 else None
    start = 0
    if cfg["engram"] and branch_step is not None:
        bp = branch_path(out)
        t_wait = time.time()
        while not bp.exists():  # parallel mode: arm2 writes it at its branch step
            if time.time() - t_wait > args.branch_wait:
                raise SystemExit(f"no arm2 branch snapshot {bp} after {args.branch_wait} s")
            time.sleep(5)
        snap = torch.load(bp, map_location=device, weights_only=False)
        missing, unexpected = net.load_state_dict(snap["net"], strict=False)
        assert "engram" in missing and set(missing) <= {"engram", "engram_hash"} and not unexpected, (missing, unexpected)
        glyph.load_state_dict(snap["glyph"])
        sd = opt.state_dict()
        sd["state"] = snap["opt"]["state"]  # same param order; the engram group starts fresh
        opt.load_state_dict(sd)
        start = snap["step"]
        log({"event": "warm_start", "arm": arm, "from": ARMS[2], "step": start,
             "engram_drop": args.engram_drop, "engram_lr": args.lr * args.engram_lr_mult,
             "engram_abs_max": float(net.engram.detach().abs().max())})
    snap_step = branch_step if arm == ARMS[2] else None
    glyph_log_tau = torch.tensor(math.log(0.05), device=device)
    if glyph is not None:
        X_img = data.stroke_x(device)
        tr = data.stroke_train.to(device)
        ycls = data.stroke_y.to(device)
        cls_ar = torch.arange(len(data.stroke_classes), device=device)[:, None]
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    t0 = time.time()
    for step in range(start, args.steps):
        if snap_step is not None and step == snap_step:
            tmp = branch_path(out).with_suffix(".tmp")
            torch.save({"step": step, "net": net.state_dict(), "glyph": glyph.state_dict() if glyph else None,
                        "opt": opt.state_dict()}, tmp)
            os.replace(tmp, branch_path(out))
            log({"event": "branch_snapshot", "arm": arm, "step": step})
        frac = step / max(1, args.steps)
        warm = min(1.0, (step + 1) / max(1, args.warmup))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if qat_start is not None:
            net.quant = step >= qat_start
        net.train()
        losses = {}
        enc = bstats = None
        # arm3 explicit lexical-alignment phase: the first --lex-phase-steps of its own steps train
        # only the Engram lexicon (+ router boundaries + glyph view), then everything jointly
        lex_phase = cfg["engram"] and step < start + args.lex_phase_steps
        if not lex_phase:
            if bpe:
                enc = net.encode(data.lm_batch(args.batch_lm))
                losses["next_token"] = net.lm_loss(enc)
            else:
                X, n, lab = data.lm_batch_t(args.batch_lm)
                enc = net.encode_tensor(X, n)
                bl = net.byte_losses(enc, recon=cfg["recon"])
                losses["next_byte"] = bl["next_byte"]
                if cfg["recon"]:
                    losses["recon"] = args.w_recon * bl["recon"]
                if args.w_boundary > 0:
                    bstats = M.boundary_stats(enc, lab, args.boundary_margin)
                else:
                    losses["ratio"] = args.w_ratio * enc["ratio_loss"]

            pi = data.pair_batch_t(args.batch_views)
            si, spos = data.sense_batch_t(args.batch_sense)
            na, ns = len(pi), (0 if si is None else len(si))
            idx = torch.cat([pi[:, 0], pi[:, 1]] + ([si[:, 0], si[:, 1]] if ns else []))
            c, z, ratio_views = encode_views(net, data, idx, args.view_buckets, bpe)
            ca, cb, za, zb = c[:na], c[na:2 * na], z[:na], z[na:2 * na]
            csa, csb = c[2 * na:2 * na + ns], c[2 * na + ns:2 * na + 2 * ns]
            if na:
                losses["contrastive"] = args.w_nce * M.info_nce(ca, cb, net.nce_log_tau)
                if cfg["lejepa"]:
                    lj, _, _ = M.lejepa_loss(za, zb, args.sigreg_lambda)
                    losses["lejepa"] = args.w_lejepa * lj
            if ns:
                losses["sense_rank"] = args.w_sense * M.sense_rank_loss(
                    csa, csb, net.sense_log_tau, args.sense_eps, pos_mask=spos, margin=args.sense_margin)
            if not bpe and args.w_boundary <= 0:
                losses["ratio_views"] = args.w_ratio * ratio_views
        if cfg["engram"] and len(data.lex):
            # Engram lexicon: every headword's own slot (FNV1a64(headword) % 65536, no slot dropout)
            # aligned with the concept of its definition (InfoNCE + LeJEPA invariance), and the
            # router taught that a headword is one chunk.  The in-context half places headwords
            # inside random corpus text: the passage concept must retrieve the headword's
            # definition, so the slots of the pieces the router actually cuts mid-sentence carry
            # the headword and dominate the passage sum (salience)
            li = torch.randint(0, len(data.lex), (args.batch_lex,), device=device, generator=data.gen)
            hl = data.lex_len[li]
            Lh = max(1, int(hl.max()))
            nctx = args.batch_lex_ctx if len(data.lex_ctx_t) else 0
            lc = data.lex_ctx_t[torch.randint(0, max(1, len(data.lex_ctx_t)), (nctx,), device=device, generator=data.gen)]
            drop, net.engram_drop = net.engram_drop, 0.0
            eh = net.encode_tensor(data.lex_X[li, :Lh].long(), hl)
            if nctx:
                Xc, nc, labc = data.lex_in_context(lc)
                ec = net.encode_tensor(Xc, nc)
            net.engram_drop = drop
            cd, zd, _ = encode_views(net, data, torch.cat([data.lex_def[li], data.lex_def[lc]]), args.view_buckets, False)
            losses["lex_nce"] = args.w_lex * M.info_nce(eh["c"], cd[:len(li)], net.nce_log_tau)
            lj, _, _ = M.lejepa_loss(eh["z"], zd[:len(li)], args.sigreg_lambda)
            losses["lex_lejepa"] = args.w_lex * args.w_lejepa * lj
            if nctx:
                losses["lex_ctx_nce"] = args.w_lex * M.info_nce(ec["c"], cd[len(li):], net.nce_log_tau)
            if args.w_boundary > 0:
                hs = M.boundary_stats(eh, data.lex_lab[li, :Lh], args.boundary_margin)
                if nctx:
                    hs = hs + M.boundary_stats(ec, labc, args.boundary_margin)
                bstats = hs if bstats is None else bstats + hs
        if bstats is not None:
            losses["boundary"] = args.w_boundary * M.boundary_loss(bstats)
        if glyph is not None:
            # glyph render -> its character concept among ALL character concepts, and each sampled
            # character concept -> its glyph renders in the batch; the character prototypes (arm3:
            # each single-character span's own Engram slot, never dropped here) are also spread
            # over the sphere (uniformity) so a glyph probe can separate them
            gidx = tr[torch.randint(0, len(tr), (args.batch_glyph,), device=device)]
            gcls = ycls[gidx]
            drop = getattr(net, "engram_drop", 0.0)
            net.engram_drop = 0.0
            protos = net.encode_tensor(*data.gather_bytes(data.glyph_tid))["c"]
            net.engram_drop = drop
            logits = glyph(X_img[gidx]) @ protos.T / glyph_log_tau.exp()
            l1 = F.cross_entropy(logits, gcls)
            pos = (gcls[None, :] == cls_ar).float()
            npos = pos.sum(1, keepdim=True)
            l2 = -(pos / npos.clamp(min=1) * F.log_softmax(logits.T, -1)).sum() / (npos > 0).sum().clamp(min=1)
            losses["glyph_view"] = args.w_glyph * 0.5 * (l1 + l2)
            if args.w_glyph_unif > 0:
                nP = len(protos)
                k = torch.exp(-2.0 * (2.0 - 2.0 * protos @ protos.T))
                losses["glyph_unif"] = args.w_glyph_unif * torch.log((k.sum() - nP) / max(1, nP * (nP - 1)) + 1e-12)
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.clip)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {"arm": arm, "step": step, "loss": float(loss.detach()),
                   "bpc": float(enc["bytes_per_chunk"]) if enc is not None else None,
                   "quant": bool(getattr(net, "quant", False)), "sec": round(time.time() - t0, 1),
                   **{k: round(float(v.detach()), 5) for k, v in losses.items()}}
            if bstats is not None:
                rec["boundary_acc"] = round(float(bstats[4].detach() / bstats[5].detach().clamp(min=1)), 4)
            if lex_phase:
                rec["lex_phase"] = True
            log(rec)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    return net


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


@torch.no_grad()
def embed(net, texts, bs: int = 128, int8: bool = False):
    net.eval()
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        if int8:
            out.append(torch.tensor(net.encode_int(chunk), dtype=torch.float32) / 127.0)
        else:
            out.append(net.encode(chunk)["c"].float().cpu())
    return torch.cat(out, 0) if out else torch.zeros(0, M.D_CONCEPT)


def eval_wsd(net, items, int8=False):
    sents = [it["sentence"] for it in items]
    cands = [c for it in items for c in it["candidates"]]
    S, C = embed(net, sents, int8=int8), embed(net, cands, int8=int8)
    correct, k = 0, 0
    for i, it in enumerate(items):
        n = len(it["candidates"])
        scores = C[k:k + n] @ S[i]
        correct += int(int(scores.argmax()) == int(it["answer"]))
        k += n
    return {"wsd_top1": correct / max(1, len(items)), "wsd_items": len(items),
            "wsd_chance": float(np.mean([1.0 / len(it["candidates"]) for it in items])) if items else 0.0}


def eval_xling(net, xl, int8=False):
    """Sentence -> definition retrieval (the benchmark) + the same ranking for the bare anchor
    term (``xling_term_alone_*``: how well the lexicon itself maps headword -> definition)."""
    cands = xl["candidates"]
    ids = {c["id"]: i for i, c in enumerate(cands)}
    C = embed(net, [c["text"] for c in cands], int8=int8)
    gold = torch.tensor([ids[q["gold"]] for q in xl["queries"]])

    def ranks(Q):
        sims = Q @ C.T
        gs = sims[torch.arange(len(gold)), gold]
        return (sims > gs[:, None]).sum(1) + 1

    rank = ranks(embed(net, [q["text"] for q in xl["queries"]], int8=int8))
    rterm = ranks(embed(net, [q["anchor"] for q in xl["queries"]], int8=int8))
    return {"xling_R@1": float((rank <= 1).float().mean()), "xling_R@5": float((rank <= 5).float().mean()),
            "xling_MRR": float((1.0 / rank.float()).mean()), "xling_queries": len(gold), "xling_candidates": len(cands),
            "xling_term_alone_R@1": float((rterm <= 1).float().mean()),
            "xling_term_alone_MRR": float((1.0 / rterm.float()).mean())}


@torch.no_grad()
def eval_segmentation(net, data: Data, texts, labs, int8=False):
    """Router vs ground-truth words on held-out windows (decision accuracy at supervised lead bytes,
    boundary precision/recall/F1 of the actual segmentation incl. the 12-byte cap) and the share of
    cross-lingual query passages in which the anchor term is exactly one chunk (= its own Engram
    slot).  ``int8``: the QAT/integer path (``quant=True`` is bit-exact with the device)."""
    net.eval()
    prev = net.quant
    net.quant = bool(int8)
    try:
        tp = fp = fn = corr = sup = 0
        for i in range(0, len(texts), 128):
            enc = net.encode(texts[i:i + 128])
            L = enc["X"].shape[1]
            lab = torch.zeros(len(enc["X"]), L, dtype=torch.uint8, device=enc["X"].device)
            for r, lb in enumerate(labs[i:i + 128]):
                lab[r, :len(lb)] = torch.from_numpy(np.asarray(lb, np.uint8).copy()).to(lab.device)
            st = M.boundary_stats(enc, lab)
            corr += float(st[4])
            sup += float(st[5])
            m = enc["valid"].clone()
            m[:, 0] = False
            gt_s = (lab == M.BND_START) & m
            gt_i = (lab == M.BND_INSIDE) & m
            pred = enc["bnd"] & m & (gt_s | gt_i)
            tp += int((pred & gt_s).sum())
            fp += int((pred & gt_i).sum())
            fn += int((~pred & gt_s).sum())
        qs = data.xling["queries"]
        hit = 0
        for i in range(0, len(qs), 128):
            batch = qs[i:i + 128]
            enc = net.encode([q["text"] for q in batch])
            st, nc, ln = enc["starts"].tolist(), enc["n_chunks"].tolist(), enc["lens"].tolist()
            for q, s, n, l in zip(batch, st, nc, ln):
                b, a = M.text_bytes(q["text"]), q["anchor"].encode("utf-8")
                o = b.find(a)
                spans = {(s[j], s[j + 1] if j + 1 < n else l) for j in range(n)}
                hit += int(o >= 0 and (o, o + len(a)) in spans)
    finally:
        net.quant = prev
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"word_boundary_acc": corr / max(1.0, sup), "word_boundary_precision": prec, "word_boundary_recall": rec,
            "word_boundary_f1": 2 * prec * rec / max(1e-9, prec + rec), "xling_anchor_chunk_hit": hit / max(1, len(qs))}


@torch.no_grad()
def eval_efficiency(net, texts, device):
    net.eval()
    enc = net.encode(texts)
    bpc = float(enc["bytes_per_chunk"])
    cpu_net = net.to("cpu")
    lat = []
    for t in texts[:50]:
        t0 = time.perf_counter()
        cpu_net.encode([t])
        lat.append(time.perf_counter() - t0)
    net.to(device)
    return {"bytes_per_chunk": bpc, "latency_ms_cpu_torch_b1": 1000 * float(np.median(lat))}


class GlyphCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(), nn.Conv2d(128, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, M.D_CONCEPT))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def eval_stroke(net, data: Data, device, steps: int, seed: int, int8=False):
    """Glyph-to-concept probe: a fresh small CNN is trained (train renders only) to land each glyph
    bitmap on the frozen concept vector of its UTF-8 character; top-1 over all characters on the
    held-out renders (same fixed split as the arm2+ glyph view)."""
    P = embed(net, data.stroke_classes, int8=int8).to(device)
    P = F.normalize(P, dim=-1)
    y = data.stroke_y.to(device)
    X = data.stroke_x(device)
    tr, te = data.stroke_train.to(device), data.stroke_test.to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    cnn = GlyphCNN().to(device)
    opt = torch.optim.AdamW(cnn.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps))
    for _ in range(steps):
        idx = tr[torch.randint(0, len(tr), (min(256, len(tr)),), device=device, generator=g)]
        loss = F.cross_entropy(cnn(X[idx]) @ P.T / 0.05, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    cnn.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(te), 512):
            idx = te[i:i + 512]
            correct += int(((cnn(X[idx]) @ P.T).argmax(1) == y[idx]).sum())
    # concept-space separability of the character prototypes themselves
    sim = P @ P.T
    sim.fill_diagonal_(-2)
    return {"stroke_top1": correct / max(1, len(te)), "stroke_classes": len(data.stroke_classes),
            "stroke_test": len(te), "stroke_proto_max_offdiag_cos": float(sim.max())}


@torch.no_grad()
def eval_bpc(net, texts, kind: str, bs: int = 64):
    """Next-byte bits per byte on held-out 256-byte windows (BPE: token NLL / bytes of the
    predicted tokens; H-Net: concept-bottleneck next-byte head)."""
    net.eval()
    bits = nb = 0.0
    for i in range(0, len(texts), bs):
        enc = net.encode(texts[i:i + bs])
        if kind == "bpe":
            b, n = net.bits_per_byte(enc)
        else:
            n = float((enc["lens"] - 1).clamp(min=0).sum())
            b = float(net.byte_losses(enc, recon=False)["next_byte"]) * n / math.log(2)
        bits += b
        nb += n
    return bits / max(1.0, nb)


def int8_latency(net, texts):
    """Device-path latency/bpc for H-Net arms: numpy integer reference on exported bytes."""
    import export_int8 as X

    tensors, scales = X.quantize_model(net)
    plhn = X.pack_plhn(tensors, scales)
    engr = None
    if net.engram is not None:
        es = scales["engram"]
        ed = np.clip(np.floor(net.engram.detach().cpu().double().numpy() / es + 0.5), -127, 127).astype(np.int8)
        engr = X.pack_engram(ed, es, np.zeros(M.ENGRAM_SLOTS, np.uint64), {})
    ref = M.PlhnIntNp(plhn, engr)
    lat, nb, nc = [], 0, 0
    for t in texts[:100]:
        t0 = time.perf_counter()
        data, spans, _ = ref.chunk_states(t)
        lat.append(time.perf_counter() - t0)
        nb += len(data)
        nc += len(spans)
    return {"int8_bytes_per_chunk": nb / max(1, nc), "latency_ms_int8_numpy_b1": 1000 * float(np.median(lat)),
            "plhn_bytes": len(plhn), "engram_bytes": len(engr) if engr else 0}


def param_breakdown(arm: str, net) -> dict:
    """Deployed (on-device) vs training-only parameters and the deployed RAM footprint."""
    total = M.count_params(net)
    if ARM_CFG[arm]["kind"] == "bpe":
        heads = sum(p.numel() for p in net.lm.parameters()) + 2  # LM head + two temperatures
        deploy = total - heads
        return {"params": total, "params_deploy": deploy, "params_train_heads": heads,
                "deploy_ram_bytes_fp32": 4 * deploy, "deploy_ram_bytes_int8": deploy}
    engram = net.engram.numel() if net.engram is not None else 0
    core = M.PLHN_PAYLOAD_BYTES
    return {"params": total, "params_deploy": core + engram, "params_plhn": core, "params_engram": engram,
            "params_train_heads": total - core - engram, "deploy_ram_bytes_fp32": 4 * (core + engram),
            "deploy_ram_bytes_int8": core + engram}


def evaluate(arm, net, data: Data, args, device):
    kind = ARM_CFG[arm]["kind"]
    if kind == "hnet":
        net.quant = False  # float (FP32) metrics; INT8 metrics come from the exact integer path below
    res = {"arm": arm, **param_breakdown(arm, net)}
    texts, labs = data.eval_texts(args.eval_windows, labels=True)
    res["next_byte_bpc"] = eval_bpc(net, texts, kind)
    res.update(eval_wsd(net, data.wsd))
    res.update(eval_xling(net, data.xling))
    res.update(eval_efficiency(net, texts[:200], device))
    res.update(eval_stroke(net, data, device, args.stroke_steps, args.seed))
    if kind == "hnet":
        res.update(eval_segmentation(net, data, texts, labs))
        res.update(int8_latency(net, texts[:200]))
        # INT8: exact integer device path (QAT-trained for arm3, post-training quantised for arm1/2)
        r8 = {"quant": "qat" if ARM_CFG[arm]["qat"] else "ptq"}
        net.quant = True
        r8["next_byte_bpc"] = eval_bpc(net, texts, kind)  # int8 encoder + float next-byte head
        net.quant = False
        r8.update(eval_wsd(net, data.wsd, int8=True))
        r8.update(eval_xling(net, data.xling, int8=True))
        r8.update(eval_stroke(net, data, device, args.stroke_steps, args.seed, int8=True))
        r8.update(eval_segmentation(net, data, texts, labs, int8=True))
        res.update({f"int8_{k}": v for k, v in r8.items()})
        net.quant = bool(ARM_CFG[arm]["qat"])
    return res


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def run_parallel(arms, out: Path, argv) -> None:
    """One process per arm on the same GPU (each arm is launch/CPU-bound, so concurrent arms fill
    the device); per-arm logs, merged summary.json."""
    import subprocess

    branch_path(out).unlink(missing_ok=True)  # arm3 must only see this run's arm2 snapshot
    procs = {}
    for arm in arms:
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
        cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--out-dir", str(out), "--worker-arm", arm]
        f = open(out / f"stdout_{arm}.log", "a")
        procs[arm] = (subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT), f)
        print(f"spawned {arm} pid {procs[arm][0].pid}", flush=True)
    failed = []
    for arm, (p, f) in procs.items():
        rc = p.wait()
        f.close()
        print(f"{arm} exit {rc}", flush=True)
        if rc:
            failed.append(arm)
    merge_summary(out)
    if failed:
        raise SystemExit(f"arms failed: {failed}")


def merge_summary(out: Path) -> dict:
    summary = {}
    for arm in ARMS:
        p = out / f"eval_{arm}.json"
        if p.exists():
            summary[arm] = json.loads(p.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    cols = ["next_byte_bpc", "wsd_top1", "xling_R@1", "xling_R@5", "xling_MRR", "stroke_top1", "bytes_per_chunk",
            "latency_ms_cpu_torch_b1"]
    print("arm".ljust(28) + "".join(c[:14].rjust(15) for c in cols))
    for arm, r in summary.items():
        print(arm.ljust(28) + "".join(f"{r.get(c, float('nan')):15.4f}" for c in cols))
    print(f"out: {out}", flush=True)
    return summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="all", help="comma list or 'all'")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--parallel", action="store_true", help="train all arms concurrently (one process each)")
    ap.add_argument("--worker-arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-lm", type=int, default=128)
    ap.add_argument("--batch-views", type=int, default=512)
    ap.add_argument("--batch-sense", type=int, default=256)
    ap.add_argument("--batch-glyph", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--view-buckets", type=int, default=4, help="length-sorted sub-batches per view encode")
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                    help="TF32 tensor-core matmuls during training only (eval/int8 always full fp32)")
    ap.add_argument("--engram-lr-mult", type=float, default=5.0)
    ap.add_argument("--engram-drop", type=float, default=0.1, help="arm3 training-time Engram slot dropout (views only)")
    ap.add_argument("--batch-lex", type=int, default=1024, help="arm3 Engram lexicon pairs per step (headword alone)")
    ap.add_argument("--batch-lex-ctx", type=int, default=512,
                    help="arm3 Engram lexicon pairs per step with the headword inside random corpus text")
    ap.add_argument("--w-lex", type=float, default=1.0, help="arm3 Engram lexicon InfoNCE (+ LeJEPA) weight")
    ap.add_argument("--lex-phase-steps", type=int, default=400,
                    help="arm3: first N of its own steps train only the lexicon + router + glyph view")
    ap.add_argument("--arm3-branch-frac", type=float, default=0.5,
                    help="arm3 warm-starts from arm2's snapshot at this fraction of steps (0 = from scratch)")
    ap.add_argument("--branch-wait", type=float, default=7200, help="max seconds arm3 waits for the arm2 snapshot")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--qat-start", type=float, default=0.3, help="fraction of steps before int8 QAT turns on (arm3)")
    ap.add_argument("--target-ratio", type=float, default=6.0)
    ap.add_argument("--w-ratio", type=float, default=0.03, help="H-Net ratio loss; only used when --w-boundary 0")
    ap.add_argument("--w-boundary", type=float, default=1.0, help="supervised word-boundary router loss (H-Net arms)")
    ap.add_argument("--boundary-margin", type=float, default=0.2)
    ap.add_argument("--w-recon", type=float, default=0.5)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-lejepa", type=float, default=1.0)
    ap.add_argument("--w-sense", type=float, default=1.5)
    ap.add_argument("--sense-margin", type=float, default=0.2)
    ap.add_argument("--w-glyph", type=float, default=0.5)
    ap.add_argument("--w-glyph-unif", type=float, default=1.0, help="uniformity of character prototypes (glyph arms)")
    ap.add_argument("--sigreg-lambda", type=float, default=0.02)
    ap.add_argument("--sense-eps", type=float, default=0.1)
    ap.add_argument("--bpe-merges", type=int, default=512)
    ap.add_argument("--bpe-train-texts", type=int, default=20000)
    ap.add_argument("--corpus-limit", type=int, default=None)
    ap.add_argument("--stroke-steps", type=int, default=2000)
    ap.add_argument("--eval-windows", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="synthetic data + 3 steps per arm")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    if args.smoke:
        if not args.worker_arm:
            args.data_dir = make_smoke_data(Path(tempfile.mkdtemp(prefix="pehnet-smoke-")))
            argv = [*argv, "--data-dir", str(args.data_dir)]
        args.steps, args.batch_lm, args.batch_views, args.batch_sense, args.batch_glyph = 3, 4, 6, 4, 8
        args.stroke_steps, args.log_every, args.bpe_merges, args.warmup = 3, 1, 32, 1
        args.qat_start, args.eval_windows, args.lex_phase_steps, args.batch_lex, args.batch_lex_ctx = 0.34, 20, 1, 8, 4
    out = args.out_dir or (ROOT / "runs" / (("smoke-" if args.smoke else "") + time.strftime("%Y%m%d-%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    arms = ARMS if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
    if args.parallel and not args.worker_arm:
        run_parallel(arms, out, argv)
        return
    if args.worker_arm:
        arms = [args.worker_arm]
    device = torch.device(args.device)
    data = Data(args.data_dir, args.seed, args.corpus_limit, device)
    tag = f"_{arms[0]}" if args.worker_arm else ""
    logf = open(out / f"train_log{tag}.jsonl", "a")

    def log(rec):
        line = json.dumps(rec, default=str, ensure_ascii=False)
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log({"event": "start", "arms": arms, "device": str(device),
         "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
         "corpus": len(data.corpus), "heldout_docs": len(data.heldout), "pairs": len(data.pairs),
         "lexical_bridge_views": data.n_bridge, "excluded_views": data.n_excluded_views,
         "excluded_docs": data.n_excluded_docs, "view_sentence_leaks": data.view_sentence_leaks,
         "text_table": len(data.text_list),
         "sense_pairs": len(data.sense), "sense_groups": len(data.sense_groups), "kinds": data.kinds,
         "wsd": len(data.wsd), "xling_q": len(data.xling["queries"]), "stroke": len(data.stroke_chars),
         "args": vars(args)})
    for arm in arms:
        t0 = time.time()
        net = train_arm(arm, data, args, device, log, out)
        train_sec = time.time() - t0
        ck = {"arm": arm, "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
              "config": ARM_CFG[arm], "args": {k: str(v) for k, v in vars(args).items()}}
        if ARM_CFG[arm]["kind"] == "bpe":
            ck["bpe_merges"] = net.bpe.merges
        tmp = out / f".{arm}.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, out / f"{arm}.pt")
        t1 = time.time()
        res = evaluate(arm, net, data, args, device)
        warm = int(args.steps * args.arm3_branch_frac) if ARM_CFG[arm]["engram"] and args.arm3_branch_frac > 0 else 0
        res.update({"train_sec": round(train_sec, 1), "eval_sec": round(time.time() - t1, 1), "steps": args.steps,
                    "own_steps": args.steps - warm, "warm_start": {"from": ARMS[2], "step": warm} if warm else None,
                    "device": str(device)})
        log({"event": "eval", **res})
        (out / f"eval_{arm}.json").write_text(json.dumps(res, indent=1))
    if not args.worker_arm:
        merge_summary(out)


if __name__ == "__main__":
    main()
