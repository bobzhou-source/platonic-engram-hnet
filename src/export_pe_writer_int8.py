"""Export a trained PE-Writer arm to ``pe-writer.bin`` (PEWR v1), ``writer-clauses.bin`` (WCLS v1) and
``pe_writer_golden.bin`` (PWGD, 32 golden cases computed by the NumPy integer reference).

    python src/export_pe_writer_int8.py --run runs/pe_writer/smoke            # -> golden/pe_writer/
    python src/export_pe_writer_int8.py --run runs/pe_writer/<run> --install  # also copy to weights/
    python src/export_pe_writer_int8.py --run runs/pe_writer/arm4_frame_locked --arm 4 --install

``--install`` also refreshes the three files in the top-level ``golden/`` (read first by the firmware ctest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_writer_model as PW  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "golden" / "pe_writer"
GOLDEN_ROOT = ROOT / "golden"
MODELS = ROOT / "weights"
N_GOLDEN = 32


def load_arm(run: Path, arm: int) -> PW.PeWriterQAT:
    ck = torch.load(run / f"arm{arm}.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    st = ck["state"]
    m = PW.PeWriterQAT(st["centroids"].numpy(), use_planner=cfg["use_planner"], use_bank=cfg["use_bank"])
    m.load_state_dict(st)
    return m.eval()


def _cut(s: str, n: int = 31) -> bytes:
    """Raw byte cut (may split a codepoint on purpose: the device reads char[32] fields verbatim)."""
    return s.encode("utf-8")[:n]


def golden_inputs(run: Path, seed: int) -> list[dict]:
    meta = np.load(run / "val_meta.npz")
    rng = np.random.default_rng(seed)
    n_val = len(meta["c_page"])
    cases = []
    for i in range(min(N_GOLDEN - 10, n_val)):
        cases.append({"c_page": meta["c_page"][i].astype(np.int64), "c_echo": meta["c_echo"][i].astype(np.int64),
                      "subject": _cut(str(meta["subject"][i])), "role": _cut(str(meta["role"][i])),
                      "echo_title": _cut(str(meta["echo_title"][i]))})

    def vec(kind):
        if kind == "zero":
            return np.zeros(PW.D, np.int64)
        if kind == "sat":
            return rng.choice([-127, 127], PW.D).astype(np.int64)
        if kind == "raw":
            return rng.integers(-128, 128, PW.D).astype(np.int64)
        return PW.normalize_i8(rng.integers(-60, 61, PW.D))

    base = cases[0] if cases else {"subject": b"Winston", "role": b"the clerk", "echo_title": b"Moby-Dick"}
    edge = [
        ("unit", "unit", b"", base["role"], base["echo_title"]),            # no subject -> <ENT_SUBJECT> masked
        ("unit", "unit", base["subject"], b"", base["echo_title"]),         # empty role -> "this concept"
        ("unit", "unit", base["subject"], base["role"], b""),               # no echo -> <BOOK_ECHO> masked
        ("unit", "zero", b"", b"", b""),
        ("zero", "zero", b"Julia", b"the lover", b"Middlemarch"),
        ("sat", "sat", b"Raskolnikov", b"the student", b"Crime and Punishment"[:31]),
        ("raw", "raw", b"Ishmael", b"narrator", b"Heart of Darkness"),
        ("unit", "unit", _cut("孙悟空大闹天宫的猴王齐天大圣"), _cut("花果山的美猴王"), _cut("西游记")),  # split codepoint
        ("unit", "raw", _cut("林黛玉"), b"", _cut("红楼梦 · Dream of the Red Chamber")),
        ("unit", "unit", b"A" * 31, b"r" * 31, b"E" * 31),
    ]
    for cp, ce, s, r, e in edge:
        cases.append({"c_page": vec(cp), "c_echo": vec(ce), "subject": s, "role": r, "echo_title": e})
    while len(cases) < N_GOLDEN:
        cases.append({"c_page": vec("unit"), "c_echo": vec("unit"), "subject": b"Darcy", "role": b"the suitor",
                      "echo_title": b"Pride and Prejudice"})
    return cases[:N_GOLDEN]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "pe_writer" / "smoke")
    ap.add_argument("--arm", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--install", action="store_true", help=f"also copy both model files to {MODELS}")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)
    if args.arm < 2:
        print(f"warning: arm {args.arm} was trained without the WCLS bank; the device kernel always uses it")
    model = load_arm(args.run, args.arm)
    vocab = json.load(open(args.run / "vocab.json"))
    tok = PW.Tokenizer(vocab)
    pewr = PW.pack_pewr(model.export_tensors(), tok.vocab_bytes())
    wcls = (args.run / "writer-clauses.bin").read_bytes()
    PW.parse_pewr(pewr)
    PW.Wcls(wcls)

    ref = PW.PeWriterIntNp(pewr, wcls)
    tref = PW.PeWriterIntTorch(pewr, wcls)
    inputs = golden_inputs(args.run, args.seed)
    cases = []
    for c in inputs:
        r = ref.generate(c["c_page"], c["c_echo"], c["subject"], c["role"], c["echo_title"])
        cases.append(c | {"c_plan": r["c_plan"], "buckets": r["buckets"], "tokens": r["tokens"], "text": r["text"]})
    tres = tref.generate(np.stack([c["c_page"] for c in inputs]), np.stack([c["c_echo"] for c in inputs]),
                         [c["subject"] for c in inputs], [c["role"] for c in inputs],
                         [c["echo_title"] for c in inputs])
    bad = sum(not (np.array_equal(a["c_plan"], b["c_plan"]) and a["buckets"] == b["buckets"] and
                   a["tokens"] == b["tokens"] and a["text"] == b["text"]) for a, b in zip(cases, tres))
    if bad:
        raise SystemExit(f"torch-int vs numpy golden mismatch on {bad} cases")
    golden = PW.pack_golden(cases)
    assert len(PW.unpack_golden(golden)) == N_GOLDEN

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pe-writer.bin").write_bytes(pewr)
    (args.out / "writer-clauses.bin").write_bytes(wcls)
    (args.out / "pe_writer_golden.bin").write_bytes(golden)
    manifest = {
        "run": str(args.run), "arm": args.arm,
        "files": {n: {"bytes": len(b), "sha256": sha(b)} for n, b in
                  (("pe-writer.bin", pewr), ("writer-clauses.bin", wcls), ("pe_writer_golden.bin", golden))},
        "shifts": {n: int(PW.parse_pewr(pewr)[0][n][0]) for n in PW.LIN_NAMES},
        "golden_samples": [{"subject": c["subject"].decode("utf-8", "replace"), "buckets": c["buckets"],
                            "n_tokens": sum(1 for v in c["tokens"] if v),
                            "text": c["text"].decode("utf-8", "replace")} for c in cases[:6]],
    }
    json.dump(manifest, open(args.out / "pe_writer_manifest.json", "w"), indent=1, ensure_ascii=False)
    if args.install:
        MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.out / "pe-writer.bin", MODELS / "pe-writer.bin")
        shutil.copyfile(args.out / "writer-clauses.bin", MODELS / "writer-clauses.bin")
        if args.out.resolve() != GOLDEN_ROOT.resolve():
            for n in ("pe-writer.bin", "writer-clauses.bin", "pe_writer_golden.bin"):
                shutil.copyfile(args.out / n, GOLDEN_ROOT / n)
    print(json.dumps({k: v for k, v in manifest.items() if k != "golden_samples"}, indent=1))
    for s in manifest["golden_samples"]:
        print(f"  {s['subject']!r} b={s['buckets']} -> {s['text']!r}")
    print(f"wrote {args.out}" + (f" and installed into {MODELS}" if args.install else ""))


if __name__ == "__main__":
    main()
