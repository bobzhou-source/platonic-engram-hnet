"""Platonic-Engram-HNet v2 quickstart: the three device paths on the published int8 weights, NumPy only.

    python examples/quickstart.py        # GitHub layout (src/ + weights/)
    python quickstart.py                 # Hugging Face layout (*.py + weights/ side by side)

(a) Tokenizer-free byte routing + 128-d int8 concept vectors on English and Chinese text.
(b) Spoiler-free entity lookup with CS-PW v2 on the public BookSum story graph at an early cutoff P and a
    later cutoff P'.
(c) Frame-locked concept-to-text generation with PE-Writer v2.

Every call below is the bit-exact integer reference of the firmware kernel (`src/verify_c_parity.py` and
`src/verify_pe_writer_parity.py` check it against the C/C++ kernels).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE / "weights").is_dir():  # Hugging Face layout
    ROOT = HERE
    sys.path.insert(0, str(HERE))
    import plhn_model as M  # noqa: E402
else:  # GitHub layout
    ROOT = HERE.parent
    sys.path.insert(0, str(ROOT / "src"))
    import model as M  # noqa: E402
import cspw_model as CM  # noqa: E402
import pe_writer_model as PW  # noqa: E402

W = ROOT / "weights"


def check_weights() -> None:
    sums = dict(line.split()[::-1] for line in (W / "SHA256SUMS").read_text().splitlines() if line.strip())
    for name, digest in sums.items():
        got = hashlib.sha256((W / name).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{name}: sha256 {got} != SHA256SUMS {digest}")
    print(f"weights OK: {len(sums)} files match SHA256SUMS")


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def demo_encoder(enc) -> None:
    print("\n(a) byte routing + 128-d int8 concepts")
    for text in ("The bank raised interest rates again.", "他在银行工作，每天很早出门。"):
        data, spans, _ = enc.chunk_states(text)
        chunks = [data[a:e].decode("utf-8", "replace") for a, e in spans]
        vec = enc.forward(text)
        print(f"  {text}\n    {len(data)} bytes -> {len(spans)} chunks: {' | '.join(chunks)}")
        print(f"    int8[128]: {vec[:12]} ...")
    pairs = [("bank (money)", "The bank raised interest rates again.", "他在银行工作，每天很早出门。"),
             ("bank (river)", "We sat on the river bank and watched the boats.", "我们坐在河岸上看船。")]
    vecs = {k: (enc.forward(en), enc.forward(zh)) for k, en, zh in pairs}
    for k, (en, zh) in vecs.items():
        print(f"  cos(en, zh) {k}: {cos(en, zh):+.3f}")
    print(f"  cos(money en, river en):  {cos(vecs['bank (money)'][0], vecs['bank (river)'][0]):+.3f}")


def demo_cspw(graph: CM.StoryGraph, w: CM.CspwWeights) -> None:
    print("\n(b) CS-PW v2 spoiler-free lookup (public BookSum story graph)")
    # an entity with several events spread over the book
    ent = max(graph.entities, key=lambda e: (len({v["cutoff"] for v in
                                                   graph.events[e["event_start"]:e["event_start"] + e["event_count"]]}),
                                             -e["term_hash"]))
    cuts = sorted({v["cutoff"] for v in graph.events[ent["event_start"]:ent["event_start"] + ent["event_count"]]})
    early, late = cuts[0], cuts[-1]
    name = ent["en_name_or_role"]
    print(f"  entity {name!r}: {len(cuts)} distinct event cutoffs, first appears at offset {ent['first_cutoff']}")
    for label, cur in (("P  (before first appearance)", max(0, ent["first_cutoff"] - 1)),
                       ("P  (early)", early), ("P' (late)", late)):
        hit = graph.lookup(ent["term_hash"], ent["book_path_hash"], cur)
        if hit is None:
            print(f"  cutoff {label} = {cur}: no recap (spoiler gate: nothing is revealed yet)")
            continue
        text, _ = CM.weave_refresher(w, hit[0], hit[1], name)
        newest = max(e["cutoff"] for e in hit[1])
        print(f"  cutoff {label} = {cur}: {len(hit[1])} eligible events, newest at {newest} <= {cur}")
        print(f"    {text}")


def demo_writer(enc, pew: PW.PeWriterIntNp) -> None:
    print("\n(c) PE-Writer v2 frame-locked concept-to-text")
    page = ("Gabriel Oak watched the flock on the hill while the storm gathered, then ran to save the ricks "
            "before the rain could ruin the harvest.")
    echo = "Ahab paces the deck at night, unable to forget the white whale that took his leg."
    r = pew.generate(np.asarray(enc.forward(page), np.int64), np.asarray(enc.forward(echo), np.int64),
                     b"Gabriel Oak", b"the shepherd", b"Moby-Dick")
    text = r["text"].decode("utf-8", "replace") if isinstance(r["text"], bytes) else r["text"]
    print(f"  page:  {page}\n  echo:  Moby-Dick\n  clause buckets {r['buckets']}, {len(r['tokens'])} tokens")
    print(f"  riff:  {text}")


def main() -> None:
    check_weights()
    enc = M.PlhnIntNp((W / "platonic-hnet.bin").read_bytes(), (W / "engram.bin").read_bytes())
    demo_encoder(enc)
    demo_cspw(CM.StoryGraph((W / "booksum-story-graph.idx").read_bytes()), CM.CspwWeights((W / "cspw.bin").read_bytes()))
    demo_writer(enc, PW.PeWriterIntNp((W / "pe-writer.bin").read_bytes(), (W / "writer-clauses.bin").read_bytes()))


if __name__ == "__main__":
    main()
