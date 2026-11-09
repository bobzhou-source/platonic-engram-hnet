"""Rebuild the public regression goldens in ``golden/`` from the published weights and public inputs only.

    python tools/make_public_goldens.py

The original export-time goldens also contained passages and book titles from the owner's private reading
library, so they are not published. This script regenerates the same file formats from public inputs:

* ``golden/golden.jsonl`` + ``golden/golden_vectors.bin``: the fixed ``GOLDEN_TEXTS`` of ``src/export_int8.py``
  (English, Chinese, mixed script, emoji, long runs) through the NumPy integer encoder.
* ``golden/cspw_golden.bin``: 32 device scene selections from ``weights/booksum-story-graph.idx`` plus 32
  random int8 scene sets, woven by the integer CS-PW reference.
* ``golden/pe_writer/pe_writer_golden.bin``: 22 BookSum entities (scene vectors from the public story graph,
  public-domain titles as echoes) plus the 10 edge cases of ``src/export_pe_writer_int8.py``.

``src/verify_c_parity.py`` and ``src/verify_pe_writer_parity.py`` then check every golden against the
independent C / C++ references and the firmware kernel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cspw_model as CM  # noqa: E402
import export_int8 as X  # noqa: E402
import model as M  # noqa: E402
import pe_writer_model as PW  # noqa: E402

W = ROOT / "weights"
OUT = ROOT / "golden"
N_PEWR = 32
TITLES = ["Far from the Madding Crowd", "Pride and Prejudice", "Moby-Dick", "Middlemarch", "Jane Eyre",
          "Great Expectations", "The Scarlet Letter", "Wuthering Heights", "Emma", "Dracula", "Frankenstein"]


def cut(s, n: int = 31) -> bytes:
    return (s if isinstance(s, bytes) else s.encode("utf-8"))[:n]


def concept_goldens(enc: M.PlhnIntNp) -> None:
    rows = [{"text": t, "hex": M.text_bytes(t).hex(), "vec": enc.forward(t),
             "spans": [list(s) for s in enc.chunk_states(t)[1]]} for t in X.GOLDEN_TEXTS]
    (OUT / "golden.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (OUT / "golden_vectors.bin").write_bytes(
        X.pack_golden_vectors([(M.text_bytes(t), enc.forward(t)) for t in X.GOLDEN_TEXTS]))


def cspw_goldens(w: CM.CspwWeights, graph: CM.StoryGraph, n: int = 64) -> None:
    rng = np.random.default_rng(7)
    sel = []
    for e in graph.entities:
        evs = graph.events[e["event_start"]:e["event_start"] + e["event_count"]]
        for cur in sorted({x["cutoff"] for x in evs})[-2:]:
            hit = graph.lookup(e["term_hash"], e["book_path_hash"], cur)
            if hit is not None:
                sc, ec = CM.select_scenes(hit[1])
                if sc:
                    sel.append(([x["vec"].tolist() for x in sc], [x["chapter_or_page"] for x in sc],
                                ec["vec"].tolist() if ec is not None else None))
    sel = sel[: n // 2]
    pool = [ev["vec"].tolist() for ev in graph.events if ev["vec"].any()]
    while len(sel) < n:
        m = int(rng.integers(1, CM.MAX_SCENES + 1))
        sc = [pool[int(rng.integers(len(pool)))] if rng.random() < 0.7
              else rng.integers(-127, 128, CM.D_SLOT).tolist() for _ in range(m)]
        pg = sorted(int(x) for x in rng.integers(0, 0x10000, m))
        ec = pool[int(rng.integers(len(pool)))] if rng.random() < 0.5 else None
        sel.append((sc, pg, ec))
    cases = [CM.pack_golden_case(sc, pg, ec, CM.int8_cspw_weave(w, sc, pg, ec)) for sc, pg, ec in sel]
    (OUT / "cspw_golden.bin").write_bytes(CM.pack_golden(cases))


def pe_writer_goldens(ref: PW.PeWriterIntNp, graph: CM.StoryGraph) -> None:
    rng = np.random.default_rng(1234)

    def vec(kind):
        if kind == "zero":
            return np.zeros(PW.D, np.int64)
        if kind == "sat":
            return rng.choice([-127, 127], PW.D).astype(np.int64)
        if kind == "raw":
            return rng.integers(-128, 128, PW.D).astype(np.int64)
        return PW.normalize_i8(rng.integers(-60, 61, PW.D))

    inputs, seen = [], set()
    for e in graph.entities:
        name = e["en_name_or_role"]
        if not name or name in seen or e["event_count"] < 2:
            continue
        seen.add(name)
        evs = graph.events[e["event_start"]:e["event_start"] + e["event_count"]]
        other = graph.events[int(rng.integers(len(graph.events)))]
        inputs.append({"c_page": PW.normalize_i8(evs[-1]["vec"]), "c_echo": PW.normalize_i8(other["vec"]),
                       "subject": cut(name), "role": cut(evs[-1]["target_or_loc"] if len(inputs) % 3 else ""),
                       "echo_title": cut(TITLES[len(inputs) % len(TITLES)])})
        if len(inputs) == N_PEWR - 10:
            break
    base = {"subject": b"Winston", "role": b"the clerk", "echo_title": b"Moby-Dick"}
    edge = [
        ("unit", "unit", b"", base["role"], base["echo_title"]),
        ("unit", "unit", base["subject"], b"", base["echo_title"]),
        ("unit", "unit", base["subject"], base["role"], b""),
        ("unit", "zero", b"", b"", b""),
        ("zero", "zero", b"Julia", b"the lover", b"Middlemarch"),
        ("sat", "sat", b"Raskolnikov", b"the student", b"Crime and Punishment"),
        ("raw", "raw", b"Ishmael", b"narrator", b"Heart of Darkness"),
        ("unit", "unit", cut("孙悟空大闹天宫的猴王齐天大圣"), cut("花果山的美猴王"), cut("西游记")),  # split codepoint
        ("unit", "raw", cut("林黛玉"), b"", cut("红楼梦 · Dream of the Red Chamber")),
        ("unit", "unit", b"A" * 31, b"r" * 31, b"E" * 31),
    ]
    for cp, ce, s, r, t in edge:
        inputs.append({"c_page": vec(cp), "c_echo": vec(ce), "subject": s, "role": r, "echo_title": t})
    assert len(inputs) == N_PEWR, len(inputs)
    cases = []
    for c in inputs:
        r = ref.generate(c["c_page"], c["c_echo"], c["subject"], c["role"], c["echo_title"])
        cases.append(c | {"c_plan": r["c_plan"], "buckets": r["buckets"], "tokens": r["tokens"], "text": r["text"]})
    (OUT / "pe_writer").mkdir(parents=True, exist_ok=True)
    (OUT / "pe_writer" / "pe_writer_golden.bin").write_bytes(PW.pack_golden(cases))
    for c in cases[:4]:
        print("  pe-writer:", c["text"].decode("utf-8", "replace") if isinstance(c["text"], bytes) else c["text"])


def main():
    OUT.mkdir(exist_ok=True)
    enc = M.PlhnIntNp((W / "platonic-hnet.bin").read_bytes(), (W / "engram.bin").read_bytes())
    graph = CM.StoryGraph((W / "booksum-story-graph.idx").read_bytes())
    concept_goldens(enc)
    cspw_goldens(CM.CspwWeights((W / "cspw.bin").read_bytes()), graph)
    pe_writer_goldens(PW.PeWriterIntNp((W / "pe-writer.bin").read_bytes(), (W / "writer-clauses.bin").read_bytes()),
                      graph)
    print("wrote", sorted(str(p.relative_to(ROOT)) for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
