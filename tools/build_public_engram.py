"""Build the public-gloss Engram (`weights/engram.bin`) from a device `engram.bin`.

    python tools/build_public_engram.py path/to/device/engram.bin [--out weights/engram.bin]

Only the 32-byte gloss column changes. Span hashes, int8 concept deltas, the header and the slot count stay
byte-identical, so encoder outputs are unchanged (the encoder reads span_hash + concept_delta only).

A gloss is kept only if it is provably a verbatim public-dictionary entry:
  * CC-CEDICT: the slot's span_hash equals FNV-1a64 of a CEDICT headword (>= 2 characters) and the stored gloss
    equals that headword's first-sense gloss as written by the exporter; or
  * Open English WordNet: the stored gloss is `~lemma: <first sense>` for an OEWN lemma, or the slot's span_hash
    equals FNV-1a64 of a hyphenated OEWN headword whose first sense is the stored gloss.
Every other gloss is blanked. That removes the curated novel terms, speech tags with character names, clan and
person names, transliterations, and the composed (`a + b`, `not x`) glosses that were derived from the private
reading corpus.
"""

import argparse
import hashlib
import json
import pathlib
import re
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import export_int8 as X  # noqa: E402
import model as M  # noqa: E402

HDR, SLOT, G0 = 512, 64, 32



def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("device_engram", type=pathlib.Path, help="device engram.bin written by src/export_int8.py")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "weights" / "engram.bin")
    ap.add_argument("--dict-dir", type=pathlib.Path, default=X.DICT_DIR, help="CC-CEDICT + OEWN StarDict files")
    args = ap.parse_args()
    DEPLOYED, OUT = args.device_engram, args.out
    X.DICT_DIR = args.dict_dir
    blob = bytearray(DEPLOYED.read_bytes())
    magic, version, num_slots, slot_bytes, _ = M.ENGR_HEADER.unpack_from(blob, 0)
    assert (magic, num_slots, slot_bytes) == (M.ENGR_MAGIC, 65536, SLOT) and len(blob) == HDR + num_slots * SLOT

    book = X.GlossBook()
    trunc = lambda s: X.gloss_trunc(s)  # noqa: E731  exporter's exact byte form
    zh_by_hash = {M.fnv1a64(w.encode()): trunc(g) for w, g in book.zh.items() if len(w) >= 2 and g}
    en_hyph = {M.fnv1a64(w.encode()): trunc(g) for w, g in book.en.items() if "-" in w and g}
    en_lemma = {trunc(f"~{l}: {g}") for l, g in book.en.items() if len(l) >= 4 and g}

    kept = {"cedict": 0, "oewn_lemma": 0, "oewn_hyphen": 0}
    blanked, samples = 0, []
    for i in range(num_slots):
        off = HDR + i * SLOT
        gloss = bytes(blob[off + G0: off + SLOT]).split(b"\0")[0]
        if not gloss:
            continue
        h = struct.unpack_from("<Q", blob, off)[0]
        if zh_by_hash.get(h) == gloss:
            kept["cedict"] += 1
        elif gloss in en_lemma:
            kept["oewn_lemma"] += 1
        elif en_hyph.get(h) == gloss:
            kept["oewn_hyphen"] += 1
        else:
            blanked += 1
            if len(samples) < 40:
                samples.append(gloss.decode("utf-8", "replace"))
            blob[off + G0: off + SLOT] = bytes(SLOT - G0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(blob)
    deployed = DEPLOYED.read_bytes()
    same_non_gloss = all(blob[HDR + i * SLOT: HDR + i * SLOT + G0] == deployed[HDR + i * SLOT: HDR + i * SLOT + G0]
                         for i in range(num_slots)) and blob[:HDR] == deployed[:HDR]
    report = {
        "source": DEPLOYED.name, "source_sha256": hashlib.sha256(deployed).hexdigest(),
        "output": OUT.name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
        "header_hash_delta_columns_identical": same_non_gloss,
        "glosses_kept": kept, "glosses_kept_total": sum(kept.values()), "glosses_blanked": blanked,
    }
    (OUT.parent / "engram.public-gloss.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "blanked_examples"}, indent=1))
    print(f"blanked {blanked} glosses; first examples (printed only, not saved):", samples[:10])


if __name__ == "__main__":
    main()
