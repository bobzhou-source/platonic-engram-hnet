---
license: mit
library_name: generic
language:
- zh
- en
pipeline_tag: sentence-similarity
tags:
- mteb
- tinyml
- esp32
- byte-level
- cross-lingual
- int8
- sentence-similarity
model-index:
- name: Platonic-Engram-HNet-v2
  results:
  - task:
      type: STS
    dataset:
      type: mteb/stsbenchmark-sts
      name: MTEB STSBenchmark
      split: test
    metrics:
    - type: cos_sim_spearman
      value: 32.8
      verified: false
  - task:
      type: STS
    dataset:
      type: C-MTEB/STSB
      name: MTEB STSB (zh)
      split: test
    metrics:
    - type: cos_sim_spearman
      value: 49.0
      verified: false
  - task:
      type: STS
    dataset:
      type: C-MTEB/BQ
      name: MTEB BQ
      split: test
    metrics:
    - type: cos_sim_spearman
      value: 22.8
      verified: false
  - task:
      type: STS
    dataset:
      type: C-MTEB/ATEC
      name: MTEB ATEC
      split: test
    metrics:
    - type: cos_sim_spearman
      value: 10.5
      verified: false
  - task:
      type: STS
    dataset:
      type: google-research-datasets/paws-x
      name: PAWS-X (zh)
      split: test
    metrics:
    - type: cos_sim_spearman
      value: 8.2
      verified: false
---

# Platonic-Engram-HNet

**No Room for a Tokenizer: Running a Bilingual Neural Reader in 95 Kilobytes of Microcontroller RAM**

Author: **Robert Zhou** ([robertzhou.dev](https://robertzhou.dev/work/platonic-engram-hnet/) · [GitHub](https://github.com/bobzhou-source/platonic-engram-hnet) · [ORCID 0009-0009-2155-7215](https://orcid.org/0009-0009-2155-7215))

When you read a book on a cheap e-reader (mine runs on an ESP32-S3 microcontroller with 8 MB of RAM), you want three simple things while you read:

1. **Tap a word in English or Chinese** and get the right definition for how it is used in that sentence.
2. **Ask "Who is this character again?"** on page 40 and get a recap of what they have done *up to page 40*, without spoiling page 300.
3. **Get short margin notes** that sum up the scene without making up fake character names or stopping mid-sentence.

A normal AI model can't even load its dictionary on a chip this small: a 32,000-word vocabulary table alone takes more memory than the e-reader has free. **Platonic-Engram-HNet** makes all three features work in **1.1 MB of RAM** by keeping a tiny 95 KB byte reader in memory and streaming the dictionary and phrase bank off the SD card the e-reader already uses to store books.

## What ships in the box

| Feature | What it does for the reader | Size |
|---|---|---:|
| **Word sense & bilingual reader** (`platonic-hnet.bin` + `engram.bin`) | Splits English and Chinese text into words using 95 KB of RAM, then looks up each word's meaning in a 4.2 MB table on the SD card | 95 KB RAM + 4.2 MB SD |
| **Spoiler-free character memory** (`cspw.bin`) | Keeps track of the characters across the book and refuses to read past your current page: 0% spoilers across 9,169 BookSum questions | 394 KB |
| **Margin notes** (`pe-writer.bin` + `writer-clauses.bin`) | Writes short notes by picking whole phrases from a 4,096-clause bank, with 0% invented names and 0% broken sentences across 4,226 public tests | 642 KB RAM + 1 MB SD |
| **E-ink screen controller** (`LiquidRefresh`, part of the [firmware](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/firmware)) | Decides when the e-ink screen needs a full flash and when a fast partial update is enough | 339 bytes |

All files are in `weights/`, with checksums in `weights/SHA256SUMS`. `weights/booksum-story-graph.idx` is a character map of 21 public-domain books from the BookSum benchmark, so the spoiler-free recap has something to read.

## Try it in 30 seconds

```bash
pip install numpy
python quickstart.py
```

It checks the weight files, splits an English and a Chinese sentence into words (and shows that a money "bank" and a river "bank" come out differently), gives a spoiler-safe character recap from a public-domain book, and writes a margin note, all with the same integer math the e-reader runs.

To get a word-meaning vector for your own text:

```python
import plhn_model as M

enc = M.PlhnIntNp(open("weights/platonic-hnet.bin", "rb").read(),
                  open("weights/engram.bin", "rb").read())
a = enc.forward("The river bank was flooded after the storm.")   # 128 small integers
b = enc.forward("河岸在暴风雨后被淹没了。")
similarity = sum(x * y for x, y in zip(a, b)) / (127 * 127)
```

## Good to know

- **It is built for a tiny chip, not for accuracy.** It matches or beats a normal model squeezed into the same memory on 11 of 12 tests, but it loses to big models and often to plain keyword search. It is not a general-purpose embedding model.
- **It can't match whole sentences across languages.** It knows that 河岸 means "river bank", but finding the English translation of a whole Chinese sentence works only 3% of the time.
- **Very long words get split.** A word of five or more Chinese characters is always cut into pieces.
- **The scores in the metadata above** come from my own re-run of the MTEB tests, not the official `mteb` package, so they are marked unverified.

## What is not published

The e-reader also keeps a character map and a concept index built from my own book library. Those are personal and are not here. The published `engram.bin` is also slightly different from the one on my device: 529 short glosses that came from my private reading were blanked, and only glosses that match a public dictionary entry word for word were kept. The model's numbers are unchanged, so it gives exactly the same results.

## Where the technical details live

- [ARCHITECTURE.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/ARCHITECTURE.md): how each model works, the file formats, and the memory budget.
- [BENCHMARKS.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/BENCHMARKS.md): the full evaluation, including where a 95 KB model still loses.
- [`src/verify_c_parity.py`](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/src/verify_c_parity.py): a test that checks the Python code and the integer-only C version of the e-reader code give bit-for-bit identical answers.

## License and attribution

Code and weights are MIT, with two exceptions:

- **The glosses inside `engram.bin`** are adapted from CC-CEDICT (CC BY-SA 4.0, © CC-CEDICT contributors, MDBG) and Open English WordNet (CC BY 4.0, © the Open English WordNet team, derived from Princeton WordNet 3.1).
- **`booksum-story-graph.idx`** is derived from BookSum (Kryściński et al., 2021). Its event notes are fragments of BookSum's chapter summaries, which come from study-guide sites (e.g. SparkNotes, GradeSaver, CliffsNotes, Shmoop). Check BookSum's terms before redistributing.
