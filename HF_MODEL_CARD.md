---
license: mit
library_name: generic
language:
- zh
- en
pipeline_tag: sentence-similarity
tags:
- deepseek-engram
- h-net
- byte-latent-transformer
- liquid-ai
- lfm2
- liquid-cfc
- qwen3
- le-jepa
- platonic-representation
- tinyml
- esp32-s3
- int8
- cross-lingual
- sentence-similarity
---

# Platonic-Engram-HNet

**Running DeepSeek Engram, H-Net Byte Routing, and Liquid AI (LFM2 / CfC) in 1.1 MB of Microcontroller Weights — Distilled from Qwen3-32B into Bit-Exact `int8` C11/C++20 Firmware for an ESP32-S3 E-Reader**

[robertzhou.dev](https://robertzhou.dev/) · [GitHub Repository](https://github.com/bobzhou-source/platonic-engram-hnet) · [Architecture](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/ARCHITECTURE.md) · [Benchmarks](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/BENCHMARKS.md)

I wanted three things on the **[Xteink X4 Pro](https://www.xteink.com)** e-ink reader I carry around (a dual-core 240 MHz **ESP32-S3** microcontroller with 8 MB of PSRAM running a fork of [CrossPoint Reader](https://github.com/CrossPoint-Reader) / [PapyriX](https://papyrix.com)):
1. Hold any English or Chinese word and get the meaning it has **in that sentence**.
2. Ask *"who is this character again?"* on page 40 **without spoiling page 300**.
3. Get short margin notes that **never invent a character name or cut off mid-sentence**.

Shrinking a standard Transformer below 1 MB fails in all three places:
- **The tokenizer eats all your RAM.** As Maxime Labonne (Liquid AI) showed in [Everything I Learned Training Frontier Small Models](https://www.slideshare.net/slideshow/everything-i-learned-training-frontier-small-models/286935677), the vocabulary embedding table alone takes **63% of the parameters** in Gemma 3 270M (`29%` in Qwen3.5-0.8B, `19%` in LFM2.5-350M). At 32,000 tokens and width 128, that table is **4.1 MB even at `int8`** (`16.4 MB` in `fp32`) before a single layer runs.
- **Standard RAG spoils the book.** Searching across a whole book means a lookup on Chapter 3 pulls plot twists from Chapter 20 ([Living the Novel](https://arxiv.org/abs/2512.07474)).
- **Sub-megabyte token-by-token decoders get stuck in "doom loops."** Labonne's slides documented `15%` repetition doom loops in small models before post-training; at `< 1 MB`, my token-by-token micro-decoder broke off mid-sentence on **51.2%** of [BookSum](https://arxiv.org/abs/2105.08209) notes and invented proper nouns on **26.1%**.

---

## How I Built It (Hardware–Model Co-Design)

Instead of cramming a standard Transformer into SRAM/PSRAM, I combined recent ideas from **DeepSeek**, **Liquid AI**, **Cartesia / CMU**, **Meta**, and **Yann LeCun's JEPA line of work**, distilled from **[Qwen3-32B](https://arxiv.org/abs/2505.09388)** and **[Qwen3-Embedding-0.6B](https://arxiv.org/abs/2506.05176)** on H200 and RTX PRO 6000 Blackwell GPUs across 14 controlled ablation arms.

The name **Platonic-Engram-HNet** comes from the three pieces of the encoder:
- **H-Net ([Hwang et al., 2025](https://arxiv.org/abs/2507.07955)) & [Byte Latent Transformer](https://arxiv.org/abs/2412.09871):** A **95 KB** causal UTF-8 byte router in PSRAM (`platonic-hnet.bin`) that learns English and Chinese word boundaries directly from raw bytes—no tokenizer or vocabulary table in RAM. Following Labonne's on-device operator ranking (`ShortConv < SWA < GDN < GLA < GQA`), the backbone uses two **[Liquid AI LFM2](https://arxiv.org/abs/2511.23404)** double-gated `ShortConv` blocks, one [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) block, and a 2-loop weight-tied **[Ouro](https://arxiv.org/abs/2510.25741)** core.
- **DeepSeek Engram ([DeepSeek-AI, 2026](https://arxiv.org/abs/2601.07372)) & [Tokenizer-Agnostic Engram](https://arxiv.org/abs/2607.29065):** Word and phrase memory is moved off RAM onto the SD card (`engram.bin`, **4.2 MB**, 65,536 FNV-1a hashed n-gram & bigram slots). Each word chunk costs a single **$O(1)$ 64-byte SD slot read**.
- **Platonic JEPA ([Balestriero & LeCun, 2025](https://www.alphaxiv.org/pdf/2610.39302); [LeJEPA / SIGReg](https://arxiv.org/abs/2511.08544)):** Trained with an isotropic Gaussian regularizer (`SIGReg`), [InfoNCE](https://arxiv.org/abs/1807.03748), and [Matryoshka](https://arxiv.org/abs/2205.13147) `Qwen3` distillation so matching English and Chinese concepts land on the same 128-d `int8` vector.

On top of that 95 KB encoder sit three tiny modules:
1. **Spoiler-Free Character Memory (`cspw.bin`, 394 KB PSRAM):** Built on four **[Liquid AI Closed-Form Continuous-Time (CfC)](https://arxiv.org/abs/2106.13898)** memory slots that decay with page distance ($2^{-\Delta p / \tau}$) behind a hard causal gate on your exact byte offset in the book.
2. **Clause-Locked Margin Writer (`pe-writer.bin`, 642 KB PSRAM + `writer-clauses.bin`, 1.0 MB SD):** Inspired by [Meta's Large Concept Models](https://arxiv.org/abs/2412.08821) and Labonne's decoupled Top-$K$ KD + loop-rejecting [DPO](https://arxiv.org/abs/2305.18290) recipe. After DPO cut 3-gram repetition loops from `9.25% → 3.50%` on validation, I locked the output grammar so a 3-step `Ouro` concept planner selects whole verified clauses from a 4,096-clause SD bank containing zero proper nouns.
3. **Physical E-Ink Refresh Controller (339 bytes in flash/SRAM):** A 16-unit **[Liquid CfC](https://arxiv.org/abs/2106.13898)** network trained on camera-calibrated ghosting ladders and 19 reading logs, with on-device [`e-prop` eligibility traces](https://arxiv.org/abs/1901.09049) that adapt when you press manual refresh.

![Memory map](https://raw.githubusercontent.com/bobzhou-source/platonic-engram-hnet/main/docs/memory-map.svg)

---

## Results at a Glance (and Where a 95 KB Model Still Loses)

Every number below runs the exact [int8 QAT](https://arxiv.org/abs/1712.05877) graph that ships on the ESP32-S3. Full tables across all 14 ablation arms are in [BENCHMARKS.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/BENCHMARKS.md).

| Benchmark / Capability | This Stack (`1.13 MB` PSRAM weights) | Matched-Budget Baseline | Reference / Upper Bound |
|---|---|---|---|
| **Word encoder RAM in firmware** (`platonic-hnet.bin` + SD `engram.bin`) | **96.8 KB RAM** (+ 4.2 MB on SD) | 550.0 KB RAM for matched `fp32` BPE (**5.7× larger**) | — |
| **Chinese word segmentation F1** ([SIGHAN05-PKU / MSR](https://aclanthology.org/I05-3017/), [UD-GSDSimp](https://universaldependencies.org/)) | **0.394 / 0.402 / 0.431** | 0.322 / 0.307 / 0.371 (BPE) | `LFM2-350M` tok 0.488 · `jieba` 0.818 |
| **English word-sense disambiguation** ([MCL-WiC](https://aclanthology.org/2021.semeval-1.3/) en-en) | **57.7%** (wins/ties **11 of 12** WiC/STS tasks vs BPE) | 52.3% (BPE) | [MiniLM-L12](https://arxiv.org/abs/2004.09813) (235 MB) 69.3% |
| **Language modeling bits/byte** ([WikiText-2](https://arxiv.org/abs/1609.07843) / PKU) | **3.4636 / 3.8351** (H-Net byte router only) · `3.5789 / 4.0487` (with Engram) | 3.4925 / 4.0200 (BPE) | `LFM2-350M` (700 MB) 1.5646 |
| **Character recap spoiler leakage** ([BookSum](https://arxiv.org/abs/2105.08209), 9,169 queries) | **0.00% spoilers** · **99.46%** scene-order accuracy | **100.0% spoilers** (same model without page-offset gate) | — |
| **Margin writer broken sentences / fake names** (4,226 tests across [BookSum](https://arxiv.org/abs/2105.08209), [NarrativeQA](https://arxiv.org/abs/1712.07040), [CommonGen](https://arxiv.org/abs/1911.03705)) | **0.00% broken** · **0.0% fake names** (clause-locked v2) | 51.2% broken · 26.1% fake names (token-by-token v1) | — |
| **E-ink refresh gate accuracy** (1,500 image episodes / 19 reading logs) | **99.9%** (`339-byte` Liquid CfC · cuts flashes `25 → 8`) | Static ink-debt counter | — |
| **PyTorch QAT → NumPy → C11 → ESP32-S3 C++20 parity** | **0 bit mismatches** (38,580 seg lines · 7,480 texts · 1,067 firmware tests) | `6.0 ms` median C query on M1 · **5.64 MB free PSRAM** on device | — |

**Where a 95 KB model still loses:**
- **5+ character Chinese compounds:** Because the byte router caps chunks at 12 UTF-8 bytes (`4` Chinese characters), it cannot emit the 214 `SIGHAN05-PKU` gold words that are 5+ characters long as a single span.
- **Whole-sentence cross-lingual retrieval:** The SD Engram aligns words and idioms in context (`71.3%` term-level recall), but a 95 KB encoder cannot match a 235 MB transformer on full-sentence [Tatoeba](https://tatoeba.org/) zh→en retrieval (`3.0%` R@1 vs `96.1%` for multilingual MiniLM-L12).
- **Template similarity:** Clause-locked decoding guarantees `0%` broken sentences and `0%` invented names, but trails a static slot template on [Sentence-BERT](https://arxiv.org/abs/1908.10084) cosine similarity (`44.17` vs `47.02` on BookSum).

---

## Quickstart (NumPy Only)

```bash
pip install numpy
python quickstart.py                                       # (a) byte routing + int8 concepts, (b) spoiler-gated recap, (c) margin note
# parity checks need the GitHub repo (C compiler + firmware sources):
git clone https://github.com/bobzhou-source/platonic-engram-hnet && cd platonic-engram-hnet
python src/verify_c_parity.py                              # encoder + character memory: C11 vs NumPy vs Python (0 mismatches)
python src/verify_pe_writer_parity.py --require-firmware   # writer: NumPy vs C++17 vs ESP32-S3 firmware kernel (0 mismatches)
```

| File in `weights/` | Size | What it is |
|---|---:|---|
| `platonic-hnet.bin` | `95,008 B` | `int8` H-Net byte router + LFM2 ShortConv + Ouro encoder (PSRAM) |
| `engram.bin` | `4,194,816 B` | 65,536-slot DeepSeek Engram hash table (SD card; public [CC-CEDICT](https://cc-cedict.org/) + [Open English WordNet](https://en-word.net/) gloss build) |
| `cspw.bin` | `393,728 B` | 4-slot Liquid CfC character memory + pointer weaver (PSRAM) |
| `pe-writer.bin` | `641,726 B` | Qwen3-32B distilled 3-step Ouro concept planner + clause selector (PSRAM) |
| `writer-clauses.bin` | `1,048,608 B` | 4,096 proper-noun-free English clauses (SD card) |
| `booksum-story-graph.idx` | `1,441,280 B` | Public [BookSum](https://arxiv.org/abs/2105.08209) character timeline used by `quickstart.py` |

See **[ARCHITECTURE.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/ARCHITECTURE.md)** for the dataflow and integer math, **[BENCHMARKS.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/BENCHMARKS.md)** for all 14 ablation arms and public benchmark tables, and **[LICENSE](LICENSE)** (MIT; dictionary glosses CC BY-SA 4.0 / CC BY 4.0; built for [CrossPoint Reader](https://github.com/CrossPoint-Reader) / [PapyriX](https://papyrix.com) / [FreeInk](https://github.com/FreeInk) on the Xteink X4 Pro).

---

## Weights

All files are in `weights/` and listed in `weights/SHA256SUMS`.

| File | Bytes | Lives in | What it is | sha256 |
|---|---:|---|---|---|
| `platonic-hnet.bin` | 95,008 | PSRAM | byte encoder and word router (94,976 int8 parameters) | `8477cd568bfd931c08ffb22588679e7e2d3589d4669b2eff8a76e5395a91b8bb` |
| `engram.bin` (public-gloss build) | 4,194,816 | SD card | 65,536 × 64-byte word slots: int8 meaning correction + short gloss | `8d6cac53f8f3038559ab71fa6e6aadc879eb9ef89cf66385cf225430615b228c` |
| `cspw.bin` | 393,728 | PSRAM | spoiler-free character memory | `cbce25c6a4fa4b215993d54c7d88c731d34004f2756fba83a9f6b8e7468dcffd` |
| `pe-writer.bin` | 641,726 | PSRAM | clause-locked margin-note writer | `962f317e22b30b8efe03aa67e5d440d6199e11ec25d96c46c9bcac6582c6cfcc` |
| `writer-clauses.bin` | 1,048,608 | SD card | 4,096 lowercase clauses, 4–10 words, no proper nouns | `7b0e3612c81d81764096c57fc6f1046abdc2a72ce9786e039861121b72d81023` |
| `booksum-story-graph.idx` | 1,441,280 | SD card | public BookSum character map: 439 characters, 5,517 events, 21 public-domain books | `cf308bcee87035c81dc00920a9224f83a7984167d6b5f81bbcb7c3bce209ef36` |

**Not published.** The device also holds a character map (`story-graph.idx`, 4,295,680 B: 2,532 characters, 16,137 event glosses and 62 aliases) and a concept index (`inbox-concepts.idx`, 96,784 B: 504 records from 126 books), both built from my personal library.

**Public-gloss `engram.bin`.** The device build (sha256 `2e7f1ec8…0be1`) has 529 glosses derived from the private reading corpus, such as character and clan names. [`tools/build_public_engram.py`](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/tools/build_public_engram.py) rewrites only the 32-byte gloss column and keeps a gloss only when it is a verbatim public-dictionary entry: 4,903 from CC-CEDICT (matched by hash and first sense), 3,090 from Open English WordNet, and 4 hyphenated WordNet headwords. The other 529 are blanked. The header, hashes and int8 corrections are byte-identical (`header_hash_delta_columns_identical: true` in `weights/engram.public-gloss.report.json`), and the encoder never reads glosses, so its output matches the device build bit for bit.

## Training data

- **Encoder and SD table.** A private Chinese/English reading corpus of 105,363 documents (not redistributed), dictionary pairs from CC-CEDICT and Open English WordNet, and jieba 0.42.1 word boundaries to teach the router. No benchmark data was used for training, tuning or selection. The table's hashes are one-way FNV-1a64 values, and the public glosses are dictionary text only.
- **Character memory and writer.** Generic template text and lowercase commentary clauses with no proper nouns; the clause bank was mined from the internal training split.
- **BookSum character map.** The BookSum chapter-level test split (`kmfoda/booksum`): 21 public-domain books (Project Gutenberg texts), with events taken from the BookSum chapter summaries.

## Technical details

- [ARCHITECTURE.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/ARCHITECTURE.md): how each model works, binary layouts, RAM budget and integer arithmetic.
- [BENCHMARKS.md](https://github.com/bobzhou-source/platonic-engram-hnet/blob/main/BENCHMARKS.md): every public benchmark table, wins and losses.
- [GitHub repo](https://github.com/bobzhou-source/platonic-engram-hnet): training code, benchmark runners, parity tests and firmware.

## License and attribution

Code and weights are MIT, with two exceptions:

- **Glosses inside `engram.bin`.** Adapted from CC-CEDICT (CC BY-SA 4.0, © CC-CEDICT contributors, MDBG) and Open English WordNet (CC BY 4.0, © the Open English WordNet team, derived from Princeton WordNet 3.1).
- **`booksum-story-graph.idx`.** Derived from BookSum (Kryściński et al., 2021). Its event glosses are fragments of BookSum's chapter summaries, which come from study-guide sites (e.g. SparkNotes, GradeSaver, CliffsNotes, Shmoop). Check BookSum's terms before redistributing.

## Citation

```bibtex
@software{zhou2026platonic_engram_hnet,
  title  = {Platonic-Engram-HNet: On-Device LLM and Concept Engine for the Xteink X4 Pro (ESP32-S3) in 1.1 MB of RAM},
  author = {Zhou, Robert},
  year   = {2026},
  url    = {https://github.com/bobzhou-source/platonic-engram-hnet}
}
```
