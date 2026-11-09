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

# Platonic-Engram-HNet v2

**Disaggregating Compute and Memory Below 1 Megabyte: Tokenizer-Free Byte Routing, O(1) Flash Engrams, and Continuous-Time State in Pure `int8` C++ (From Qwen3-32B H200 Distillation & 4-Arm Ablations to Bit-Exact ESP32-S3 Firmware)**

Standard language models spend megabytes of RAM on a static vocabulary table, leak spoilers when used for book retrieval, and hallucinate when shrunk below 100M parameters. I split the problem along the memory hierarchy instead of shrinking a transformer. **Compute** is a 95 KB looped byte-router in PSRAM. **Lexical memory** is a 4.2 MB $O(1)$ Engram hash table on the SD card, read one 64-byte slot per word. **Episodic state** is 4 continuous-time Chrono-CfC entity slots behind hard causal page-offset spoiler gates. **Syntax** is 4,096 frame-locked clauses on SD. With that split, an ESP32-S3 e-reader runs bilingual word sense, spoiler-proof character recaps and margin notes with no invented names, in 1,130,462 bytes (< 1.2 MB) of PSRAM weights.

I built the whole stack myself: Qwen3-32B teacher distillation and controlled multi-arm ablations on H200 GPUs, storage/compute disaggregation, bit-exact `int8`/Q15 C and C++20 kernels, and the production ESP32-S3 firmware that ships them.

## The 4-pillar co-design

| Pillar | Where it lives | Component | File | Bytes |
|---|---|---|---|---:|
| **1. Compute**: looped `int8` byte router | PSRAM | **Platonic-Engram-HNet v2** (PLHN): UTF-8 bytes → learned word-like chunks → 128-d int8 concept | `platonic-hnet.bin` | 95,008 |
| **2. Lexical memory**: $O(1)$ 64-byte lookups | SD / flash | Engram hash table (ENGR): 65,536 hashed span slots, int8 concept delta + 32-byte gloss | `engram.bin` | 4,194,816 |
| **3. Episodic state**: continuous-time slots + causal spoiler gates | PSRAM | **Chrono-Slot Pointer Weaver** (CS-PW v2): 4 Chrono-CfC slots, typed frame/phrase pointers | `cspw.bin` | 393,728 |
| **4a. Syntax**: frame-locked decoding | PSRAM + SD | **Frame-Locked PE-Writer v2**: 3-step Ouro planner + frame-locked decoder over the WCLS clause bank | `pe-writer.bin` + `writer-clauses.bin` | 641,726 + 1,048,608 |
| **4b. Physical control**: e-ink refresh | internal RAM | **LiquidRefresh CfC v2**: continuous-time refresh/dither policy | compiled into firmware | 339 |

Target hardware: the X4 Pro e-reader (ESP32-S3, dual-core 240 MHz, 8 MB embedded PSRAM). With every model loaded, minimum free PSRAM during the night-reading workload stays at 5,640,084 bytes.

## Three headline results

1. **Byte router + SD Engram vs subword BPE, with 5.7× less RAM.** The encoder needs 96.8 KB of RAM, against 550.0 KB for a matched-RAM BPE model. It beats BPE on all 3 Chinese segmentation benchmarks (+7.2 F1 on SIGHAN05-PKU, +9.5 on MSR, +6.0 on UD-GSDSimp). It wins or ties 11 of 12 public WiC/STS/paraphrase tasks (5 wins, 6 ties, 1 loss). The H-Net without the Engram also beats BPE on WikiText-2 bits/byte (3.4636 vs 3.4925). The deployed Engram encoder does not (3.5789; see the limitations below).
2. **Causal Chrono-CfC slots: 0.00% spoilers across 9,169 BookSum queries.** With the gate removed, spoiler leakage is 100%. On my internal eval the full weaver keeps temporal-swap accuracy at 0.9946.
3. **Frame-locked concept decoder: 0.00% broken sentences and 0.0% invented names across 4,226 public tests.** My token-by-token micro-LM decoder (PE-Writer v1) had 51.2% broken tails and 26.1% invented names on BookSum.

## Controlled multi-arm ablations

Every component was chosen by a controlled ablation: same data, same steps, one change per arm. The teachers for CS-PW and PE-Writer were Qwen3-32B (generation) and Qwen3-Embedding-0.6B (embeddings), run on H200 GPUs.

| Study | Arms (one change each) | What moved |
|---|---|---|
| Encoder, 4 arms | BPE baseline → H-Net router + ShortConv + Ouro → + LeJEPA/SIGReg → + Engram + int8 QAT | WSD top-1 (FP32) 0.201 → 0.278 → 0.302 → 0.303. Term-alone cross-lingual R@1 0.0033 → **0.713** with the Engram. Cross-lingual R@1 (int8) 0 → 0.0017 → 0.0317. |
| CS-PW, 5 arms | v1 encoder → + Bigram-Span Engram + Chunk-GQA → d=192 → + Qwen3-32B relational KD → + typed pointers + spoiler gate | WSD 0.2842 → **0.3133** with Qwen KD. Spoiler leakage 0.856 → **0.000** only with the gate. Dossier fact precision 0.751 → 0.888. Cross-book concept F1 0.058 → 0.385. Temporal-swap 0.000 → 0.9946. |
| PE-Writer, 5 arms | flat AR → + Ouro 3-step planner → + WCLS clause engram → + DPO + KD + int8 QAT → + frame-locked decoding (v2) | Invented names 13.75% → 1.0% (arm 3) → **0.0%** (v2, public sets). Grounded 70% → 98%. Rep-3 9.25% → 3.50%. BookSum broken tails 51.2% → **0%**. |

## Bit-exact parity: PyTorch QAT → NumPy → C → C++20 firmware

The device arithmetic is integer-only (int8 weights, power-of-two shifts, Q15 decay). The Python references (`plhn_model.py`, `cspw_model.py`, `pe_writer_model.py`), the standalone C/C++ references and the firmware kernel (`firmware/ConceptEngine.cpp` in the GitHub repo) must agree bit for bit.

| Check | Paths compared | Result |
|---|---|---|
| Encoder segmentation, benchmark lines | C int8 kernel vs NumPy | 38,580 / 38,580 lines identical |
| Encoder vs device binary, benchmark texts | PyTorch QAT graph vs exported `platonic-hnet.bin` + `engram.bin` | 7,480 / 7,480 texts identical |
| `src/verify_c_parity.py` (GitHub repo, public weights) | C11 vs NumPy vs pure Python: 12 goldens + 500 fuzz inputs incl. malformed UTF-8, with/without Engram and bigrams | 0 mismatches |
| CS-PW weave (GitHub repo) | C11 vs Python: 500 cases on `cspw.bin` + 500 on random weights + 64 goldens | 0 mismatches |
| CS-PW device text (GitHub repo) | C11 vs Python: 500 queries on the BookSum graph + 480 on random graphs | 0 mismatches |
| `src/verify_pe_writer_parity.py` (GitHub repo) | NumPy vs batched torch-int vs standalone C++17 vs firmware C++20 `generateMarginRiff`: 32 goldens + 200 fuzz + 8 random models | 0 mismatches |
| Export time (training checkpoints) | PyTorch QAT vs integer reference for PLHN, CS-PW and PE-Writer | 0 mismatches |
| Firmware | `ConceptEngineTest` / host `ctest` | 30 / 30 · 1067 / 1067 (1 skipped) |

Host timings are from an Apple M1, not the ESP32-S3. Encoding one query with the C int8 kernel takes a median 6,030 µs at 8,496,818 int8 MACs. A CS-PW weave with M=4 takes 399 µs. A PE-Writer riff takes 7.9 ms.

## Scope

It is **not** a general-purpose embedding model. On STS and paraphrase tasks it beats a matched-RAM BPE model, but it loses to lexical baselines (BM25, character 3-grams) on most tasks and to MiniLM-class models on all but two. On Tatoeba zh→en retrieval it reaches R@1 0.030, against 0.961 for multilingual MiniLM-L12. Use it where 100 KB of RAM is the constraint, not where accuracy is.

## Weights manifest

All files are in `weights/` and listed in `weights/SHA256SUMS`.

| File | Format | Bytes | sha256 | Device location | Role |
|---|---|---:|---|---|---|
| `platonic-hnet.bin` | PLHN v1 | 95,008 | `8477cd568bfd931c08ffb22588679e7e2d3589d4669b2eff8a76e5395a91b8bb` | PSRAM | byte encoder (94,976 int8 params) |
| `engram.bin` | ENGR v1, **public-gloss build** | 4,194,816 | `8d6cac53f8f3038559ab71fa6e6aadc879eb9ef89cf66385cf225430615b228c` | SD | 65,536 × 64 B hashed span slots: int8 concept delta + gloss |
| `cspw.bin` | CSPW v2 | 393,728 | `cbce25c6a4fa4b215993d54c7d88c731d34004f2756fba83a9f6b8e7468dcffd` | PSRAM | Chrono-Slot Pointer Weaver (spoiler-free entity refreshers) |
| `pe-writer.bin` | PEWR v1 (v2 frame-locked weights) | 641,726 | `962f317e22b30b8efe03aa67e5d440d6199e11ec25d96c46c9bcac6582c6cfcc` | PSRAM | Frame-Locked PE-Writer v2 |
| `writer-clauses.bin` | WCLS v1 | 1,048,608 | `7b0e3612c81d81764096c57fc6f1046abdc2a72ce9786e039861121b72d81023` | SD | 4,096 lowercase clauses, 4–10 words, no proper nouns |
| `booksum-story-graph.idx` | SGRF v1 | 1,441,280 | `cf308bcee87035c81dc00920a9224f83a7984167d6b5f81bbcb7c3bce209ef36` | SD | public BookSum story graph: 439 entities, 5,517 events, 21 public-domain books |

### Explicitly excluded (never published)

| File | Why |
|---|---|
| `story-graph.idx` (SGRF, 4,295,680 B on the device) | Built from the owner's personal library: 2,532 character/entity records, 16,137 event glosses and 62 aliases from books the owner reads. Every reader builds their own. |
| `inbox-concepts.idx` (ICPT, 96,784 B) | Concept index of the owner's book inbox (504 records from 126 books). Personal data. |
| the device `engram.bin` (sha256 `2e7f1ec8…0be1`) | Its gloss column has 529 glosses derived from the private reading corpus: character names, clan names, curated web-fiction terms, speech tags with names, and composed glosses. The public build replaces it (see below). |

### What "public-gloss" means

`tools/build_public_engram.py` (GitHub repo) rewrites only the 32-byte gloss column of the device `engram.bin`. A gloss is kept only when it is provably a verbatim public-dictionary entry:

- a CC-CEDICT headword whose FNV-1a64 equals the slot's span hash and whose first sense equals the stored gloss (4,903 kept);
- an Open English WordNet `~lemma: first sense` gloss (3,090 kept);
- a hyphenated OEWN headword (4 kept).

The other 529 glosses are blanked. The header, span hashes and int8 concept deltas are byte-identical to the device file (`header_hash_delta_columns_identical: true` in `weights/engram.public-gloss.report.json`). The encoder never reads the gloss column, so embeddings from the public build equal the device build bit for bit.

## Architecture

```
UTF-8 bytes (≤256) → byte_embed 256×64 → 2 × LFM2 double-gated ShortConv(k=3)
  → H-Net cosine router (causal, 6-byte receptive field, 12-byte span cap, jieba-supervised)
  → chunk mean → chunk_proj 64→128
  → + Engram delta (FNV-1a64(span) mod 65,536, dims 0–23) + Bigram-span Engram (hash-verified)
  → Ouro core × 2 (shared weights) → sum → integer-sqrt L2 → int8 × 128
```

Training: LeJEPA identity + SIGReg (λ = 0.02), InfoNCE cross-lingual alignment between Chinese context sentences and English dictionary definitions, and quantisation-aware training. Binary layouts, the integer arithmetic and the companion models are documented in `ARCHITECTURE.md` in the GitHub repo.

## How to use (Python integer reference)

```bash
pip install numpy
python quickstart.py   # verifies SHA256SUMS, then runs (a) byte routing + int8 concepts, (b) a spoiler-gated BookSum recap, (c) a PE-Writer riff
```

`plhn_model.py`, `cspw_model.py` and `pe_writer_model.py` are the bit-exact integer references of the firmware. They need only numpy.

```python
import plhn_model as M

enc = M.PlhnIntNp(open("weights/platonic-hnet.bin", "rb").read(),
                  open("weights/engram.bin", "rb").read())
a = enc.forward("The river bank was flooded after the storm.")   # list of 128 int8
b = enc.forward("河岸在暴风雨后被淹没了。")
cos = sum(x * y for x, y in zip(a, b)) / (127 * 127)
data, spans, _ = enc.chunk_states("他笑着说，明天再去山上看看。")  # learned byte chunks
```

Inputs longer than 256 bytes should be split into sentence-packed windows of at most 256 bytes. Sum the per-window int8 vectors, then re-normalise with `M.l2_int8`.

## Evaluation

These results come from the project's own harness, which reimplements the MTEB/C-MTEB protocols. They were **not** produced by the `mteb` package, so the `model-index` entries are marked `verified: false`. Each score is the deployed encoder (`Arm4-device`).

| Task | Platonic-Engram-HNet v2 | Matched-RAM BPE (140,800 params) | Best lexical | MiniLM-L12 multilingual (235.31 MB) |
|---|---:|---:|---:|---:|
| STS-B en (Spearman ×100) | 32.8 | 29.3 | 64.8 | 84.4 |
| STS-B zh | 49.0 | 39.4 | 55.6 | 78.9 |
| PAWS-X en | 14.3 | 17.8 | 17.4 | 17.3 |
| PAWS-X zh | 8.2 | 15.9 | 15.7 | 10.1 |
| BQ | 22.8 | 22.2 | 38.3 | 38.5 |
| ATEC | 10.5 | 8.8 | 20.8 | 18.4 |
| Tatoeba cmn→eng R@1 | 0.030 | 0.001 | 0.018 | 0.961 |
| MCL-WiC en-en acc | 0.577 | 0.523 | 0.632 | 0.693 |
| SIGHAN05-PKU segmentation word F1 | 0.394 | 0.322 | — | — (jieba 0.818) |

Against matched-RAM BPE across 12 tasks: 5 wins, 6 ties, 1 loss. Against the best lexical baseline: 1 win, 2 ties, 9 losses.

Companion models on public data:

- **CS-PW**, BookSum, 9,169 cutoff queries: spoiler leakage 0.000 (1.000 with the gate removed) and fact precision 0.703. A cutoff-filtered BM25 is also leak-free, with FP 0.772.
- **PE-Writer v2**, 4,226 examples: 0.00% broken tails and 0.0% invented proper nouns. It still scores below a no-learning slot template on MiniLM similarity (BookSum 44.17 vs 47.02).

## Training data

- **Encoder.** A private Chinese/English reading corpus of 105,363 documents (not redistributed). Dictionary pairs come from CC-CEDICT and Open English WordNet. Router supervision uses jieba 0.42.1 word boundaries. No benchmark data was used for training, tuning or selection.
- **Engram.** Span hashes and concept deltas were learned on the same corpus. The hashes are one-way FNV-1a64 values, and the public glosses are dictionary text only.
- **CS-PW frames/phrases and PE-Writer clauses.** Generic template text and lowercase commentary clauses with no proper nouns. The clause bank was mined from the internal training split.
- **BookSum story graph.** Entities and events come from the BookSum chapter-level test split (`kmfoda/booksum`): 21 public-domain books (Project Gutenberg texts), with events taken from the BookSum chapter summaries.

## Limitations

- **Cross-lingual retrieval.** The model does not do sentence-level cross-lingual retrieval (Tatoeba R@1 0.030).
- **12-byte chunk cap.** Words longer than 12 UTF-8 bytes (five or more Chinese characters) cannot be emitted as one chunk. SIGHAN05-PKU has 214 such gold words.
- **Bits per byte.** The deployed Engram encoder loses to matched-RAM BPE on bits/byte (WikiText-2 3.5789 vs 3.4925). The H-Net without Engram wins (3.4636).
- **Timings.** All timings in the project reports are Apple M1 host timings for the C int8 kernel (median encode 6,030 µs), not ESP32-S3 measurements.

## License and attribution

Code and weights are released under the MIT license, with these exceptions:

- **Gloss column of `engram.bin`.** Adapted from CC-CEDICT (CC BY-SA 4.0, © CC-CEDICT contributors, MDBG) and Open English WordNet (CC BY 4.0, © the Open English WordNet team, derived from Princeton WordNet 3.1).
- **`booksum-story-graph.idx`.** Derived from BookSum (Kryściński et al., 2021). Its event glosses are fragments of BookSum's chapter summaries, which come from study-guide sites (e.g. SparkNotes, GradeSaver, CliffsNotes, Shmoop). Check BookSum's terms before redistributing.
