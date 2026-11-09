# Platonic-Engram-HNet

**Running Bilingual Word Sense, Spoiler-Free Character Memory, and Zero-Hallucination Generation in 1.1 MB of Microcontroller RAM — From Qwen3-32B H200 Distillation & 14 Controlled Ablations to Bit-Exact int8 C11/C++20 Firmware**

Author: **Robert Zhou** ([robertzhou.dev](https://robertzhou.dev/work/platonic-engram-hnet/) · [GitHub](https://github.com/bobzhou-source/platonic-engram-hnet) · [ORCID 0009-0009-2155-7215](https://orcid.org/0009-0009-2155-7215))

I wanted three things from the cheap e-reader I read on (an ESP32-S3 microcontroller with 8 MB of PSRAM): tap an English or Chinese word and get the meaning it has in that sentence, ask "who is this character again?" on page 40 without spoiling page 300, and get short margin notes that never invent a name or stop mid-sentence.

Shrinking a standard language model does not get you there. Below a megabyte it fails in three separate places:

- **The dictionary doesn't fit.** A 32,000-token vocabulary table at width 128 takes 4,096,000 bytes even at 8-bit precision, before a single layer runs. That alone is more RAM than the reader can spare next to its page buffers.
- **Search spoils the book.** Ordinary retrieval looks through the whole book, so a question asked on page 40 can be answered from page 300.
- **Tiny models can't finish a sentence.** My own token-by-token micro-model broke off mid-sentence in 51.2% of its notes and invented a character name in 26.1%.

So I designed the models and the hardware memory layout together. The models were distilled from Qwen3-32B teachers on H200 GPUs and chosen through 14 controlled ablation arms; each one is placed where the device can afford it, in fast PSRAM or on the SD card the reader already uses to store books:

1. **Split compute from the dictionary — 96.8 KB of RAM vs 550 KB for a subword (BPE) model, 5.7× smaller.** A 95 KB 8-bit byte router in PSRAM reads raw UTF-8 and finds English and Chinese word boundaries, beating a BPE tokenizer given the same RAM by +7.2, +9.5 and +6.0 F1 points on three Chinese segmentation benchmarks. Word meanings live in a 4.2 MB hash table on the SD card, read one 64-byte slot per word.
2. **Hard-causal character memory — 0.00% spoilers across 9,169 BookSum queries.** Four continuous-time character slots in PSRAM (394 KB) fade with page distance and sit behind a gate on your current page offset. It still keeps each character's earliest and latest scenes in the right order 99.46% of the time (temporal-swap accuracy 0.9946). Remove the gate and spoiler leakage is 100%.
3. **Clause-locked generation — 0.00% broken sentences and 0.0% invented names across 4,226 public tests.** Instead of decoding token by token, a 3-step planner (642 KB) picks whole clauses from a 4,096-entry bank on the SD card, none of which contains a proper noun.
4. **Bit-exact integer parity — 0 mismatches from PyTorch QAT → NumPy → C11 → C++20 firmware.** Verified on 38,580 / 38,580 benchmark lines and 1,067 / 1,067 firmware unit tests. The C kernel answers a query in a median 6.0 ms on an Apple M1 host, and with every model loaded the device keeps 5.64 MB of PSRAM free.

![Memory map](docs/memory-map.svg)

## Results

Every number comes from public benchmarks run on the exact 8-bit model that ships on the device. "Matched-RAM BPE" is a conventional subword model given the same memory budget. Losses are listed next to wins; full tables are in [BENCHMARKS.md](BENCHMARKS.md).

| What is measured | Platonic-Engram-HNet | Matched-RAM BPE | Reference |
|---|---:|---:|---|
| Chinese word segmentation, SIGHAN05-PKU (word F1) | **0.394** | 0.322 | LFM2-350M tokenizer 0.488 · jieba 0.818 |
| Chinese word segmentation, MSR / UD-GSDSimp (word F1) | **0.402 / 0.431** | 0.307 / 0.371 | jieba 0.813 / 0.799 |
| English text prediction, WikiText-2 (bits/byte; byte router without SD table) | **3.4636** | 3.4925 | — |
| Chinese text prediction, PKU (bits/byte; byte router without SD table) | **3.8351** | 4.0200 | — |
| Same two, **deployed** model with SD table (bits/byte) | 3.5789 / 4.0487 | **3.4925 / 4.0200** | the deployed model loses here |
| English word-in-context sense, MCL-WiC (accuracy) | **0.577** | 0.523 | MiniLM-L12 (235 MB) 0.693 |
| 12 public word-sense, similarity and paraphrase tasks | 5 wins · 6 ties · 1 loss | | |
| Chinese→English sentence matching, Tatoeba (R@1) | 0.030 | 0.001 | MiniLM-L12 **0.961** |
| Spoiler leakage, 9,169 BookSum queries | **0.000** | — | gate removed: 1.000 |
| Broken sentences / invented names, 4,226 public examples | **0.00% / 0.0%** | — | token-by-token v1: 51.2% / 26.1% |
| Margin-note similarity to reference, BookSum (MiniLM) | 44.17 | — | no-learning slot template **47.02** |
| E-ink refresh decision accuracy, 1,500 held-out episodes | **99.9%** | — | full-screen clears 10 → 8 vs fixed rule |
| Encoder RAM in firmware | **96.8 KB** | 550.0 KB | |

## Controlled ablations

Every component earned its place in a controlled comparison: same data, same training steps, exactly one change between neighbouring arms. Fourteen arms in total. Teachers were Qwen3-32B (text) and Qwen3-Embedding-0.6B (embeddings), run on H200 GPUs. Technique names are in [ARCHITECTURE.md](ARCHITECTURE.md).

| Study | Arms (one change each) | What moved |
|---|---|---|
| Byte encoder, 4 arms | subword (BPE) baseline → byte router + short convolutions + looped core → + self-supervised alignment objective (LeJEPA/SIGReg) → + SD hash table + 8-bit quantization-aware training | Word-sense top-1 (FP32) 0.201 → 0.278 → 0.302 → 0.303. Single-word Chinese↔English retrieval R@1 0.0033 → **0.713** once the SD table is added. Cross-lingual R@1 (int8) 0 → 0.0017 → 0.0317. |
| Character memory, 5 arms | v1 encoder → + two-chunk hash entries + chunk-grouped attention → width 192 → + Qwen3-32B relational distillation → + typed pointers + page gate | Word-sense 0.2842 → **0.3133** with Qwen3 distillation. Spoiler leakage 0.856 → **0.000**, and only once the gate is added. Character-fact precision 0.751 → 0.888. Cross-book concept F1 0.058 → 0.385. Scene-order (temporal-swap) accuracy 0.000 → 0.9946. |
| Margin-note writer, 5 arms | token-by-token model → + 3-step planner → + SD clause bank → + preference tuning (DPO) + distillation + 8-bit QAT → + clause-locked decoding (v2) | Invented names 13.75% → 1.0% (arm 3) → **0.0%** (v2, public sets). Grounded notes 70% → 98%. Repeated phrases 9.25% → 3.50%. BookSum broken sentences 51.2% → **0%**. |

## Bit-exact parity from PyTorch to firmware

The device does no floating-point math: 8-bit weights, power-of-two shifts, and 16-bit fixed-point decay. That allows a stronger test than "close enough": the training graph, the Python references in `src/`, standalone C/C++ references, and the firmware kernel in `firmware/ConceptEngine.cpp` must return the same bits.

| Check | Paths compared | Result |
|---|---|---|
| Encoder segmentation, benchmark lines | C int8 kernel vs NumPy | 38,580 / 38,580 lines identical |
| Encoder vs device binary, benchmark texts | PyTorch QAT graph vs exported `platonic-hnet.bin` + `engram.bin` | 7,480 / 7,480 texts identical |
| `src/verify_c_parity.py` (public weights) | C11 vs NumPy vs pure Python: 12 goldens + 500 fuzz inputs incl. malformed UTF-8, with/without SD table and two-chunk entries | 0 mismatches |
| Character memory step | C11 vs Python: 500 cases on `cspw.bin` + 500 on random weights + 64 goldens | 0 mismatches |
| Character recap text | C11 vs Python: 500 queries on the BookSum graph + 480 on random graphs | 0 mismatches |
| `src/verify_pe_writer_parity.py` | NumPy vs batched torch-int vs standalone C++17 vs firmware C++20 `generateMarginRiff`: 32 goldens + 200 fuzz + 8 random models | 0 mismatches |
| Export time (training checkpoints) | PyTorch QAT vs integer reference for all three models | 0 mismatches |
| Firmware | `ConceptEngineTest` / host `ctest` | 30 / 30 · 1067 / 1067 (1 skipped) |

Timings are from an Apple M1 host, not the ESP32-S3: a median 6,030 µs to encode one query (8,496,818 int8 multiply-accumulates), 399 µs per character-memory step, 7.9 ms per margin note.

## Where a 95 KB model still loses

- **Words longer than 12 bytes.** The router never lets a chunk run past 12 bytes. A Chinese character is 3 bytes in UTF-8, so words of five or more characters (long names, compound terms, idioms with a suffix) get split. SIGHAN05-PKU has 214 such words.
- **Whole-sentence translation.** The model knows that 河岸 means "river bank", but matching a whole Chinese sentence to its English translation works 3.0% of the time (Tatoeba R@1 0.030), against 96.1% for the 235 MB multilingual MiniLM-L12.
- **Bits per byte.** Without the SD table, the byte router predicts text better than matched-RAM BPE. The deployed model trades that margin for better segmentation and word sense, and loses to BPE on bits per byte (WikiText-2 3.5789 vs 3.4925).
- **Note quality.** The writer never breaks a sentence or invents a name, but a no-learning slot template still scores higher on similarity to the reference (BookSum 44.17 vs 47.02).
- **Keyword search.** BM25 or character-trigram overlap beats the deployed encoder on 9 of the 12 sentence tasks.

## Quickstart

```bash
pip install numpy
python examples/quickstart.py                     # word splitting + meaning vectors, spoiler-safe recap, margin note
python src/verify_c_parity.py                     # Python vs C11 parity (needs a C compiler)
python src/verify_pe_writer_parity.py --require-firmware   # writer: NumPy vs C++17 vs firmware kernel (needs clang++)
```

Training (`src/train.py`, `src/train_cspw.py`, `src/train_pe_writer.py`) and the benchmark runners (`benchmarks/`) need PyTorch plus the public datasets each script names; `src/train.py --smoke` runs on synthetic data on a CPU. The private training corpus and checkpoints are not published.

| Path | Contents |
|---|---|
| `src/` | models, trainers, int8 exporters, integer references, parity tests |
| `firmware/` | device C++ kernel (`ConceptEngine.*`), e-ink refresh controller (`AdaptiveRefreshPolicy.*`), host build stubs |
| `benchmarks/` | public benchmark runners |
| `examples/quickstart.py` | the three reader features on the published weights |
| `weights/` | published int8 weights + `SHA256SUMS` (same files as the Hugging Face repo) |

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): how each model works, file formats, RAM budget and integer arithmetic.
- [BENCHMARKS.md](BENCHMARKS.md): every public benchmark table, wins and losses.
- [HF_MODEL_CARD.md](HF_MODEL_CARD.md): weight files, checksums, and what was left out to protect personal reading data.

## License

Code and weights: MIT (see [LICENSE](LICENSE)). Dictionary glosses come from CC-CEDICT (CC BY-SA 4.0) and Open English WordNet (CC BY 4.0); see the model card for attribution.
