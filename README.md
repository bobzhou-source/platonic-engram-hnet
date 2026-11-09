# Platonic-Engram-HNet

**Running Bilingual Word Sense, Spoiler-Free Character Memory, and Zero-Hallucination Generation in 1.1 MB of Microcontroller RAM — From Qwen3-32B H200 Distillation & 14 Controlled Ablations to Bit-Exact int8 C11/C++20 Firmware**

[Project Page](https://robertzhou.dev/work/platonic-engram-hnet/) · [Hugging Face Weights](https://huggingface.co/bobzhou0/Platonic-Engram-HNet-v2) · [Architecture](ARCHITECTURE.md) · [Benchmarks](BENCHMARKS.md)

I wanted three things from the cheap e-reader I read on (an ESP32-S3 microcontroller with 8 MB of PSRAM): tap an English or Chinese word and get the meaning it has in that sentence, ask "who is this character again?" on page 40 without spoiling page 300, and get short margin notes that never invent a name or stop mid-sentence.

Shrinking a standard language model does not get you there. Below a megabyte it fails in three separate places:

- **The dictionary doesn't fit.** A 32,000-token vocabulary table at width 128 takes 4,096,000 bytes even at 8-bit precision, before a single layer runs. That alone is more RAM than the reader can spare next to its page buffers.
- **Search spoils the book.** Ordinary retrieval looks through the whole book, so a question asked on page 40 can be answered from page 300.
- **Tiny models can't finish a sentence.** My micro-model break off mid-sentence in 51.2% of its notes and invents a character name in 26.1%.

So I designed the models and the hardware memory layout together. The models were distilled from Qwen3-32B teachers on H200 GPUs and chosen through 14 controlled ablation arms; each one is placed where the device can afford it, in fast PSRAM or on the SD card the reader already uses to store books:

1. **Split compute from the dictionary — 96.8 KB of RAM vs 550 KB for a subword (BPE) model, 5.7× smaller.** A 95 KB 8-bit byte router in PSRAM reads raw UTF-8 and finds English and Chinese word boundaries, beating a BPE tokenizer given the same RAM by +7.2, +9.5 and +6.0 F1 points on three Chinese segmentation benchmarks. Word meanings live in a 4.2 MB hash table on the SD card, read one 64-byte slot per word.
2. **Hard-causal character memory — 0.00% spoilers across 9,169 BookSum queries.** Four continuous-time character slots in PSRAM (394 KB) fade with page distance and sit behind a gate on your current page offset. It still keeps each character's earliest and latest scenes in the right order 99.46% of the time (temporal-swap accuracy 0.9946). Remove the gate and spoiler leakage is 100%.
3. **Clause-locked generation — 0.00% broken sentences and 0.0% invented names across 4,226 public tests.** Instead of decoding token by token, a 3-step planner (642 KB) picks whole clauses from a 4,096-entry bank on the SD card, none of which contains a proper noun.
4. **Bit-exact integer parity — 0 mismatches from PyTorch QAT → NumPy → C11 → C++20 firmware.** Verified on 38,580 / 38,580 benchmark lines and 1,067 / 1,067 firmware unit tests. The C kernel answers a query in a median 6.0 ms on an Apple M1 host, and with every model loaded the device keeps 5.64 MB of PSRAM free.

![Memory map](docs/memory-map.svg)

## At a glance

| What it does | This system (`1.13 MB` total RAM) | Standard baseline at the same budget |
|---|---|---|
| **Word reader RAM in firmware** (`platonic-hnet.bin` + `engram.bin`) | **96.8 KB RAM** (+ 4.2 MB on SD card) | 550.0 KB RAM for a subword (BPE) tokenizer (**5.7× larger**) |
| **Chinese word splitting & English word sense** | **+7.2 / +9.5 / +6.0 F1** over BPE on 3 Chinese benchmarks · **5 wins, 6 ties, 1 loss** across 12 public sentence tasks | Matched-RAM BPE (140,800 parameters) |
| **Spoiler-free character recaps** (`cspw.bin`, `394 KB`) | **0.00% spoilers** across 9,169 BookSum questions · **99.46%** scene-order accuracy | **100% spoiler leakage** when the page gate is removed |
| **Margin notes** (`pe-writer.bin`, `642 KB` + `1 MB` SD clause bank) | **0.00% broken sentences · 0.0% invented names** across 4,226 public tests | **51.2% broken sentences · 26.1% invented names** with token-by-token decoding |
| **E-ink screen refresh controller** (`LiquidRefresh`, `339 B`) | **99.9%** refresh decision accuracy across 1,500 episodes (cuts full-screen flashes) | Fixed page-count rule |
| **Python → C11 → C++20 firmware match** | **0 bit mismatches** across 38,580 benchmark lines & 1,067 firmware tests (`6.0 ms` median C query) | — |

**Where a 95 KB model still loses:** Because the byte router caps chunks at 12 UTF-8 bytes and fits in 95 KB of RAM, it splits 5+ character Chinese compounds, loses to 200 MB+ server models on whole-sentence translation, and trails a static slot template on reference similarity. Every win, loss, and 14-arm ablation table is published in [BENCHMARKS.md](BENCHMARKS.md), and the model internals are in [ARCHITECTURE.md](ARCHITECTURE.md).
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
