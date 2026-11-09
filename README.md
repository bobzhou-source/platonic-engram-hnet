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

## Why no tokenizer

Platonic-Engram-HNet keeps a 95,008-byte int8 encoder in PSRAM and a 65,536-slot, 4,194,816-byte Engram hash table on the SD card. Each chunk costs one 64-byte slot read. The encoder reads raw UTF-8 bytes through two LFM2-style double-gated short-convolution blocks, then an H-Net cosine router with a 12-byte span cap, an FNV-1a hashed Engram lookup, and a 2-loop shared-weight Ouro core. In firmware the encoder needs 96.8 KB of RAM. The matched-RAM BPE baseline needs 550.0 KB.

The router is causal: it sees 6 bytes back and never cuts inside a UTF-8 character. It is trained with a hinge on jieba word boundaries, and the encoder is trained with a LeJEPA identity + SIGReg objective plus InfoNCE cross-lingual alignment. The Engram gives the model megabytes of lexical memory for the price of one SD read per chunk. That memory sits on storage the reader already has, not in RAM.

![Memory map](docs/memory-map.svg)

## Results at a glance

Every number below comes from public benchmarks run on the exact int8 graph that ships on the device. The losses are reported too. Full tables are in [BENCHMARKS.md](BENCHMARKS.md).

| Benchmark | Platonic-Engram-HNet v2 | Matched-RAM BPE | Reference |
|---|---:|---:|---|
| SIGHAN05-PKU word F1 (deployed router) | **0.394** | 0.322 | LFM2-350M tokenizer 0.488 · jieba 0.818 |
| SIGHAN05-MSR / UD-GSDSimp word F1 | **0.402 / 0.431** | 0.307 / 0.371 | jieba 0.813 / 0.799 |
| WikiText-2 bits/byte (H-Net, no Engram, FP32) | **3.4636** | 3.4925 | — |
| SIGHAN05-PKU bits/byte (H-Net, no Engram, FP32) | **3.8351** | 4.0200 | — |
| WikiText-2 / PKU bits/byte (**deployed** Engram encoder) | 3.5789 / 4.0487 | **3.4925 / 4.0200** | the deployed encoder loses here |
| MCL-WiC en-en accuracy | **0.577** | 0.523 | MiniLM-L12 (235 MB) 0.693 |
| 12 public sentence tasks vs BPE | 5 wins · 6 ties · 1 loss (PAWS-X zh 0.082 vs 0.159) | | |
| Tatoeba zh→en R@1 | 0.030 | 0.001 | MiniLM-L12 **0.961** |
| CS-PW spoiler leakage, BookSum (9,169 queries) | **0.000** | — | gate removed: 1.000 |
| PE-Writer v2 broken tails / invented proper nouns (4,226 public examples) | **0.00% / 0.0%** | — | v1 on BookSum: 51.2% / 26.1% |
| PE-Writer v2 MiniLM similarity, BookSum | 44.17 | — | no-learning slot template **47.02** |
| LiquidRefresh v2 refresh-gate accuracy (1,500 held-out episodes) | **99.9%** | — | non-initial clears 10 → 8 vs static rule |
| RAM in firmware (encoder) | **96.8 KB** | 550.0 KB | |

## Where a 95 KB encoder still loses

- **The 12-byte span cap.** 214 SIGHAN05-PKU gold words are longer than 12 UTF-8 bytes, so the router cannot emit them as one chunk. A Chinese character is 3 bytes, so these are words of five or more characters: long names, compound terms and idioms with a suffix attached.
- **Cross-lingual retrieval.** Tatoeba zh→en R@1 is 0.030, against 0.961 for multilingual MiniLM-L12. The InfoNCE alignment carries known dictionary entries into new sentences. It does not produce a sentence-level bilingual space.
- **The writer loses to a template.** PE-Writer v2 never breaks a sentence or invents a name, but a no-learning slot template still scores higher on MiniLM similarity (BookSum 44.17 vs 47.02; CommonGen 35.81 vs 44.46).
- **The Engram costs bits/byte.** Without Engram, the H-Net beats matched-RAM BPE on WikiText-2 and PKU. The deployed Engram encoder trades that margin for better segmentation and word-sense scores, and loses to BPE on bits/byte.
- **Lexical baselines.** BM25 or character-3-gram Jaccard beats the deployed encoder on 9 of the 12 sentence tasks.

## Controlled multi-arm ablations

Every component was chosen by a controlled ablation: same data, same steps, one change per arm. The teachers for CS-PW and PE-Writer were Qwen3-32B (generation) and Qwen3-Embedding-0.6B (embeddings), run on H200 GPUs.

| Study | Arms (one change each) | What moved |
|---|---|---|
| Encoder, 4 arms | BPE baseline → H-Net router + ShortConv + Ouro → + LeJEPA/SIGReg → + Engram + int8 QAT | WSD top-1 (FP32) 0.201 → 0.278 → 0.302 → 0.303. Term-alone cross-lingual R@1 0.0033 → **0.713** with the Engram. Cross-lingual R@1 (int8) 0 → 0.0017 → 0.0317. |
| CS-PW, 5 arms | v1 encoder → + Bigram-Span Engram + Chunk-GQA → d=192 → + Qwen3-32B relational KD → + typed pointers + spoiler gate | WSD 0.2842 → **0.3133** with Qwen KD. Spoiler leakage 0.856 → **0.000** only with the gate. Dossier fact precision 0.751 → 0.888. Cross-book concept F1 0.058 → 0.385. Temporal-swap 0.000 → 0.9946. |
| PE-Writer, 5 arms | flat AR → + Ouro 3-step planner → + WCLS clause engram → + DPO + KD + int8 QAT → + frame-locked decoding (v2) | Invented names 13.75% → 1.0% (arm 3) → **0.0%** (v2, public sets). Grounded 70% → 98%. Rep-3 9.25% → 3.50%. BookSum broken tails 51.2% → **0%**. |

## Bit-exact parity: PyTorch QAT → NumPy → C → C++20 firmware

The device arithmetic is integer-only (int8 weights, power-of-two shifts, Q15 decay). The Python references in `src/`, the standalone C/C++ references and the firmware kernel in `firmware/ConceptEngine.cpp` must agree bit for bit.

| Check | Paths compared | Result |
|---|---|---|
| Encoder segmentation, benchmark lines | C int8 kernel vs NumPy | 38,580 / 38,580 lines identical |
| Encoder vs device binary, benchmark texts | PyTorch QAT graph vs exported `platonic-hnet.bin` + `engram.bin` | 7,480 / 7,480 texts identical |
| `src/verify_c_parity.py` (this repo, public weights) | C11 vs NumPy vs pure Python: 12 goldens + 500 fuzz inputs incl. malformed UTF-8, with/without Engram and bigrams | 0 mismatches |
| CS-PW weave (this repo) | C11 vs Python: 500 cases on `cspw.bin` + 500 on random weights + 64 goldens | 0 mismatches |
| CS-PW device text (this repo) | C11 vs Python: 500 queries on the BookSum graph + 480 on random graphs | 0 mismatches |
| `src/verify_pe_writer_parity.py` (this repo) | NumPy vs batched torch-int vs standalone C++17 vs firmware C++20 `generateMarginRiff`: 32 goldens + 200 fuzz + 8 random models | 0 mismatches |
| Export time (training checkpoints) | PyTorch QAT vs integer reference for PLHN, CS-PW and PE-Writer | 0 mismatches |
| Firmware | `ConceptEngineTest` / host `ctest` | 30 / 30 · 1067 / 1067 (1 skipped) |

Host timings are from an Apple M1, not the ESP32-S3. Encoding one query with the C int8 kernel takes a median 6,030 µs at 8,496,818 int8 MACs. A CS-PW weave with M=4 takes 399 µs. A PE-Writer riff takes 7.9 ms.

## Quickstart

```bash
pip install numpy
python examples/quickstart.py                     # (a) byte routing + int8 concepts, (b) spoiler-gated recap, (c) PE-Writer riff
python src/verify_c_parity.py                     # encoder + CS-PW: C11 vs NumPy vs Python (needs a C compiler)
python src/verify_pe_writer_parity.py --require-firmware   # PE-Writer: NumPy vs C++17 vs firmware kernel (needs clang++)
```

Training (`src/train.py`, `src/train_cspw.py`, `src/train_pe_writer.py`) and the benchmark runners (`benchmarks/`) need PyTorch plus the public datasets each script names. `src/train.py --smoke` runs on synthetic data on a CPU. The training corpus and checkpoints are not published (see the model card).

## Repository layout

| Path | Contents |
|---|---|
| `src/` | models, trainers, int8 exporters, NumPy/pure-Python integer references, parity gates |
| `firmware/` | the device C++ kernel (`ConceptEngine.*`), LiquidRefresh (`AdaptiveRefreshPolicy.*`) and the host stubs used to build it on a desktop |
| `benchmarks/` | public benchmark runners (WSD/cross-lingual, segmentation + efficiency, narrative/spoiler, PE-Writer) |
| `examples/quickstart.py` | the three device paths on the published weights |
| `tools/` | `make_public_goldens.py` (rebuilds `golden/` from public inputs), `build_public_engram.py` (public-gloss Engram) |
| `golden/` | regression goldens read by the parity gates |
| `weights/` | the published int8 weights + `SHA256SUMS` (the same files as the Hugging Face repo) |

## Files and checksums

| File | Bytes | sha256 |
|---|---:|---|
| `platonic-hnet.bin` | 95,008 | `8477cd568bfd931c08ffb22588679e7e2d3589d4669b2eff8a76e5395a91b8bb` |
| `engram.bin` (device build) | 4,194,816 | `2e7f1ec8e707be06478b4e9b944e02dec2b4743579534c938c2f89f9ea9e0be1` |
| `engram.bin` (public-gloss build, published) | 4,194,816 | `8d6cac53f8f3038559ab71fa6e6aadc879eb9ef89cf66385cf225430615b228c` |
| `cspw.bin` | 393,728 | `cbce25c6a4fa4b215993d54c7d88c731d34004f2756fba83a9f6b8e7468dcffd` |
| `pe-writer.bin` | 641,726 | `962f317e22b30b8efe03aa67e5d440d6199e11ec25d96c46c9bcac6582c6cfcc` |
| `writer-clauses.bin` | 1,048,608 | `7b0e3612c81d81764096c57fc6f1046abdc2a72ce9786e039861121b72d81023` |
| `booksum-story-graph.idx` (public BookSum benchmark graph) | 1,441,280 | `cf308bcee87035c81dc00920a9224f83a7984167d6b5f81bbcb7c3bce209ef36` |

The published `engram.bin` differs from the device build only in its gloss column. Span hashes, concept deltas and the header are byte-identical, so encoder outputs do not change. See the [model card](HF_MODEL_CARD.md) for what was removed and why. Two per-reader indexes are **never** published: `story-graph.idx`, built from the owner's library, and `inbox-concepts.idx`, built from the owner's book inbox.

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md): the pipeline, binary layouts, RAM budget and integer arithmetic.
- [BENCHMARKS.md](BENCHMARKS.md): every public benchmark table, wins and losses.
- [HF_MODEL_CARD.md](HF_MODEL_CARD.md): the Hugging Face model card and public weights manifest.

## License

Code and weights: MIT (see [LICENSE](LICENSE)). Engram glosses come from CC-CEDICT (CC BY-SA 4.0) and Open English WordNet (CC BY 4.0). See the model card for attribution.
