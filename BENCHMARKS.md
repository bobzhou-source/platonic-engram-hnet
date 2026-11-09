# Benchmarks

Every number here is copied from the reports written by the runners in `benchmarks/` (WSD/cross-lingual, segmentation + efficiency, narrative/spoiler, PE-Writer) and from the firmware LiquidRefresh evaluation. Each report is generated from its raw JSON. My encoders run the exact int8 QAT graph that ships on the device. The deployed encoder was checked bit-exact against the device binary on 7,480/7,480 benchmark texts. The matched-RAM BPE baseline (`Arm0-BPE`, 140,800 parameters, 1.82 bytes per token) has no int8 export and runs in FP32. No model was trained or tuned on any benchmark set. Thresholds are tuned on dev splits only.

Encoders compared: **v2s_arm1** is the H-Net with no Engram. **E3 / Arm4-device** is the deployed encoder with Engram + Bigram-span Engram + Qwen3 relational KD.

## 1. Chinese word segmentation (word F1, gold human annotation)

| System | SIGHAN05-PKU | SIGHAN05-MSR | UD-GSDSimp | PKU boundary F1 |
|---|---:|---:|---:|---:|
| **router v2s_E3 (deployed)** | **0.394** | **0.402** | **0.431** | 0.692 |
| router v2s_arm1 (no Engram) | 0.360 | 0.350 | 0.399 | 0.745 |
| matched-RAM BPE (Arm0) | 0.322 | 0.307 | 0.371 | 0.737 |
| char-unigram heuristic | 0.343 | 0.330 | 0.394 | 0.750 |
| MiniLM-L6-v2 tokenizer | 0.349 | 0.334 | 0.428 | 0.757 |
| LFM2-350M tokenizer | 0.488 | 0.499 | 0.485 | 0.803 |
| BGE-M3 tokenizer | 0.649 | 0.659 | 0.614 | 0.865 |
| Qwen3-Embedding-0.6B tokenizer | 0.656 | 0.671 | 0.599 | 0.860 |
| jieba 0.42.1 (supervised) | 0.818 | 0.813 | 0.799 | 0.932 |
| oracle: gold + 12-byte cap | 0.997 | 0.972 | 0.996 | 0.999 |

The deployed router beats matched-RAM BPE on word F1 on all three sets, but it loses to BPE on boundary F1 (0.692 vs 0.737). The LFM2-350M, BGE-M3 and Qwen3 tokenizers all beat it. The 12-byte cap alone costs very little: the gold + cap oracle scores 0.997 on PKU, even though 214 PKU gold words exceed 12 bytes.

## 2. Bits per byte (lower is better)

| Model | WikiText-2 (en) | SIGHAN05-PKU (zh) | SIGHAN05-MSR (zh) | UD-GSDSimp (zh) | Tatoeba (zh/en) |
|---|---:|---:|---:|---:|---:|
| matched-RAM BPE, FP32 | 3.4925 | 4.0200 | 3.9243 | 3.9944 | **4.4297** |
| **H-Net v2s_arm1, FP32** | **3.4636** | **3.8351** | **3.7774** | **3.8338** | 4.6553 |
| H-Net v2s_arm1, INT8 | 3.5028 | 3.8772 | 3.8202 | 3.8806 | 4.4860 |
| deployed v2s_E3, FP32 | 3.5789 | 4.0487 | 3.9675 | 4.0604 | 4.6010 |
| deployed v2s_E3, INT8 | 3.6038 | 4.0697 | 3.9843 | 4.0793 | 4.7322 |
| gzip -9, whole corpus | 2.7617 | 3.4249 | 3.2641 | 4.2780 | 3.9139 |

The H-Net without Engram beats matched-RAM BPE on the four monolingual sets. The deployed Engram encoder loses to BPE on all five. Whole-corpus compressors are not window-limited, so they are a reference only.

## 3. Word sense, cross-lingual retrieval and sentence similarity

Primary metric per task. WiC tasks use test accuracy of the dev-selected feature. Tatoeba uses mean R@1 over both directions. STS and paraphrase tasks use Spearman.

| Task | Arm4-device | vs best lexical | vs matched BPE | vs MiniLM-L12 (235.31 MB) |
|---|---:|---|---|---|
| WiC (en) | 0.507 | loss (0.540, char-3) | tie (0.521) | loss (0.608) |
| XL-WiC (zh) | 0.532 | loss (0.608, BM25) | **win** (0.509) | loss (0.632) |
| MCL-WiC en-en | 0.577 | loss (0.632, BM25) | **win** (0.523) | loss (0.693) |
| MCL-WiC zh-zh | 0.546 | loss (0.610, BM25) | tie (0.548) | loss (0.712) |
| MCL-WiC en-zh | 0.503 | tie (0.501) | tie (0.500) | loss (0.709) |
| Tatoeba cmn-eng | 0.023 | **win** (0.016) | **win** (0.003) | loss (0.962) |
| STS-B (en) | 0.328 | loss (0.648, char-3) | tie (0.293) | loss (0.844) |
| STS-B (zh) | 0.490 | loss (0.556, BM25) | **win** (0.394) | loss (0.789) |
| PAWS-X (en) | 0.143 | tie (0.174) | tie (0.178) | tie (0.173) |
| PAWS-X (zh) | 0.082 | loss (0.157, char-3) | **loss** (0.159) | tie (0.101) |
| BQ (zh) | 0.228 | loss (0.383, BM25) | tie (0.222) | loss (0.385) |
| ATEC (zh) | 0.105 | loss (0.208, BM25) | **win** (0.088) | loss (0.184) |

Against matched-RAM BPE: 5 wins, 6 ties, 1 loss. Against the best lexical baseline: 1 win, 2 ties, 9 losses. Against MiniLM-L12: 0 wins, 2 ties, 10 losses.

Tatoeba cmn-eng detail (1,000 pairs): zh→en R@1 0.030, R@10 0.103. en→zh R@1 0.016. MiniLM-L12 scores 0.961 / 0.996 / 0.964.

## 4. Spoiler-free entity tracking (CS-PW), public BookSum

There are 57 (book, study-guide source) sequences and 439 entity timelines. That gives **9,169 cutoff queries**, including 245 pre-introduction probes. SL is spoiler leakage. FP is token fact precision against the original book text up to the cutoff.

| System | SL | SL pre-intro | FP | Spoiler-token rate | Coverage |
|---|---:|---:|---:|---:|---:|
| **CS-PW arm4 (SGRF gate + int8 weave)** | **0.000** | **0.000** | 0.703 | 0.084 | 0.973 |
| CS-PW arm4, gate removed | 1.000 | 1.000 | 0.683 | 0.106 | 1.000 |
| CS-PW arms 0–3 (flat, ungated) | 1.000 | 1.000 | 0.529–0.560 | 0.106–0.123 | 1.000 |
| BM25 top-1 sentence, ungated | 0.442 | 1.000 | 0.742 | 0.095 | 1.000 |
| BM25 top-1 sentence, cutoff-filtered | 0.000 | 0.000 | **0.772** | 0.061 | 0.973 |

The gate is what makes CS-PW spoiler-free. A cutoff-filtered BM25 is also spoiler-free and has higher fact precision. Internal eval: SL 0.000 on 3,631 lookups (ungated arms 85.6%), temporal-swap 0.9946, dossier fact precision 0.888, cross-book concept F1 0.385. The public BookSum story graph used here is published as `booksum-story-graph.idx` (1,441,280 B), and re-running the benchmark from it reproduces SL 0.000 / FP 0.703.

## 5. Concept-to-text (Frame-Locked PE-Writer v2), 4,226 public examples

MiniLM cos×100 against the human reference, plus defect rates. Arm 3 is PE-Writer v1.

| Set (n) | v2 MiniLM | Slot template (no learning) | v2 broken tail | v1 broken tail | v2 invented PN | v1 invented PN |
|---|---:|---:|---:|---:|---:|---:|
| BookSum (723) | 44.17 | **47.02** | 0.00% | 51.18% | 0.0% | 26.1% |
| NarrativeQA (710) | 19.26 | **21.21** | 0.00% | 56.20% | 0.0% | 21.7% |
| Tatoeba zh→en (1,000) | 11.13 | **13.60** | 0.00% | 21.50% | 0.0% | 19.6% |
| UD-GSDSimp (500) | 12.99 | **15.00** | 0.00% | 27.80% | 0.0% | 13.8% |
| SIGHAN05-PKU (300) | 19.26 | **23.43** | 0.00% | 26.67% | 0.0% | 10.3% |
| CommonGen val (993) | 35.81 | **44.46** | 0.00% | 26.28% | 0.0% | 22.9% |

CommonGen coverage: v2 83.94%, slot template 83.97%. v2 is 100% grounded on every set. It runs in 7.9 ms per riff on the host, with 641,726 B in PSRAM and 1,048,608 B on SD. The no-learning slot template beats it on MiniLM similarity everywhere. The writer's contribution is structural: it never breaks a sentence and never invents a name.

## 6. LiquidRefresh CfC v2 (rc10 firmware)

| Metric | Value |
|---|---|
| Parameters | 339 int8 (8 inputs → 16 CfC units → 3 readouts) |
| Refresh-gate accuracy, 1,500 held-out episodes | 99.9% (TP 860, FN 2, FP 2, TN 3021; recall 99.8%, precision 99.8%) |
| Non-initial full-screen clears, card config | 10 → 8 vs the static rule (−20%) |
| Agreement with LiquidRefresh v1, frame by frame | 712 / 712 |

## 7. Efficiency (Apple M1 host, C int8 kernel at -O2; not ESP32-S3 timings)

| Config | Encode µs (median) | p90 µs | int8 MACs/query | Bytes/chunk | CS-PW weave (M=4) µs |
|---|---:|---:|---:|---:|---:|
| Arm4 deployed (E3 + engram.bin + cspw.bin) | 6,030 | 11,240 | 8,496,818 | 4.60 | 399 |
| v2s_arm1 PLHN only | 6,112 | 12,388 | 9,224,502 | 3.96 | — |

| Model | Deploy params | Flash/SD KB | RAM KB | MACs/query |
|---|---:|---:|---:|---:|
| mine: v2s_arm1 (no Engram) | 94,976 | 92.8 | 96.8 | 9,224,502 |
| mine: v2s_E3 (deployed encoder) | 1,667,840 | 4,189.3 | 96.8 | 8,496,818 |
| mine: Arm4 full (E3 + Engram + CS-PW) | 2,061,568 | 4,573.8 | 487.6 | 9,502,258 |
| matched-RAM BPE (Arm0) | 140,800 | 137.5 | 550.0 | 8,192 |

The 1,667,840 deploy parameters are 94,976 encoder weights plus 65,536 × 24 Engram deltas. The deltas sit on SD and are read one slot at a time.

## Parity and tests

C int8 vs numpy segmentation: 38,580 / 38,580 lines. Encoder vs device binary: 7,480 / 7,480 texts. PE-Writer Python int8 vs firmware: 64 / 64. CS-PW parity: 0 mismatches. Firmware `ConceptEngineTest`: 30 / 30. Host `ctest`: 1067 / 1067 (1 skipped).
