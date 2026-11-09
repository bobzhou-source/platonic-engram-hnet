# Benchmarks

Every number here is copied from the reports written by the runners in `benchmarks/` (word sense / cross-lingual, segmentation + efficiency, narrative / spoiler, margin writer) and from the firmware Liquid CfC refresh-controller evaluation. Each report is generated from its raw JSON. My encoders run the exact [int8 QAT](https://arxiv.org/abs/1712.05877) graph that ships on the device. The shipped encoder was checked bit-exact against the device binary on 7,480/7,480 benchmark texts. The matched-RAM BPE baseline (140,800 parameters, 1.82 bytes per token) has no int8 export and runs in FP32. No model was trained or tuned on any benchmark set. Thresholds are tuned on dev splits only.

Encoders compared:

- **H-Net byte router only (no Engram):** the 95 KB [H-Net](https://arxiv.org/abs/2507.07955) byte encoder by itself.
- **Shipped encoder (H-Net + DeepSeek Engram + Qwen3 KD):** the same router plus the SD-card [DeepSeek Engram](https://arxiv.org/abs/2601.07372) table, the bigram-span Engram, and [Qwen3](https://arxiv.org/abs/2505.09388) relational knowledge distillation. This is what runs on the device.
- **Matched-RAM BPE baseline:** a subword (BPE) encoder given the same RAM budget.

## 1. Chinese word segmentation (word F1, gold human annotation)

Datasets: [SIGHAN 2005](https://aclanthology.org/I05-3017/) PKU and MSR, and [UD-GSDSimp](https://universaldependencies.org/treebanks/zh_gsdsimp/index.html).

| System | SIGHAN05-PKU | SIGHAN05-MSR | UD-GSDSimp | PKU boundary F1 |
|---|---:|---:|---:|---:|
| **Shipped encoder (H-Net + DeepSeek Engram + Qwen3 KD)** | **0.394** | **0.402** | **0.431** | 0.692 |
| H-Net byte router only (no Engram) | 0.360 | 0.350 | 0.399 | 0.745 |
| Matched-RAM BPE baseline | 0.322 | 0.307 | 0.371 | 0.737 |
| char-unigram heuristic | 0.343 | 0.330 | 0.394 | 0.750 |
| MiniLM-L6-v2 tokenizer | 0.349 | 0.334 | 0.428 | 0.757 |
| LFM2-350M tokenizer | 0.488 | 0.499 | 0.485 | 0.803 |
| BGE-M3 tokenizer | 0.649 | 0.659 | 0.614 | 0.865 |
| Qwen3-Embedding-0.6B tokenizer | 0.656 | 0.671 | 0.599 | 0.860 |
| [jieba](https://github.com/fxsjy/jieba) 0.42.1 (supervised) | 0.818 | 0.813 | 0.799 | 0.932 |
| oracle: gold + 12-byte cap | 0.997 | 0.972 | 0.996 | 0.999 |

The shipped router beats matched-RAM BPE on word F1 on all three sets, but it loses to BPE on boundary F1 (0.692 vs 0.737). The LFM2-350M, BGE-M3 and Qwen3 tokenizers all beat it. The 12-byte cap alone costs very little: the gold + cap oracle scores 0.997 on PKU, even though 214 PKU gold words exceed 12 bytes.

## 2. Bits per byte (lower is better)

Datasets: [WikiText-2](https://arxiv.org/abs/1609.07843), SIGHAN 2005, UD-GSDSimp and [Tatoeba](https://tatoeba.org/).

| Model | WikiText-2 (en) | SIGHAN05-PKU (zh) | SIGHAN05-MSR (zh) | UD-GSDSimp (zh) | Tatoeba (zh/en) |
|---|---:|---:|---:|---:|---:|
| Matched-RAM BPE baseline, FP32 | 3.4925 | 4.0200 | 3.9243 | 3.9944 | **4.4297** |
| **H-Net byte router only (no Engram), FP32** | **3.4636** | **3.8351** | **3.7774** | **3.8338** | 4.6553 |
| H-Net byte router only (no Engram), INT8 | 3.5028 | 3.8772 | 3.8202 | 3.8806 | 4.4860 |
| Shipped encoder (H-Net + DeepSeek Engram + Qwen3 KD), FP32 | 3.5789 | 4.0487 | 3.9675 | 4.0604 | 4.6010 |
| Shipped encoder (H-Net + DeepSeek Engram + Qwen3 KD), INT8 | 3.6038 | 4.0697 | 3.9843 | 4.0793 | 4.7322 |
| gzip -9, whole corpus | 2.7617 | 3.4249 | 3.2641 | 4.2780 | 3.9139 |

The H-Net without Engram beats matched-RAM BPE on the four monolingual sets. The shipped Engram encoder loses to BPE on all five. Whole-corpus compressors are not window-limited, so they are a reference only.

## 3. Word sense, cross-lingual retrieval and sentence similarity

Datasets: [WiC](https://arxiv.org/abs/1808.09121), [XL-WiC](https://arxiv.org/abs/2010.06478), [MCL-WiC](https://aclanthology.org/2021.semeval-1.3/), [Tatoeba](https://tatoeba.org/), [STS-B](https://arxiv.org/abs/1708.00055) (Chinese STS-B, BQ and ATEC from [C-MTEB](https://arxiv.org/abs/2309.07597)), and [PAWS-X](https://arxiv.org/abs/1908.11828).

Primary metric per task. WiC tasks use test accuracy of the dev-selected feature. Tatoeba uses mean R@1 over both directions. STS and paraphrase tasks use Spearman. The upper-bound reference is multilingual [MiniLM-L12](https://arxiv.org/abs/2004.09813).

| Task | Shipped encoder | vs best lexical | vs matched-RAM BPE | vs MiniLM-L12 (235.31 MB) |
|---|---:|---|---|---|
| WiC (en) | 0.507 | loss (0.540, char-3) | tie (0.521) | loss (0.608) |
| XL-WiC (zh) | 0.532 | loss (0.608, BM25) | **win** (0.509) | loss (0.632) |
| MCL-WiC en-en | 0.577 | loss (0.632, BM25) | **win** (0.523) | loss (0.693) |
| MCL-WiC zh-zh | 0.546 | loss (0.610, BM25) | tie (0.548) | loss (0.712) |
| MCL-WiC en-zh | 0.503 | tie (0.501) | tie (0.500) | loss (0.709) |
| Tatoeba cmn-eng | 0.023 | **win** (0.016) | **win** (0.003) | loss (0.962) |
| STS-B (en) | 0.328 | loss (0.648, char-3) | tie (0.293) | loss (0.844) |
| STS-B (zh, C-MTEB) | 0.490 | loss (0.556, BM25) | **win** (0.394) | loss (0.789) |
| PAWS-X (en) | 0.143 | tie (0.174) | tie (0.178) | tie (0.173) |
| PAWS-X (zh) | 0.082 | loss (0.157, char-3) | **loss** (0.159) | tie (0.101) |
| BQ (zh, C-MTEB) | 0.228 | loss (0.383, BM25) | tie (0.222) | loss (0.385) |
| ATEC (zh, C-MTEB) | 0.105 | loss (0.208, BM25) | **win** (0.088) | loss (0.184) |

Against matched-RAM BPE: 5 wins, 6 ties, 1 loss. Against the best lexical baseline: 1 win, 2 ties, 9 losses. Against MiniLM-L12: 0 wins, 2 ties, 10 losses.

Tatoeba cmn-eng detail (1,000 pairs): zh→en R@1 0.030, R@10 0.103. en→zh R@1 0.016. MiniLM-L12 scores 0.961 / 0.996 / 0.964.

## 4. Spoiler-free character tracking, public BookSum

Dataset: [BookSum](https://arxiv.org/abs/2105.08209) chapter-level test split. There are 57 (book, study-guide source) sequences and 439 character timelines. That gives **9,169 cutoff queries**, including 245 probes asked before the character is introduced. **Spoiler leakage** is the share of answers that reveal anything past the reader's position. **Fact precision** is token precision against the original book text up to the cutoff.

| System | Spoiler leakage | Spoiler leakage (before introduction) | Fact precision | Spoiler-token rate | Coverage |
|---|---:|---:|---:|---:|---:|
| **Spoiler-gated Liquid CfC memory (page gate + int8 weave)** | **0.000** | **0.000** | 0.703 | 0.084 | 0.973 |
| Same memory, page gate removed | 1.000 | 1.000 | 0.683 | 0.106 | 1.000 |
| Flat ungated memory variants (4 ablations) | 1.000 | 1.000 | 0.529–0.560 | 0.106–0.123 | 1.000 |
| BM25 top-1 sentence, ungated | 0.442 | 1.000 | 0.742 | 0.095 | 1.000 |
| BM25 top-1 sentence, cutoff-filtered | 0.000 | 0.000 | **0.772** | 0.061 | 0.973 |

The page gate is what makes the character memory spoiler-free. A cutoff-filtered BM25 is also spoiler-free and has higher fact precision. Internal eval: spoiler leakage 0.000 on 3,631 lookups (ungated variants 85.6%), temporal-swap accuracy 0.9946, dossier fact precision 0.888, cross-book concept F1 0.385. The public BookSum story graph used here is published as `booksum-story-graph.idx` (1,441,280 B), and re-running the benchmark from it reproduces spoiler leakage 0.000 / fact precision 0.703.

## 5. Concept-to-text (clause-locked margin writer v2), 4,226 public examples

Datasets: [BookSum](https://arxiv.org/abs/2105.08209), [NarrativeQA](https://arxiv.org/abs/1712.07040), [Tatoeba](https://tatoeba.org/), UD-GSDSimp, SIGHAN 2005 PKU and [CommonGen](https://arxiv.org/abs/1911.03705).

[Sentence-BERT](https://arxiv.org/abs/1908.10084) MiniLM cosine ×100 against the human reference, plus defect rates. v1 is the earlier token-by-token writer.

| Set (n) | v2 MiniLM | Slot template (no learning) | v2 broken sentences | v1 broken sentences | v2 invented names | v1 invented names |
|---|---:|---:|---:|---:|---:|---:|
| BookSum (723) | 44.17 | **47.02** | 0.00% | 51.18% | 0.0% | 26.1% |
| NarrativeQA (710) | 19.26 | **21.21** | 0.00% | 56.20% | 0.0% | 21.7% |
| Tatoeba zh→en (1,000) | 11.13 | **13.60** | 0.00% | 21.50% | 0.0% | 19.6% |
| UD-GSDSimp (500) | 12.99 | **15.00** | 0.00% | 27.80% | 0.0% | 13.8% |
| SIGHAN05-PKU (300) | 19.26 | **23.43** | 0.00% | 26.67% | 0.0% | 10.3% |
| CommonGen val (993) | 35.81 | **44.46** | 0.00% | 26.28% | 0.0% | 22.9% |

CommonGen coverage: v2 83.94%, slot template 83.97%. v2 is 100% grounded on every set. It runs in 7.9 ms per note on the host, with 641,726 B in PSRAM and 1,048,608 B on SD. The no-learning slot template beats it on MiniLM similarity everywhere. The writer's contribution is structural: it never breaks a sentence and never invents a name.

## 6. Liquid CfC e-ink refresh controller (rc10 firmware)

| Metric | Value |
|---|---|
| Parameters | 339 int8 (8 inputs → 16 [CfC](https://arxiv.org/abs/2106.13898) units → 3 readouts) |
| Refresh-gate accuracy, 1,500 held-out episodes | 99.9% (TP 860, FN 2, FP 2, TN 3021; recall 99.8%, precision 99.8%) |
| Non-initial full-screen clears, card config | 10 → 8 vs the static rule (−20%) |
| Agreement with the v1 controller, frame by frame | 712 / 712 |

## 7. Efficiency (Apple M1 host, C int8 kernel at -O2; not ESP32-S3 timings)

| Config | Encode µs (median) | p90 µs | int8 MACs/query | Bytes/chunk | Character memory weave (M=4) µs |
|---|---:|---:|---:|---:|---:|
| Shipped stack (encoder + `engram.bin` + `cspw.bin`) | 6,030 | 11,240 | 8,496,818 | 4.60 | 399 |
| H-Net byte router only (no Engram) | 6,112 | 12,388 | 9,224,502 | 3.96 | — |

| Model | Deploy params | Flash/SD KB | RAM KB | MACs/query |
|---|---:|---:|---:|---:|
| H-Net byte router only (no Engram) | 94,976 | 92.8 | 96.8 | 9,224,502 |
| Shipped encoder (H-Net + DeepSeek Engram + Qwen3 KD) | 1,667,840 | 4,189.3 | 96.8 | 8,496,818 |
| Shipped encoder + spoiler-gated Liquid CfC memory | 2,061,568 | 4,573.8 | 487.6 | 9,502,258 |
| Matched-RAM BPE baseline | 140,800 | 137.5 | 550.0 | 8,192 |

The 1,667,840 deploy parameters are 94,976 encoder weights plus 65,536 × 24 Engram deltas. The deltas sit on SD and are read one slot at a time.

## Parity and tests

C int8 vs NumPy segmentation: 38,580 / 38,580 lines. Encoder vs device binary: 7,480 / 7,480 texts. Margin writer Python int8 vs firmware: 64 / 64. Character memory parity: 0 mismatches. Firmware `ConceptEngineTest`: 30 / 30. Host `ctest`: 1067 / 1067 (1 skipped).
