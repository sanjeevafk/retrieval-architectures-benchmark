# Retrieval Architectures Benchmark & Evaluation Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-BEIR%20%7C%20Cybersecurity-orange.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Colab%20T4%20%7C%20Kaggle%20Dual%20T4-green.svg)]()

An empirical benchmark comparing lexical, dense single-vector, multi-vector late interaction, and full cross-chunk contextual bi-encoder retrieval architectures across academic datasets and in-domain technical documentation.

---

## 🏆 Key Findings at a Glance

| Benchmark Suite / Domain | BM25 (Lexical) | EmbeddingGemma 2 (270M) | Late-0.6b (340M) | Context-9B (9B FP16) | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BEIR SciFact** (Scientific Claims) | 0.6532 | **0.8524** | 0.7770 | 0.8183 | **EmbeddingGemma 2** (+19.9 vs BM25) |
| **BEIR NFCorpus** (Medical / Nutrition) | 0.3060 | 0.3646 | **0.4060** | 0.3988 | **Late-0.6b** (+10.0 vs BM25) |
| **Security Books (Exact Technical CLI)** | **93.3%** | 86.7% | 86.7% | **93.3%** | **Context-9B & BM25** |
| **Security Books (Conceptual Zero-Keyword)** | 60.0% (-33.3%) | **86.7%** (+0.0%) | 80.0% (-6.7%) | **86.7%** (-6.7%) | **Context-9B & Gemma 2** |
| **Total 30-Query Top-1 Accuracy** | **76.7%** | **86.7%** | **83.3%** | **90.0%** | **Context-9B** |

---

## 🔬 Evaluated Architectures

1. **BM25 (Okapi)**: Classical inverted index baseline matching exact term frequencies.
2. **`google/embeddinggemma-2` (270M)**: Modern compact dense bi-encoder with 768 dimensions and Matryoshka capability.
3. **`perplexity-ai/pplx-embed-v2-late-0.6b` (340M)**: Multi-vector token-level late interaction based on ColBERT MaxSim architecture.
4. **`perplexity-ai/pplx-embed-v2-context-9b-preview` (9B FP16)**: Autoregressive bi-encoder utilizing shared cross-chunk attention across sliding window document segments.

---

## 📂 Repository Structure

```text
├── SHOWCASE_RETRIEVAL_BENCHMARKS.ipynb   # Master presentation notebook with outputs & analysis
├── BENCHMARK_REPORT.md                   # Full empirical analysis, hardware profiles & audit
│
├── notebook_runs/                        # Verified execution notebooks with recorded outputs
│   ├── 01_embeddinggemma_scifact.ipynb
│   ├── 02_embeddinggemma_nfcorpus.ipynb
│   ├── 03_late_interaction_scifact.ipynb
│   ├── 04_late_interaction_nfcorpus.ipynb
│   ├── 05_security_books_bm25_gemma_late.ipynb
│   └── 06_kaggle_dual_t4_context9b_full_eval.ipynb
│
└── sec_eval/                             # In-domain cybersecurity corpus & generators
    ├── corpus.jsonl                      # 1,243 chunks from 3 cybersecurity reference manuals
    ├── queries.json                      # 15 exact technical CLI evaluation queries
    ├── queries_conceptual.json           # 15 conceptual zero-keyword queries
    ├── make_colab_sec_eval.py            # Colab notebook generator script
    ├── make_kaggle_context9b.py          # Kaggle Dual-T4 notebook generator script
    └── run_sec_eval.py                   # Standalone Python BM25 baseline script
```

---

## 🧠 Architectural Insights

### 1. The "Lexical Drag" of Naive Reciprocal Rank Fusion (RRF)
- Naive 50/50 RRF (`k=60`) degraded performance on conceptual queries from **90.0% down to 83.3%**.
- When a neural model is significantly stronger than BM25 on conceptual intent (86.7% vs 60.0%), unweighted reciprocal fusion allows lexical false-positives to drag down true semantic hits.
- **Solution**: Use **Alpha-Weighted Fusion** ($\ge 0.85$ neural weight) or a query-intent router.

### 2. When Late Interaction (ColBERT MaxSim) Wins
- On **NFCorpus**, Late-0.6b outperformed dense single-vector models (`0.4060` vs `0.3646` vs `0.3988`).
- Fine-grained token matching excels on specialized medical and scientific entities that lose fidelity when pooled into a single sentence vector.

### 3. The Power of Cross-Chunk Context
- On continuous multi-chunk manuals, Context-9B was the **only neural model to achieve 93.3% on exact CLI commands** while matching Gemma on **conceptual queries (86.7%)**, achieving **90.0% total accuracy**.
- Cross-chunk attention resolves references across neighboring chunks ("the above payload", "step 2 in previous section") that cause isolated single-chunk retrievers to fail.

---

## 🚀 Running the Benchmarks

To run the local BM25 evaluation:
```bash
python3 sec_eval/run_sec_eval.py
```

To inspect the master showcase notebook:
```bash
jupyter notebook SHOWCASE_RETRIEVAL_BENCHMARKS.ipynb
```
