# Comprehensive Retrieval Benchmark Report: Neural vs. Multi-Vector vs. Lexical RAG

**Date:** October 10, 2026  
**Hardware Platforms:** Google Colab (NVIDIA T4 16GB) & Kaggle (Dual NVIDIA T4 15GB x 2)  
**Evaluator:** Antigravity AI Engineering Suite  

---

## Executive Summary

This study conducted an empirical end-to-end evaluation across **academic IR benchmarks (BEIR SciFact & NFCorpus)** and a **production-scale technical domain (Cybersecurity Corpus: 1,243 chunks across 3 reference books)**. We compared 4 distinct retrieval architectures:

1. **BM25 (Okapi)**: Lexical exact-term matching (classic CPU inverted index).
2. **google/embeddinggemma-2 (270M)**: Compact single-vector dense bi-encoder (768-dim, Matryoshka-capable).
3. **perplexity-ai/pplx-embed-v2-late-0.6b (340M)**: Token-level multi-vector late interaction (ColBERT MaxSim architecture).
4. **perplexity-ai/pplx-embed-v2-context-9b-preview (9B)**: Full autoregressive cross-chunk attention contextual bi-encoder (2048-dim, unquantized FP16).

---

## 1. Master Evaluation Scoreboard

### A. Academic Benchmarks (BEIR)

| Benchmark Suite | Metric | BM25 (Lexical) | EmbeddingGemma 2 (270M Dense) | Late-0.6b (340M Multi-Vector) | Context-9B (9B Cross-Chunk) | Architecture Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SciFact** (Scientific Claims) | **nDCG@10** | 0.6532 | **0.8524** | 0.7770 | **0.8183** | **EmbeddingGemma 2** (+19.9 vs BM25) |
| SciFact (5,183 docs, 300 queries) | Recall@10 | 0.7891 | **0.9521** | 0.8920 | **0.9367** | EmbeddingGemma 2 |
| SciFact | MAP@10 | 0.6120 | **0.8045** | 0.7254 | **0.7778** | EmbeddingGemma 2 |
| **NFCorpus** (Medical / Nutrition) | **nDCG@10** | 0.3060 | 0.3646 | **0.4060** | **0.3988** | **Late-0.6b** (+10.0 vs BM25) |
| NFCorpus (3,633 docs, 323 queries)| Recall@10 | 0.2230 | 0.2450 | **0.2710** | **0.1975** | Late-0.6b |
| NFCorpus | MAP@10 | 0.1250 | 0.1420 | **0.1680** | **0.1550** | Late-0.6b |

---

### B. In-Domain Cybersecurity Technical Manuals (1,243 Chunks)
Corpus: *Metasploit: The Penetration Tester's Guide*, *SQL Injection Attacks and Defense*, *The Web Application Hacker's Handbook (WAHH)*.

| Query Test Category | BM25 (Lexical) | EmbeddingGemma 2 (270M) | Late-0.6b (340M) | Context-9B (9B FP16) | Hybrid RRF (50/50) | Hybrid Alpha (0.9 Dense) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Technical CLI (15 Queries)** | **93.3%** (14/15) | 86.7% (13/15) | 86.7% (13/15) | **93.3%** (14/15) | 86.7% (13/15) | **93.3%** (14/15) |
| **Conceptual Zero-Keyword (15 Queries)**| 60.0% (9/15) | **86.7%** (13/15) | 80.0% (12/15) | **86.7%** (13/15) | 80.0% (12/15) | **86.7%** (13/15) |
| **Total 30-Query Top-1 Accuracy** | **76.7%** (23/30) | **86.7%** (26/30) | **83.3%** (25/30) | **90.0%** (27/30) | **83.3%** (25/30) | **90.0%** (27/30) |
| **Vocabulary Drop Delta** | **-33.3%** | **0.0%** (Invariant) | **-6.7%** | **-6.7%** | -6.7% | -6.7% |

---

### C. Granular Query-Level Failure & Hit Analysis (Context-9B)

Across all 30 queries, Context-9B achieved a **90.0% overall Top-1 accuracy** (27 hits / 3 misses). Inspecting the exact failures reveals the subtle mechanics of semantic retrieval:

#### 1. Exact CLI Queries (14/15 Hits - 93.3%)
- **14 Hits**: Perfect disambiguation on technical tools, commands, and options across Metasploit (`getsystem`, `shikata_ga_nai`, `msfpayload`), SQLi (`WAITFOR DELAY`, second-order, DNS out-of-band, comment tricks), and WAHH (`double submit cookie`, `DOM sources/sinks`, `null byte traversal`).
- **The Single Miss (Q13 - Padding Oracle)**:
  - *Query*: `"padding oracle attack cbc mode block cipher decryption without key"`
  - *Expected*: `[WAHH]` (Chapter on Cryptography)
  - *Retrieved Top-1*: `[SQLi_Attacks]` (Score: 0.378)
  - *Diagnosis*: WAHH mentions padding oracle attacks briefly in older editions, but the extracted textbook chunks had low lexical concentration. SQLi Attacks contained a chapter on database password hash decryption (MD4/MD5/SHA1) that scored 0.378 vs WAHH's 0.344 (which placed at Rank #9).

#### 2. Conceptual Zero-Keyword Queries (13/15 Hits - 86.7%)
- **13 Hits**: Exceptional zero-keyword semantic reasoning:
  - *Q01* ("escalating a compromised process without credentials on windows") -> correctly mapped to `getsystem` in Metasploit (Score 0.538).
  - *Q05* ("inferring secret database contents one character at a time by making the server pause") -> correctly mapped to Time-based blind SQLi (Score 0.608).
  - *Q15* ("accessing restricted files outside web root by terminating string prematurely") -> correctly mapped to Null byte path truncation in WAHH (Score 0.593).
- **Miss 1 (Q06 - Stored/Second-Order Attack Ambiguity)**:
  - *Query*: `"saving malicious data into user profile records that trigger code evaluation only when reviewed by an administrator"`
  - *Expected*: `[SQLi_Attacks]` (Second-order SQL injection)
  - *Retrieved Top-1*: `[WAHH]` (Score: 0.430) | Target SQLi appeared at Rank #4 (Score: 0.397).
  - *Diagnosis*: **Cross-domain conceptual overlap**. The query describes storing malicious input in a profile to execute on an admin screen. This is both *Second-Order SQLi* AND *Stored XSS*! WAHH explicitly discusses stored malicious input in user profiles attacking administrators, so Context-9B retrieving WAHH was semantically valid.
- **Miss 2 (Q13 - Conceptual Padding Oracle)**:
  - *Query*: `"decrypting encrypted data blocks without knowing the secret key by analyzing error responses from validation routines"`
  - *Expected*: `[WAHH]`
  - *Retrieved Top-1*: `[SQLi_Attacks]` (Score: 0.411) | Target WAHH appeared at Rank #2 (Score: 0.409 - a margin of only 0.002!).
  - *Diagnosis*: Target WAHH was essentially tied for Rank #1, missing by only two thousandths of a score point against an error-based database decryption chunk.

---

## 2. Deep Technical Analysis & Findings

### Finding 1: The "Lexical Drag" of Naive RRF
- **Why Naive RRF degraded performance (90.0% -> 83.3%)**:
  Standard Reciprocal Rank Fusion assumes that both rankers have roughly balanced discriminatory signal. When queries contain **zero overlapping keywords** with the ground truth chunk, BM25 assigns high ranks to random documents containing generic filler tokens (e.g., "attack", "data", "system"). A 50/50 RRF rank-sum dragged down Context-9B's clean semantic retrieval.
- **Production Solution**: Use **Alpha Weighted Fusion** ($\ge 0.85$ neural weight) or a **Classifier Gate** (only engage BM25 when query entity density is high, such as exact function names or CVE numbers).

### Finding 2: Where Late-Interaction (ColBERT MaxSim) Excels
- On **NFCorpus**, Late-0.6b decisively beat both single-vector dense models and the 9B model (`0.4060` vs `0.3646` vs `0.3988`).
- **Reason**: Medical documents feature highly specific, multi-token terminology (drug names, biochemical compounds, dosage protocols). ColBERT's per-token MaxSim allows query tokens to individually align against target sub-spans without being squashed into a single pooled sentence embedding.

### Finding 3: The Context-9B Advantage on Long Continuous Texts
- On BEIR's short standalone abstracts, Context-9B is competitive but does not beat Gemma-2.
- On **1,243 continuous book chunks**, Context-9B was the **only neural model to hit 93.3% on exact CLI commands** while matching Gemma's **86.7% on conceptual abstraction**, achieving the highest overall standalone accuracy (**90.0%**).
- **Reason**: Cross-chunk shared attention resolves cross-boundary references ("this command", "the preceding payload", "step 2 above") that cause single-chunk models to lose context.

---

## 3. Engineering & Operational Profiles

| Model | Weights Size | Active VRAM Footprint | Docs / Sec Throughput (T4) | Index Storage (1M Docs) | Recommended Deployment Tier |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25** | < 5 MB | ~200 MB CPU RAM | > 10,000 docs/s | ~50 MB | Low-cost CPU worker / Elasticsearch |
| **EmbeddingGemma 2** | 540 MB (FP16) | ~1.2 GB VRAM | ~350 docs/s | 3.0 GB (768-d float32) | Edge, Mobile, Single T4, Cheap Cloud |
| **Late-0.6b** | 680 MB (FP16) | ~1.8 GB VRAM | ~120 docs/s | 32 GB (Multi-vector) | Enterprise Search, Medical / Legal RAG |
| **Context-9B** | 18.0 GB (FP16) | ~22.0 GB VRAM (Dual T4) | ~2.1 docs/s | 8.0 GB (2048-d float32) | Offline Document Ingestion / Re-ranking |

---

## 4. Production Architectural Blueprint

For building an optimal, cost-efficient RAG system over technical manuals:

```
                  ┌──────────────────────┐
                  │      User Query      │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Lexical Pipeline]                [Semantic Pipeline]
      BM25 (CPU)                    EmbeddingGemma 2 (Dense)
     Index: < 50MB                     Latency: < 15ms
            │                                 │
            └────────────────┬────────────────┘
                             ▼
               [Hybrid Gated Re-Ranker]
                 - Exact code/CLI: 80% BM25
                 - Conceptual questions: 90% Dense
                             ▼
                Top-50 Candidate Chunks
                             ▼
             [Optional Deep Re-scoring]
             Late-0.6b MaxSim / Context-9B
                             ▼
                 Final Top-3 RAG Context
```
