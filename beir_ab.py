"""
BEIR A/B harness: EmbeddingGemma 2 vs pplx-embed-v2-late-0.6b

Design notes
------------
pylate is NOT usable here: pylate 1.6.0 pins sentence-transformers==5.3.0 and
transformers<=5.3.0, while pplx-embed-v2-late requires >=6.0.0 / >=5.4.0.
So scoring is brute-force MaxSim, which is fine at BEIR scale
(SciFact 5.2k docs, NFCorpus 3.6k docs) and needs no index.

Two silent-failure traps are handled explicitly:
  * EmbeddingGemma 2 needs task instruction prefixes, and needs
    documents formatted as "title: {title} | text: {content}".
  * fp16 returns NaN / degraded embeddings. We use bf16 if the GPU
    supports it, else fp32. Never fp16.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- data

DATASETS = {
    "scifact":  {"corpus": 5183, "queries": 300},
    "nfcorpus": {"corpus": 3633, "queries": 323},
    "scidocs":  {"corpus": 25657, "queries": 1000},
}


def load_beir(name: str, cache_dir: str | None = None):
    """Load a BEIR dataset as (corpus_ids, corpus_texts, query_ids, queries, qrels).

    corpus_texts are already formatted for EmbeddingGemma 2 document input.
    """
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus",
                          cache_dir=cache_dir)
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries",
                           cache_dir=cache_dir)
    # qrels live in the *_qrels dataset; fall back to the eval split
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", split="test",
                            cache_dir=cache_dir)

    c_ids = [str(x) for x in corpus["_id"]]
    titles = corpus["title"] if "title" in corpus.column_names else [""] * len(c_ids)
    texts = corpus["text"]
    c_texts = [
        f"title: {(t or '').strip() or 'none'} | text: {x.strip()}"
        for t, x in zip(titles, texts)
    ]

    q_ids, q_texts = [], []
    for row in queries:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        q_ids.append(str(row["_id"]))
        q_texts.append(text)

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid, did, rel = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if rel > 0:
            qrels.setdefault(qid, {})[did] = rel

    # keep only queries that have at least one positive judgement
    keep = [i for i, qid in enumerate(q_ids) if qid in qrels]
    q_ids = [q_ids[i] for i in keep]
    q_texts = [q_texts[i] for i in keep]

    return c_ids, c_texts, q_ids, q_texts, qrels


# ------------------------------------------------------------------ metrics

def evaluate(run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]],
             k: int = 10) -> dict[str, float]:
    """nDCG@k / Recall@k / MRR@k via pytrec_eval, matching BEIR conventions."""
    import pytrec_eval

    ev = pytrec_eval.RelevanceEvaluator(
        {q: {d: int(r) for d, r in docs.items()} for q, docs in qrels.items()},
        {f"ndcg_cut.{k}", f"recall_{k}", f"map_cut.{k}", "recip_rank"},
    )
    per = ev.evaluate(run)

    out = {}
    for metric in (f"ndcg_cut_{k}", f"recall_{k}", f"map_cut_{k}", "recip_rank"):
        vals = [v[metric] for v in per.values() if metric in v]
        out[metric] = float(np.mean(vals)) if vals else 0.0
    return out


# ------------------------------------------------------------- retrievers

class BM25Retriever:
    """Lexical baseline. Needs no GPU; sanity-checks the eval path."""

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b

    def run(self, c_texts, q_texts, batch=64):
        from rank_bm25 import BM25Okapi
        tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        bm = BM25Okapi([tok(t) for t in c_texts])
        return {qi: {di: float(s) for di, s in
                     zip(range(len(c_texts)), bm.get_scores(tok(q)))}
                for qi, q in enumerate(q_texts)}


class GemmaRetriever:
    """Dense single-vector. Correct prefixes, bf16-or-fp32, never fp16."""

    def __init__(self, model_id="google/embeddinggemma-2", dim=768, batch=16):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.dim, self.batch = dim, batch
        self.model = SentenceTransformer(
            model_id,
            model_kwargs={"torch_dtype": (
                torch.bfloat16 if torch.cuda.is_available()
                and torch.cuda.is_bf16_supported() else torch.float32)},
            config_kwargs={"vision_config": None, "audio_config": None},
        )

    def encode(self, texts, is_query: bool):
        # Queries get the SearchQuery task prefix. Documents are already
        # formatted as "title: {title} | text: {content}" by load_beir,
        # which IS the document-side instruction for titled documents, so
        # they must NOT receive the query prefix. Getting this wrong
        # degrades quality silently rather than raising.
        kw = dict(batch_size=self.batch, truncate_dim=self.dim,
                  normalize_embeddings=True, show_progress_bar=True)
        if is_query:
            kw["prompt_name"] = "SearchQuery"
        return self.model.encode(texts, **kw).astype("float32")

    def run(self, c_texts, q_texts):
        c = self.encode(c_texts, False)
        q = self.encode(q_texts, True)
        sims = q @ c.T                      # (nq, ndoc)
        return {i: dict(enumerate(row)) for i, row in enumerate(sims)}


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype="float32")


class LateRetriever:
    """ColBERT-style late interaction, brute-force MaxSim, no index needed.

    128 dims per token; score = sum over query tokens of max over doc tokens.
    """

    def __init__(self, model_id="perplexity-ai/pplx-embed-v2-late-0.6b",
                 batch=8, doc_batch=64):
        from sentence_transformers import MultiVectorEncoder
        self.model = MultiVectorEncoder(model_id, device="cuda" if _cuda() else "cpu")
        self.batch, self.doc_batch = batch, doc_batch

    def run(self, c_texts, q_texts):
        import torch
        print(f"  encoding {len(q_texts)} queries...", flush=True)
        qv = self.model.encode_query(q_texts, batch_size=self.batch)
        # MultiVectorEncoder returns a list of (n_tokens, dim) CUDA tensors or arrays
        qv = [_to_numpy(x) for x in qv]

        scores = np.zeros((len(q_texts), len(c_texts)), dtype="float32")
        n_batches = (len(c_texts) + self.doc_batch - 1) // self.doc_batch
        print(f"  encoding {len(c_texts)} docs in {n_batches} batches...", flush=True)
        for b_idx, d0 in enumerate(range(0, len(c_texts), self.doc_batch)):
            d1 = min(d0 + self.doc_batch, len(c_texts))
            dv = [_to_numpy(x)
                  for x in self.model.encode_document(c_texts[d0:d1],
                                                      batch_size=self.batch)]
            for qi in range(len(q_texts)):
                scores[qi, d0:d1] = maxsim(qv[qi], dv)
            if (b_idx + 1) % 10 == 0 or b_idx == n_batches - 1:
                print(f"    processed docs {d1}/{len(c_texts)} (batch {b_idx+1}/{n_batches})", flush=True)
        return {i: dict(enumerate(row)) for i, row in enumerate(scores)}


def maxsim(q: np.ndarray, docs: list[np.ndarray]) -> np.ndarray:
    """sum_t max_d (q_t . d_d) for one query against many docs."""
    out = np.empty(len(docs), dtype="float32")
    for j, d in enumerate(docs):
        if q.size == 0 or d.size == 0:
            out[j] = -1e9
            continue
        out[j] = (d @ q.T).max(axis=0).sum()
    return out


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact", choices=list(DATASETS))
    ap.add_argument("--models", default="bm25,gemma,late")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    print(f"loading {args.dataset} ...", flush=True)
    c_ids, c_texts, q_ids, q_texts, qrels = load_beir(args.dataset)
    if args.limit_queries:
        c_ids, c_texts = c_ids, c_texts
        q_ids, q_texts = q_ids[:args.limit_queries], q_texts[:args.limit_queries]
        qrels = {q: r for q, r in qrels.items() if q in set(q_ids)}
    print(f"  corpus={len(c_ids):,}  queries={len(q_ids)}  "
          f"judged queries={len(qrels)}", flush=True)

    results = {"dataset": args.dataset, "corpus": len(c_ids),
               "queries": len(q_ids),
               "judged": len(qrels), "k": args.k, "models": {}}

    for name in args.models.split(","):
        name = name.strip()
        print(f"\n=== {name} ===", flush=True)
        t0 = time.perf_counter()
        try:
            if name == "bm25":
                r = BM25Retriever()
            elif name == "gemma":
                r = GemmaRetriever(dim=args.dim)
            elif name == "late":
                r = LateRetriever()
            else:
                raise SystemExit(f"unknown model {name}")

            run = r.run(c_texts, q_texts)
            # map int keys back to real doc ids
            mapped = {q_ids[qi]: {c_ids[di]: float(s)
                                  for di, s in docs.items()}
                      for qi, docs in run.items() if q_ids[qi] in qrels}
            m = evaluate(mapped, qrels, k=args.k)
            m["seconds"] = round(time.perf_counter() - t0, 1)
            results["models"][name] = m
            print(f"  nDCG@{args.k}={m[f'ndcg_cut_{args.k}']:.4f}  "
                  f"R@{args.k}={m[f'recall_{args.k}']:.4f}  "
                  f"MRR={m['recip_rank']:.4f}  ({m['seconds']}s)", flush=True)
        except Exception as e:
            results["models"][name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()