"""Generate self-contained Colab notebooks from beir_ab.py.

Embedding the harness via %%writefile keeps each notebook runnable on its own
while guaranteeing it matches the local file byte for byte.
"""
import json
import pathlib

HARNESS = pathlib.Path("beir_ab.py").read_text()

PREAMBLE = """# %md
# BEIR A/B: {title}

Notes specific to this notebook are in the markdown cell below. The harness
(`beir_ab.py`) is written by the next cell, so this notebook is self-contained.
"""


def md(text):
    # .ipynb already encodes cell type; strip the editor-level "# %md" magic
    text = text.strip()
    if text.startswith("# %md"):
        text = text[len("# %md"):].strip()
    return {"cell_type": "markdown", "metadata": {},
            "source": text.split("\n")}


def code(text, tag=""):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def harness_cell():
    """ipynb source is a list of newline-terminated lines; no join."""
    lines = HARNESS.rstrip("\n").split("\n")
    src = ["%%writefile beir_ab.py\n"] + [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


GPU_CHECK = """
import torch, sys, transformers, sentence_transformers
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("sentence-transformers", sentence_transformers.__version__)

# embedding_gemma2 landed in a recent transformers release. If this is older
# the Gemma run dies deep inside model loading with a confusing message.
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
ok = "embedding_gemma2" in CONFIG_MAPPING_NAMES
print("embedding_gemma2 supported:", ok)
if not ok:
    raise RuntimeError(
        "transformers %s is too old for google/embeddinggemma-2. "
        "Run: %%pip install -q --upgrade 'transformers>=5.19.0'"
        % transformers.__version__
    )

print()
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"vram: {gb:.1f} GB")
    print()
    if not torch.cuda.is_bf16_supported():
        print("!! No native bf16 (Turing-class, e.g. T4).")
        print("!! EmbeddingGemma 2 MUST run fp32 -- fp16 returns NaN silently.")
else:
    print("!! No GPU. This notebook will be very slow or fail.")
"""

SUMMARY = """
# %md
## Reading the output

`ndcg_cut_10` is the headline BEIR number, identical to what papers report.
`recall_10`, `map_cut_10` and `recip_rank` (MRR) come along for free.

Two things to check before trusting a run:

1. **BM25 should land in a known range.** If it does not, the data or eval
   path is broken and every model number in this run is suspect. Reference
   nDCG@10 at `k1=0.9, b=0.4` on BEIR's own splits: SciFact ~0.66,
   NFCorpus ~0.32. NFCorpus is *supposed* to be far lower — it uses graded
   relevance and has many relevant documents per query.

Whichever model wins here answers only one question: *does a rerank pass
earn its cost on short documents?* Both datasets here are short and
self-contained, which favours single-vector dense retrieval. That is not the
regime your 250M-token book library lives in.
"""


def build(title, dataset, install, model_args, notes):
    cells = [
        md(PREAMBLE.format(title=title)),
        md(notes),
        code(install),
        code(GPU_CHECK),
        harness_cell(),
        md("## Run\n\nBM25 runs first as a control — it needs no GPU and finishes in seconds."),
        code(f"%run beir_ab.py --dataset {dataset} --models {model_args} "
             f"--k 10 --out results_{dataset}.json"),
        code("""
import json
r = json.load(open("RESULTS_JSON"))
rows = [("model", "nDCG@10", "Recall@10", "MRR", "secs")]
for name, m in r["models"].items():
    if "error" in m:
        rows.append((name, "ERROR", m["error"][:44], "", ""))
    else:
        rows.append((name, f"{m['ndcg_cut_10']:.4f}",
                     f"{m['recall_10']:.4f}", f"{m['recip_rank']:.4f}",
                     str(m["seconds"])))
w = [max(len(x[i]) for x in rows) for i in range(5)]
for row in rows:
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(row)))
print()
print(f"corpus={r['corpus']}  queries={r['queries']}  judged={r['judged']}")
""".replace("RESULTS_JSON", f"results_{dataset}.json")),
        md(SUMMARY),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": title, "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


GEMMA_NOTES = """
## What this notebook tests

Does **EmbeddingGemma 2** (270M text core, 768-dim) beat a lexical BM25
baseline on BEIR, and how does it compare when a late-interaction reranker
is added (see the companion notebook)?

**Model:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2),
Apache-2.0. MTEB code v1 NDCG@10 78.68, MIEB 64.64.

**Two silent failure modes are handled in the harness, not left to you:**

- *Task prefixes.* Gemma is trained with instruction prefixes. Queries use
  `task: search result | query: ...`; documents are formatted
  `title: {title} | text: {content}`. Omitting them still returns plausible
  vectors, just worse ones — no error is raised.
- *fp16.* The model card warns fp16 yields NaN or silently degraded
  embeddings. The harness selects bf16 when available and fp32 otherwise,
  and never fp16. On a T4 there is no native bf16, so you get fp32 and
  slower throughput.

`config_kwargs={"vision_config": None, "audio_config": None}` drops the
vision and audio encoders, taking the model from 740M to 270M params.
"""

LATE_NOTES = """
## What this notebook tests

Does **pplx-embed-v2-late-0.6b** (ColBERT-style late interaction, 340M active
params) beat plain dense retrieval enough to justify a rerank stage?

**Model:** [perplexity-ai/pplx-embed-v2-late-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-late-0.6b),
MIT. Scores 128-dim vectors **per token** and ranks with MaxSim
(`sum over query tokens of max over doc tokens`). The 9B sibling scores
ViDoRe v3 65.2% image / 64.7% markdown and shares this embedding space, so
you can index with the 9B and query with the 0.6B.

**Why there is no PLAID index here.** `pylate` 1.6.0 pins
`sentence-transformers==5.3.0` and `transformers<=5.3.0`, but this model
requires `sentence-transformers>=6.0.0` and `transformers>=5.4.0`. Those
constraints are mutually exclusive, so this notebook scores MaxSim by brute
force instead. At SciFact (5.2k docs) and NFCorpus (3.6k docs) that is fast.
It would not be at MS MARCO scale — a real late-interaction index is a
PLAID-style structure, roughly 128 floats per token of every document.

**Cost shape.** Indexing is expensive (per-token vectors, ~100x the storage
of a single vector). Query time is cheap: one document encode per query,
not one per document. That asymmetry is the entire argument for reranking.
"""

DATASET_NOTES = {
    "scifact": """
**SciFact** — 5,183 scientific-claim documents, 300 judged queries. Claims are
short and self-contained. Published BEIR BM25 nDCG@10 sits near 0.66-0.69.
""",
    "nfcorpus": """
**NFCorpus** — 3,633 medical/NutritionFacts documents, 323 judged queries.
Short documents, but graded relevance (0/1/2) and far more queries per
document than SciFact, so recall behaves differently. Published BEIR BM25
nDCG@10 is around 0.32 — much lower than SciFact, and that is expected,
not a bug.
""",
}

INSTALL = ("%pip install -q --upgrade 'transformers>=5.19.0' "
           "'sentence-transformers>=6.0.0' datasets rank_bm25 pytrec_eval-terrier")

MODELS = [
    ("embeddinggemma", "EmbeddingGemma 2", "bm25,gemma", GEMMA_NOTES),
    ("late_interaction", "pplx-embed-v2-late-0.6b", "bm25,late", LATE_NOTES),
]

if __name__ == "__main__":
    out = pathlib.Path(".")
    for ds in ("scifact", "nfcorpus"):
        for slug, title, margs, notes in MODELS:
            full = f"{title} on BEIR ({ds})"
            body = notes.rstrip() + "\n" + DATASET_NOTES[ds].rstrip()
            name = f"colab_{slug}_{ds}.ipynb"
            nb = build(full, ds, INSTALL, margs, body)
            (out / name).write_text(json.dumps(nb, indent=1))
            print("wrote", name)