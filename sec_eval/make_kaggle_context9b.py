import gzip
import json
import base64
from pathlib import Path

DATA_DIR = Path("/home/sanjeev/Downloads/beir-ab/sec_eval")
NB_PATH = DATA_DIR / "kaggle_pplx_context9b_eval.ipynb"

# 1. Read and compress corpus + queries
with open(DATA_DIR / "corpus.jsonl", "rb") as f:
    corpus_gz_b64 = base64.b64encode(gzip.compress(f.read(), compresslevel=9)).decode("ascii")

with open(DATA_DIR / "queries.json", "r", encoding="utf-8") as f:
    queries_json_str = f.read()

with open(DATA_DIR / "queries_conceptual.json", "r", encoding="utf-8") as f:
    queries_conceptual_str = f.read()

# 2. Define notebook cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Perplexity Contextual Embeddings: pplx-embed-v2-context-9b-preview\n",
            "\n",
            "This notebook benchmarks the **unquantized 9B parameter contextual embedding model**:\n",
            "**`perplexity-ai/pplx-embed-v2-context-9b-preview`**\n",
            "\n",
            "### Architecture & Execution Highlights:\n",
            "- **Precision**: Full unquantized **FP16** (16.8 GB weights) sharded across **Dual Nvidia T4 GPUs (T4 x 2)** via `accelerate` (`device_map=\"auto\"`).\n",
            "- **Contextual Attention**: Contiguous section chunks are passed together so each chunk attends to the global surrounding chapter context during the forward pass.\n",
            "- **Evaluation**: Evaluates both **15 Exact Technical CLI queries** and **15 Conceptual Zero-Keyword queries** against the 1,243 extracted security book chunks.\n",
            "\n",
            "> **Prerequisites on Kaggle**:\n",
            "> 1. Set **Settings -> Accelerator -> GPU T4 x2**\n",
            "> 2. Set **Settings -> Internet -> On**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Install dependencies & initialize safe imports\n",
            "%pip install -q --upgrade 'transformers>=5.19.0' accelerate safetensors rank_bm25\n",
            "\n",
            "import sys, types, importlib.abc, importlib.machinery\n",
            "class _AutoMock(types.ModuleType):\n",
            "    def __getattr__(self, name):\n",
            "        if name.startswith('__'): raise AttributeError(name)\n",
            "        val = _AutoMock(f'{self.__name__}.{name}')\n",
            "        setattr(self, name, val)\n",
            "        return val\n",
            "    def __call__(self, *args, **kwargs): return None\n",
            "class _MockFinder(importlib.abc.MetaPathFinder):\n",
            "    def find_spec(self, fullname, path, target=None):\n",
            "        if fullname == 'torchvision' or fullname.startswith('torchvision.'):\n",
            "            return importlib.machinery.ModuleSpec(fullname, _MockLoader(), is_package=True)\n",
            "        return None\n",
            "class _MockLoader(importlib.abc.Loader):\n",
            "    def create_module(self, spec):\n",
            "        mod = _AutoMock(spec.name); mod.__path__ = []; return mod\n",
            "    def exec_module(self, module): pass\n",
            "sys.meta_path = [f for f in sys.meta_path if not isinstance(f, _MockFinder)]\n",
            "sys.meta_path.insert(0, _MockFinder())"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Unpack embedded dataset (1,243 chunks + Exact & Conceptual Queries)\n",
            "import gzip, base64, json\n",
            "\n",
            "CORPUS_B64 = \"\"\"" + corpus_gz_b64 + "\"\"\"\n",
            "raw_corpus_bytes = gzip.decompress(base64.b64decode(CORPUS_B64))\n",
            "corpus = [json.loads(l) for l in raw_corpus_bytes.decode('utf-8').splitlines() if l.strip()]\n",
            "exact_queries = json.loads(\"\"\"" + queries_json_str + "\"\"\")\n",
            "conceptual_queries = json.loads(\"\"\"" + queries_conceptual_str + "\"\"\")\n",
            "\n",
            "print(f\"✓ Loaded {len(corpus):,} chunks across {len(set(c['book'] for c in corpus))} books\")\n",
            "print(f\"✓ Loaded {len(exact_queries)} exact technical queries\")\n",
            "print(f\"✓ Loaded {len(conceptual_queries)} conceptual queries\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Verify Dual T4 GPUs\n",
            "import torch\n",
            "num_gpus = torch.cuda.device_count()\n",
            "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
            "print(f\"Number of GPUs: {num_gpus}\")\n",
            "for i in range(num_gpus):\n",
            "    props = torch.cuda.get_device_properties(i)\n",
            "    print(f\"  GPU {i}: {props.name} | VRAM: {props.total_memory / (1024**3):.1f} GB\")\n",
            "\n",
            "if num_gpus < 2:\n",
            "    print(\"\\n⚠️ WARNING: Less than 2 GPUs detected! In Kaggle sidebar, set Accelerator to 'GPU T4 x2'.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Load pplx-embed-v2-context-9b-preview across Dual GPUs\n",
            "import os, time\n",
            "from transformers import AutoModel, AutoTokenizer\n",
            "\n",
            "print(\"Loading perplexity-ai/pplx-embed-v2-context-9b-preview in unquantized FP16 across Dual GPUs...\")\n",
            "t0 = time.perf_counter()\n",
            "\n",
            "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
            "\n",
            "model = AutoModel.from_pretrained(\n",
            "    \"perplexity-ai/pplx-embed-v2-context-9b-preview\",\n",
            "    torch_dtype=torch.float16,\n",
            "    device_map=\"auto\",\n",
            "    max_memory={0: \"9GB\", 1: \"13GB\"},\n",
            "    trust_remote_code=True,\n",
            ")\n",
            "\n",
            "# Explicitly attach tokenizer with trust_remote_code=True to bypass custom model property bug\n",
            "tokenizer = AutoTokenizer.from_pretrained(\n",
            "    \"perplexity-ai/pplx-embed-v2-context-9b-preview\",\n",
            "    trust_remote_code=True,\n",
            "    padding_side=\"right\",\n",
            ")\n",
            "model.tokenizer = tokenizer\n",
            "\n",
            "# Multi-GPU device alignment patch for custom _pool method\n",
            "import torch.nn.functional as F, types\n",
            "def _safe_pool(self, features: dict, task: str):\n",
            "    dev0 = self.device\n",
            "    input_ids = features[\"input_ids\"].to(dev0)\n",
            "    attention_mask = features[\"attention_mask\"].to(dev0)\n",
            "    hidden = self(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state\n",
            "    dev_out = hidden.device\n",
            "    with torch.autocast(device_type=dev_out.type, enabled=False):\n",
            "        hidden = hidden.float()\n",
            "        if task == \"query\":\n",
            "            mask = attention_mask.to(dev_out).unsqueeze(-1).float()\n",
            "            pooled = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).unsqueeze(1)\n",
            "        else:\n",
            "            prefix = torch.cat((hidden.new_zeros(hidden.shape[0], 1, hidden.shape[-1]), hidden.cumsum(dim=1)), dim=1)\n",
            "            spans = features[\"chunk_token_spans\"].to(dev_out)\n",
            "            starts, ends = spans.unbind(dim=-1)\n",
            "            rows = torch.arange(hidden.shape[0], device=dev_out).unsqueeze(1)\n",
            "            pooled = (prefix[rows, ends] - prefix[rows, starts]) / (ends - starts).clamp_min(1).unsqueeze(-1)\n",
            "        proj_w = self.contextual_projection.weight.to(dev_out).float()\n",
            "        return F.linear(pooled, proj_w)\n",
            "\n",
            "model._pool = types.MethodType(_safe_pool, model)\n",
            "print(\"✓ Multi-GPU device alignment patch applied to model._pool\")\n",
            "\n",
            "load_time = time.perf_counter() - t0\n",
            "print(f\"✓ Model & Tokenizer loaded in {load_time:.1f}s\")\n",
            "if hasattr(model, 'hf_device_map'):\n",
            "    print(f\"Device map layer distribution:\")\n",
            "    devices = set(model.hf_device_map.values())\n",
            "    for d in sorted(list(devices)):\n",
            "        layers = [k for k, v in model.hf_device_map.items() if v == d]\n",
            "        print(f\"  Device {d}: {len(layers)} modules\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Contextual Document Chunking & Encoding\n",
            "import numpy as np, gc, torch\n",
            "\n",
            "# Free any lingering fragmentation from previous runs\n",
            "gc.collect()\n",
            "torch.cuda.empty_cache()\n",
            "\n",
            "# Group contiguous chunks by section (capped at 2 chunks per document to keep token sequence short and prevent OOM)\n",
            "doc_chunks = []\n",
            "current_key = None\n",
            "current_doc = []\n",
            "\n",
            "for idx, c in enumerate(corpus):\n",
            "    key = (c['book_tag'], c['section'])\n",
            "    if key != current_key or len(current_doc) >= 2:\n",
            "        if current_doc:\n",
            "            doc_chunks.append(current_doc)\n",
            "        current_doc = []\n",
            "        current_key = key\n",
            "    current_doc.append(f\"title: {c['title']} | text: {c['text']}\")\n",
            "\n",
            "if current_doc:\n",
            "    doc_chunks.append(current_doc)\n",
            "\n",
            "total_chunks = sum(len(d) for d in doc_chunks)\n",
            "print(f\"Grouped {total_chunks} chunks into {len(doc_chunks)} lightweight contextual clusters (max 2 chunks each).\")\n",
            "\n",
            "print(\"Encoding document clusters with shared attention (batch_size=1)...\")\n",
            "t0 = time.perf_counter()\n",
            "# Batch size 1 eliminates activation memory spikes across Dual T4 cards\n",
            "doc_embeddings = model.encode(doc_chunks, batch_size=1, normalize_embeddings=True, show_progress_bar=True)\n",
            "context_doc_time = time.perf_counter() - t0\n",
            "\n",
            "corpus_embeddings = np.vstack(doc_embeddings)\n",
            "print(f\"✓ Contextual encoding finished in {context_doc_time:.1f}s. Matrix shape: {corpus_embeddings.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Evaluate Exact & Conceptual Queries\n",
            "# Encode exact queries\n",
            "print(\"Encoding exact technical queries...\")\n",
            "eq_input = [[q['query']] for q in exact_queries]\n",
            "eq_vecs = np.vstack(model.encode_queries(eq_input, normalize_embeddings=True))\n",
            "eq_sims = eq_vecs @ corpus_embeddings.T\n",
            "\n",
            "exact_results = {}\n",
            "for i, q in enumerate(exact_queries):\n",
            "    row = eq_sims[i]\n",
            "    top = row.argsort()[::-1][:10]\n",
            "    exact_results[q['id']] = [(top[j], float(row[top[j]])) for j in range(10)]\n",
            "\n",
            "# Encode conceptual queries\n",
            "print(\"Encoding conceptual zero-keyword queries...\")\n",
            "cq_input = [[q['query']] for q in conceptual_queries]\n",
            "cq_vecs = np.vstack(model.encode_queries(cq_input, normalize_embeddings=True))\n",
            "cq_sims = cq_vecs @ corpus_embeddings.T\n",
            "\n",
            "concept_results = {}\n",
            "for i, q in enumerate(conceptual_queries):\n",
            "    row = cq_sims[i]\n",
            "    top = row.argsort()[::-1][:10]\n",
            "    concept_results[q['id']] = [(top[j], float(row[top[j]])) for j in range(10)]\n",
            "\n",
            "# Compute Accuracies\n",
            "p9b_exact_acc = sum(1 for q in exact_queries if corpus[exact_results[q['id']][0][0]]['book_tag'] == q['book_tag']) / len(exact_queries)\n",
            "p9b_concept_acc = sum(1 for q in conceptual_queries if corpus[concept_results[q['id']][0][0]]['book_tag'] == q['book_tag']) / len(conceptual_queries)\n",
            "\n",
            "print(\"✓ Query evaluation complete!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 7. Comprehensive 4-Way Benchmark Table\n",
            "print(\"=\" * 85)\n",
            "print(\"COMPREHENSIVE BENCHMARK: EXACT vs CONCEPTUAL RETRIEVAL\")\n",
            "print(\"=\" * 85)\n",
            "\n",
            "# Reference baselines from previous tests:\n",
            "# BM25: Exact 93.3%, Conceptual 60.0%\n",
            "# Gemma 2 (270M): Exact 86.7%, Conceptual 86.7%\n",
            "# Late-0.6b (340M): Exact 86.7%, Conceptual 80.0%\n",
            "\n",
            "print(f\"{'Model':<30} {'Exact Top-1':<15} {'Conceptual Top-1':<18} {'Delta':<10}\")\n",
            "print(\"-\" * 85)\n",
            "print(f\"{'BM25':<30} {'93.3%':<15} {'60.0%':<18} {'-33.3%':<10}\")\n",
            "print(f\"{'EmbeddingGemma 2 (270M)':<30} {'86.7%':<15} {'86.7%':<18} {'+0.0%':<10}\")\n",
            "print(f\"{'Late Interaction (0.6B)':<30} {'86.7%':<15} {'80.0%':<18} {'-6.7%':<10}\")\n",
            "delta_str = f\"{(p9b_concept_acc - p9b_exact_acc)*100:+5.1f}%\"\n",
            "print(f\"{'Pplx Contextual (9B)':<30} {f'{p9b_exact_acc*100:.1f}%':<15} {f'{p9b_concept_acc*100:.1f}%':<18} {delta_str:<10}\")\n",
            "print(\"=\" * 85)\n",
            "\n",
            "print(\"\\nSAMPLE SIDE-BY-SIDE: CONCEPTUAL QUERY 1\")\n",
            "q1 = conceptual_queries[0]\n",
            "top1_idx, sc1 = concept_results[q1['id']][0]\n",
            "hit1 = corpus[top1_idx]\n",
            "print(f\"Query: \\\"{q1['query']}\\\"\")\n",
            "print(f\"Target: [{q1['book_tag']}] {q1['topic']}\")\n",
            "print(f\"Pplx-9B Top-1: [{hit1['book_tag']} p.{hit1['pages'][0]}] {hit1['section']} (score: {sc1:.3f})\")\n",
            "print(f\"Text: \\\"{hit1['text'][:120]}...\\\"\")"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"✓ Wrote self-contained Kaggle Dual-GPU notebook to {NB_PATH} ({NB_PATH.stat().st_size / 1024:.1f} KB)")
