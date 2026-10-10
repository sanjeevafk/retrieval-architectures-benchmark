import json
import re
import time
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path("/home/sanjeev/Downloads/beir-ab/sec_eval")


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 1]


def run_bm25():
    corpus_file = DATA_DIR / "corpus.jsonl"
    queries_file = DATA_DIR / "queries.json"

    print("Loading corpus...")
    corpus = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"Loaded {len(corpus)} chunks.")
    
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print("Building BM25 index...")
    t0 = time.perf_counter()
    corpus_tokens = [tokenize(f"{c['title']} {c['text']}") for c in corpus]
    bm25 = BM25Okapi(corpus_tokens)
    index_time = time.perf_counter() - t0
    print(f"Index built in {index_time:.3f}s")

    print("\n" + "=" * 80)
    print("BM25 Retrieval on Security Books (Top-2 previews per query)")
    print("=" * 80)

    for q in queries:
        qid = q["id"]
        q_text = q["query"]
        expected_book = q["book_tag"]
        topic = q["topic"]

        q_tokens = tokenize(q_text)
        scores = bm25.get_scores(q_tokens)
        top_indices = scores.argsort()[::-1][:5]

        print(f"\nQuery [{qid}]: \"{q_text}\"")
        print(f"  Target: [{expected_book}] - {topic}")
        print("  Top Hits:")
        for rank, idx in enumerate(top_indices[:2], 1):
            doc = corpus[idx]
            score = scores[idx]
            match_flag = "✓ MATCH" if doc["book_tag"] == expected_book else "✗ OTHER BOOK"
            preview = doc["text"][:120].replace("\n", " ")
            print(f"    #{rank} [Score: {score:5.2f}] [{doc['book_tag']} - Pages: {doc['pages'][:3]}] {match_flag}")
            print(f"       Section: {doc['section']}")
            print(f"       Text: \"{preview}...\"")


if __name__ == "__main__":
    run_bm25()
