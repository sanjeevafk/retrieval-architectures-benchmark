"""Validate maxsim and the eval wrapper without downloading models."""
import numpy as np
import beir_ab as H

# ---- 1. maxsim vs brute force, and vs torch.tensordot
rng = np.random.default_rng(0)
q = rng.normal(size=(7, 128)).astype("float32")
docs = [rng.normal(size=(int(n), 128)).astype("float32") for n in (3, 40, 128, 5)]
got = H.maxsim(q, docs)

ref = np.array([sum(max(float(q[t] @ d[j]) for j in range(d.shape[0]))
                    for t in range(q.shape[0])) for d in docs])
print("maxsim matches brute force:", np.allclose(got, ref, atol=1e-3),
      "| max abs diff", float(np.abs(got - ref).max()))

# known-answer: perfect match doc must score highest
same = q.copy()
sc = H.maxsim(q, [rng.normal(size=(20, 128)).astype("float32"), same])
print("perfect-match doc ranks first:", sc[1] > sc[0])

# edge cases must not explode
print("empty query ->", H.maxsim(np.zeros((0, 128), "float32"), [docs[0]])[0])
print("empty doc   ->", H.maxsim(q, [np.zeros((0, 128), "float32")])[0])

# ---- 2. metrics vs hand-computed values on a tiny known case
# docs a,b,c ; q1 relevant = a(2), b(1). rank a first -> ideal nDCG@3
qrels = {"q1": {"a": 2, "b": 1}}
m = H.evaluate({"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}, qrels, k=3)
disc = 1 / np.log2(np.arange(2, 5))
ideal = (2**2 - 1) * disc[0] + (2**1 - 1) * disc[1]
expect_ndcg = ((2**2 - 1) * disc[0] + (2**1 - 1) * disc[1]) / ideal
print(f"ndcg@3 got {m['ndcg_cut_3']:.6f} expect {expect_ndcg:.6f}",
      "->", abs(m["ndcg_cut_3"] - expect_ndcg) < 1e-6)
print(f"recall@3 {m['recall_3']:.4f} (expect 1.0) ->", abs(m["recall_3"] - 1) < 1e-9)
print(f"recip_rank {m['recip_rank']:.4f} (expect 1.0) ->", abs(m["recip_rank"] - 1) < 1e-9)

# reversed ranking should score lower nDCG, same recall
m2 = H.evaluate({"q1": {"b": 3.0, "a": 2.0, "c": 1.0}}, qrels, k=3)
print("reversed ranks lower ndcg:", m2["ndcg_cut_3"] < m["ndcg_cut_3"])

# ---- 3. cross-check against pytrec_eval's own ndcg_cut directly
import pytrec_eval
rel = {"q1": {"a": 2, "b": 1}}
for run in ({"q1": {"a": 3.0, "b": 2.0, "c": 1.0}},
            {"q1": {"c": 9.0, "b": 5.0, "a": 1.0}}):
    ev = pytrec_eval.RelevanceEvaluator(rel, {"ndcg_cut.3"}).evaluate(run)
    ours = H.evaluate(run, rel, k=3)["ndcg_cut_3"]
    print(f"  direct={ev['q1']['ndcg_cut_3']:.6f} ours={ours:.6f} "
          f"-> {abs(ev['q1']['ndcg_cut_3']-ours) < 1e-9}")