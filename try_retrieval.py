from dataset import load_dataset
from retrieval.retriever import build_index, retrieve
from pipeline import run_pipeline

data = load_dataset("mas_scifact_dataset (1).json")
print("Building index over", len(data["corpus"]), "abstracts (downloads the model on first run)...")
index = build_index(data["corpus"])

q = data["questions"][0]
print("\nClaim:", q["claim"])
print("Gold label:", q["gold_label"], "| Gold passages:", q["gold_passages"])

top = retrieve(index, q["claim"], k=5)
print("\nTop retrieved passages:")
for p in top:
    print(f"  {p['passage_id']}  score={p['_score']:.3f}  {p['title']}")
found_gold = any(p["passage_id"] in q["gold_passages"] for p in top)
print("\nGold passage found in top-5:", found_gold)

report, result = run_pipeline(q["claim"], condition="C_proposed", retrieval_index=index, top_k=5)
print("\nPipeline report:", report)
print("Run result:", result.model_dump())