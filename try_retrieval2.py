from dataset import load_dataset
from retrieval.retriever import build_index, retrieve

data = load_dataset("mas_scifact_dataset.json")
print("Building index over", len(data["corpus"]), "abstracts (only needs to run once per session)...")
index = build_index(data["corpus"])


def check(q, k=5):
    print(f"\nClaim: {q['claim']}")
    print(f"Gold label: {q['gold_label']} | Gold passages: {q['gold_passages']} | Cited: {q['cited_passages']}")
    top = retrieve(index, q["claim"], k=k)
    for p in top:
        print(f"  {p['passage_id']}  score={p['_score']:.3f}  {p['title']}")
    ids = [p["passage_id"] for p in top]
    if q["gold_passages"]:
        print("Gold passage found in top-k:", any(g in ids for g in q["gold_passages"]))
    else:
        print("(NOINFO claim -- no gold passage to check against; SciFact defines NOINFO as having none)")
    print("Cited passage found in top-k:", any(c in ids for c in q["cited_passages"]))


# Check a handful of SUPPORT and CONTRADICT claims, where gold_passages is non-empty
support_claims = [q for q in data["questions"] if q["gold_label"] == "SUPPORT"][:3]
contradict_claims = [q for q in data["questions"] if q["gold_label"] == "CONTRADICT"][:3]

for q in support_claims + contradict_claims:
    check(q)