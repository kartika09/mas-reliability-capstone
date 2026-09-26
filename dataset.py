import json

def load_dataset(path="mas_scifact_dataset(1).json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["passages"] = {p["passage_id"]: p for p in data["corpus"]}
    return data
