import pandas as pd
import json

def read_jsonl(path):
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

path = "data/processed/catapi_breed_documents.jsonl"
breeds = ["British Shorthair", "Ragamuffin", "Ragdoll"]
cols = ["Temperament", "Weight", "Grooming level"]

df = pd.read_json(path, lines=True)
test = df[df['breed_name'].isin(breeds)]

print(test["metadata"])