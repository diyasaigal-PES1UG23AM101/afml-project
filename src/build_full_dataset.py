import json
import os

OUTPUT_PATH = "data/processed/all_passages.jsonl"

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

all_items = []

if os.path.exists("data/processed/passages.jsonl"):
    all_items += load_jsonl("data/processed/passages.jsonl")

if os.path.exists("data/processed/bible_passages.jsonl"):
    all_items += load_jsonl("data/processed/bible_passages.jsonl")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in all_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Created {OUTPUT_PATH} with {len(all_items)} total passages.")