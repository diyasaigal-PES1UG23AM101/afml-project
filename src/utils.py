# src/utils.py

import json

def load_jsonl(path):
    """
    Loads a JSONL file and returns a list of dicts.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    """
    Saves a list of dicts to a JSONL file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_text(text):
    """
    Minimal text cleaning used across modules.
    (More advanced cleaning can be done inside data_cleaning.py)
    """
    text = text.strip()
    text = " ".join(text.split())
    return text
