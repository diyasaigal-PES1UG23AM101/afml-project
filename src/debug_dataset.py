import json

bible = 0
wiki = 0
news = 0
unknown = 0

with open("../data/processed/all_passages.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        src = obj.get("source", "")

        if "bible" in src:
            bible += 1
        elif "wiki" in src:
            wiki += 1
        elif "news" in src:
            news += 1
        else:
            unknown += 1

print("Bible passages:", bible)
print("Wiki passages:", wiki)
print("News passages:", news)
print("Unknown:", unknown)