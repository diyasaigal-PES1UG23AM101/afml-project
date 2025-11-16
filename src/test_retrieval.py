from sentence_transformers import SentenceTransformer
import faiss, json

# MUST MATCH embed_index.py
model = SentenceTransformer("sentence-transformers/LaBSE")

# Load FAISS index
index = faiss.read_index("data/processed/faiss_index.bin")

# Load text dataset
texts = []
with open("data/processed/all_passages.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

# Query
query = "Norwegian Air Force World War 2"
query_vec = model.encode([query])

# Search
D, I = index.search(query_vec, 3)

print("Top results:")
for idx in I[0]:
    print(texts[idx][:300])
    print("------")