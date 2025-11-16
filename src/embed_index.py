from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

DATASET_PATH = "data/processed/all_passages.jsonl"

# Load model
model = SentenceTransformer("sentence-transformers/LaBSE")
print("Model loaded.")

# Load dataset
texts = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])

print("Loaded", len(texts), "text chunks.")

# Embed
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
np.save("data/processed/embeddings.npy", embeddings)
print("Embeddings saved.")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "data/processed/faiss_index.bin")
print("FAISS index saved.")