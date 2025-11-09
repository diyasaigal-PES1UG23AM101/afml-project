# src/embed_index.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from tqdm import tqdm

# STEP 1: Load the multilingual embedding model
# (Use either distiluse-base-multilingual-cased-v2 or LaBSE)
# LaBSE is heavier but more accurate for bilingual tasks
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(model_name)

print(f"Loaded model: {model_name}")

# STEP 2: Load your cleaned Tulu text dataset
texts = []
ids = []

with open("data/processed/passages.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        texts.append(data["text"])
        ids.append(data["id"])

print(f"Loaded {len(texts)} text chunks for embedding.")

# STEP 3: Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

# STEP 4: Save embeddings for later evaluation
np.save("data/processed/embeddings.npy", embeddings)
print("Saved embeddings to data/processed/embeddings.npy")

# STEP 5: Create FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance metric

index.add(embeddings)
faiss.write_index(index, "data/processed/faiss_index.bin")
print("FAISS index created and saved at data/processed/faiss_index.bin")

# Optional: Check one retrieval test
query = "Tulu language"  # English example query
query_vector = model.encode([query])
D, I = index.search(query_vector, 3)

print("\n🔍 Sample retrieval results:")
for idx in I[0]:
    print(f"- {texts[idx][:120]}...")
