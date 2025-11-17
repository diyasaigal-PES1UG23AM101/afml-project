# src/embed_index.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import os

# --- Robust Pathing ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

DATASET_PATH = os.path.join(DATA_DIR, "all_passages.jsonl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# --- Main Logic ---
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    print("Please run 'build_full_dataset.py' first.")
    exit()

# Load model
model = SentenceTransformer("sentence-transformers/LaBSE")
print("Model loaded.")

# Load dataset
texts = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

if not texts:
    print("ERROR: The dataset file is empty. Cannot build index.")
    exit()

print(f"Loaded {len(texts)} text chunks.")

# Embed
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
np.save(EMBEDDINGS_PATH, embeddings)
print(f"Embeddings saved to {EMBEDDINGS_PATH}")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
# FAISS requires float32
index.add(embeddings.astype('float32'))
faiss.write_index(index, INDEX_PATH)
print(f"FAISS index saved to {INDEX_PATH}")
