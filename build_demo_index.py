"""
Build a demo FAISS index with sample Tulu data
Run this if you don't have the full dataset yet
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample Tulu passages for demo
SAMPLE_PASSAGES = [
    "Tulu is a Dravidian language spoken mainly in the southwestern part of India, particularly in the coastal districts of Karnataka and Kasaragod district of Kerala.",
    "The Tulu script is called Tigalari, which was historically used to write the Tulu language. However, today Tulu is mostly written in Kannada script.",
    "Tulu has a rich oral literature tradition including folk songs, Yakshagana performances, and traditional ballads called Paddanas.",
    "The Tulu-speaking region is known as Tulu Nadu, which roughly corresponds to the undivided Dakshina Kannada district.",
    "Mangalore is the largest city in Tulu Nadu and is considered the cultural hub of Tulu-speaking people.",
    "Tulu cuisine is known for its seafood dishes, including fish curry, prawn ghee roast, and neer dosa.",
    "The Tulu language has several dialects including Common Tulu, Brahmin Tulu, and Jain Tulu.",
    "Tulu has been recognized as a minority language by the Government of Karnataka.",
    "The history of Tulu literature dates back to at least the 14th-15th century with works like Devi Mahatmyam.",
    "Tulu Yakshagana is a traditional theater form that combines dance, music, dialogue, and elaborate costumes.",
    "ತುಳು ಭಾಷೆ ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬದ ಭಾಗವಾಗಿದೆ ಮತ್ತು ಕರ್ನಾಟಕದ ಕರಾವಳಿ ಜಿಲ್ಲೆಗಳಲ್ಲಿ ಮಾತನಾಡಲಾಗುತ್ತದೆ.",
    "Tulu is known for its unique phonology and has sounds that are not present in other Dravidian languages.",
    "The Tulu Sahitya Academy was established in 1994 to promote Tulu language and literature.",
    "Tulu has influenced the culture and cuisine of coastal Karnataka and northern Kerala.",
    "Many Tulu speakers are bilingual or multilingual, also speaking Kannada, Konkani, or Malayalam.",
    "The Tulu calendar follows the Saka calendar and begins in the month of Bisu (April).",
    "Traditional Tulu festivals include Kambala (buffalo race), Bhuta Kola (spirit worship), and Karavali Utsav.",
    "Tulu cinema has a growing industry with several films produced annually in the language.",
    "The Tulu speaking population is estimated to be around 2-3 million people.",
    "Tulu has a Subject-Object-Verb (SOV) word order, typical of Dravidian languages.",
]

def create_demo_index():
    """Create a demo FAISS index with sample passages"""
    
    print("Creating demo FAISS index with sample Tulu data...")
    
    # Create directories
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    index_path = data_dir / "faiss_index.bin"
    data_path = data_dir / "all_passages.jsonl"
    
    # Check if already exists
    if index_path.exists() and data_path.exists():
        response = input(f"Index already exists at {index_path}. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print("\n1. Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    print("   ✓ Model loaded")
    
    print("\n2. Encoding passages...")
    embeddings = model.encode(SAMPLE_PASSAGES, show_progress_bar=True, convert_to_numpy=True)
    print(f"   ✓ Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    print("\n3. Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"   ✓ Added {index.ntotal} vectors to index")
    
    print("\n4. Saving FAISS index...")
    faiss.write_index(index, str(index_path))
    print(f"   ✓ Saved to {index_path}")
    
    print("\n5. Saving passages...")
    with open(data_path, "w", encoding="utf-8") as f:
        for passage in SAMPLE_PASSAGES:
            f.write(json.dumps({"text": passage}, ensure_ascii=False) + "\n")
    print(f"   ✓ Saved {len(SAMPLE_PASSAGES)} passages to {data_path}")
    
    print("\n" + "="*60)
    print("✅ Demo index created successfully!")
    print("="*60)
    print(f"\nIndex location: {index_path}")
    print(f"Data location: {data_path}")
    print(f"Total passages: {len(SAMPLE_PASSAGES)}")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run app/app.py")
    print("\nOr test retrieval:")
    print("  python src/test_retrieval.py")
    
    # Test the index
    print("\n" + "="*60)
    print("Testing retrieval...")
    print("="*60)
    
    test_query = "What is Tulu language?"
    print(f"\nQuery: '{test_query}'")
    
    q_vec = model.encode([test_query], convert_to_numpy=True)
    D, I = index.search(q_vec.astype('float32'), 3)
    
    print("\nTop 3 results:")
    for i, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        print(f"\n{i}. [Distance: {dist:.3f}]")
        print(f"   {SAMPLE_PASSAGES[idx][:150]}...")
    
    print("\n✓ Retrieval test successful!")


if __name__ == "__main__":
    try:
        create_demo_index()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
