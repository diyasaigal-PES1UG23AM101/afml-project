#!/usr/bin/env python3
"""
rag_pipeline.py

RAG (Retrieval-Augmented Generation) Pipeline for Tulu-English Translation.

This script implements a RAG architecture for translation:
1. Loads Tulu text passages and builds a FAISS index with embeddings
2. Retrieves relevant context for input queries
3. Uses retrieval-augmented prompts with translation models (mT5, mBART, or GPT-4)
4. Evaluates translation quality using BLEU and METEOR metrics

Usage:
    # Build index from passages
    python src/rag_pipeline.py --build_index --passages_file data/processed/passages.jsonl
    
    # Translate a single sentence
    python src/rag_pipeline.py --translate "ಪ್ರಾರಂಭಿಸಿ" --model mT5
    
    # Evaluate on test set
    python src/rag_pipeline.py --evaluate --test_file data/test.jsonl --model mBART
"""

import os
import json
import argparse
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Try importing required libraries
try:
    import faiss
except ImportError:
    print("[ERROR] faiss-cpu not installed. Install with: pip install faiss-cpu")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[ERROR] sentence-transformers not installed. Install with: pip install sentence-transformers")
    exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
except ImportError:
    print("[ERROR] transformers or torch not installed. Install with: pip install transformers torch")
    exit(1)

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
except ImportError:
    print("[WARN] nltk not installed. Evaluation metrics will not work. Install with: pip install nltk")
    nltk = None

# Configuration
DEFAULT_PASSAGES_FILE = "data/processed/passages.jsonl"
DEFAULT_INDEX_DIR = "data/rag_index"
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Good multilingual model
DEFAULT_INDEX_FILE = os.path.join(DEFAULT_INDEX_DIR, "faiss_index.index")
DEFAULT_METADATA_FILE = os.path.join(DEFAULT_INDEX_DIR, "metadata.pkl")

# Translation model options
AVAILABLE_MODELS = {
    "mT5": "google/mt5-base",
    "mBART": "facebook/mbart-large-50-many-to-many-mmt",
    # GPT-4 is handled via API
}

class RAGPipeline:
    """RAG Pipeline for Tulu-English Translation."""
    
    def __init__(self, 
                 embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                 index_dir: str = DEFAULT_INDEX_DIR,
                 device: Optional[str] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model_name: Name of the embedding model
            index_dir: Directory to store/load FAISS index
            device: Device to use ('cuda' or 'cpu')
        """
        self.embedding_model_name = embedding_model_name
        self.index_dir = index_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embedding model
        print(f"[INFO] Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Initialize FAISS index and metadata
        self.index = None
        self.metadata = []
        self.translation_model = None
        self.translation_tokenizer = None
        self.translation_pipeline = None
        
        # Load index if it exists
        self.index_file = os.path.join(index_dir, "faiss_index.index")
        self.metadata_file = os.path.join(index_dir, "metadata.pkl")
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.load_index()
    
    def build_index(self, passages_file: str, batch_size: int = 32):
        """
        Build FAISS index from passages JSONL file.
        
        Args:
            passages_file: Path to passages.jsonl file
            batch_size: Batch size for embedding generation
        """
        if not os.path.exists(passages_file):
            raise FileNotFoundError(f"Passages file not found: {passages_file}")
        
        print(f"[INFO] Loading passages from {passages_file}")
        passages = []
        texts = []
        
        with open(passages_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        text = record.get("text", "").strip()
                        if text and len(text) > 10:  # Filter very short texts
                            passages.append(record)
                            texts.append(text)
                    except json.JSONDecodeError:
                        continue
        
        if not texts:
            raise ValueError("No valid passages found in file")
        
        print(f"[INFO] Loaded {len(texts)} passages")
        print(f"[INFO] Generating embeddings (batch_size={batch_size})...")
        
        # Generate embeddings in batches
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        print(f"[INFO] Creating FAISS index (dimension={dimension})...")
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        
        # Normalize embeddings for cosine similarity (optional but recommended)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = passages
        
        # Save index
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        
        print(f"[OK] Index built and saved:")
        print(f"  Index: {self.index_file}")
        print(f"  Metadata: {self.metadata_file}")
        print(f"  Total vectors: {self.index.ntotal}")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("Index files not found. Run --build_index first.")
        
        print(f"[INFO] Loading index from {self.index_file}")
        self.index = faiss.read_index(self.index_file)
        
        with open(self.metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        
        print(f"[OK] Index loaded: {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most similar passages for a query.
        
        Args:
            query: Query text in Tulu
            k: Number of passages to retrieve
            
        Returns:
            List of retrieved passages with metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded. Run --build_index or load_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Retrieve passages
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = {
                    "text": self.metadata[idx].get("text", ""),
                    "score": float(1.0 / (1.0 + distances[0][i])),  # Convert distance to similarity
                    "metadata": self.metadata[idx]
                }
                results.append(result)
        
        return results
    
    def load_translation_model(self, model_name: str, api_key: Optional[str] = None):
        """
        Load translation model.
        
        Args:
            model_name: Model name ('mT5', 'mBART', or 'gpt4')
            api_key: API key for GPT-4 (if using GPT-4)
        """
        if model_name.lower() == "gpt4":
            self.translation_model = "gpt4"
            self.api_key = api_key
            if not api_key:
                print("[WARN] GPT-4 API key not provided. Set OPENAI_API_KEY environment variable.")
        elif model_name in AVAILABLE_MODELS:
            model_path = AVAILABLE_MODELS[model_name]
            print(f"[INFO] Loading translation model: {model_path}")
            
            self.translation_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.translation_model.to(self.device)
            self.translation_model.eval()
            
            # Create pipeline for easier inference
            self.translation_pipeline = pipeline(
                "translation",
                model=self.translation_model,
                tokenizer=self.translation_tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            print(f"[OK] Translation model loaded on {self.device}")
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())} + ['gpt4']")
    
    def translate_with_rag(self, 
                          tulu_text: str, 
                          model_name: str = "mT5",
                          k: int = 3,
                          api_key: Optional[str] = None,
                          max_context_length: int = 512) -> str:
        """
        Translate Tulu text to English using RAG.
        
        Args:
            tulu_text: Input text in Tulu
            model_name: Translation model to use
            k: Number of retrieved passages to use as context
            api_key: API key for GPT-4 (if using)
            max_context_length: Maximum context length for the model
            
        Returns:
            Translated English text
        """
        # Load model if not loaded or different model requested
        if self.translation_model != model_name.lower() and model_name.lower() != "gpt4":
            self.load_translation_model(model_name)
        elif model_name.lower() == "gpt4" and self.translation_model != "gpt4":
            self.load_translation_model("gpt4", api_key)
        
        # Retrieve relevant context
        retrieved = self.retrieve(tulu_text, k=k)
        
        # Build context from retrieved passages
        context_parts = []
        for r in retrieved:
            text = r["text"]
            metadata = r.get("metadata", {})
            # Check for parallel English text (can be in metadata or top-level)
            parallel_text = None
            if "parallel" in metadata:
                parallel_text = metadata["parallel"]
            elif "parallel" in r:
                parallel_text = r["parallel"]
            
            # Use parallel English text if available
            if parallel_text and parallel_text.strip():
                context_parts.append(f"Tulu: {text}\nEnglish: {parallel_text.strip()}")
            else:
                context_parts.append(text)
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length]
        
        # Build prompt based on model
        if model_name.lower() == "gpt4":
            return self._translate_gpt4(tulu_text, context, api_key)
        else:
            return self._translate_hf_model(tulu_text, context, model_name)
    
    def _translate_gpt4(self, tulu_text: str, context: str, api_key: Optional[str] = None) -> str:
        """Translate using GPT-4 API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("GPT-4 API key not provided. Set OPENAI_API_KEY environment variable.")
        
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a translator specializing in Tulu to English translation. 
Here are some example Tulu-English translation pairs for context:

{context}

Now translate the following Tulu text to English:

{tulu_text}

Translation:"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"GPT-4 API error: {e}")
    
    def _translate_hf_model(self, tulu_text: str, context: str, model_name: str) -> str:
        """Translate using HuggingFace models (mT5 or mBART)."""
        if model_name == "mT5":
            # mT5: Use prompt-based approach with RAG context
            # Build prompt with context examples
            if context:
                # Use context to create few-shot examples
                context_lines = context.split("\n\n")[:2]  # Use top 2 examples
                examples = []
                for line in context_lines:
                    if "Tulu:" in line and "English:" in line:
                        examples.append(line)
                
                if examples:
                    context_prompt = "Examples:\n" + "\n".join(examples) + "\n\n"
                    prompt = f"{context_prompt}translate Tulu to English: {tulu_text}"
                else:
                    prompt = f"translate Tulu to English: {tulu_text}"
            else:
                prompt = f"translate Tulu to English: {tulu_text}"
            
            inputs = self.translation_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.translation_model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
            
            translation = self.translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        
        elif model_name == "mBART":
            # mBART: Use many-to-many format
            # mBART-50 supports many languages but Tulu might not be directly supported
            # We'll try to use Hindi (hi_IN) as proxy or direct translation
            # First, try to set source language (mBART-50 supports en_XX, hi_IN, etc.)
            # Since Tulu is not in mBART-50, we'll treat input as generic text
            inputs = self.translation_tokenizer(
                tulu_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Try to use English as target (mBART-50 supports en_XX)
            try:
                # Get available language codes
                if hasattr(self.translation_tokenizer, "lang_code_to_id"):
                    # Try to use en_XX as target
                    if "en_XX" in self.translation_tokenizer.lang_code_to_id:
                        en_id = self.translation_tokenizer.lang_code_to_id["en_XX"]
                        generated_tokens = self.translation_model.generate(
                            **inputs,
                            forced_bos_token_id=en_id,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True
                        )
                    else:
                        # Fallback: generate without forced bos token
                        generated_tokens = self.translation_model.generate(
                            **inputs,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True
                        )
                else:
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
            except Exception as e:
                # Fallback: try without forced_bos_token_id
                print(f"[WARN] mBART translation warning: {e}")
                generated_tokens = self.translation_model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
            
            translation = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return translation
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def evaluate(self, test_file: str, model_name: str = "mT5", k: int = 3, api_key: Optional[str] = None):
        """
        Evaluate translation quality using BLEU and METEOR metrics.
        
        Args:
            test_file: JSONL file with test examples (each line: {"tulu": "...", "english": "..."})
            model_name: Translation model to use
            k: Number of retrieved passages
            api_key: API key for GPT-4 (if using)
        """
        if nltk is None:
            raise ImportError("nltk not installed. Install with: pip install nltk")
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        # Load test data
        test_data = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "tulu" in record and "english" in record:
                            test_data.append(record)
                    except json.JSONDecodeError:
                        continue
        
        if not test_data:
            raise ValueError("No valid test examples found")
        
        print(f"[INFO] Evaluating on {len(test_data)} examples")
        print(f"[INFO] Model: {model_name}, k={k}")
        
        # Load translation model
        self.load_translation_model(model_name, api_key)
        
        # Translate and evaluate
        bleu_scores = []
        meteor_scores = []
        translations = []
        
        smoothing = SmoothingFunction().method1
        
        for i, example in enumerate(test_data):
            tulu_text = example["tulu"]
            reference = example["english"]
            
            # Translate
            try:
                translation = self.translate_with_rag(tulu_text, model_name, k, api_key)
                translations.append({
                    "tulu": tulu_text,
                    "reference": reference,
                    "translation": translation
                })
                
                # Calculate BLEU
                ref_tokens = nltk.word_tokenize(reference.lower())
                hyp_tokens = nltk.word_tokenize(translation.lower())
                bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
                bleu_scores.append(bleu)
                
                # Calculate METEOR
                meteor = meteor_score([ref_tokens], hyp_tokens)
                meteor_scores.append(meteor)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(test_data)} examples...")
            
            except Exception as e:
                print(f"[ERROR] Failed to translate example {i+1}: {e}")
                continue
        
        # Calculate averages
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
        avg_meteor = np.mean(meteor_scores) if meteor_scores else 0.0
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Test examples: {len(test_data)}")
        print(f"Successfully translated: {len(translations)}")
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average METEOR: {avg_meteor:.4f}")
        print("="*60)
        
        # Save translations
        output_file = f"data/evaluations/translations_{model_name}_{k}.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for t in translations:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\n[OK] Translations saved to: {output_file}")
        
        # Save metrics
        metrics_file = f"data/evaluations/metrics_{model_name}_{k}.json"
        metrics = {
            "model": model_name,
            "k": k,
            "num_examples": len(test_data),
            "num_translated": len(translations),
            "avg_bleu": float(avg_bleu),
            "avg_meteor": float(avg_meteor),
            "bleu_scores": [float(s) for s in bleu_scores],
            "meteor_scores": [float(s) for s in meteor_scores]
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[OK] Metrics saved to: {metrics_file}")
        
        return {
            "avg_bleu": avg_bleu,
            "avg_meteor": avg_meteor,
            "translations": translations
        }


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Tulu-English Translation")
    parser.add_argument("--build_index", action="store_true", help="Build FAISS index from passages")
    parser.add_argument("--passages_file", type=str, default=DEFAULT_PASSAGES_FILE, help="Path to passages.jsonl")
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR, help="Directory for index files")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name")
    
    parser.add_argument("--translate", type=str, help="Translate a single Tulu sentence")
    parser.add_argument("--model", type=str, default="mT5", choices=["mT5", "mBART", "gpt4"], help="Translation model")
    parser.add_argument("--k", type=int, default=3, help="Number of retrieved passages")
    parser.add_argument("--api_key", type=str, help="API key for GPT-4")
    
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--test_file", type=str, help="Path to test JSONL file")
    
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline_obj = RAGPipeline(
        embedding_model_name=args.embedding_model,
        index_dir=args.index_dir,
        device=args.device
    )
    
    # Build index
    if args.build_index:
        pipeline_obj.build_index(args.passages_file)
        return
    
    # Translate
    if args.translate:
        if pipeline_obj.index is None:
            print("[INFO] Loading index...")
            pipeline_obj.load_index()
        
        translation = pipeline_obj.translate_with_rag(
            args.translate,
            model_name=args.model,
            k=args.k,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY")
        )
        print(f"\nTulu: {args.translate}")
        print(f"English: {translation}")
        return
    
    # Evaluate
    if args.evaluate:
        if not args.test_file:
            print("[ERROR] --test_file required for evaluation")
            return
        
        if pipeline_obj.index is None:
            print("[INFO] Loading index...")
            pipeline_obj.load_index()
        
        pipeline_obj.evaluate(
            args.test_file,
            model_name=args.model,
            k=args.k,
            api_key=args.api_key or os.getenv("OPENAI_API_KEY")
        )
        return
    
    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()

