#!/usr/bin/env python3
"""
evaluate_rag.py

Convenience script for evaluating the RAG pipeline with different models and configurations.

Usage:
    python src/evaluate_rag.py --test_file data/test.jsonl --models mT5 mBART --k_values 1 3 5
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with different configurations")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--models", nargs="+", default=["mT5"], choices=["mT5", "mBART", "gpt4"], help="Models to evaluate")
    parser.add_argument("--k_values", nargs="+", type=int, default=[3], help="Number of retrieved passages (k)")
    parser.add_argument("--index_dir", type=str, default="data/rag_index", help="Index directory")
    parser.add_argument("--embedding_model", type=str, default="paraphrase-multilingual-MiniLM-L12-v2", help="Embedding model")
    parser.add_argument("--api_key", type=str, help="GPT-4 API key (if using GPT-4)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_file):
        print(f"[ERROR] Test file not found: {args.test_file}")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_model_name=args.embedding_model,
        index_dir=args.index_dir
    )
    
    if pipeline.index is None:
        print("[INFO] Loading index...")
        pipeline.load_index()
    
    # Evaluate each model and k combination
    results = []
    for model in args.models:
        for k in args.k_values:
            print(f"\n{'='*60}")
            print(f"Evaluating: Model={model}, k={k}")
            print(f"{'='*60}")
            
            try:
                result = pipeline.evaluate(
                    args.test_file,
                    model_name=model,
                    k=k,
                    api_key=args.api_key or os.getenv("OPENAI_API_KEY")
                )
                results.append({
                    "model": model,
                    "k": k,
                    "bleu": result["avg_bleu"],
                    "meteor": result["avg_meteor"]
                })
            except Exception as e:
                print(f"[ERROR] Evaluation failed: {e}")
                continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'k':<5} {'BLEU':<10} {'METEOR':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<10} {r['k']:<5} {r['bleu']:<10.4f} {r['meteor']:<10.4f}")
    print("="*60)

if __name__ == "__main__":
    main()

