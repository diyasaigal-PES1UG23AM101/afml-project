#!/usr/bin/env python3
"""
create_test_set.py

Create a test set for evaluation from passages.jsonl that have parallel English translations
(like Bible passages with English parallel text).

Usage:
    python src/create_test_set.py --input data/processed/passages.jsonl --output data/test.jsonl --num_samples 100
"""

import json
import argparse
import random
import os

def main():
    parser = argparse.ArgumentParser(description="Create test set from passages with parallel translations")
    parser.add_argument("--input", type=str, default="data/processed/bible_passages.jsonl", help="Input passages file (use bible_passages.jsonl for parallel translations)")
    parser.add_argument("--output", type=str, default="data/test.jsonl", help="Output test file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load passages with parallel translations
    candidates = []
    input_files = [args.input]
    
    # Also check main passages file if bible_passages not found
    if "bible_passages" not in args.input:
        bible_file = args.input.replace("passages.jsonl", "bible_passages.jsonl")
        if os.path.exists(bible_file):
            input_files.append(bible_file)
    
    for input_file in input_files:
        if not os.path.exists(input_file):
            continue
            
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        # Check if it has parallel English text (can be at top level or in meta)
                        parallel_text = None
                        if "parallel" in record:
                            parallel_text = record["parallel"]
                        elif "meta" in record and isinstance(record["meta"], dict) and "parallel" in record["meta"]:
                            parallel_text = record["meta"]["parallel"]
                        
                        if parallel_text and parallel_text.strip():
                            tulu_text = record.get("text", "").strip()
                            if tulu_text and len(tulu_text) > 10:
                                candidates.append({
                                    "tulu": tulu_text,
                                    "english": parallel_text.strip()
                                })
                    except json.JSONDecodeError:
                        continue
    
    if not candidates:
        print("[ERROR] No passages with parallel translations found")
        return
    
    print(f"[INFO] Found {len(candidates)} candidates with parallel translations")
    
    # Sample test set
    num_samples = min(args.num_samples, len(candidates))
    test_set = random.sample(candidates, num_samples)
    
    # Write test set
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for example in test_set:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"[OK] Created test set with {len(test_set)} examples: {args.output}")
    print(f"[INFO] Sample examples:")
    for i, ex in enumerate(test_set[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    Tulu: {ex['tulu'][:100]}...")
        print(f"    English: {ex['english'][:100]}...")

if __name__ == "__main__":
    main()

