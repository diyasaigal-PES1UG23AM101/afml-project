#!/usr/bin/env python3
"""
build_dataset.py

Reads the consolidated data/processed/passages.jsonl file and
converts it into train.txt and valid.txt files for model training.

Each passage is written on its own, separated by two newlines.

Usage:
    python src/build_dataset.py
"""

import os
import json
import argparse
import random

DEFAULT_PASSAGES_FILE = "data/processed/passages.jsonl"
DEFAULT_OUT_DIR = "data/final"
VALIDATION_SPLIT = 0.02 # Use 2% of the data for validation

def main():
    parser = argparse.ArgumentParser(description="Build final train/valid text files")
    parser.add_argument("--input_file", type=str, default=DEFAULT_PASSAGES_FILE, help="Input passages.jsonl file")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for train/valid files")
    parser.add_argument("--val_split", type=float, default=VALIDATION_SPLIT, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()
    
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    train_file_path = os.path.join(args.out_dir, "train.txt")
    valid_file_path = os.path.join(args.out_dir, "valid.txt")

    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        print("Please run 'add_bible.py --merge' and 'process_raw.py' first.")
        return

    try:
        # Read all passages into memory to shuffle
        with open(args.input_file, "r", encoding="utf-8") as f:
            passages = [json.loads(line) for line in f]
            
        if not passages:
            print("[ERROR] Input file is empty. No data to process.")
            return
            
        print(f"Loaded {len(passages)} passages from {args.input_file}")
        
        # Shuffle the passages
        random.shuffle(passages)
        
        # Split
        split_index = int(len(passages) * (1.0 - args.val_split))
        train_passages = passages[:split_index]
        valid_passages = passages[split_index:]

        # Write files
        doc_separator = "\n\n" # Separator between documents
        
        with open(train_file_path, "w", encoding="utf-8") as f_train:
            for p in train_passages:
                text = p.get("text", "").strip()
                if text:
                    f_train.write(text + doc_separator)
                    
        with open(valid_file_path, "w", encoding="utf-8") as f_valid:
            for p in valid_passages:
                text = p.get("text", "").strip()
                if text:
                    f_valid.write(text + doc_separator)

        print(f"\n[OK] Successfully created dataset:")
        print(f"  Train: {train_file_path} ({len(train_passages)} passages)")
        print(f"  Valid: {valid_file_path} ({len(valid_passages)} passages)")

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()