#!/usr/bin/env python3
"""
process_raw.py

Reads raw text files from wiki and news directories, cleans them,
and appends them as JSONL records to data/processed/passages.jsonl.

This file should be run *after* scraping but *before* building the final dataset.
It is designed to merge with the output of add_bible.py.

Usage:
    python src/process_raw.py
"""

import os
import glob
import json
import argparse
import uuid
from pathlib import Path

DEFAULT_WIKI_DIR = "data/raw/wiki"
DEFAULT_NEWS_DIR = "data/raw/news"
DEFAULT_OUT_DIR = "data/processed"
DEFAULT_MAIN_PASSAGES = os.path.join(DEFAULT_OUT_DIR, "passages.jsonl")

MIN_TEXT_LENGTH = 50 # Min characters to be considered a valid passage

def clean_text(text):
    """Apply basic cleaning rules."""
    # Strip leading/trailing whitespace
    text = text.strip()
    # You could add more rules here, e.g., regex replacements
    # for now, we'll just normalize whitespace (optional)
    # text = re.sub(r'\s+', ' ', text)
    return text

def process_directory(dir_path, source_type, out_file_handle):
    """
    Reads all .txt files in a directory, cleans them, and writes
    to the open output file handle.
    """
    print(f"Processing directory: {dir_path} (source_type: {source_type})")
    
    # Use glob to find all .txt files, including in subdirs if any
    files = glob.glob(os.path.join(dir_path, "**", "*.txt"), recursive=True)
    
    if not files:
        print(f"[WARN] No .txt files found in {dir_path}")
        return 0

    count = 0
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            cleaned_text = clean_text(raw_text)
            
            if len(cleaned_text) < MIN_TEXT_LENGTH:
                print(f"[SKIP] File too short: {fpath}")
                continue
                
            # Create a record that matches the structure from add_bible.py
            record = {
                "id": str(uuid.uuid4()),
                "text": cleaned_text,
                "source": fpath.replace("\\", "/"), # Normalize path separators
                "meta": {
                    "lang": "tcy", # Tulu ISO 639-3 code
                    "source_type": source_type,
                    "filename": Path(fpath).name
                }
            }
            
            # Write as a JSON line
            out_file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process file {fpath}: {e}")
            
    print(f"Processed and added {count} passages from {source_type}.")
    return count

def main():
    parser = argparse.ArgumentParser(description="Process raw wiki/news files into passages.jsonl")
    parser.add_argument("--wiki_dir", type=str, default=DEFAULT_WIKI_DIR, help="Directory with raw wiki .txt files")
    parser.add_argument("--news_dir", type=str, default=DEFAULT_NEWS_DIR, help="Directory with raw news .txt files")
    parser.add_argument("--out_file", type=str, default=DEFAULT_MAIN_PASSAGES, help="Output JSONL file (will be appended)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    total_added = 0
    
    # Open the output file in append mode ("a")
    with open(args.out_file, "a", encoding="utf-8") as f:
        # Process Wiki files
        total_added += process_directory(args.wiki_dir, "wiki", f)
        
        # Process News files
        total_added += process_directory(args.news_dir, "news", f)
        
    print(f"\n[OK] Done. Added a total of {total_added} new passages to {args.out_file}")

if __name__ == "__main__":
    main()