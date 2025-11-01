#!/usr/bin/env python3
r"""
add_bible.py

Ingest Tulu Bible text files from data/raw/bible/ (and optional English in data/raw/bible_en/)
and convert to JSONL passages for downstream processing.

Usage:
    python src/add_bible.py --merge

Options:
    --merge    Append simplified bible passages into data/processed/passages.jsonl
    --bible_dir PATH     (default: data/raw/bible)
    --bible_en_dir PATH  (default: data/raw/bible_en)
    --out_dir PATH       (default: data/processed)
"""
import os
import re
import json
import argparse
import glob
import uuid
from pathlib import Path

# Defaults (change if needed)
DEFAULT_BIBLE_DIR = "data/raw/bible"
DEFAULT_BIBLE_EN_DIR = "data/raw/bible_en"
DEFAULT_OUT_DIR = "data/processed"
DEFAULT_OUT_FILE = os.path.join(DEFAULT_OUT_DIR, "bible_passages.jsonl")
DEFAULT_MAIN_PASSAGES = os.path.join(DEFAULT_OUT_DIR, "passages.jsonl")

VERSE_RE = re.compile(r'^\s*(\d{1,3})\s*[:.\)]\s*(.+)$')  # matches "1: text" or "1. text" or "1) text"

def parse_chapter_text(text):
    """
    Try to split a chapter into verse id -> verse text.
    If no verse markers found, return the whole text as one verse (verse 1).
    Returns a list of tuples: (verse_number_or_None, verse_text)
    """
    if not text:
        return []
    verses = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        m = VERSE_RE.match(line)
        if m:
            verse_num = int(m.group(1))
            verse_text = m.group(2).strip()
            verses.append((verse_num, verse_text))
        else:
            # If the line doesn't match the verse pattern, still keep it as text.
            # We'll set verse number to None and later normalize sequentially if needed.
            verses.append((None, line))
    # if all verse numbers are None, give them sequential numbers starting at 1
    if verses and all(v[0] is None for v in verses):
        verses = [(i+1, v[1]) for i, v in enumerate(verses)]
    return verses

def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def book_chapter_from_filename(fn):
    """
    Expect filenames like 'matthew_1.txt' or 'genesis_02.txt'
    Fallback: treat the whole stem as the book and chapter 0.
    """
    name = Path(fn).stem
    parts = name.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        chapter = int(parts[-1])
        book = "_".join(parts[:-1])
    else:
        book = name
        chapter = 0
    return book, chapter

def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_to_main_passages(main_path, records):
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    with open(main_path, "a", encoding="utf-8") as f:
        for rec in records:
            # create a simpler record for the main passages file
            small = {
                "id": rec["id"],
                "text": rec["text"],
                "source": rec["source"],
                "meta": {"book": rec.get("book"), "chapter": rec.get("chapter"), "verse": rec.get("verse"), "lang": rec.get("lang")}
            }
            f.write(json.dumps(small, ensure_ascii=False) + "\n")

def process_bible(bible_dir=DEFAULT_BIBLE_DIR, bible_en_dir=DEFAULT_BIBLE_EN_DIR, out_dir=DEFAULT_OUT_DIR, merge_main=False):
    out_file = os.path.join(out_dir, "bible_passages.jsonl")
    main_passages = os.path.join(out_dir, "passages.jsonl")

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(bible_dir, "*.txt")))
    if not files:
        print(f"[WARN] No files found in {bible_dir}. Place chapter files like matthew_1.txt")
        return

    # load english map if exists
    en_map = {}
    if os.path.isdir(bible_en_dir):
        en_files = sorted(glob.glob(os.path.join(bible_en_dir, "*.txt")))
        for ef in en_files:
            key = Path(ef).stem.lower()
            en_map[key] = load_text_file(ef)

    out_lines = []
    stats = {
        "chapters_processed": 0,
        "total_tulu_verses": 0,
        "total_en_verses": 0,
        "chapters_with_en": 0,
        "chapters_with_mismatch": 0,
        "mismatch_details": []  # list of tuples (chapter_key, tulu_count, en_count)
    }

    for fp in files:
        text = load_text_file(fp)
        book, chapter = book_chapter_from_filename(fp)
        key = f"{book}_{chapter}".lower()
        en_text = en_map.get(key)
        verses = parse_chapter_text(text)
        en_verses = parse_chapter_text(en_text) if en_text else None

        tulu_count = len(verses)
        en_count = len(en_verses) if en_verses else 0

        stats["chapters_processed"] += 1
        stats["total_tulu_verses"] += tulu_count
        if en_verses:
            stats["chapters_with_en"] += 1
            stats["total_en_verses"] += en_count

        # mismatch warning if counts differ
        if en_verses and tulu_count != en_count:
            diff = abs(tulu_count - en_count)
            stats["chapters_with_mismatch"] += 1
            stats["mismatch_details"].append((key, tulu_count, en_count))
            print(f"[WARN] Chapter {key}: Tulu {tulu_count} verses, English {en_count} verses ({diff} unmatched)")

        # create records
        for i, (vnum, vtext) in enumerate(verses, start=1):
            # if vnum is None, assign sequential number i
            verse_num = vnum if vnum is not None else i
            rec = {
                "id": str(uuid.uuid4()),
                "text": vtext,
                "source": fp.replace("\\", "/"),
                "book": book,
                "chapter": chapter,
                "verse": verse_num,
                "lang": "tcy"
            }
            # attempt to attach parallel english verse (best-effort)
            if en_verses:
                # first try matching by verse number
                match = next((ev for ev in en_verses if ev[0] == vnum), None)
                if match:
                    rec["parallel"] = match[1]
                else:
                    # fallback by index: try to match by position
                    try:
                        # verse_num may be 0 for some cases; ensure index valid
                        if verse_num > 0 and len(en_verses) >= verse_num:
                            rec["parallel"] = en_verses[verse_num - 1][1]
                    except Exception:
                        # ignore if alignment not possible
                        pass
            out_lines.append(rec)

    # write bible_passages.jsonl
    write_jsonl(out_file, out_lines)
    print(f"[OK] Wrote {len(out_lines)} bible passages to {out_file}")

    # optionally merge into main passages.jsonl (append)
    if merge_main:
        append_to_main_passages(main_passages, out_lines)
        print(f"[OK] Merged bible passages into {main_passages}")

    # summary
    print("\n=== Bible Import Summary ===")
    print(f"Chapters processed: {stats['chapters_processed']}")
    print(f"Total Tulu verses: {stats['total_tulu_verses']}")
    print(f"Chapters with English parallel: {stats['chapters_with_en']}")
    print(f"Total English verses (sum across chapters with English): {stats['total_en_verses']}")
    print(f"Chapters with mismatched verse counts: {stats['chapters_with_mismatch']}")
    if stats['mismatch_details']:
        print("\nMismatch details (chapter_key, tulu_count, en_count):")
        for detail in stats['mismatch_details']:
            print("  -", detail)
    print("============================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="append bible passages to data/processed/passages.jsonl")
    parser.add_argument("--bible_dir", type=str, default=DEFAULT_BIBLE_DIR, help="path to tulu bible txt files")
    parser.add_argument("--bible_en_dir", type=str, default=DEFAULT_BIBLE_EN_DIR, help="path to english bible txt files (optional)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="output directory for processed files")
    args = parser.parse_args()
    process_bible(bible_dir=args.bible_dir, bible_en_dir=args.bible_en_dir, out_dir=args.out_dir, merge_main=args.merge)
