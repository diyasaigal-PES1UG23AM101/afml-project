#!/usr/bin/env python3
"""
scrape_wiki.py
Download pages from Tulu Wikipedia (https://tcy.wikipedia.org/)
Saves each page as data/raw/wiki/<safe_title>.txt

Usage:
    python src\scrape_wiki.py --limit 200
"""

import requests
import time
import os
import argparse
import sys
import json
import re

WIKI_API = "https://tcy.wikipedia.org/w/api.php"
# *** CHANGE 1: Define a polite User-Agent header ***
# Replace the email with your actual contact information
HEADERS = {
    'User-Agent': 'TuluWikiScraper/1.0 (Contact at your_email@example.com) - Educational project to gather Tulu data'
}

def safe_filename(s):
    # remove characters that are problematic on Windows filenames
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    s = s.strip()
    if len(s) == 0:
        s = "page"
    return s[:200]

def fetch_all_pages(limit=None):
    pages = []
    apcontinue = None
    fetched = 0
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "allpages",
            "aplimit": 50  # fairly small, safe
        }
        if apcontinue:
            params["apcontinue"] = apcontinue

        # *** CHANGE 2: Pass HEADERS to the request ***
        resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status() # This is where the 403 error happened
        data = resp.json()
        batch = data.get("query", {}).get("allpages", [])
        for p in batch:
            pages.append(p["title"])
            fetched += 1
            if limit and fetched >= limit:
                return pages
        # continuation?
        cont = data.get("continue")
        if cont and cont.get("apcontinue"):
            apcontinue = cont["apcontinue"]
            # *** CHANGE 3: Increase politeness pause to 1.0s ***
            time.sleep(1.0) 
        else:
            break
    return pages

def fetch_and_save(title, out_dir):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": title
    }
    try:
        # *** CHANGE 4: Pass HEADERS to the request ***
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return False
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        if not text or len(text.strip()) < 10:
            return False
        fname = safe_filename(title) + ".txt"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"[WARN] failed to fetch '{title}': {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="max number of pages to fetch (for testing)")
    parser.add_argument("--out", type=str, default="data/raw/wiki",
                        help="output directory")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    print("Fetching list of pages from Tulu Wikipedia...")
    titles = fetch_all_pages(limit=args.limit)
    print(f"Found {len(titles)} page titles (will try to download).")

    success = 0
    fail = 0
    for idx, title in enumerate(titles, start=1):
        print(f"[{idx}/{len(titles)}] Downloading: {title}")
        ok = fetch_and_save(title, out_dir)
        if ok:
            success += 1
        else:
            fail += 1
        # *** CHANGE 5: Increase politeness pause to 1.0s ***
        time.sleep(1.0) 

    print("Done.")
    print(f"Successful: {success}, Failed/Empty: {fail}")
    # small index file for provenance
    meta = {"total_titles": len(titles), "success": success, "fail": fail}
    with open(os.path.join(out_dir, "scrape_meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()