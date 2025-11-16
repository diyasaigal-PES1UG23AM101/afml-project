#!/usr/bin/env python3
"""
scrape_news.py
Reads 'src/news_urls.txt' (one URL per line) and extracts article text using trafilatura.
Saves each article as data/raw/news/<safe_title>.txt

Usage:
    python src/scrape_news.py
"""

import os
import re
import argparse
import sys
import time
import urllib.parse
import trafilatura
import requests

def safe_filename(s):
    # Remove characters illegal in filepaths
    s = re.sub(r'[<>:"/\\|?*]', '_', s)
    s = s.strip()
    if len(s) == 0:
        s = "article"
    # truncate
    return s[:200]

def load_urls(path="src/news_urls.txt"):
    if not os.path.exists(path):
        print(f"[ERROR] URL file not found: {path}. Create it with one article URL per line.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return urls

def fetch_and_save(url, out_dir):
    try:
        # 1. Fetch the URL content
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # 2. Extract the main text using trafilatura
        # We pass the HTML content, not a language code, making it robust for Tulu.
        text = trafilatura.extract(response.text)
        
        if not text or len(text.strip()) < 50:
            print(f"[WARN] Article too short or empty: {url}")
            return False
            
        # 3. Use the URL path to create a filename (most reliable method now)
        url_path = urllib.parse.urlparse(url).path.replace("/", "_").strip("_")
        
        # If the path is empty (e.g., scraping a root domain), use the domain name
        if not url_path:
            url_path = urllib.parse.urlparse(url).netloc
            
        fname = safe_filename(url_path) + ".txt"
        path = os.path.join(out_dir, fname)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"[OK] Saved: {path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] HTTP/Network error fetching {url}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] failed to extract content from {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/news", help="output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    urls = load_urls()

    success = 0
    fail = 0
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] Fetching: {url}")
        ok = fetch_and_save(url, args.out)
        if ok:
            success += 1
        else:
            fail += 1
        time.sleep(0.7)
    print("Done. Success:", success, "Failed:", fail)

if __name__ == "__main__":
    main()
