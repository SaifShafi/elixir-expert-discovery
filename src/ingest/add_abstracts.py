#!/usr/bin/env python3
"""
Enrich a CSV with PubMed abstracts by PMID, in batches, with resume support.

Usage (run separately for each approach):
  python enrich_abstracts.py \
    --in seed_publications.csv \
    --pmid-col pmid \
    --out seed_publications_with_abstracts.csv \
    --checkpoint seed_abstracts_checkpoint.csv

  python enrich_abstracts.py \
    --in theme_publications.csv \
    --pmid-col pmid \
    --out theme_publications_with_abstracts.csv \
    --checkpoint theme_abstracts_checkpoint.csv

Notes:
- Keeps approaches separate (run twice).
- If your CSV already has an 'abstract' column, this only fills the missing ones.
- Safe to stop/restart; it will skip PMIDs already in the checkpoint.
"""

import os
import sys
import time
import math
import argparse
import csv
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200  # PubMed allows up to 200 IDs per efetch call
PAUSE_SEC = 0.05  # ~3 req/sec; bump down if you see 429s
TIMEOUT = 30


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fetch PubMed abstracts by PMID and enrich a CSV."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument(
        "--pmid-col", dest="pmid_col", default="pmid", help="Column name holding PMIDs"
    )
    ap.add_argument(
        "--out", dest="out", required=True, help="Output CSV path (enriched)"
    )
    ap.add_argument(
        "--checkpoint",
        dest="checkpoint",
        required=True,
        help="CSV to cache pmid->abstract (resumable)",
    )
    ap.add_argument(
        "--email",
        dest="email",
        default=None,
        help="Contact email for NCBI (optional but recommended)",
    )
    ap.add_argument(
        "--api-key-env",
        dest="api_key_env",
        default=os.getenv("NCBI_API_KEY", ""),
        help="Env var name for NCBI API key",
    )
    return ap.parse_args()


def read_checkpoint(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str)
    if "pmid" not in df.columns or "abstract" not in df.columns:
        return {}
    # pmid as string keys
    return dict(zip(df["pmid"].astype(str), df["abstract"].astype(str)))


def append_checkpoint(path, rows):
    # rows: list of dicts with keys ['pmid', 'abstract']
    hdr_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "abstract"])
        if not hdr_exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def extract_abstract_from_article(article_el):
    """
    Pulls together AbstractText blocks. Preserves section labels if present.
    Returns '' if no abstract found.
    """
    ab = []
    for abstract in article_el.findall(".//Abstract"):
        for at in abstract.findall(".//AbstractText"):
            txt = "".join(at.itertext()).strip()
            label = at.attrib.get("Label") or at.attrib.get("NlmCategory")
            if label and txt:
                ab.append(f"{label}: {txt}")
            elif txt:
                ab.append(txt)
    return "\n\n".join(ab).strip()


def fetch_batch(pmids, api_key=None, email=None):
    """
    Fetch one batch of PMIDs via efetch; returns dict: pmid -> abstract ('' if none).
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    r = requests.get(EFETCH_URL, params=params, timeout=TIMEOUT)
    if r.status_code == 429:
        # too many requests -> backoff
        time.sleep(2)
        r = requests.get(EFETCH_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    out = {}
    for art in root.findall(".//PubmedArticle"):
        # pmid
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None
        if not pmid:
            continue
        abstract = extract_abstract_from_article(art)
        out[str(pmid)] = abstract
    # Some PMIDs might not be returned (retracted/invalid/etc.) -> fill missing
    for p in pmids:
        out.setdefault(str(p), "")
    return out


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, None)

    # Read input
    df = pd.read_csv(args.inp, low_memory=False)
    if args.pmid_col not in df.columns:
        print(f"PMID column '{args.pmid_col}' not found in {args.inp}", file=sys.stderr)
        sys.exit(1)

    # Ensure we have an 'abstract' column
    if "abstract" not in df.columns:
        df["abstract"] = pd.NA

    # Normalize PMIDs to strings (drop obvious NAs)
    pmids_all = df[args.pmid_col].dropna().astype(str).str.strip()
    pmids_all = pmids_all[pmids_all != ""].unique().tolist()

    # Only fetch for rows with missing abstracts
    missing_mask = df["abstract"].isna() | (
        df["abstract"].astype(str).str.strip() == ""
    )
    pmids_need = df.loc[missing_mask, args.pmid_col].dropna().astype(str).str.strip()
    pmids_need = sorted(set([p for p in pmids_need if p != ""]))

    print(f"Total rows: {len(df)}")
    print(f"Unique PMIDs: {len(set(pmids_all))}")
    print(f"Abstracts missing: {len(pmids_need)}")

    # Load checkpoint (pmid -> abstract)
    cache = read_checkpoint(args.checkpoint)
    print(f"Checkpoint loaded: {len(cache)} PMIDs cached")

    # Filter out PMIDs already cached
    pmids_todo = [p for p in pmids_need if p not in cache]
    print(f"To fetch now: {len(pmids_todo)} PMIDs")

    if not pmids_todo:
        print("Nothing to fetch. Will just merge checkpoint and save output.")

    # Fetch in batches
    done = 0
    total = len(pmids_todo)
    email = args.email

    for i in range(0, total, BATCH_SIZE):
        batch = pmids_todo[i : i + BATCH_SIZE]
        try:
            res = fetch_batch(batch, api_key=api_key, email=email)
        except Exception as e:
            # simple retry once after a brief nap
            print(f"Fetch error on batch {i//BATCH_SIZE+1}: {e}. Retrying in 5s...")
            time.sleep(5)
            res = fetch_batch(batch, api_key=api_key, email=email)

        # update cache + checkpoint on disk immediately (so we can resume safely)
        rows = [{"pmid": k, "abstract": v} for k, v in res.items()]
        append_checkpoint(args.checkpoint, rows)
        cache.update(res)

        done += len(batch)
        print(f"Fetched {done}/{total}  | last batch size={len(batch)}")
        time.sleep(PAUSE_SEC)  # be nice to NCBI

    # Merge abstracts back into the dataframe
    # Only fill where empty; keep existing abstracts as-is
    def fill_fn(row):
        if pd.isna(row["abstract"]) or str(row["abstract"]).strip() == "":
            key = str(row[args.pmid_col]).strip()
            if key in cache:
                return cache[key]
        return row["abstract"]

    df["abstract"] = df.apply(fill_fn, axis=1)

    # Save output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved enriched CSV to: {out_path}")

    # Small stats
    filled = df["abstract"].notna().sum()
    print(f"Abstracts present after enrichment: {filled}/{len(df)}")


if __name__ == "__main__":
    main()
