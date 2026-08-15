#!/usr/bin/env python3
"""
fetch_pubmed_uk_resumable_v4b.py
--------------------------------
- History paging (WebEnv+QueryKey) with automatic fallback to ID-chunk mode.
- Correctly resumes even if you change --retmax between runs (covered_until).
- Detects empty/error EFETCH pages and falls back automatically.
- Append-only CSVs, per-theme checkpoints, disk-aware raw retention.
"""

import argparse, csv, os, sys, time, json, math, re, shutil, gzip
from pathlib import Path
from typing import List, Dict, Any, Iterable
from datetime import datetime
from xml.etree import ElementTree as ET
import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------- Logging ----------------
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write(f"=== Session start {now()} ===")

    def write(self, msg: str):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ---------------- Disk helpers ----------------
def disk_free_mb(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free / (1024*1024))

def raw_files_sorted(raw_dir: Path):
    if not raw_dir.exists(): return []
    files = [p for p in raw_dir.glob("efetch_*.xml*") if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime)

def enforce_raw_cap(logger: Logger, raw_dir: Path, raw_cap_mb: int):
    if raw_cap_mb <= 0 or not raw_dir.exists():
        return
    files = raw_files_sorted(raw_dir)
    total = sum(int(p.stat().st_size/(1024*1024)) for p in files)
    if total > raw_cap_mb:
        logger.write(f"Raw cap exceeded: {total} MB > {raw_cap_mb} MB; pruning …")
        for p in files:
            try:
                logger.write(f"  deleting raw: {p.name}")
                p.unlink(missing_ok=True)
            except Exception as e:
                logger.write(f"  could not delete {p}: {e}")
            files = raw_files_sorted(raw_dir)
            total = sum(int(q.stat().st_size/(1024*1024)) for q in files)
            if total <= raw_cap_mb:
                break

def ensure_space(logger: Logger, outdir: Path, raw_dir: Path, min_free_mb: int, raw_cap_mb: int, raw_policy: str) -> bool:
    enforce_raw_cap(logger, raw_dir, raw_cap_mb)
    free_mb = disk_free_mb(outdir)
    logger.write(f"Free disk: {free_mb} MB (min required: {min_free_mb} MB)")
    if free_mb >= min_free_mb:
        return True
    if raw_policy != "purge":
        logger.write("Low disk; deleting oldest raw chunks …")
        for p in raw_files_sorted(raw_dir):
            try:
                logger.write(f"  deleting raw: {p.name}")
                p.unlink(missing_ok=True)
            except Exception as e:
                logger.write(f"  could not delete {p}: {e}")
            if disk_free_mb(outdir) >= min_free_mb:
                return True
    return disk_free_mb(outdir) >= min_free_mb

# ---------------- CSV writer ----------------
def safe_csv_append(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header: w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

# ---------------- PubMed calls ----------------
def esearch_history(term: str, email: str, api_key: str) -> Dict[str, Any]:
    params = {"db":"pubmed","term":term,"retmode":"json","usehistory":"y","retmax":0,
              "tool":"msc_project","email":email}
    if api_key: params["api_key"] = api_key
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def esearch_id_page(term: str, email: str, api_key: str, retstart: int, retmax: int) -> Dict[str, Any]:
    params = {"db":"pubmed","term":term,"retmode":"json","retstart":retstart,"retmax":retmax,
              "tool":"msc_project","email":email}
    if api_key: params["api_key"] = api_key
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def efetch_history_chunk(webenv: str, query_key: str, email: str, api_key: str, retstart: int, retmax: int) -> str:
    data = {"db":"pubmed","query_key":query_key,"WebEnv":webenv,
            "retstart":retstart,"retmax":retmax,"retmode":"xml",
            "tool":"msc_project","email":email}
    if api_key: data["api_key"] = api_key
    r = requests.post(f"{BASE}/efetch.fcgi", data=data, timeout=180)
    r.raise_for_status()
    return r.text

def efetch_by_ids(pmids: Iterable[str], email: str, api_key: str) -> str:
    ids = ",".join(pmids)
    data = {"db":"pubmed","id":ids,"retmode":"xml","tool":"msc_project","email":email}
    if api_key: data["api_key"] = api_key
    r = requests.post(f"{BASE}/efetch.fcgi", data=data, timeout=180)
    r.raise_for_status()
    return r.text

# ---------------- Parsing ----------------
def parse_pubmed_xml(xml_text: str) -> List[Dict[str,Any]]:
    root = ET.fromstring(xml_text)
    recs = []
    for art in root.findall(".//PubmedArticle"):
        pmid = (art.findtext(".//PMID") or "").strip()
        art_title = (art.findtext(".//ArticleTitle") or "").strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = art.findtext(".//JournalIssue/PubDate/Year")
        medline_date = art.findtext(".//JournalIssue/PubDate/MedlineDate")
        date = (year or medline_date or "").strip()
        doi = None
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = (aid.text or "").strip()
                break
        authors = []
        for i, au in enumerate(art.findall(".//AuthorList/Author")):
            last = (au.findtext("LastName") or "").strip()
            fore = (au.findtext("ForeName") or "").strip()
            init = (au.findtext("Initials") or "").strip()
            affs = [ (a.findtext("Affiliation") or "").strip() for a in au.findall("AffiliationInfo") ]
            affs = [a for a in affs if a]
            authors.append({"pmid": pmid, "pos": i+1, "last": last, "fore": fore, "initials": init, "affiliations": affs})
        recs.append({"pmid": pmid, "title": art_title, "journal": journal, "year_or_date": date, "doi": doi, "authors": authors})
    return recs

# ---------------- UK affiliation heuristic ----------------
UK_PAT = re.compile(r"\b(United Kingdom|England|Scotland|Wales|Northern Ireland|Great Britain|GBR)\b", re.I)
KENTUCKY_PAT = re.compile(r"\b(Kentucky|Lexington)\b", re.I)
UK_DOMAIN_PAT = re.compile(r"\b(\.ac\.uk|\.nhs\.uk|\.gov\.uk|\.org\.uk|\.co\.uk)\b", re.I)
def is_uk_aff(aff: str) -> bool:
    if not aff: return False
    if KENTUCKY_PAT.search(aff): return False
    return bool(UK_PAT.search(aff) or UK_DOMAIN_PAT.search(aff))

# ---------------- Misc ----------------
def backoff_sleep(attempt: int):
    time.sleep(min(60, 2 ** attempt))

def sanitize_term(term: str) -> str:
    t = (term or "").strip()
    t = t.replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
    return re.sub(r"\s+", " ", t)

def batched(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def xml_has_pubmed_articles(xml_text: str) -> bool:
    if not xml_text:
        return False
    if "<ERROR>" in xml_text or "Empty result" in xml_text:
        return False
    return "<PubmedArticle" in xml_text

def ranges_from_done_pages(done_pages: set, page_size_at_checkpoint: int) -> int:
    # exclusive upper bound of covered items given recorded pages with that page size
    if not done_pages or page_size_at_checkpoint <= 0:
        return 0
    return max(done_pages) * page_size_at_checkpoint

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="CSV with columns: theme,pubmed_query")
    ap.add_argument("--outdir", required=True, help="Output folder (absolute path recommended)")
    ap.add_argument("--email", required=True, help="Your email per NCBI policy")
    ap.add_argument("--api_key", default="", help="NCBI API key")
    ap.add_argument("--theme", default="", help="Only run a single theme (exact match)")
    ap.add_argument("--retmax", type=int, default=5000, help="History page size (efetch page)")
    ap.add_argument("--id_chunk", type=int, default=200, help="Fallback EFETCH by IDs chunk size")
    ap.add_argument("--rate_sleep", type=float, default=0.1, help="Sleep between requests (sec)")
    ap.add_argument("--force_refresh", action="store_true", help="Ignore cached esearch history")
    ap.add_argument("--skip_edges", action="store_true", help="Skip writing coauthor edges (big)")
    # disk-aware options
    ap.add_argument("--raw_policy", choices=["keep","gzip","purge"], default="gzip", help="Retention for raw XML pages")
    ap.add_argument("--raw_cap_mb", type=int, default=1024, help="Cap raw storage (MB) per theme")
    ap.add_argument("--min_free_mb", type=int, default=400, help="Minimum free space required before/after each page")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir / "_status.log")
    logger.write(f"Config: outdir={outdir} retmax={args.retmax} id_chunk={args.id_chunk} email={args.email} theme_only={bool(args.theme)} raw_policy={args.raw_policy} raw_cap_mb={args.raw_cap_mb} min_free_mb={args.min_free_mb}")

    pubs_path = outdir / "publications.csv"
    auth_path = outdir / "authors.csv"
    aff_path  = outdir / "author_affiliations.csv"
    edges_path= outdir / "coauthor_edges.csv"
    pubs_fn = ["pmid","title","journal","year_or_date","doi","theme"]
    auth_fn = ["pmid","author_pos","last","fore","initials","theme"]
    aff_fn  = ["pmid","author_pos","affiliation","is_uk","theme"]
    edge_fn = ["pmid","a","b","theme"]

    import pandas as pd
    Q = pd.read_csv(args.queries)
    if args.theme:
        Q = Q[Q["theme"] == args.theme]
        if Q.empty:
            logger.write(f"No theme named '{args.theme}' in {args.queries}. Exiting.")
            return

    # prevent duplicate writes within a single process run
    seen_pmids = set()

    for _, row in Q.iterrows():
        theme = row["theme"]
        term  = sanitize_term(row["pubmed_query"])
        tdir = outdir / "themes" / theme
        tdir.mkdir(parents=True, exist_ok=True)
        logger.write(f"=== THEME: {theme} ===")

        history_file = tdir / "history.json"
        ckpt_file    = tdir / "checkpoint.json"
        chunks_dir   = tdir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        if not ensure_space(logger, outdir, chunks_dir, args.min_free_mb, args.raw_cap_mb, args.raw_policy):
            logger.write("Not enough free space; pause and resume later.")
            return

        # History (cached)
        history = None
        if history_file.exists() and not args.force_refresh:
            try:
                history = json.loads(history_file.read_text(encoding="utf-8"))
                logger.write(f"Loaded cached history (count={history.get('Count','?')})")
            except Exception:
                history = None
        if history is None:
            logger.write("Querying esearch (usehistory=y) …")
            for attempt in range(1, 6):
                try:
                    data = esearch_history(term, args.email, args.api_key)
                    break
                except Exception as e:
                    logger.write(f"esearch error (attempt {attempt}): {e}")
                    backoff_sleep(attempt)
            else:
                logger.write("esearch failed after retries. Skipping theme.")
                continue
            es = data.get("esearchresult", {})
            history = {"Count": int(es.get("count","0")),
                       "WebEnv": es.get("webenv",""),
                       "QueryKey": es.get("querykey","")}
            history_file.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.write(f"Esearch history saved -> {history_file}")

        total = int(history.get("Count", 0))
        if total == 0 or not history.get("WebEnv") or not history.get("QueryKey"):
            logger.write("No results or history missing; continuing.")
            continue

        # Checkpoint (init BEFORE we read from it!)
        page_size = args.retmax
        ckpt = {"done_pages": [], "fetched": 0, "total": total, "retmax": page_size}
        if ckpt_file.exists():
            try:
                ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        done_pages = set(ckpt.get("done_pages", []))
        ckpt_retmax = int(ckpt.get("retmax", page_size))
        covered_until = ranges_from_done_pages(done_pages, ckpt_retmax)
        logger.write(f"Resume logic: covered_until={covered_until} (was retmax={ckpt_retmax}, now retmax={page_size})")

        # Page loop
        for retstart in range(0, total, page_size):
            # Skip any region fully covered already under old page size
            if retstart < covered_until:
                continue
            page = (retstart // page_size) + 1
            if page in done_pages:
                logger.write(f"  [page {page}] skip (already done)")
                continue

            # Disk check
            if not ensure_space(logger, outdir, chunks_dir, args.min_free_mb, args.raw_cap_mb, args.raw_policy):
                logger.write("Low disk before page; pausing to resume later.")
                ckpt["done_pages"] = sorted(done_pages)
                ckpt["retmax"] = page_size
                ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
                return

            label = f"{retstart:09d}_{min(retstart+page_size, total):09d}"
            xml_path = chunks_dir / f"efetch_{label}.xml"
            xml_gz_path = chunks_dir / f"efetch_{label}.xml.gz"

            # --- Try history-based EFETCH (fallback on empty/error) ---
            xml_text = None
            history_ok = False
            for attempt in range(1, 6):
                try:
                    xml_text = efetch_history_chunk(history["WebEnv"], history["QueryKey"], args.email, args.api_key, retstart, page_size)
                    if not xml_has_pubmed_articles(xml_text):
                        raise ValueError("History efetch returned empty or error XML")
                    history_ok = True
                    break
                except Exception as e:
                    logger.write(f"  history efetch issue (attempt {attempt}, page {page}): {e}")
                    backoff_sleep(attempt)

            if not history_ok:
                # --- Fallback to ID-chunk mode for this page ---
                logger.write(f"  FALLBACK: retrieving IdList for page {page}")
                ids = []
                for attempt in range(1, 6):
                    try:
                        j = esearch_id_page(term, args.email, args.api_key, retstart=retstart, retmax=page_size)
                        ids = j.get("esearchresult", {}).get("IdList", [])
                        break
                    except Exception as e:
                        logger.write(f"  esearch IdList error (attempt {attempt}): {e}")
                        backoff_sleep(attempt)
                if not ids:
                    logger.write("  could not retrieve IdList for this page; marking done to avoid spin.")
                    done_pages.add(page)
                    ckpt["done_pages"] = sorted(done_pages)
                    ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue

                combined_xml = []
                chunk_idx = 0
                for chunk_ids in batched(ids, args.id_chunk):
                    chunk_idx += 1
                    for attempt in range(1, 6):
                        try:
                            chunk_xml = efetch_by_ids(chunk_ids, args.email, args.api_key)
                            combined_xml.append(chunk_xml)
                            break
                        except Exception as e:
                            logger.write(f"    efetch by IDs error (attempt {attempt}, chunk {chunk_idx}): {e}")
                            backoff_sleep(attempt)
                    time.sleep(args.rate_sleep)
                xml_text = "<PubmedArticleSet>" + "".join(combined_xml) + "</PubmedArticleSet>"
                if not xml_has_pubmed_articles(xml_text):
                    logger.write("  fallback produced no articles; marking page done to avoid spin.")
                    done_pages.add(page)
                    ckpt["done_pages"] = sorted(done_pages)
                    ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue

            # Save raw per policy
            try:
                if args.raw_policy == "keep":
                    xml_path.write_text(xml_text, encoding="utf-8")
                elif args.raw_policy == "gzip":
                    with gzip.open(xml_gz_path, "wt", encoding="utf-8") as fh:
                        fh.write(xml_text)
            except Exception as e:
                logger.write(f"  warning: could not save raw page: {e} (continuing)")

            # Parse & append (dedupe within this process)
            try:
                recs = parse_pubmed_xml(xml_text)
                pubs_rows, auth_rows, aff_rows, edge_rows = [], [], [], []
                for r in recs:
                    pmid = r["pmid"]
                    if not pmid or pmid in seen_pmids: 
                        continue
                    seen_pmids.add(pmid)
                    pubs_rows.append({"pmid": pmid, "title": r["title"], "journal": r["journal"],
                                      "year_or_date": r["year_or_date"], "doi": r["doi"], "theme": theme})
                    ids_for_edges = []
                    for a in r["authors"]:
                        aid = f"{pmid}:{a['pos']}"
                        auth_rows.append({"pmid": pmid, "author_pos": a["pos"], "last": a["last"],
                                          "fore": a["fore"], "initials": a["initials"], "theme": theme})
                        ids_for_edges.append(aid)
                        if a["affiliations"]:
                            for aff in a["affiliations"]:
                                aff_rows.append({"pmid": pmid, "author_pos": a["pos"],
                                                 "affiliation": aff, "is_uk": int(is_uk_aff(aff)), "theme": theme})
                        else:
                            aff_rows.append({"pmid": pmid, "author_pos": a["pos"], "affiliation": "", "is_uk": 0, "theme": theme})
                    if not args.skip_edges:
                        n = len(ids_for_edges)
                        for u in range(n):
                            for v in range(u+1, n):
                                edge_rows.append({"pmid": pmid, "a": ids_for_edges[u], "b": ids_for_edges[v], "theme": theme})

                safe_csv_append(pubs_path, pubs_fn, pubs_rows)
                safe_csv_append(auth_path, auth_fn, auth_rows)
                safe_csv_append(aff_path,  aff_fn,  aff_rows)
                if not args.skip_edges:
                    safe_csv_append(edges_path, edge_fn, edge_rows)

                done_pages.add(page)
                ckpt["done_pages"] = sorted(done_pages)
                ckpt["retmax"] = page_size
                ckpt["fetched"] = len(done_pages)
                ckpt["total"] = total
                ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.write(f"  [page {page}] wrote: {len(pubs_rows)} pubs, {len(auth_rows)} authors, {len(aff_rows)} affs"
                             + ("" if args.skip_edges else f", {len(edge_rows)} edges"))

                if not ensure_space(logger, outdir, chunks_dir, args.min_free_mb, args.raw_cap_mb, args.raw_policy):
                    logger.write("Low disk after page; pausing to resume later.")
                    return
            except Exception as e:
                logger.write(f"  parse/append error on page {page}: {e}")
            finally:
                time.sleep(args.rate_sleep)

        logger.write(f"=== THEME DONE: {theme} ===")

    logger.write("ALL THEMES COMPLETE. Outputs: publications.csv, authors.csv, author_affiliations.csv, coauthor_edges.csv")

if __name__ == "__main__":
    main()
