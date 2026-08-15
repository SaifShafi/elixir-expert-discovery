#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UK + ELIXIR-theme PubMed harvester (parallel + throttled + resumable)

Speed/features
- Parallel EFetch with QPS limiter (requests + ThreadPoolExecutor)
- Threaded XML parse + per-batch UK resolution (ROR online serialized + throttled)
- Offline-first ROR; online only when ambiguous; enforce min_ror_score
- Primary org name only (avoid alias explosion)
- Regex theme matcher (precompiled; plural/hyphen tolerant)
- Stream write + (optional) checkpoints with pruning
- Logging fixed (no "%,d" formatting crash)
- UK decision tightened (no false UK on non-UK affiliations)

ENV: NCBI_TOOL, NCBI_EMAIL, NCBI_API_KEY
"""

import os, re, json, time, argparse, logging, csv, threading, math, glob
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from rapidfuzz import process, fuzz
from tqdm import tqdm
import xml.etree.ElementTree as ET

# -------------------- Defaults --------------------
DEFAULT_MIN_ROR_SCORE   = 0.90
DEFAULT_MIN_FUZZY       = 88
DEFAULT_RULES_THRESHOLD = 7
DEFAULT_SLEEP_SECONDS   = 0.05
DEFAULT_BATCH_SIZE      = 120
DEFAULT_MAX_PER_ORG     = 0
DEFAULT_MAX_TOTAL       = 0
DEFAULT_AUDIT_SAMPLE    = 200
DEFAULT_MIN_YEAR        = 2000
DEFAULT_CACHE_FILE      = "ror_api_cache.json"

# Parallelism + throttling
DEFAULT_WORKERS   = 8           # threads for efetch/parse/resolve
DEFAULT_QPS       = 8           # max E-utilities requests/sec WITH API key (3 if no key)
DEFAULT_ROR_QPS   = 4           # max ROR requests/sec

# Checkpoints
DEFAULT_CK_EVERY  = 0           # 0 = off
DEFAULT_CK_KEEP   = 3

NCBI_TOOL    = os.getenv("NCBI_TOOL",  "uk-kg-builder")
NCBI_EMAIL   = os.getenv("NCBI_EMAIL", "me@example.com")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# -------------------- Logging ---------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# -------------------- QPS Limiter -----------------
class RateLimiter:
    """Simple token-bucket-ish limiter for threads."""
    def __init__(self, qps: float):
        self.qps = max(0.1, qps)
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            min_interval = 1.0 / self.qps
            delta = now - self.last
            if delta < min_interval:
                time.sleep(min_interval - delta)
            self.last = time.time()

# -------------------- ROR snapshot ----------------
def read_ror_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)

    col_cc_candidates = [
        "country_code","country","country_code/primary","country/primary",
        "country.country_code","country.country-name","country.country_name"
    ]
    cc_col = next((c for c in col_cc_candidates if c in df.columns), None)

    if cc_col:
        df_uk = df[df[cc_col].str.upper().str.contains(r"\bGB\b", na=False)].copy()
    else:
        name_cols = [c for c in df.columns if "country" in c.lower() and "name" in c.lower()]
        if name_cols:
            mask = False
            for c in name_cols:
                mask = mask | df[c].str.contains(r"United Kingdom|UK|Great Britain", case=False, na=False)
            df_uk = df[mask].copy()
        else:
            df_uk = df.copy()  # fallback

    def collect_names(row):
        names = [row.get("name","")]
        for k in ["aliases","labels","acronyms"]:
            if k in row and row[k]:
                try:
                    val = row[k]
                    if isinstance(val, str) and val.strip().startswith("["):
                        items = [x.strip(" '\"") for x in json.loads(val) if x]
                    else:
                        items = [x.strip() for x in str(val).split(";") if x.strip()]
                    names.extend(items)
                except Exception:
                    pass
        out, seen = [], set()
        for n in names:
            if isinstance(n, str):
                nn = n.strip()
                if nn and nn.lower() not in seen:
                    out.append(nn); seen.add(nn.lower())
        return out

    df_uk["all_names"] = df_uk.apply(collect_names, axis=1)
    for c in ["id","name","all_names"]:
        if c not in df_uk.columns: df_uk[c] = ""
    return df_uk[["id","name","all_names"]].reset_index(drop=True)

def normalize_aff(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

UK_RULE_PATTERNS = [
    r"\bUK\b","United Kingdom","England","Scotland","Wales","Northern Ireland",
    r"London","Oxford","Cambridge","Edinburgh","Glasgow","Cardiff","Belfast",
    r"Manchester","Birmingham","Leeds","Sheffield","Newcastle","Nottingham",
    r"Imperial College", r"\bUCL\b","King's College London","University of Edinburgh",
    "University of Oxford","University of Cambridge"
]
NON_UK_COUNTRIES = [
    "USA","United States","Germany","France","Spain","Italy","Netherlands","Belgium","Switzerland",
    "China","Japan","Canada","Australia","Sweden","Denmark","Norway","Finland","Austria","Ireland",
    "India","Brazil","Russia","Turkey","Israel","Saudi Arabia","UAE","Qatar","Singapore","Hong Kong"
]

def rules_score_aff(aff: str) -> Tuple[int,bool]:
    txt = aff or ""
    score = 0
    for pat in UK_RULE_PATTERNS:
        if re.search(pat, txt, flags=re.I):
            score += 1
    multi = any(re.search(rf"\b{c}\b", txt, flags=re.I) for c in NON_UK_COUNTRIES)
    return score, multi

class ROROfflineMatcher:
    def __init__(self, ror_df: pd.DataFrame):
        self.names = []
        self.ids = []
        for _,row in ror_df.iterrows():
            rid = row.get("id","")
            for n in row.get("all_names",[]) or []:
                if n:
                    self.names.append(n)
                    self.ids.append(rid)

    def best_match(self, aff: str) -> Tuple[Optional[str], float]:
        if not aff: return None, 0.0
        res = process.extractOne(aff, self.names, scorer=fuzz.WRatio, score_cutoff=DEFAULT_MIN_FUZZY-10)
        if not res: return None, 0.0
        cand, score, _ = res
        try:
            idx = self.names.index(cand)
            return self.ids[idx], score/100.0
        except ValueError:
            return None, score/100.0

class RORClient:
    def __init__(self, cache_path: str, limiter: RateLimiter):
        self.cache_path = cache_path
        self.cache_lock = threading.Lock()
        self.req_lock   = threading.Lock()
        self.limiter    = limiter
        if os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path,"r"))
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

    def save(self):
        with self.cache_lock:
            tmp = self.cache_path + ".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(self.cache, f)
            os.replace(tmp, self.cache_path)

    def query(self, text: str) -> Dict[str,Any]:
        key = normalize_aff(text).lower()
        with self.cache_lock:
            if key in self.cache:
                return self.cache[key]
        try:
            self.limiter.wait()
            with self.req_lock:
                r = requests.get("https://api.ror.org/organizations",
                                 params={"affiliation": text, "query": text},
                                 timeout=30)
            r.raise_for_status()
            js = r.json()
        except Exception as e:
            js = {"error": str(e)}
        with self.cache_lock:
            self.cache[key] = js
        self.save()
        return js

# -------------------- Theme matching -----------------
def _term_to_regex(term: str) -> str:
    t = term.strip()
    tokens = re.split(r"[\s\-]+", t)
    esc = [re.escape(tok) for tok in tokens if tok]
    if not esc: return r"$^"
    between = r"[\s\-]+"
    last = esc[-1]
    if re.search(r"[A-Za-z]$", tokens[-1]): last = last + r"(?:es|s)?"
    core = between.join(esc[:-1] + [last]) if len(esc) > 1 else last
    return rf"\b{core}\b"

def load_themes(path: Optional[str]):
    if not path: return {}, []
    obj = json.load(open(path,"r",encoding="utf-8"))
    compiled = {}
    all_includes = []
    for name,conf in obj.items():
        inc = [x for x in conf.get("include",[]) if isinstance(x,str) and x.strip()]
        exc = [x for x in conf.get("exclude",[]) if isinstance(x,str) and x.strip()]
        compiled[name] = {
            "include":[re.compile(_term_to_regex(x), re.I) for x in inc],
            "exclude":[re.compile(_term_to_regex(x), re.I) for x in exc]
        }
        all_includes.extend(inc)
    # Build query OR list (unique)
    uq, seen = [], set()
    for t in all_includes:
        tt = t.strip().lower()
        if tt and tt not in seen:
            uq.append(t.strip()); seen.add(tt)
    return compiled, uq

def build_theme_query(include_terms: List[str]) -> str:
    parts = [f"\"{t}\"[tiab]" if re.search(r"\s", t) else f"{t}[tiab]" for t in include_terms]
    return " OR ".join(parts) if parts else ""

def themes_match_text(text: str, compiled_themes) -> List[str]:
    t = text or ""
    hits=[]
    for name,pat in compiled_themes.items():
        inc = pat["include"]; exc = pat["exclude"]
        inc_ok = any(p.search(t) for p in inc) if inc else False
        exc_hit= any(p.search(t) for p in exc) if exc else False
        if inc_ok and not exc_hit:
            hits.append(name)
    return sorted(set(hits))

# -------------------- PubMed helpers -----------------
def esearch_affiliation(org: str, theme_query: Optional[str], retmax: int, retstart: int,
                        limiter: RateLimiter, headers: dict) -> Dict[str,Any]:
    term = f"\"{org}\"[Affiliation]"
    if theme_query:
        term = f"{term} AND ({theme_query})"
    payload = {
        "tool": NCBI_TOOL, "email": NCBI_EMAIL,
        "db": "pubmed", "retmode": "json",
        "term": term, "retmax": retmax, "retstart": retstart
    }
    if NCBI_API_KEY:
        payload["api_key"] = NCBI_API_KEY
    limiter.wait()
    r = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                      data=payload, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def efetch_chunks(pmids: List[str], limiter: RateLimiter, headers: dict) -> str:
    ids = [str(x).strip() for x in pmids if str(x).strip().isdigit()]
    if not ids: return ""

    def _post(sub: List[str]) -> str:
        payload = {"tool":NCBI_TOOL, "email":NCBI_EMAIL, "db":"pubmed",
                   "retmode":"xml", "id":",".join(sub)}
        if NCBI_API_KEY:
            payload["api_key"] = NCBI_API_KEY
        limiter.wait()
        r = requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                          data=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.text

    queue=[ids]; out=[]
    while queue:
        sub = queue.pop()
        try:
            out.append(_post(sub))
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 400 and len(sub) > 1:
                mid=len(sub)//2
                logging.warning(f"EFetch 400 on {len(sub)} PMIDs; splitting into {len(sub[:mid])}+{len(sub[mid:])}")
                queue.append(sub[:mid]); queue.append(sub[mid:])
            else:
                logging.error(f"EFetch failed for {len(sub)} PMIDs (status {status}); skipping.")
        except Exception as e:
            logging.error(f"EFetch error {e}; skipping {len(sub)} PMIDs.")
    return "\n".join(out)

def parse_pubmed_xml(xml_text: str) -> List[Dict[str,Any]]:
    if not xml_text: return []
    try:
        root = ET.fromstring(xml_text)
        arts_iter = root.findall(".//PubmedArticle")
    except ET.ParseError:
        xml_text_wrapped = f"<ROOT>{xml_text}</ROOT>"
        root = ET.fromstring(xml_text_wrapped)
        arts_iter = root.findall(".//PubmedArticle")

    out=[]
    for art in arts_iter:
        try:
            pmid = art.findtext(".//PMID") or ""
            title = art.findtext(".//ArticleTitle") or ""
            abstracts = [a.text for a in art.findall(".//AbstractText") if a.text]
            abstract = " ".join(abstracts) if abstracts else ""
            journal = art.findtext(".//Journal/Title") or ""
            year = (art.findtext(".//ArticleDate/Year")
                    or art.findtext(".//PubDate/Year")
                    or (re.search(r"\b(19|20)\d{2}\b", art.findtext(".//PubDate/MedlineDate") or "") or [None])[0]
                    or "")
            doi = None
            for el in art.findall(".//ELocationID"):
                if (el.attrib or {}).get("EIdType","").lower()=="doi" and el.text:
                    doi = el.text.strip(); break

            authors=[]
            for a in art.findall(".//Author"):
                name = " ".join([(a.findtext("ForeName") or ""), (a.findtext("LastName") or "")]).strip()
                if not name: name = a.findtext("CollectiveName") or ""
                orcid = None
                for ident in a.findall(".//Identifier"):
                    if (ident.attrib or {}).get("Source","").lower()=="orcid" and ident.text:
                        orcid = ident.text.strip()
                affs = [normalize_aff(af.findtext("Affiliation") or "") for af in a.findall(".//AffiliationInfo")]
                authors.append({"name":name, "orcid":orcid, "affiliations":affs})

            out.append({"pmid":pmid,"title":title,"abstract":abstract,"journal":journal,"year":year,"doi":doi,"authors":authors})
        except Exception as e:
            logging.warning(f"XML parse error for one article: {e}")
    return out

# -------------------- Resolver (offline-first, tight UK) ---------------
def resolve_affiliation(aff: str, offline: ROROfflineMatcher, ror_client: Optional[RORClient],
                        min_ror_score: float, min_fuzzy: int, rules_threshold: int) -> Tuple[bool, Optional[str], Dict[str,Any]]:
    if not aff:
        return False, None, {"method":"none","note":"empty_aff"}

    rules, multi = rules_score_aff(aff)
    rid_off, fuzzy_score = offline.best_match(aff)
    fuzzy_pts = int(round((fuzzy_score or 0)*100))

    positive_offline = (rid_off is not None) and (fuzzy_pts >= min_fuzzy)
    positive_rules   = (rules >= rules_threshold)

    rid_api = None; uk_api=False; api_score=0.0
    if ror_client and not (positive_offline or positive_rules):
        js = ror_client.query(aff)
        try:
            items = js.get("items") or js.get("data") or []
            if items:
                best = items[0]
                rid_api = best.get("id") or best.get("identifier") or best.get("ror")
                cc  = (best.get("country") or {}).get("country_code","")
                cn  = (best.get("country") or {}).get("country_name","")
                api_score = best.get("score") or best.get("confidence") or 0.0
                uk_api = (str(cc).upper()=="GB") or bool(re.search(r"\b(United Kingdom|UK|Great Britain)\b", str(cn), re.I))
                if api_score < min_ror_score:
                    rid_api = None; uk_api = False
        except Exception:
            pass

    is_uk = positive_rules or uk_api or positive_offline
    if not is_uk and multi:
        is_uk = False

    rid = rid_api or (rid_off if positive_offline else None)
    return is_uk, rid, {
        "method": ("online_mix" if ror_client else "offline_only"),
        "fuzzy": fuzzy_pts, "rules": rules,
        "rid_off": rid_off, "rid_api": rid_api,
        "uk_api": uk_api, "api_score": api_score
    }

# -------------------- Main -------------------------
def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="UK+ELIXIR-theme PubMed harvester (parallel+throttled)")
    ap.add_argument("--ror-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--themes-file", default=None)
    ap.add_argument("--require-theme-match", action="store_true")
    ap.add_argument("--query-with-themes", action="store_true")
    ap.add_argument("--use-online", action="store_true")

    ap.add_argument("--min-ror-score", type=float, default=DEFAULT_MIN_ROR_SCORE)
    ap.add_argument("--min-fuzzy", type=int, default=DEFAULT_MIN_FUZZY)
    ap.add_argument("--rules-threshold", type=int, default=DEFAULT_RULES_THRESHOLD)
    ap.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)

    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)    # PMIDs per batch
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--qps", type=float, default=DEFAULT_QPS)                # E-utilities QPS
    ap.add_argument("--ror-qps", type=float, default=DEFAULT_ROR_QPS)        # ROR QPS

    ap.add_argument("--max-per-org", type=int, default=DEFAULT_MAX_PER_ORG)
    ap.add_argument("--max-total", type=int, default=DEFAULT_MAX_TOTAL)
    ap.add_argument("--max-orgs", type=int, default=0, help="Limit number of orgs (for quick tests)")

    ap.add_argument("--cache-file", default=DEFAULT_CACHE_FILE)
    ap.add_argument("--resume-from-out", default=None)

    ap.add_argument("--checkpoint-every", type=int, default=DEFAULT_CK_EVERY)
    ap.add_argument("--checkpoint-keep",  type=int, default=DEFAULT_CK_KEEP)

    args = ap.parse_args()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    header_needed = not os.path.exists(out_path)

    # HTTP defaults
    headers = {"Accept-Encoding":"gzip"}

    # ROR + offline matcher
    logging.info("Loading ROR GB snapshot…")
    ror_df = read_ror_snapshot(args.ror_csv)
    offline = ROROfflineMatcher(ror_df)
    ror_limiter = RateLimiter(args.ror_qps)
    ror_client = RORClient(args.cache_file, limiter=ror_limiter) if args.use_online else None

    # Themes
    compiled_themes, include_terms = load_themes(args.themes_file)
    if compiled_themes:
        logging.info("Loaded themes: %s", ", ".join(sorted(compiled_themes.keys())))
    theme_query = build_theme_query(include_terms) if (args.query_with_themes and include_terms) else None
    if args.query_with_themes and theme_query:
        logging.info("Using theme narrowing at query-time.")

    # Resume
    already=set()
    if args.resume_from_out and os.path.exists(args.resume_from_out):
        try:
            df_prev=pd.read_csv(args.resume_from_out, dtype=str, low_memory=False)
            pmid_col=next((c for c in df_prev.columns if c.lower()=="pmid"), None)
            if pmid_col:
                already={str(x).strip() for x in df_prev[pmid_col].dropna().astype(str) if str(x).strip().isdigit()}
                logging.info("Resume: skipping %d PMIDs from %s", len(already), args.resume_from_out)
            if os.path.abspath(args.resume_from_out)==os.path.abspath(out_path):
                header_needed=False
        except Exception as e:
            logging.warning("Could not resume from %s: %s", args.resume_from_out, e)

    # Output writer
    write_lock = threading.Lock()
    batch_idx = {"i": -1}

    def write_rows(rows: List[Dict[str,Any]]):
        if not rows: return
        df=pd.DataFrame(rows)
        with write_lock:
            nonlocal header_needed
            with open(out_path,"a",encoding="utf-8",newline="") as f:
                df.to_csv(f, index=False, header=header_needed)
                f.flush(); os.fsync(f.fileno())
            header_needed=False

            # checkpoints (optional)
            if args.checkpoint_every and ((batch_idx["i"]+1) % args.checkpoint_every == 0):
                chk_dir = os.path.join(os.path.dirname(out_path) or ".", "checkpoints")
                os.makedirs(chk_dir, exist_ok=True)
                ck = os.path.join(chk_dir, f"batch_{batch_idx['i']+1:05d}.csv")
                df.to_csv(ck, index=False)
                if args.checkpoint_keep>0:
                    files = sorted(glob.glob(os.path.join(chk_dir,"batch_*.csv")))
                    if len(files) > args.checkpoint_keep:
                        for old in files[:-args.checkpoint_keep]:
                            try: os.remove(old)
                            except: pass

    # Org list (primary names only)
    org_names = sorted(ror_df["name"].dropna().unique().tolist())
    logging.info("UK ROR orgs to query: %d", len(org_names))
    if args.max_orgs and args.max_orgs < len(org_names):
        org_names = org_names[:args.max_orgs]
        logging.info("Limiting to first %d orgs for test", len(org_names))

    # Gather PMIDs
    eutils_limiter = RateLimiter(args.qps if NCBI_API_KEY else min(args.qps, 3))
    pmids: Set[str] = set()
    per_org_limit = args.max_per_org
    max_total     = args.max_total
    ret_step      = 200  # bigger step = fewer esearch calls

    for org in tqdm(org_names, desc="Querying PubMed by org"):
        retstart = 0
        while True:
            js = esearch_affiliation(org, theme_query, retmax=ret_step, retstart=retstart,
                                     limiter=eutils_limiter, headers=headers)
            ids = js.get("esearchresult",{}).get("idlist",[]) or []
            if not ids: break
            for pid in ids:
                if max_total and len(pmids) >= max_total: break
                pmids.add(str(pid).strip())
            retstart += len(ids)
            if len(ids) < ret_step or (per_org_limit and retstart >= per_org_limit) or (max_total and len(pmids) >= max_total):
                break
        if max_total and len(pmids) >= max_total:
            break

    logging.info("Candidate PMIDs: %d", len(pmids))

    # Remove already-done
    if already:
        before=len(pmids)
        pmids={p for p in pmids if p not in already}
        logging.info("After resume-skip: %d remaining (skipped %d)", len(pmids), before-len(pmids))

    pmids_list=[p for p in pmids if p.isdigit()]
    total=len(pmids_list)
    if total==0:
        logging.info("No PMIDs to fetch. Done.")
        return

    # Parallel fetch/parse/resolve pipeline
    batches = [pmids_list[i:i+args.batch_size] for i in range(0, total, args.batch_size)]
    logging.info("Total batches: %d (size=%d)", len(batches), args.batch_size)

    def worker(chunk: List[str]) -> List[Dict[str,Any]]:
        xml = efetch_chunks(chunk, limiter=eutils_limiter, headers=headers)
        arts = parse_pubmed_xml(xml)

        # cache unique affs per worker
        uniq_affs = set()
        for art in arts:
            for au in art.get("authors",[]):
                for aff in au.get("affiliations",[]) or []:
                    if aff: uniq_affs.add(normalize_aff(aff))

        aff_resolved: Dict[str, Tuple[bool, Optional[str], Dict[str,Any]]] = {}
        for aff in uniq_affs:
            is_uk, rid, info = resolve_affiliation(
                aff=aff, offline=offline, ror_client=ror_client,
                min_ror_score=args.min_ror_score, min_fuzzy=args.min_fuzzy,
                rules_threshold=args.rules_threshold
            )
            aff_resolved[aff]=(is_uk, rid, info)

        rows=[]
        for art in arts:
            pmid=art["pmid"]
            y = art.get("year","")
            if args.min_year and str(y).isdigit() and int(y) < int(args.min_year):
                continue

            title = art.get("title",""); abstract = art.get("abstract","")
            fulltext = f"{title} {abstract}".strip()
            matched = themes_match_text(fulltext, compiled_themes) if compiled_themes else []

            if args.require_theme_match and not matched:
                continue

            any_uk=False; authors_out=[]; methods_used=set()
            for au in art.get("authors",[]):
                affs = au.get("affiliations",[]) or []
                uk_flags=[]; ror_ids=[]
                for aff in affs:
                    if not aff:
                        uk_flags.append(False); ror_ids.append(None); continue
                    is_uk, rid, info = aff_resolved.get(aff, (False,None,{"method":"none"}))
                    uk_flags.append(bool(is_uk)); ror_ids.append(rid)
                    methods_used.add(info.get("method","none"))
                    if is_uk: any_uk=True
                authors_out.append({
                    "name": au.get("name",""),
                    "orcid": au.get("orcid"),
                    "affiliations": affs,
                    "uk_flags": uk_flags,
                    "ror_ids": ror_ids
                })

            if not any_uk:
                continue

            rows.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": y,
                "journal": art.get("journal",""),
                "doi": art.get("doi"),
                "themes": ";".join(matched),
                "authors_json": json.dumps(authors_out, ensure_ascii=False),
                "any_uk_author": 1,
                "methods_json": json.dumps({
                    "query_with_themes": bool(args.query_with_themes and (compiled_themes is not None)),
                    "resolver": "online_mix" if args.use_online else "offline_only",
                    "notes": ";".join(sorted(met for met in methods_used if met!="none"))
                }, ensure_ascii=False)
            })
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(worker, chunk) for chunk in batches]
        for i, fut in enumerate(tqdm(as_completed(futs), total=len(futs), desc="Fetching+Parsing+Resolving")):
            try:
                rows = fut.result()
            except Exception as e:
                logging.error("Batch failed: %s", e)
                rows = []
            batch_idx["i"] += 1
            write_rows(rows)
            logging.info("WROTE batch %05d: +%d rows → %s", batch_idx["i"], len(rows), out_path)

    logging.info("Done. Output at %s", out_path)

if __name__ == "__main__":
    main()
