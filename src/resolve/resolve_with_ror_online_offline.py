#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve UK affiliations using ROR (online + offline) with a deterministic rules fallback.

Inputs
------
- authors_discovered.csv   Columns: pmid (str/int, optional but recommended), author_name, affiliation, orcid (optional)
- publications.csv         (optional) Same PMIDs as authors; enables uk_publications_clean.csv output
- ROR CSV snapshot         e.g., v1.70-2025-08-26-ror-data.csv  (download from ROR releases)

Outputs
-------
- uk_authors_combined.csv
  Adds: ror_id, ror_name, ror_country, ror_score, affil_method∈{ror_api, ror_offline, rules, none}, rules_score, is_uk
- affiliations_near_threshold_sample.csv  (borderline RULES rows for manual audit)
- uk_publications_clean.csv               (if publications.csv present & joinable by PMID)
- resolve_summary.json                    (reproducibility + counts)

Usage
-----
No command-line arguments needed. Edit the USER CONFIG block below.

Dependencies
------------
pip install pandas requests rapidfuzz ujson

Notes
-----
- Online step is cached in ror_api_cache.json to avoid repeat lookups.
- Weights and thresholds are explicit for dissertation defensibility.
- Timestamp is timezone-aware; snapshot + args logged in resolve_summary.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd

# Optional deps (script runs without them, but functionality degrades)
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

try:
    from rapidfuzz import process, fuzz  # type: ignore
except Exception:  # pragma: no cover
    process = None
    fuzz = None

# -----------------------------
# USER CONFIG — edit these values instead of using CLI flags
# -----------------------------
AUTHORS_PATH = "/Users/suhashafi/authors_discovered.csv"
PUBLICATIONS_PATH = "/Users/suhashafi/publications.csv"  # optional
ROR_CSV_PATH = "/Users/suhashafi/Documents/Uom/masters_project/code/v1.70-2025-08-26-ror-data/v1.70-2025-08-26-ror-data.csv"
USE_ONLINE = False  # set True to call the ROR API
MIN_ROR_SCORE = 0.90  # confidence for online API matches
MIN_FUZZY = 88  # 0..100; offline fuzzy threshold
RULES_THRESHOLD = 5  # rules cutoff when ROR fails
SLEEP_SECONDS = 0.25  # politeness delay for API
CACHE_FILE = "ror_api_cache.json"  # cache for online API
AUDIT_SAMPLE = 200  # rows to export for manual audit

# Example absolute paths (uncomment & edit to use)
# AUTHORS_PATH      = "/Users/you/authors_discovered.csv"
# PUBLICATIONS_PATH = "/Users/you/publications.csv"
# ROR_CSV_PATH      = "/Users/you/v1.70-2025-08-26-ror-data.csv"

# -----------------------------
# Configurable, versioned gazetteers for rules fallback
# -----------------------------
GAZ_VERSION = "v1.0-2025-08-27"

UK_COUNTRY = [
    "united kingdom",
    "u.k.",
    "uk",
    "great britain",
    "gb",
    "england",
    "scotland",
    "wales",
    "northern ireland",
]

UK_CITIES = [
    "london",
    "oxford",
    "cambridge",
    "manchester",
    "edinburgh",
    "glasgow",
    "birmingham",
    "bristol",
    "leeds",
    "liverpool",
    "newcastle",
    "nottingham",
    "sheffield",
    "southampton",
    "leicester",
    "coventry",
    "cardiff",
    "swansea",
    "belfast",
    "dundee",
    "aberdeen",
    "york",
    "bath",
    "exeter",
    "reading",
    "warwick",
    "lancaster",
    "durham",
    "brighton",
    "norwich",
    "plymouth",
    "portsmouth",
    "keele",
    "loughborough",
    "st andrews",
    "st. andrews",
    "st-andrews",
    "strathclyde",
    "ulster",
]

UK_ORGS = [
    "university of oxford",
    "oxford university",
    "university of cambridge",
    "cambridge university",
    "imperial college london",
    "kings college london",
    "king's college london",
    "university college london",
    "ucl",
    "university of manchester",
    "university of edinburgh",
    "roslin institute",
    "earlham institute",
    "quadram institute",
    "quadram institute bioscience",
    "rothamsted research",
    "uk centre for ecology and hydrology",
    "centre for ecology & hydrology",
    "ukceh",
    "university of birmingham",
    "university of bradford",
    "university of dundee",
    "university of east anglia",
    "uea",
    "university of exeter",
    "university of leicester",
    "university of liverpool",
    "university of nottingham",
    "newcastle university",
    "open university",
    "hdruk",
    "hdr uk",
    "health data research uk",
    "pirbright institute",
    "the pirbright institute",
    "queen mary university of london",
    "qmul",
    "swansea university",
    "cardiff university",
    "heriot watt university",
    "heriot-watt university",
]

NEGATIVE_PATTERNS = [
    r"cambridge[, ]\s*(ma|massachusetts|usa)",
    r"oxford[, ]\s*(ms|mississippi|usa)",
    r"birmingham[, ]\s*(al|alabama|usa)",
    r"london[, ]\s*(on|ontario|canada)",
    r"newcastle[, ]\s*(nsw|new south wales|australia)",
    r"\bauckland\b",
    r"\bsydney\b",
    r"\bmelbourne\b",
    r"\bcanada\b",
    r"\bunited states\b|\busa\b|\bu\.s\.a\.\b",
]

COUNTRY_PAT = re.compile(r"\b(" + "|".join(map(re.escape, UK_COUNTRY)) + r")\b")
CITIES_PAT = re.compile(r"\b(" + "|".join(map(re.escape, UK_CITIES)) + r")\b")
ORGS_PAT = re.compile("|".join(map(re.escape, UK_ORGS)))
NEG_PAT = re.compile("|".join(NEGATIVE_PATTERNS))

# -----------------------------
# Helpers
# -----------------------------


def norm(s: Any) -> str:
    """ASCII-folding + lower + whitespace squeeze for robust matching."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower()).strip()


def rules_score(aff: str) -> int:
    s = norm(aff)
    score = 0
    if not s:
        return score
    if re.search(NEG_PAT, s):
        score -= 6
    if re.search(COUNTRY_PAT, s):
        score += 8
    if re.search(ORGS_PAT, s):
        score += 5
    if re.search(CITIES_PAT, s):
        score += 3
    if re.search(r"\buk\b|\bgb\b", s):
        score += 2
    return score


# -----------------------------
# ROR online matching
# -----------------------------
ROR_MATCH_URL = "https://api.ror.org/organizations"


def ror_api_match(
    affiliation: str, min_score: float, only_gb: bool, session=None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    if not requests:
        return None, None, None, None
    if not affiliation or not affiliation.strip():
        return None, None, None, None
    sess = session or requests.Session()
    try:
        r = sess.get(ROR_MATCH_URL, params={"affiliation": affiliation}, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = data.get("items") or data.get("data") or []
        best = None
        for it in items:
            org = it.get("organization") or it
            score = it.get("score") or it.get("matching_score") or 0.0
            cc = (org.get("country") or {}).get("country_code")
            rid = org.get("id")
            name = org.get("name")
            if rid and cc is not None and score is not None:
                if score >= min_score and (not only_gb or cc == "GB"):
                    if best is None or score > best[3]:
                        best = (rid, name, cc, float(score))
        return best if best else (None, None, None, None)
    except Exception:
        return None, None, None, None


# -----------------------------
# ROR offline CSV snapshot matching
# -----------------------------


def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if isinstance(c, str) and ("country" in c and "code" in c):
            return c
    return None


def parse_list_cell(cell: Any) -> List[str]:
    if cell is None or str(cell).strip() == "":
        return []
    s = str(cell).strip()
    # Try JSON
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            out: List[str] = []
            for x in arr:
                if isinstance(x, dict) and "value" in x:
                    out.append(x["value"])
                elif isinstance(x, dict) and "label" in x:
                    out.append(x["label"])
                elif isinstance(x, str):
                    out.append(x)
            return [t for t in out if t]
    except Exception:
        pass
    # Fallback split on | or ;
    parts = re.split(r"[|;]", s)
    return [p.strip() for p in parts if p.strip()]


def load_ror_gb_names(ror_csv_path: str) -> Optional[pd.DataFrame]:
    if not ror_csv_path or not os.path.exists(ror_csv_path):
        print(f"[WARN] ROR CSV not found: {ror_csv_path}")
        return None
    df = pd.read_csv(ror_csv_path, low_memory=False)
    id_col = _pick(df, ["id", "ror_id"])
    name_col = _pick(df, ["name"])
    cc_col = _pick(df, ["country.country_code", "country_country_code", "country_code"])
    aliases_c = _pick(df, ["aliases"])
    labels_c = _pick(df, ["labels"])
    acr_c = _pick(df, ["acronyms", "acronym"])

    if not (id_col and name_col and cc_col):
        print(
            "[WARN] ROR CSV headers not recognised. First columns:",
            df.columns.tolist()[:20],
        )
        return None

    keep = df[df[cc_col] == "GB"].copy()
    rows: List[Dict[str, Any]] = []
    for _, r in keep.iterrows():
        rid = r[id_col]
        base = [r[name_col]] if pd.notna(r[name_col]) else []
        aliases = parse_list_cell(r[aliases_c]) if aliases_c else []
        labels = parse_list_cell(r[labels_c]) if labels_c else []
        acrs = parse_list_cell(r[acr_c]) if acr_c else []
        names = list({n for n in (base + aliases + labels + acrs) if n})
        for nm in names:
            rows.append({"ror_id": rid, "country": "GB", "name": nm})
    out = pd.DataFrame(rows).drop_duplicates()
    print(f"[INFO] Loaded {len(out):,} GB organisation names from ROR snapshot.")
    return out


def ror_offline_match(
    affiliation: str, gb_df: Optional[pd.DataFrame], min_score: int
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    if gb_df is None or process is None or fuzz is None:
        return None, None, None, None
    if not affiliation or not affiliation.strip():
        return None, None, None, None
    cand = process.extractOne(affiliation, gb_df["name"], scorer=fuzz.WRatio)
    if not cand:
        return None, None, None, None
    best_name, score, idx = cand
    if score < min_score:
        return None, None, None, None
    row = gb_df.iloc[idx]
    return row["ror_id"], row["name"], row["country"], float(score)


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    use_online = bool(USE_ONLINE)
    authors_path = AUTHORS_PATH
    publications = PUBLICATIONS_PATH
    ror_csv_path = ROR_CSV_PATH
    min_ror_score = float(MIN_ROR_SCORE)
    min_fuzzy = int(MIN_FUZZY)
    rules_threshold = int(RULES_THRESHOLD)
    sleep_seconds = float(SLEEP_SECONDS)
    cache_file = CACHE_FILE
    audit_sample = int(AUDIT_SAMPLE)

    # Load authors
    if not os.path.exists(authors_path):
        print(f"[ERROR] Missing authors file: {authors_path}", file=sys.stderr)
        return 2
    auth = pd.read_csv(authors_path)
    if "affiliation" not in auth.columns:
        print("[ERROR] authors file must include 'affiliation' column", file=sys.stderr)
        return 2

    # Preload ROR offline names (GB only)
    gb_df = load_ror_gb_names(ror_csv_path)
    if gb_df is None or gb_df.empty:
        print(
            "[WARN] No GB names loaded from ROR snapshot; offline matching will be skipped."
        )

    # Prepare ROR API cache
    cache: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    sess = requests.Session() if (requests and use_online) else None

    # Resolve affiliations
    out_rows: List[Dict[str, Any]] = []
    for _, r in auth.iterrows():
        raw_aff = r.get("affiliation", "")
        aff = "" if pd.isna(raw_aff) else str(raw_aff)
        key = aff.strip()

        rid = rname = rcc = None
        rscore: Optional[float] = None
        method = "none"

        # 1) ROR online (high confidence, GB only)
        if sess is not None and key:
            if key in cache:
                tmp = cache[key]
                rid, rname, rcc, rscore = (
                    tmp.get("rid"),
                    tmp.get("rname"),
                    tmp.get("rcc"),
                    tmp.get("rscore"),
                )
            else:
                rid, rname, rcc, rscore = ror_api_match(
                    aff, min_score=min_ror_score, only_gb=True, session=sess
                )
                cache[key] = {"rid": rid, "rname": rname, "rcc": rcc, "rscore": rscore}
                time.sleep(sleep_seconds)
            if rid and rcc == "GB" and (rscore is None or rscore >= min_ror_score):
                method = "ror_api"

        # 2) ROR offline snapshot + fuzzy (GB only)
        if (
            method == "none"
            and gb_df is not None
            and not gb_df.empty
            and process is not None
        ):
            rid2, rname2, rcc2, rscore2 = ror_offline_match(
                aff, gb_df, min_score=min_fuzzy
            )
            if rid2 and rcc2 == "GB":
                rid, rname, rcc, rscore = rid2, rname2, rcc2, rscore2
                method = "ror_offline"

        # 3) Rules fallback (deterministic, transparent)
        rs = rules_score(aff)
        is_uk = False
        if method in ("ror_api", "ror_offline"):
            is_uk = True
        else:
            if rs >= rules_threshold:
                method = "rules"
                is_uk = True

        out = r.to_dict()
        out.update(
            {
                "ror_id": rid,
                "ror_name": rname,
                "ror_country": rcc,
                "ror_score": rscore,
                "affil_method": method,
                "rules_score": rs,
                "is_uk": bool(is_uk),
            }
        )
        out_rows.append(out)

    # Save cache
    if use_online and cache:
        try:
            with open(cache_file, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    # DataFrames + outputs
    res = pd.DataFrame(out_rows)
    res.to_csv("uk_authors_combined.csv", index=False)

    # Borderline RULES rows for audit (for dissertation precision/recall)
    ambig = res[
        (res["affil_method"].eq("rules"))
        & (res["rules_score"].between(rules_threshold - 2, rules_threshold + 1))
    ]
    ambig[["author_name", "affiliation", "rules_score"]].head(audit_sample).to_csv(
        "affiliations_near_threshold_sample.csv", index=False
    )

    # Publications (optional): keep PMIDs that have >=1 UK author
    pubs_out = "not_created"
    if os.path.exists(publications):
        pubs = pd.read_csv(publications)
        if "pmid" in pubs.columns and "pmid" in res.columns:
            pubs["pmid"] = pubs["pmid"].astype(str)
            res["pmid"] = res["pmid"].astype(str)
            pmids_uk = set(res.loc[res["is_uk"], "pmid"])
            pubs[pubs["pmid"].isin(pmids_uk)].to_csv(
                "uk_publications_clean.csv", index=False
            )
            pubs_out = "uk_publications_clean.csv"

    # Summary JSON (for Methods reproducibility)
    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gazetteer_version": GAZ_VERSION,
        "args": {
            "min_ror_score": min_ror_score,
            "min_fuzzy": min_fuzzy,
            "rules_threshold": rules_threshold,
            "use_online": use_online,
        },
        "inputs": {
            "authors": os.path.abspath(authors_path),
            "publications": (
                os.path.abspath(publications) if os.path.exists(publications) else None
            ),
            "ror_csv": os.path.abspath(ror_csv_path),
        },
        "counts": {
            "total_author_rows": int(len(res)),
            "uk_rows": int(res["is_uk"].sum()),
            "by_method_uk": res.loc[res["is_uk"], "affil_method"]
            .value_counts()
            .to_dict(),
        },
        "outputs": {
            "uk_authors_combined.csv": os.path.abspath("uk_authors_combined.csv"),
            "affiliations_near_threshold_sample.csv": os.path.abspath(
                "affiliations_near_threshold_sample.csv"
            ),
            "uk_publications_clean.csv": (
                os.path.abspath(pubs_out)
                if pubs_out != "not_created"
                else "not_created"
            ),
        },
    }
    with open("resolve_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Also print summary to stdout for convenience
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
