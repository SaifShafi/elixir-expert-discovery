#!/usr/bin/env python3
"""
build_uk_queries.py
-------------------
Reads an ELIXIR themes JSON and emits PubMed queries per theme with a UK affiliation filter.
- Adds [tiab] to normal phrases (Title/Abstract)
- Preserves explicit boolean terms by tagging each token with [All Fields]
- Appends UK affiliation filter based on [ad] field (avoids 'UK[ad]' to reduce Univ. of Kentucky false positives)

Example:
  python build_uk_queries.py --themes_json elixir_themes.json --out_csv uk_pubmed_queries.csv
"""
import argparse, json, re, sys, csv, os, textwrap
from datetime import datetime

UK_AD = '(United Kingdom[ad] OR England[ad] OR Scotland[ad] OR Wales[ad] OR "Northern Ireland"[ad] OR "Great Britain"[ad])'

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def tag_token(token: str, field="tiab"):
    token = token.strip()
    if not token:
        return ""
    # If token already contains boolean operators, tag words with [All Fields]
    if re.search(r"\b(AND|OR|NOT)\b", token, flags=re.I):
        parts = re.split(r"(\bAND\b|\bOR\b|\bNOT\b)", token, flags=re.I)
        out = []
        for p in parts:
            if re.fullmatch(r"\b(AND|OR|NOT)\b", p, flags=re.I):
                out.append(p.upper())
            else:
                t = p.strip().strip('"')
                if not t: 
                    continue
                out.append(f'("{t}"[All Fields])')
        return "(" + " ".join(out) + ")"
    # Plain phrase -> tag to field
    tok = token.strip('"')
    return f'("{tok}"[{field}])'

def dedup(seq):
    seen = set(); out = []
    for x in seq or []:
        if x not in seen and x is not None:
            out.append(x)
            seen.add(x)
    return out

def build_query(include_terms, exclude_terms=None, synonyms=None, field="tiab"):
    include_terms = dedup((include_terms or []) + (synonyms or []))
    exclude_terms = dedup(exclude_terms or [])

    include_exprs = [tag_token(t, field=field) for t in include_terms if t and t.strip()]
    include_block = "(" + " OR ".join(include_exprs) + ")" if include_exprs else ""

    exclude_exprs = [tag_token(t, field=field) for t in exclude_terms if t and t.strip()]
    exclude_block = (" NOT (" + " OR ".join(exclude_exprs) + ")") if exclude_exprs else ""

    base_query = f"{include_block}{exclude_block}".strip()
    if base_query:
        final = f"({base_query}) AND {UK_AD}"
    else:
        final = UK_AD
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--themes_json", default="elixir_themes.json", help="Path to elixir_themes.json")
    ap.add_argument("--out_csv", default="uk_pubmed_queries.csv", help="Where to write the theme/query table")
    ap.add_argument("--field", default="tiab", choices=["tiab","tw","all"], help="Which field to tag include/synonym terms with")
    args = ap.parse_args()

    log(f"Reading themes from: {args.themes_json}")
    with open(args.themes_json, "r") as f:
        themes = json.load(f)

    rows = []
    for theme, cfg in sorted(themes.items(), key=lambda kv: kv[0].lower()):
        q = build_query(cfg.get("include"), cfg.get("exclude"), cfg.get("synonyms"), field="tiab")
        rows.append({"theme": theme, "pubmed_query": q})

    out = args.out_csv
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["theme","pubmed_query"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    log(f"Wrote {len(rows)} queries -> {out}")
    for r in rows[:3]:
        log(f"Example [{r['theme']}]: {textwrap.shorten(r['pubmed_query'], width=160, placeholder='…')}")

if __name__ == "__main__":
    main()
