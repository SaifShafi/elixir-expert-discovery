import os, time, json, csv, argparse, sys, re
from pathlib import Path
from typing import List, Dict, Any
from xml.etree import ElementTree as ET

import requests
import pandas as pd

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = os.getenv("NCBI_EMAIL", "")
API_KEY = os.getenv("NCBI_API_KEY", "")
RATE_SLEEP = 0.05  # ~8 req/sec if API_KEY present; otherwise slower automatically

def esearch(term, retstart=0, retmax=100000):
    params = {"db":"pubmed","term":term,"retmode":"json","retstart":retstart,"retmax":retmax}
    if EMAIL: params["email"]=EMAIL
    if API_KEY: params["api_key"]=API_KEY
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def efetch_xml(pmids: List[str]) -> str:
    params = {"db":"pubmed","id":",".join(pmids),"retmode":"xml"}
    if EMAIL: params["email"]=EMAIL
    if API_KEY: params["api_key"]=API_KEY
    r = requests.post(f"{BASE}/efetch.fcgi", data=params, timeout=120)
    r.raise_for_status()
    return r.text

def parse_pubmed_xml(xml_text: str) -> List[Dict[str,Any]]:
    root = ET.fromstring(xml_text)
    ns = {}  # PubMed XML has no strict namespaces for core fields
    recs = []
    for art in root.findall(".//PubmedArticle", ns):
        pmid = (art.findtext(".//PMID") or "").strip()
        art_title = (art.findtext(".//ArticleTitle") or "").strip()
        journal = (art.findtext(".//Journal/Title") or "").strip()
        year = art.findtext(".//JournalIssue/PubDate/Year")
        medline_date = art.findtext(".//JournalIssue/PubDate/MedlineDate")
        date = (year or medline_date or "").strip()

        doi = None
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = aid.text
                break

        authors = []
        for i, au in enumerate(art.findall(".//AuthorList/Author")):
            last = au.findtext("LastName") or ""
            fore = au.findtext("ForeName") or ""
            init = au.findtext("Initials") or ""
            affs = [a.findtext("Affiliation") for a in au.findall("AffiliationInfo")]
            affs = [a for a in affs if a]
            authors.append({"pmid": pmid, "pos": i+1, "last": last, "fore": fore, "initials": init, "affiliations": affs})

        recs.append({
            "pmid": pmid, "title": art_title, "journal": journal, "year_or_date": date, "doi": doi, "authors": authors
        })
    return recs

UK_PAT = re.compile(r"\b(United Kingdom|England|Scotland|Wales|Northern Ireland|Great Britain|GBR)\b", re.I)
KENTUCKY_PAT = re.compile(r"\b(Kentucky|Lexington)\b", re.I)
UK_DOMAIN_PAT = re.compile(r"\b(\.ac\.uk|\.nhs\.uk|\.gov\.uk|\.org\.uk|\.co\.uk)\b", re.I)

def is_uk_aff(aff: str) -> bool:
    if not aff: return False
    if KENTUCKY_PAT.search(aff):  # safety against Univ. of Kentucky
        return False
    return bool(UK_PAT.search(aff) or UK_DOMAIN_PAT.search(aff))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="uk_pubmed_queries.csv")
    ap.add_argument("--outdir", default="pubmed_uk")
    ap.add_argument("--email", default="")
    ap.add_argument("--api_key", default="")
    args = ap.parse_args()

    global EMAIL, API_KEY
    EMAIL = args.email or os.environ.get("NCBI_EMAIL")
    API_KEY = args.api_key or os.environ.get("NCBI_API_KEY")
    if not EMAIL:
        print("Set --email or NCBI_EMAIL env var.", file=sys.stderr); sys.exit(2)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pubs_path = outdir/"publications.csv"
    auth_path = outdir/"authors.csv"
    aff_path  = outdir/"author_affiliations.csv"
    edges_path= outdir/"coauthor_edges.csv"

    pubs_rows, auth_rows, aff_rows, edge_rows = [], [], [], []

    Q = pd.read_csv(args.queries)

    for _, row in Q.iterrows():
        theme = row["theme"]
        term  = row["pubmed_query"]
        print(f"Searching: {theme}")
        # paginate
        retstart = 0; batch = 10000
        pmids = []
        while True:
            j = esearch(term, retstart=retstart, retmax=batch)
            ids = j.get("esearchresult", {}).get("IdList", [])
            pmids.extend(ids)
            count = int(j.get("esearchresult", {}).get("count", "0"))
            retstart += len(ids)
            print(f"  got {len(ids)} / total {count}")
            time.sleep(RATE_SLEEP)
            if retstart >= count or not ids: break

        # fetch in chunks (200 ids per efetch is healthy)
        for i in range(0, len(pmids), 200):
            chunk = pmids[i:i+200]
            xml = efetch_xml(chunk)
            recs = parse_pubmed_xml(xml)
            for r in recs:
                pubs_rows.append({"pmid": r["pmid"], "title": r["title"], "journal": r["journal"],
                                  "year_or_date": r["year_or_date"], "doi": r["doi"], "theme": theme})
                # author rows & affiliations
                ids_for_edges = []
                for a in r["authors"]:
                    auth_id = f'{r["pmid"]}:{a["pos"]}'
                    auth_rows.append({"pmid": r["pmid"], "author_pos": a["pos"], "last": a["last"],
                                      "fore": a["fore"], "initials": a["initials"], "theme": theme})
                    ids_for_edges.append(auth_id)
                    for aff in a["affiliations"] or [""]:
                        aff_rows.append({"pmid": r["pmid"], "author_pos": a["pos"],
                                         "affiliation": aff, "is_uk": int(is_uk_aff(aff)), "theme": theme})
                # coauthor edges (undirected simple)
                for x in range(len(ids_for_edges)):
                    for y in range(x+1, len(ids_for_edges)):
                        edge_rows.append({"pmid": r["pmid"], "a": ids_for_edges[x], "b": ids_for_edges[y], "theme": theme})
            time.sleep(RATE_SLEEP)

    # write outputs
    pd.DataFrame(pubs_rows).drop_duplicates(subset=["pmid","theme"]).to_csv(pubs_path, index=False)
    pd.DataFrame(auth_rows).to_csv(auth_path, index=False)
    pd.DataFrame(aff_rows).to_csv(aff_path, index=False)
    pd.DataFrame(edge_rows).drop_duplicates().to_csv(edges_path, index=False)

    print(f"Done. Files in {outdir}/:")
    print(" - publications.csv")
    print(" - authors.csv")
    print(" - author_affiliations.csv")
    print(" - coauthor_edges.csv")

if __name__ == "__main__":
    main()
