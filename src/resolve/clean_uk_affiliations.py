# clean_uk_affiliations.py
import os, re, unicodedata, pandas as pd

BASE = "."
AUTH_PATH = os.path.join(BASE, "authors_discovered.csv")
PUBS_PATH = os.path.join(BASE, "publications.csv")

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


def norm(s: str) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.lower()).strip()


def main():
    auth = pd.read_csv(AUTH_PATH)
    if "affiliation" not in auth.columns:
        raise KeyError("authors_discovered.csv must contain an 'affiliation' column")

    auth["aff_norm"] = auth["affiliation"].fillna("").map(norm)

    country_pat = re.compile(r"\b(" + "|".join(map(re.escape, UK_COUNTRY)) + r")\b")
    cities_pat = re.compile(r"\b(" + "|".join(map(re.escape, UK_CITIES)) + r")\b")
    orgs_pat = re.compile("|".join(map(re.escape, UK_ORGS)))
    neg_pat = re.compile("|".join(NEGATIVE_PATTERNS))

    auth["neg_hit"] = auth["aff_norm"].str.contains(neg_pat, regex=True, na=False)
    auth["hit_country"] = auth["aff_norm"].str.contains(
        country_pat, regex=True, na=False
    )
    auth["hit_city"] = auth["aff_norm"].str.contains(cities_pat, regex=True, na=False)
    auth["hit_org"] = auth["aff_norm"].str.contains(orgs_pat, regex=True, na=False)

    auth["uk_score"] = (
        auth["hit_country"].astype(int) * 8
        + auth["hit_org"].astype(int) * 5
        + auth["hit_city"].astype(int) * 3
        - auth["neg_hit"].astype(int) * 6
    )
    auth["is_uk_affiliation_clean"] = auth["uk_score"] >= 5

    uk_auth = auth[auth["is_uk_affiliation_clean"]].copy()

    # Prefer ORCID as author ID; else a stable hash of name+affiliation
    def make_author_id(row):
        orcid = str(row.get("orcid", "") or "").strip()
        if orcid:
            return f"orcid:{orcid}"
        name = norm(row.get("author_name", ""))
        return f"hash:{abs(hash((name, row['aff_norm'])))}"

    uk_auth["author_id"] = uk_auth.apply(make_author_id, axis=1)

    uk_auth.to_csv("uk_authors_clean.csv", index=False)
    ambig = auth[auth["uk_score"].between(3, 6)].copy().head(100)
    ambig[["author_name", "affiliation", "uk_score"]].to_csv(
        "affiliations_near_threshold_sample.csv", index=False
    )

    # Optional: build uk_publications_clean.csv if pmid exists in both tables
    if os.path.exists(PUBS_PATH) and "pmid" in auth.columns:
        pubs = pd.read_csv(PUBS_PATH)
        if "pmid" in pubs.columns:
            pubs["pmid"] = pubs["pmid"].astype(str)
            auth["pmid"] = auth["pmid"].astype(str)
            pmids_uk = set(auth.loc[auth["is_uk_affiliation_clean"], "pmid"])
            pubs[pubs["pmid"].isin(pmids_uk)].to_csv(
                "uk_publications_clean.csv", index=False
            )


if __name__ == "__main__":
    main()
