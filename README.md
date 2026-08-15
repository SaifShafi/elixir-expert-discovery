# ELIXIR Expert Discovery

Two methods for finding the UK biomedical researchers who belong in
[ELIXIR Europe](https://elixir-europe.org/) communities — built, evaluated against each
other, and found to disagree almost completely.

MSc Advanced Computer Science (AI) dissertation, University of Manchester, 2025.
*Mapping ELIXIR Communities: Evaluating Content-Based and Network-Based Approaches for
Systematic Expert Discovery in UK Biomedical Research.*

---

## The problem

ELIXIR coordinates 18 life-sciences communities across Europe. It has no systematic way
to identify which UK researchers' work aligns with which community — membership comes
from passive self-selection through known research groups, which structurally misses
everyone outside those groups.

Doing it by hand would mean reading tens of thousands of publications and judging
research alignment across 18 distinct domains.

## Two pipelines

Both end in a Retrieval-Augmented Generation system over BioBERT embeddings and a FAISS
index. They differ in how the corpus is built, which turns out to matter enormously.

| | **Content-based** ("themes") | **Network-based** ("seeds") |
|---|---|---|
| Corpus | Theme-defining query patterns across PubMed | Expands outward from 86 verified ELIXIR UK authors via co-authorship |
| Publications | 80,849 UK-affiliated | 370,282 |
| Assumes | Community membership shows in what you write about | Community membership shows in who you work with |
| Experts found | 26,111 (100% confidence) | 28,567 (99.97% confidence) |

Author-institution affiliation records processed: **8,373,485**.

## The finding

Both systems work. Both produce high-confidence expert lists of comparable size.

**Only 924 experts — 5.6% — were validated by both.**

The two approaches identify fundamentally different populations. Not noise, not a tuning
artefact: a structural consequence of what each method treats as evidence of belonging.

- **Network-based** finds collaborative, early-career researchers (70.9% vs 49.6%) in
  established computational domains like Galaxy workflows.
- **Content-based** finds focused specialists (99.1% single-theme) and mid-career
  researchers in emerging interdisciplinary areas like Rare Diseases.
- Author overlap between systems: 24.4%. Publication overlap: **3.6%**.
- The 924 who appear in both turn out to be cross-domain researchers.

The practical implication is the useful part: **choosing one method silently chooses which
kind of researcher you are able to find.** An organisation running expert discovery with
a single approach will systematically miss a population it has no way of knowing it
missed.

Combined output: **17,333 unique UK experts** across **54,500+ researcher–community
mappings**.

Robustness: results hold across **600+ parameter combinations** of thresholds and expert
criteria.

---

## Evaluation

Four complementary frameworks: coverage validation, overlap analysis, literature-based
expert evaluation using bibliometric criteria (10+ publications, multi-institutional
collaboration, activity since 2020), and head-to-head RAG querying.

On the RAG comparison — 17 focused queries scored on relevance, diversity, recency and
completeness — **content-based won 12, network-based won 5**. Query types: community-specific
(6), institutional (5), general (4), AI/ML (2).

Raw evaluation output ships in [`eval/`](eval/) rather than being summarised away, so the
scoring can be checked instead of taken on trust.

---

## Pipeline

```
PubMed E-utilities ──▶ ingest ──▶ resolve ──▶ dataset ──▶ BioBERT + FAISS ──▶ RAG
                        │           │           │              │              │
                   resumable    ROR affil.  structured     embeddings     filtered
                   harvesting   matching    parquet        + index        semantic
                                                                          search
```

### `src/ingest/` — PubMed harvesting

The engineering problem was runtime: the first working version projected to ~100 hours.

- Parallel E-utilities calls via `ThreadPoolExecutor` (32 workers) with a QPS limiter
- Resumable: append-only CSVs, per-theme checkpoints, disk-aware raw-XML retention
- Batched EFetch (200 PMIDs/request) with history-server paging
- `lru_cache` on rules scoring and fuzzy matching

Result, written up in [`docs/performance.md`](docs/performance.md): **~100h → <20h**.
`build_uk_elixir_theme_papers.py` is the pre-optimisation baseline, kept deliberately so
the comparison is inspectable rather than asserted.

### `src/resolve/` — affiliation resolution

PubMed affiliation strings are free text. Deciding "is this a UK institution" quietly
determines corpus quality. Three tiers, offline-first:

1. **Offline** — fuzzy match against a [ROR](https://ror.org/) snapshot
2. **Online** — ROR API only when the offline match is ambiguous, cached to disk
3. **Rules fallback** — deterministic scoring when ROR returns nothing

Every row records which tier decided it (`affil_method ∈ {ror_api, ror_offline, rules,
none}`) and its score. Borderline rules-tier matches are written to a separate
near-threshold file for manual audit rather than silently accepted.

### `src/network/` — co-authorship analysis

Betweenness centrality over the co-authorship graph — the structural signal underlying
the network-based pipeline.

### `src/rag/` — retrieval systems

`SeedsRAGSystem` and `ThemesRAGSystem`. Semantic search with post-retrieval filtering by
year, author, institution, theme, journal and UK affiliation, plus lookup helpers.

BioBERT was chosen over PubMedBERT and SciBERT: PubMedBERT has more specialised
vocabulary and SciBERT broader scientific coverage, but BioBERT's combined training on
PubMed abstracts and PMC full texts was the better balance for this corpus.

API docs: [`docs/seeds-rag.md`](docs/seeds-rag.md) · [`docs/themes-rag.md`](docs/themes-rag.md)

### `src/rag_llm/` — LLM layer

An answer layer over the retrieval systems, plus the harness that produced `eval/`.

---

## Running it

```bash
pip install -r requirements.txt
cp env.example .env      # then fill in NCBI_EMAIL and NCBI_API_KEY
```

An [NCBI API key](https://www.ncbi.nlm.nih.gov/account/settings/) is free and raises the
E-utilities rate limit from 3 to 10 requests/second. Config is read from the environment
only — there are no credentials in this repository.

```bash
python src/ingest/build_uk_elixir_theme_papers_optimized.py \
    --ror-csv <ror_snapshot.csv> \
    --out output.csv \
    --themes-file themes.json \
    --max-workers 32 --batch-size 200 --sleep-seconds 0.01
```

The harvest resumes from existing output, so an interrupted run restarts with the same
command.

### Not included

Generated artefacts — the ROR API cache (~116MB), the harvested corpus, FAISS indices and
embedding parquets — are reproducible from the pipeline and are not in the repository.
The two RAG loaders in `src/rag/` were authored as Colab cells; their shell magics are
commented `# [colab magic]` so the files parse as plain Python.

## Stack

Python · BioBERT (`transformers`) · FAISS · PyTorch · pandas · rapidfuzz · NCBI
E-utilities · ROR · NumPy

## Licence

MIT — see [LICENSE](LICENSE).
