# ELIXIR Expert Discovery

Mapping UK biomedical researchers to [ELIXIR Europe](https://elixir-europe.org/)
communities, by building two competing retrieval systems over PubMed and evaluating
them head to head.

MSc Advanced Computer Science (AI) dissertation, University of Manchester, 2025.

---

## The problem

ELIXIR Europe organises bioinformatics work into communities (Galaxy, Proteomics,
Microbiome, Human Data, and others). There is no reliable index of *which UK researchers
actually work in which community*. Membership lists are incomplete and self-reported;
publication records are complete but unstructured.

So: infer the mapping from the literature.

## Two approaches, evaluated against each other

The interesting question was not "can RAG retrieve papers" — it was **which corpus
construction strategy produces better expert discovery**. Two pipelines were built to
find out.

| | **Seeds** | **Themes** |
|---|---|---|
| Corpus | Expands outward from known ELIXIR member publications | Built from theme-defining query patterns across all of PubMed |
| Bias | Anchored to existing membership — inherits its gaps | Anchored to topic language — finds people no list knows about |
| Risk | Misses unaffiliated researchers | Topic drift, false positives |

Both use BioBERT embeddings, a FAISS index, and a filtered semantic-search layer
(year / author / institution / theme / journal / UK-affiliation).

### Result

Across 17 focused evaluation queries scored on relevance, diversity, recency,
completeness, and result count:

**Themes won 12, Seeds won 5.**

Query types covered: community-specific (6), institutional (5), general (4), AI/ML (2).
A second broader evaluation set covers 10 further queries with per-query expert lists.

Raw evaluation output is in [`eval/`](eval/) — not summarised away, so the scoring can
be checked rather than taken on trust.

<!-- CONFIRM BEFORE PUBLISHING: the dissertation-scale figures below come from the
     project write-up. Verify each against the final report before this goes public. -->
Corpus scale: ~370,000 publications processed; 17,000+ experts mapped across ~54,500
researcher–community relationships.

---

## Pipeline

```
PubMed E-utilities  ──▶  ingest  ──▶  resolve  ──▶  dataset  ──▶  embed + index  ──▶  RAG
                          │            │              │              │                │
                    resumable      ROR affil.     structured     BioBERT          filtered
                    harvesting     matching       parquet        + FAISS          semantic
                                                                                   search
```

### `src/ingest/` — PubMed harvesting

Long-running harvest of theme-matched UK publications. The engineering problem here was
runtime: the first working version projected to ~100 hours.

- Parallel E-utilities calls via `ThreadPoolExecutor` (32 workers) with a QPS limiter
- Resumable: append-only CSVs, per-theme checkpoints, disk-aware raw-XML retention
- Batched EFetch (200 PMIDs/request) with history-server paging
- `lru_cache` on rules scoring and fuzzy matching

Measured effect is written up in [`docs/performance.md`](docs/performance.md):
**~100h → <20h**, with the largest single win from parallelising the API calls.

`build_uk_elixir_theme_papers.py` is the pre-optimisation baseline, kept deliberately so
the comparison is inspectable rather than asserted.

### `src/resolve/` — affiliation resolution

PubMed affiliation strings are free text. Deciding "is this a UK institution" is the
step that quietly determines corpus quality.

Three-tier strategy, offline-first:

1. **Offline** — fuzzy match against a [ROR](https://ror.org/) snapshot
2. **Online** — ROR API only when the offline match is ambiguous, cached to disk
3. **Rules fallback** — deterministic scoring when ROR returns nothing

Every row records which tier decided it (`affil_method ∈ {ror_api, ror_offline, rules,
none}`) plus the score. Borderline rules-tier matches are written to a separate
near-threshold file for manual audit instead of being silently accepted.

### `src/network/` — co-authorship analysis

Betweenness centrality over the co-authorship graph, as a structural signal for community
membership independent of the text.

### `src/rag/` — retrieval systems

`SeedsRAGSystem` and `ThemesRAGSystem`. Semantic search with post-retrieval filtering, plus
lookup helpers (papers by theme, institutions by author, metadata by PMID).

API documentation: [`docs/seeds-rag.md`](docs/seeds-rag.md) ·
[`docs/themes-rag.md`](docs/themes-rag.md)

### `src/rag_llm/` — LLM integration layer

Adds an LLM answer layer over the retrieval systems, and the harness that produced the
evaluation in `eval/`.

---

## Running it

```bash
pip install -r requirements.txt
cp env.example .env     # then fill in NCBI_EMAIL and NCBI_API_KEY
```

An [NCBI API key](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
is free and raises the E-utilities rate limit from 3 to 10 requests/second. Config is
read from the environment only — there are no credentials in this repository.

```bash
python src/ingest/build_uk_elixir_theme_papers_optimized.py \
    --ror-csv <ror_snapshot.csv> \
    --out output.csv \
    --themes-file themes.json \
    --max-workers 32 \
    --batch-size 200 \
    --sleep-seconds 0.01
```

The harvest resumes from existing output, so an interrupted run is restarted with the
same command.

### Not included

The generated artefacts — the ROR API cache (~116MB), the harvested corpus, the FAISS
indices, and the embedding parquets — are not in the repository. They are reproducible
from the pipeline. The two RAG loaders in `src/rag/` were authored as Colab cells;
their shell magics are commented with `# [colab magic]` so the files parse as plain
Python.

---

## Stack

Python · BioBERT (`transformers`) · FAISS · PyTorch · pandas · rapidfuzz · NCBI
E-utilities · ROR · NumPy

## Licence

MIT — see [LICENSE](LICENSE).
