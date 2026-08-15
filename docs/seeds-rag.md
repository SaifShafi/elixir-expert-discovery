## Seeds RAG System Documentation

This documentation explains how to use the `SeedsRAGSystem` class, which was developed to enable semantic search and retrieval over the Seeds publication data.

### 1. Setting up the Seeds RAG System

**Instructions:**

1.  **Copy the code:** Copy the entire code block from start_seeds_rag_system_final.py.
2.  **Paste into a new notebook:** Create a new Google Colab notebook and paste the copied code into a single code cell.
3.  **Update paths:** **Crucially**, update the `seeds_pipeline_output_base_dir` variable at the beginning of the code cell to the correct path where you have saved the outputs of the Seeds pipeline.
4.  **Run the cell:** Execute the code cell. This will load the necessary components (metadata, embeddings, FAISS index, lookup dictionaries) and instantiate the `SeedsRAGSystem` class as the variable `seeds_rag_system`.

Once the cell has run successfully, the `seeds_rag_system` object will be available for use.

### 2. Performing Semantic Search

The primary method for searching the Seeds data is `semantic_search()`.

**`semantic_search(query_text, k=10, year_filter=None, author_filter=None, institution_filter=None, theme_filter=None, journal_filter=None, uk_filter=None, output_format='dataframe')`**

- **`query_text`** (`str`): The text query you want to use for semantic search. The system will find papers semantically similar to this query.
- **`k`** (`int`, optional, default=10): The number of top relevant papers you want to retrieve _after_ applying any filters.
- **`year_filter`** (`int` or `List[int]`, optional): Filter results to include only papers published in the specified year(s).
  - Example: `year_filter=2023` or `year_filter=[2022, 2023, 2024]`
- **`author_filter`** (`str` or `List[str]`, optional): Filter results to include only papers with the specified author(s). Case-insensitive match.
  - Example: `author_filter="Jane Doe"` or `author_filter=["Jane Doe", "John Smith"]`
- **`institution_filter`** (`str` or `List[str]`, optional): Filter results to include only papers associated with the specified institution(s). Case-insensitive match. Note: This uses the 'institutions' data, which currently contains the raw affiliation strings.
  - Example: `institution_filter="University of Edinburgh"` or `institution_filter=["Cardiff University", "University College London"]`
- **`theme_filter`** (`str` or `List[str]`, optional): Filter results to include only papers associated with the specified theme(s). Case-insensitive match.
  - Example: `theme_filter="Genomics"` or `theme_filter=["Microbiome", "Proteomics"]`
- **`journal_filter`** (`str` or `List[str]`, optional): Filter results to include only papers published in the specified journal(s). Case-insensitive match.
  - Example: `journal_filter="Nature Genetics"` or `journal_filter=["Science", "Nature"]`
- **`uk_filter`** (`bool`, optional): Filter results based on whether the paper has any UK affiliation. `True` for UK papers, `False` for non-UK papers.
  - Example: `uk_filter=True`
- **`output_format`** (`str`, optional, default='dataframe'): Specifies the format of the output.
  - `'dataframe'`: Returns a pandas DataFrame containing the metadata of the top k results, sorted by similarity score.
  - `'formatted_string'`: Returns a formatted text string containing the details of the top k results, similar to the demonstration output.

**Example Usage:**

# Example 1: Find institutions for a specific author

author = "Michael Glinka" # Replace with an author from your data
institutions = seeds_rag_system.get_institutions_by_author(author)
print(f"Institutions associated with {author}: {institutions}")

# Example 2: Get metadata for papers associated with a theme

theme = "Proteomics" # Replace with a theme from your data
paper_indices = seeds_rag_system.get_papers_by_theme(theme)
if paper_indices:
print(f"\nSample metadata for papers on '{theme}':")
sample_indices = paper_indices[:5] # Look at first 5 papers
papers_metadata = seeds_rag_system.get_metadata_by_indices(sample_indices)
display(papers_metadata[['pmid', 'title', 'year', 'authors']])
else:
print(f"\nNo papers found for theme '{theme}'.")

# Example 3: Get metadata for a specific paper by PMID

pmid = "39807940" # Replace with a PMID from your data
paper_info = seeds_rag_system.get_paper_by_pmid(pmid)
if paper_info is not None:
print(f"\nMetadata for PMID {pmid}:")
display(paper_info)
else:
print(f"\nPaper with PMID {pmid} not found.")
