## Themes RAG System Documentation (Revised)

This documentation explains how to use the `ThemesRAGSystem_Revised` class, which was developed to enable semantic search and retrieval over the Themes publication data.

### 1. Setting up the Themes RAG System

**Instructions:**

1.  **Copy the code:** Copy the entire code block from the self-contained setup cell for the Themes pipeline in start_themes_rag_system_final.py
2.  **Paste into a new notebook:** Create a new Google Colab notebook and paste the copied code into a single code cell.
3.  **Update paths:** **Crucially**, update the `themes_pipeline_output_base_dir` variable at the beginning of the code cell to the correct path where you have saved the outputs of the Themes pipeline.
4.  **Run the cell:** Execute the code cell. This will load the necessary components (metadata, embeddings, FAISS index, lookup dictionaries) and instantiate the `ThemesRAGSystem_Revised` class as the variable `themes_rag_system_revised`.

Once the cell has run successfully, the `themes_rag_system_revised` object will be available for use.

### 2. Performing Semantic Search

The primary method for searching the Themes data is `semantic_search()`.

**`semantic_search(query_text, k=10, year_filter=None, author_filter=None, institution_filter=None, theme_filter=None, journal_filter=None, uk_filter=None, output_format='dataframe')`**

- **`query_text`** (`str`): The text query you want to use for semantic search. The system will find papers semantically similar to this query.
- **`k`** (`int`, optional, default=10): The number of top relevant papers you want to retrieve _after_ applying any filters.
- **`year_filter`** (`int` or `List[int]`, optional): Filter results to include only papers published in the specified year(s).
  - Example: `year_filter=2023` or `year_filter=[2022, 2023, 2024]`
- **`author_filter`** (`str` or `List[str]`, optional): Filter results to include only papers with the specified author(s). Case-insensitive match.
  - Example: `author_filter="Jane Doe"` or `author_filter=["Jane Doe", "John Smith"]`
- **`institution_filter`** (`str` or `List[str]`, optional): Filter results to include only papers associated with the specified institution(s). Case-insensitive match. Note: This uses the 'institutions' data from the Themes pipeline.
  - Example: `institution_filter="University of Oxford"` or `institution_filter=["Imperial College London", "University of Manchester"]`
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

# Assuming 'themes_rag_system_revised' object is already instantiated from the setup cell

# Example 1: Basic semantic search

query = "artificial intelligence applications in healthcare"
results_df = themes_rag_system_revised.semantic_search(query, k=5, output_format='dataframe')
print("Top 5 results (DataFrame):")
display(results_df[['pmid', 'title', 'year', 'themes', 'similarity_score']])

# Example 2: Semantic search with filter and formatted output

query = "CRISPR-cas9 gene editing"
filtered_results_string = themes_rag_system_revised.semantic_search(query, k=3, year_filter=[2023, 2024], output_format='formatted_string')
print(f"\nTop 3 results for '{query}' published in 2023 or 2024 (Formatted String):")
print(filtered_results_string)

# Example 3: Semantic search filtered by institution and UK affiliation

query = "cancer research breakthroughs"
filtered_results_uk_institution = themes_rag_system_revised.semantic_search(query, k=5, institution_filter="Cancer Research UK", uk_filter=True, output_format='dataframe')
print(f"\nTop 5 results for '{query}' from Cancer Research UK (UK affiliated):")
display(filtered_results_uk_institution[['pmid', 'title', 'year', 'institutions', 'any_uk', 'similarity_score']])
