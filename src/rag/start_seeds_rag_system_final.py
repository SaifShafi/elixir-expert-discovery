import faiss
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel # Need these for the get_embedding method in the class
from pathlib import Path
import json
import ast # Needed for ast.literal_eval if metadata includes string representations of lists
from typing import List, Dict, Any, Optional, Union

# --- Configuration: Define Paths to Saved Components ---
# IMPORTANT: Replace with the actual paths where your Seeds pipeline outputs are saved
seeds_pipeline_output_base_dir = "/content/drive/MyDrive/outputs/rebuild_seeds_pipeline" # Ensure this matches your base output directory

seeds_metadata_path = f"{seeds_pipeline_output_base_dir}/seeds_step5_embeddings/seeds_embedding_metadata.parquet"
seeds_embeddings_path = f"{seeds_pipeline_output_base_dir}/seeds_step5_embeddings/seeds_embeddings.npy"
seeds_faiss_index_path = f"{seeds_pipeline_output_base_dir}/seeds_step6_faiss_index/seeds_faiss_index.faiss"
seeds_lookup_dicts_path = f"{seeds_pipeline_output_base_dir}/seeds_step7_lookup_dicts/seeds_lookup_dicts.json"

# --- Function to Load Components ---
def load_seeds_rag_components(metadata_path: str,
                              embeddings_path: str,
                              faiss_index_path: str,
                              lookup_dicts_path: str):
    """Loads saved components for the Seeds RAG system."""
    print("Loading Seeds RAG components...")
    try:
        metadata_df = pd.read_parquet(metadata_path)
        print(f"  Loaded metadata DataFrame: {metadata_df.shape}")

        embeddings = np.load(embeddings_path)
        print(f"  Loaded embeddings: {embeddings.shape}")

        faiss_index = faiss.read_index(faiss_index_path)
        print(f"  Loaded FAISS index with {faiss_index.ntotal} vectors.")

        with open(lookup_dicts_path, 'r') as f:
            lookup_dicts = json.load(f)
        print(f"  Loaded lookup dictionaries with keys: {list(lookup_dicts.keys())}")

        print("✓ All components loaded successfully.")
        return metadata_df, embeddings, faiss_index, lookup_dicts

    except FileNotFoundError as e:
        print(f"❌ Error loading component: {e}. Please ensure the paths are correct.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ An error occurred during component loading: {e}")
        return None, None, None, None


# --- SeedsRAGSystem Class Definition (Consistent with ThemesRAGSystem_Revised and formatted output) ---
class SeedsRAGSystem:
    """
    A RAG system for Seeds publications, integrating metadata, BioBERT embeddings,
    FAISS index, and lookup dictionaries, consistent with ThemesRAGSystem_Revised
    and including formatted search output.
    """
    def __init__(self,
                 metadata_df: pd.DataFrame,
                 embeddings: np.ndarray,
                 faiss_index: faiss.Index,
                 lookup_dicts: Dict[str, Any]):
        """
        Initializes the SeedsRAGSystem.
        Note: Model, tokenizer, and device are handled internally within get_embedding.

        Args:
            metadata_df: DataFrame containing publication metadata.
            embeddings: Normalized numpy array of BioBERT embeddings.
            faiss_index: Loaded FAISS index.
            lookup_dicts: Dictionary containing various lookup mappings.
        """
        print("   Initializing SeedsRAGSystem...")
        self.metadata_df = metadata_df
        self.embeddings = embeddings
        self.faiss_index = faiss_index
        self.lookup_dicts = lookup_dicts

        # Store individual lookup dictionaries for easier access
        self.pmid_to_index = self.lookup_dicts.get('pmid_to_index', {})
        self.index_to_pmid = self.lookup_dicts.get('index_to_pmid', {})
        self.author_to_papers = self.lookup_dicts.get('author_to_papers', {})
        self.institution_to_papers = self.lookup_dicts.get('institution_to_papers', {})
        self.theme_to_papers = self.lookup_dicts.get('theme_to_papers', {})
        self.year_to_papers = self.lookup_dicts.get('year_to_papers', {})
        self.journal_to_papers = self.lookup_dicts.get('journal_to_papers', {})
        self.author_to_institutions = self.lookup_dicts.get('author_to_institutions', {})
        self.institution_to_authors = self.lookup_dicts.get('institution_to_authors', {})
        self.theme_to_authors = self.lookup_dicts.get('theme_to_authors', {})

        # --- Internal Model and Tokenizer Setup ---
        # Load BioBERT model and tokenizer internally when the class is instantiated
        # This requires transformers and torch to be installed and potentially CUDA
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model_and_tokenizer()


        print("   ✓ SeedsRAGSystem initialized.")

    def _load_model_and_tokenizer(self):
        """Loads the BioBERT model and tokenizer internally."""
        model_name = "dmis-lab/biobert-v1.1"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Loading BioBERT model: {model_name} on device: {self.device}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            print(f"   ✓ BioBERT model and tokenizer loaded internally.")
        except Exception as e:
            print(f"   ❌ Error loading BioBERT model/tokenizer internally: {e}")
            print("      Embedding generation will not be available.")
            self.model = None
            self.tokenizer = None
            self.device = 'cpu' # Reset device if model loading fails


    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generates a BioBERT embedding for a given text.

        Args:
            text: The input text string.

        Returns:
            A normalized numpy array representing the embedding, or None if model is not loaded.
        """
        if self.model is None or self.tokenizer is None:
            print("Error: BioBERT model or tokenizer not loaded. Cannot generate embedding.")
            return None

        self.model.eval()
        with torch.no_grad():
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            output = self.model(input_ids=encoded_input['input_ids'],
                                attention_mask=encoded_input['attention_mask'],
                                token_type_ids=encoded_input['token_type_ids']) # Include token_type_ids
            last_hidden_states = output[0]

            # Mean pooling
            embedding = torch.sum(last_hidden_states * encoded_input['attention_mask'].unsqueeze(-1), dim=1) / torch.clamp(encoded_input['attention_mask'].sum(1, keepdim=True), min=1e-9)

            embedding_np = embedding.cpu().numpy()
            # Normalize the embedding
            embedding_normalized = embedding_np / np.linalg.norm(embedding_np, axis=1, keepdims=True)

        return embedding_normalized.flatten() # Return as a 1D array


    def semantic_search(self,
                          query_text: str,
                          k: int = 10,
                          year_filter: Optional[Union[int, List[int]]] = None,
                          author_filter: Optional[Union[str, List[str]]] = None,
                          institution_filter: Optional[Union[str, List[str]]] = None,
                          theme_filter: Optional[Union[str, List[str]]] = None,
                          journal_filter: Optional[Union[str, List[str]]] = None,
                          uk_filter: Optional[bool] = None,
                          output_format: str = 'dataframe' # Added output_format parameter
                         ) -> Union[pd.DataFrame, str]: # Return type can be DataFrame or string
        """
        Performs a semantic search using the FAISS index with optional metadata filtering.
        Consistent with ThemesRAGSystem_Revised.

        Args:
            query_text: The text query for semantic search.
            k: The number of nearest neighbors to retrieve initially from FAISS.
            year_filter: Optional year or list of years to filter by.
            author_filter: Optional author name or list of names to filter by.
            institution_filter: Optional institution name or list of names to filter by.
            theme_filter: Optional theme name or list of names to filter by.
            journal_filter: Optional journal name or list of names to filter by.
            uk_filter: Optional boolean to filter by UK affiliation (True for UK, False for non-UK).
            output_format: 'dataframe' to return pandas DataFrame, 'formatted_string' to return
                            a string formatted like the example. Defaults to 'dataframe'.

        Returns:
            A DataFrame containing the metadata of the top k relevant papers
            that match the optional filters, sorted by similarity score,
            OR a formatted string of the results, depending on `output_format`.
        """
        if self.model is None or self.tokenizer is None:
             return "Error: BioBERT model or tokenizer not loaded. Cannot perform semantic search." if output_format == 'formatted_string' else pd.DataFrame()

        # print(f"\nPerforming semantic search for query: '{query_text}'") # Removed for cleaner output
        query_embedding = self.get_embedding(query_text)

        if query_embedding is None: # Check if embedding generation failed
             return "Error: Failed to generate query embedding. Cannot perform semantic search." if output_format == 'formatted_string' else pd.DataFrame()


        # Perform initial FAISS search to get nearest neighbors based on embedding similarity
        # D is the distance/similarity matrix, I is the index matrix
        # Search for more initially (e.g., k * 100) to increase the chance of finding
        # enough results that match the subsequent metadata filters.
        search_k = k * 100 if k > 0 else 100 # Ensure search_k is reasonable
        D, I = self.faiss_index.search(query_embedding.reshape(1, -1).astype('float32'), search_k)

        # Get the indices of the nearest neighbors
        nearest_neighbor_indices = I.flatten()

        # Retrieve metadata for these indices
        # Ensure indices are within the valid range of metadata_df
        valid_indices = [idx for idx in nearest_neighbor_indices if idx < len(self.metadata_df)]
        # Map back to original DataFrame index if the metadata_df was re-indexed
        # (In this pipeline, metadata_df index IS the paper index, so direct iloc is fine)
        search_results_df = self.metadata_df.iloc[valid_indices].copy()

        # Add similarity scores to the results (cosine similarity = inner product for normalized vectors)
        # The scores in D are already inner products since embeddings are normalized
        # Ensure scores match the valid_indices by mapping based on the original index
        score_mapping = dict(zip(nearest_neighbor_indices, D.flatten()))
        search_results_df['similarity_score'] = search_results_df.index.map(score_mapping)


        # print(f"   Initial FAISS search returned {len(search_results_df):,} results.") # Removed for cleaner output

        # --- Apply Metadata Filters ---
        # Start with the search results and progressively filter
        filtered_results_df = search_results_df.copy()
        initial_filter_count = len(filtered_results_df)
        # print(f"   Applying metadata filters...") # Removed for cleaner output

        # Helper to get paper indices for a given filter value(s) using lookups
        def get_indices_for_filter(filter_value: Optional[Union[str, int, List[Union[str, int]]]],
                                   lookup_dict: Dict[Union[str, int], List[int]]) -> Optional[set]:
            if filter_value is None:
                return None # No filter applied

            # Ensure filter_value is a list for consistent processing
            if not isinstance(filter_value, list):
                filter_value = [filter_value]

            # Get indices for each item in the filter list
            all_indices = []
            for item in filter_value:
                if isinstance(item, str):
                     # Use lower case for string lookups
                     all_indices.extend(lookup_dict.get(item.strip().lower(), []))
                elif isinstance(item, int): # Handle integer keys for year
                     all_indices.extend(lookup_dict.get(item, []))
                # Add handling for other potential filter types if needed

            # Return unique indices as a set for efficient checking
            return set(all_indices)


        # Apply Year Filter
        if year_filter is not None:
            year_indices = get_indices_for_filter(year_filter, self.year_to_papers)
            if year_indices is not None:
                # Filter the DataFrame based on whether the paper index is in the set of allowed indices
                # The index of search_results_df is the original metadata_df index
                filtered_results_df = filtered_results_df[filtered_results_df.index.isin(year_indices)]
                # print(f"      - After year filter: {len(filtered_results_df):,} results") # Removed

        # Apply Author Filter
        if author_filter is not None:
            author_indices = get_indices_for_filter(author_filter, self.author_to_papers)
            if author_indices is not None:
                 filtered_results_df = filtered_results_df[filtered_results_df.index.isin(author_indices)]
                 # print(f"      - After author filter: {len(filtered_results_df):,} results") # Removed

        # Apply Institution Filter
        if institution_filter is not None:
            institution_indices = get_indices_for_filter(institution_filter, self.institution_to_papers)
            if institution_indices is not None:
                 filtered_results_df = filtered_results_df[filtered_results_df.index.isin(institution_indices)]
                 # print(f"      - After institution filter: {len(filtered_results_df):,} results") # Removed

        # Apply Theme Filter
        if theme_filter is not None:
            theme_indices = get_indices_for_filter(theme_filter, self.theme_to_papers)
            if theme_indices is not None:
                 filtered_results_df = filtered_results_df[filtered_results_df.index.isin(theme_indices)]
                 # print(f"      - After theme filter: {len(filtered_results_df):,} results") # Removed

        # Apply Journal Filter
        if journal_filter is not None:
            journal_indices = get_indices_for_filter(journal_filter, self.journal_to_papers)
            if journal_indices is not None:
                 filtered_results_df = filtered_results_df[filtered_results_df.index.isin(journal_indices)]
                 # print(f"      - After journal filter: {len(filtered_results_df):,} results") # Removed

        # Apply UK Filter
        if uk_filter is not None:
             # Filter based on the 'any_uk' boolean column in the metadata_df
             # Ensure 'any_uk' column exists before filtering
             if 'any_uk' in filtered_results_df.columns:
                filtered_results_df = filtered_results_df[filtered_results_df['any_uk'] == uk_filter]
                # print(f"      - After UK filter: {len(filtered_results_df):,} results") # Removed
             # else:
                # print("      - Warning: 'any_uk' column not found for UK filter.") # Removed


        # print(f"   ✓ Metadata filters applied. {initial_filter_count - len(filtered_results_df):,} results filtered out.") # Removed


        # Sort the filtered results by similarity score in descending order and take the top k
        final_results_df = filtered_results_df.sort_values(by='similarity_score', ascending=False).head(k).reset_index(drop=False) # Keep original index for reference

        # print(f"   Final search returned {len(final_results_df):,} results.") # Removed for cleaner output

        # --- Format Output based on output_format ---
        if output_format == 'dataframe':
            return final_results_df
        elif output_format == 'formatted_string':
            output_string = f"--- Testing Semantic Search ---\n"
            output_string += f"Performing semantic search for query: '{query_text}' (k={k})\n\n"

            # Initial candidates count is difficult to get accurately here, use placeholder or remove
            # output_string += f"Initial candidates: {search_k}\n" # Use search_k as an approximation
            output_string += f"   Searching the main index and filtering results...\n"
            output_string += f"   Search completed. Found {len(final_results_df)} results.\n\n"

            if not final_results_df.empty:
                output_string += "--- Search Results Sample ---\n"
                for i, row in final_results_df.iterrows():
                    output_string += f"{i+1}. Title: {row.get('title', 'N/A')}\n"
                    output_string += f"   Score: {row.get('similarity_score', 0):.3f}\n"
                    output_string += f"   PMID: {row.get('pmid', 'N/A')}\n"
                    year_val = row.get('year')
                    output_string += f"   Year: {int(year_val) if pd.notna(year_val) else 'N/A'}\n"

                    themes = row.get('themes', [])
                    institutions = row.get('institutions', [])
                    authors = row.get('authors', [])

                    output_string += f"   Themes: {', '.join(themes) if themes else 'N/A'}\n"
                    # Limit institutions and authors to a few for display if lists are long
                    display_inst = institutions[:5] if len(institutions) > 5 else institutions
                    output_string += f"   Institutions: {', '.join(display_inst) if display_inst else 'N/A'}{'...' if len(institutions) > 5 else ''}\n"

                    display_authors = authors[:5] if len(authors) > 5 else authors
                    output_string += f"   Authors: {', '.join(display_authors) if display_authors else 'N/A'}{'...' if len(authors) > 5 else ''}\n"

                    journal = row.get('journal', 'N/A')
                    output_string += f"   Journal: {journal if journal else 'N/A'}\n"

                    output_string += "-" * 20 + "\n" # Separator for each result
            else:
                output_string += "No results found.\n"
            output_string += "-" * 20 + "\n" # Final separator

            return output_string

        else:
            # Handle invalid output_format
            print(f"Warning: Invalid output_format '{output_format}'. Returning DataFrame.")
            return final_results_df


    # --- Getter Methods for Lookup Dictionaries ---
    # Provide methods to access the lookup dictionaries, consistent with ThemesRAGSystem_Revised
    def get_authors_by_institution(self, institution_name: str) -> List[str]:
        """Returns a list of authors associated with a given institution (case-insensitive)."""
        return self.institution_to_authors.get(institution_name.strip().lower(), [])

    def get_institutions_by_author(self, author_name: str) -> List[str]:
        """Returns a list of institutions associated with a given author (case-insensitive)."""
        return self.author_to_institutions.get(author_name.strip().lower(), [])

    def get_authors_by_theme(self, theme_name: str) -> List[str]:
        """Returns a list of authors associated with a given theme (case-insensitive)."""
        return self.theme_to_authors.get(theme_name.strip().lower(), [])

    def get_papers_by_author(self, author_name: str) -> List[int]:
         """Returns a list of paper indices for a given author (case-insensitive)."""
         return self.author_to_papers.get(author_name.strip().lower(), [])

    def get_papers_by_institution(self, institution_name: str) -> List[int]:
         """Returns a list of paper indices for a given institution (case-insensitive)."""
         return self.institution_to_papers.get(institution_name.strip().lower(), [])

    def get_papers_by_theme(self, theme_name: str) -> List[int]:
         """Returns a list of paper indices for a given theme (case-insensitive)."""
         return self.theme_to_papers.get(theme_name.strip().lower(), [])

    def get_papers_by_year(self, year: int) -> List[int]:
         """Returns a list of paper indices for a given year."""
         # Ensure year is an integer for lookup
         try:
             year_int = int(year)
             return self.year_to_papers.get(year_int, [])
         except (ValueError, TypeError):
             return []

    def get_papers_by_journal(self, journal_name: str) -> List[int]:
         """Returns a list of paper indices for a given journal (case-insensitive)."""
         return self.journal_to_papers.get(journal_name.strip().lower(), [])

    def get_paper_by_pmid(self, pmid: Union[str, int]) -> Optional[pd.Series]:
         """Returns the metadata for a paper given its PMID."""
         pmid_str = str(pmid).strip()
         if pmid_str in self.pmid_to_index:
              index = self.pmid_to_index[pmid_str]
              # Ensure index is valid before accessing
              if index < len(self.metadata_df):
                   # .iloc returns a Series for a single row
                   return self.metadata_df.iloc[index].copy() # Return a copy to prevent SettingWithCopyWarning
         return None # Return None if PMID not found or index invalid

    def get_metadata_by_indices(self, indices: List[int]) -> pd.DataFrame:
         """Returns the metadata for a list of paper indices."""
         # Filter indices to ensure they are within the valid range
         valid_indices = [idx for idx in indices if idx < len(self.metadata_df)]
         # Use .loc with the index (which corresponds to original index)
         return self.metadata_df.loc[valid_indices].copy() if valid_indices else pd.DataFrame() # Return a copy


    # Add any other methods needed for consistency or functionality


# --- Load Components and Instantiate SeedsRAGSystem ---
print("--- Setting up Seeds RAG System ---")
seeds_metadata_df_loaded, seeds_embeddings_loaded, seeds_faiss_index_loaded, seeds_lookup_dicts_loaded = load_seeds_rag_components(
    seeds_metadata_path,
    seeds_embeddings_path,
    seeds_faiss_index_path,
    seeds_lookup_dicts_path
)

seeds_rag_system = None # Initialize to None
if all([seeds_metadata_df_loaded is not None, seeds_embeddings_loaded is not None, seeds_faiss_index_loaded is not None, seeds_lookup_dicts_loaded is not None]):
    try:
        # Pass loaded components to the constructor
        seeds_rag_system = SeedsRAGSystem(
            metadata_df=seeds_metadata_df_loaded,
            embeddings=seeds_embeddings_loaded,
            faiss_index=seeds_faiss_index_loaded,
            lookup_dicts=seeds_lookup_dicts_loaded
        )
        print("\n✓ SeedsRAGSystem instantiated successfully using loaded components.")
        print("The 'seeds_rag_system' object is now available for use.")

    except Exception as e:
        print(f"\n❌ Error instantiating SeedsRAGSystem with loaded components: {e}")
else:
    print("\n❌ SeedsRAGSystem could not be instantiated due to loading errors.")

# The 'seeds_rag_system' variable is now available if instantiation was successful