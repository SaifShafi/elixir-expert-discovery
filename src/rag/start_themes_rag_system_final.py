# @title
# Define the base directory where your pipeline outputs are saved
# IMPORTANT: Update this path if you saved your outputs elsewhere!
pipeline_output_base_dir = "/content/drive/MyDrive/outputs/rebuild_themes_pipeline"

# Define paths to the saved components
step2_output_dir = f"{pipeline_output_base_dir}/step2_structured"
step5_output_dir = f"{pipeline_output_base_dir}/step5_embeddings"
step6_output_dir = f"{pipeline_output_base_dir}/step6_faiss_index"
step7_output_dir = f"{pipeline_output_base_dir}/step7_lookups"

master_data_path = f"{step2_output_dir}/themes_master_structured.parquet"
metadata_path = f"{step5_output_dir}/themes_metadata_for_embedding.parquet"
faiss_index_path = f"{step6_output_dir}/themes_faiss.index"
lookups_path = f"{step7_output_dir}/themes_lookup_dictionaries.pkl"


# --- Step 2: Import Necessary Libraries ---
# [colab magic] !pip install faiss-cpu # Use faiss-cpu if you don't have a GPU
import pandas as pd
import numpy as np
import faiss # Make sure faiss-gpu or faiss-cpu is installed in this notebook's environment (`!pip install faiss-gpu`)
import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
import time
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


# --- Step 3: Define the ThemesRAGSystem_Revised Class and Helper Function ---
# This is the definition of the RAG system class, including all methods

def ensure_list(item):
    """Ensures an item is a list, handling None, NaN, and numpy arrays."""
    if isinstance(item, list):
        return item
    elif isinstance(item, np.ndarray):
        return item.tolist() # Convert numpy array to list
    elif pd.notna(item):
        return [item] # Wrap non-NaN scalar in a list
    else:
        return [] # Return empty list for NaN or None


class ThemesRAGSystem_Revised:
    def __init__(self,
                 master_data_path,
                 metadata_path,
                 faiss_index_path,
                 lookups_path,
                 model_name="dmis-lab/biobert-v1.1"):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for RAG system: {self.device}")

        # --- Load Components ---
        print("Loading RAG system components...")

        # Load Master Data (for rich result details)
        try:
            self.master_df = pd.read_parquet(master_data_path)
            print(f"   Loaded master DataFrame: {self.master_df.shape}")
        except FileNotFoundError:
            print(f"   Master DataFrame not found at {master_data_path}")
            self.master_df = pd.DataFrame()
        except Exception as e:
            print(f"   Error loading master DataFrame: {e}")
            self.master_df = pd.DataFrame()


        # Load Metadata (aligns with embeddings and FAISS index)
        try:
            self.metadata_df = pd.read_parquet(metadata_path)
            print(f"   Loaded metadata DataFrame: {self.metadata_df.shape}")
            # Ensure index is reset as embeddings were built on a re-indexed df
            self.metadata_df = self.metadata_df.reset_index(drop=True)
            print(f"   Metadata DataFrame re-indexed.")
        except FileNotFoundError:
            print(f"   Metadata DataFrame not found at {metadata_path}")
            self.metadata_df = pd.DataFrame()
        except Exception as e:
            print(f"   Error loading metadata DataFrame: {e}")
            self.metadata_df = pd.DataFrame()

        # Load Embeddings (optional, could load only when needed for testing/validation)
        # For now, we assume the index contains the embeddings.

        # Load FAISS Index
        try:
            self.index = faiss.read_index(faiss_index_path)
            print(f"   Loaded FAISS index with {self.index.ntotal:,} vectors.")
        except FileNotFoundError:
            print(f"   FAISS index not found at {faiss_index_path}")
            self.index = None
        except Exception as e:
            print(f"   Error loading FAISS index: {e}")
            self.index = None


        # Load Lookup Dictionaries
        try:
            with open(lookups_path, 'rb') as f:
                self.lookups = pickle.load(f)
            print(f"   Loaded lookup dictionaries: {list(self.lookups.keys())}")
            # Assign lookups with default empty dict if key is missing
            self.pmid_to_index = self.lookups.get('pmid_to_index', {})
            self.author_to_papers = self.lookups.get('author_to_papers', {})
            self.institution_to_papers = self.lookups.get('institution_to_papers', {})
            self.theme_to_papers = self.lookups.get('theme_to_papers', {})
            self.year_to_papers = self.lookups.get('year_to_papers', {})
            self.journal_to_papers = self.lookups.get('journal_to_papers', {})
            self.city_to_papers = self.lookups.get('city_to_papers', {})
            self.author_to_institutions = self.lookups.get('author_to_institutions', {})
            self.institution_to_authors = self.lookups.get('institution_to_authors', {})
            self.theme_to_authors = self.lookups.get('theme_to_authors', {})

        except FileNotFoundError:
            print(f"   Lookup dictionaries not found at {lookups_path}")
            self.lookups = {}
        except Exception as e:
            print(f"   Error loading lookup dictionaries: {e}")
            self.lookups = {}


        # Load BioBERT Model and Tokenizer (for query embedding)
        print("Loading BioBERT model and tokenizer for query embedding...")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"   ✓ BioBERT model and tokenizer loaded for query embedding.")
        except Exception as e:
            print(f"   ❌ Error loading BioBERT model/tokenizer: {e}")
            self.tokenizer = None
            self.model = None

        # Create a PMID -> Master DataFrame Index mapping for efficient result lookup
        # This is done once during initialization
        self.master_pmid_to_index = {}
        if not self.master_df.empty and 'pmid' in self.master_df.columns:
             print("\nCreating PMID to Master DataFrame index mapping...")
             # Ensure pmid is string in master_df for consistent mapping
             self.master_df['pmid_str'] = self.master_df['pmid'].astype(str).str.strip()
             self.master_pmid_to_index = pd.Series(self.master_df.index, index=self.master_df['pmid_str']).to_dict()
             self.master_df = self.master_df.drop(columns=['pmid_str']) # Clean up temp column
             print(f"   ✓ Master PMID mapping created: {len(self.master_pmid_to_index):,}")
        else:
             print("   ⚠️  'pmid' column not found in master DataFrame. Cannot create master_pmid_to_index.")
             self.master_pmid_to_index = {}


        if self.master_df.empty or self.metadata_df.empty or self.index is None or not self.lookups or self.model is None:
             print("\n⚠️  Warning: One or more essential components failed to load. RAG system may not be fully functional.")
        else:
             print("\n✓ Themes RAG System components loaded successfully.")


    def _encode_query(self, query_text):
        """Encodes a query string into an embedding using the BioBERT model."""
        if self.model is None or self.tokenizer is None:
            print("Error: BioBERT model or tokenizer not available for query encoding.")
            return None

        self.model.eval()
        with torch.no_grad():
            encoded_input = self.tokenizer(
                query_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            outputs = self.model(**encoded_input)
            # Use [CLS] token embedding and move back to CPU for numpy conversion
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Normalize the query embedding
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

        return query_embedding


    def get_paper_details_by_metadata_index(self, metadata_index):
        """
        Retrieves full paper details from the master_df given an index from the metadata_df.
        Uses the PMID mapping to ensure correctness.
        """
        if self.metadata_df.empty or self.master_df.empty or not self.master_pmid_to_index:
            # print("DEBUG: get_paper_details_by_metadata_index: Data not loaded or mapping missing.") # Debug print
            return None

        if metadata_index < len(self.metadata_df) and 'pmid' in self.metadata_df.columns:
            pmid_from_metadata = str(self.metadata_df.iloc[metadata_index]['pmid']).strip()
            if pmid_from_metadata in self.master_pmid_to_index:
                master_df_index = self.master_pmid_to_index[pmid_from_metadata]
                paper_details = self.master_df.iloc[master_df_index].to_dict()
                # Ensure list columns are lists
                for col in ['authors', 'institutions', 'themes']:
                     if col in paper_details:
                          paper_details[col] = ensure_list(paper_details[col])
                # Ensure boolean 'any_uk'
                if 'any_uk' in paper_details:
                     paper_details['any_uk'] = bool(paper_details['any_uk'])
                # Ensure year is nullable integer
                if 'year' in paper_details and pd.notna(paper_details['year']):
                     paper_details['year'] = int(paper_details['year'])
                else:
                     paper_details['year'] = None
                # Include journal information
                paper_details['journal'] = paper_details.get('journal', 'N/A')


                return paper_details
            # else:
                # print(f"DEBUG: get_paper_details_by_metadata_index: PMID '{pmid_from_metadata}' from metadata_df not found in master_df_index.") # Debug print
        # else:
            # print(f"DEBUG: get_paper_details_by_metadata_index: Invalid metadata index {metadata_index} or pmid not in metadata_df.") # Debug print
        return None


    def semantic_search(self, query, k=10, theme_filter=None, institution_filter=None,
                        year_filter=None, author_filter=None, journal_filter=None,
                        any_uk_filter=None, print_results=None):
        """
        Performs a semantic search using the FAISS index with optional filtering.

        Args:
            query (str): The search query string.
            k (int): The number of nearest neighbors to retrieve from FAISS.
            theme_filter (list, optional): List of themes to filter by (papers must have AT LEAST ONE of these).
            institution_filter (list, optional): List of institutions to filter by (papers must have AT LEAST ONE of these).
            year_filter (tuple or int, optional): Year or range (start_year, end_year) to filter by.
            author_filter (list, optional): List of authors to filter by (papers must have AT LEAST ONE of these).
            journal_filter (list, optional): List of journals to filter by (papers must be in AT LEAST ONE of these).
            any_uk_filter (bool, optional): Filter for papers with (True) or without (False) any UK affiliation.
            print_results (bool, optional): Whether to print the search results.


        Returns:
            list: A list of dictionaries, each representing a search result with paper details and score.
        """
        if self.index is None or self.metadata_df.empty or not self.lookups or self.model is None:
            print("Error: RAG system is not fully initialized. Cannot perform search.")
            return []

        query_embedding = self._encode_query(query)
        if query_embedding is None:
            return []

        # --- Apply Filters to get Candidate Indices ---
        candidate_indices = set(self.metadata_df.index) # Start with all indices in the metadata_df

        print(f"\nInitial candidates: {len(candidate_indices):,}")

        # Apply Theme Filter
        if theme_filter and 'theme_to_papers' in self.lookups:
            theme_candidate_indices = set()
            theme_filter_list = ensure_list(theme_filter)
            if theme_filter_list:
                theme_filter_lower = [str(t).strip().lower() for t in theme_filter_list if pd.notna(t) and str(t).strip()]
                print(f"   Applying theme filter: {theme_filter_lower}")
                for theme in theme_filter_lower:
                    # Case-insensitive matching for themes
                    matching_themes = [key for key in self.theme_to_papers.keys() if key.lower() == theme]
                    for matching_theme in matching_themes:
                         theme_candidate_indices.update(self.theme_to_papers.get(matching_theme, []))
                candidate_indices = candidate_indices.intersection(theme_candidate_indices)
                print(f"   Candidates after theme filter: {len(candidate_indices):,}")
            else:
                 print("   ⚠️  Theme filter list is empty or invalid.")


        # Apply Institution Filter
        if institution_filter and 'institution_to_papers' in self.lookups:
            institution_candidate_indices = set()
            institution_filter_list = ensure_list(institution_filter)
            if institution_filter_list:
                institution_filter_lower = [str(i).strip().lower() for i in institution_filter_list if pd.notna(i) and str(i).strip()]
                print(f"   Applying institution filter: {institution_filter_lower}")
                for institution in institution_filter_lower:
                    # Case-insensitive matching for institutions
                    matching_institutions = [key for key in self.institution_to_papers.keys() if key.lower() == institution]
                    for matching_institution in matching_institutions:
                         institution_candidate_indices.update(self.institution_to_papers.get(matching_institution, []))
                candidate_indices = candidate_indices.intersection(institution_candidate_indices)
                print(f"   Candidates after institution filter: {len(candidate_indices):,}")
            else:
                 print("   ⚠️  Institution filter list is empty or invalid.")


        # Apply Year Filter
        if year_filter and 'year_to_papers' in self.lookups:
            year_candidate_indices = set()
            years_to_include = set()
            if isinstance(year_filter, int) and pd.notna(year_filter):
                years_to_include.add(int(year_filter))
                print(f"   Applying year filter: {int(year_filter)}")
            elif isinstance(year_filter, tuple) and len(year_filter) == 2:
                try:
                    start_year, end_year = sorted([int(y) for y in year_filter if pd.notna(y)])
                    years_to_include.update(range(start_year, end_year + 1))
                    print(f"   Applying year filter range: {start_year}-{end_year}")
                except ValueError:
                     print(f"   ⚠️  Invalid year filter format for range: {year_filter}")
            else:
                print(f"   ⚠️  Invalid year filter format: {year_filter}")

            if years_to_include:
                for year in years_to_include:
                    year_candidate_indices.update(self.year_to_papers.get(year, []))

                candidate_indices = candidate_indices.intersection(year_candidate_indices)
                print(f"   Candidates after year filter: {len(candidate_indices):,}")
            else:
                 print("   ⚠️  Year filter resulted in no valid years to include.")


        # Apply Author Filter
        if author_filter and 'author_to_papers' in self.lookups:
            author_candidate_indices = set()
            author_filter_list = ensure_list(author_filter)
            if author_filter_list:
                author_filter_lower = [str(a).strip().lower() for a in author_filter_list if pd.notna(a) and str(a).strip()]
                print(f"   Applying author filter: {author_filter_lower}")
                for author in author_filter_lower:
                    # Case-insensitive matching for authors
                    matching_authors = [key for key in self.author_to_papers.keys() if key.lower() == author]
                    for matching_author in matching_authors:
                         author_candidate_indices.update(self.author_to_papers.get(matching_author, []))
                candidate_indices = candidate_indices.intersection(author_candidate_indices)
                print(f"   Candidates after author filter: {len(candidate_indices):,}")
            else:
                print("   ⚠️  Author filter list is empty or invalid.")


        # Apply Journal Filter
        if journal_filter and 'journal_to_papers' in self.lookups:
            journal_candidate_indices = set()
            journal_filter_list = ensure_list(journal_filter)
            if journal_filter_list:
                journal_filter_lower = [str(j).strip().lower() for j in journal_filter_list if pd.notna(j) and str(j).strip()]
                print(f"   Applying journal filter: {journal_filter_lower}")
                for journal in journal_filter_lower:
                    # Case-insensitive matching for journals
                    matching_journals = [key for key in self.journal_to_papers.keys() if key.lower() == journal]
                    for matching_journal in matching_journals:
                        journal_candidate_indices.update(self.journal_to_papers.get(matching_journal, []))
                candidate_indices = candidate_indices.intersection(journal_candidate_indices)
                print(f"   Candidates after journal filter: {len(candidate_indices):,}")
            else:
                print("   ⚠️  Journal filter list is empty or invalid.")


        # Apply Any UK Filter
        if any_uk_filter is not None and 'any_uk' in self.metadata_df.columns:
            if isinstance(any_uk_filter, bool):
                print(f"   Applying any_uk filter: {any_uk_filter}")
                uk_filtered_indices = set(self.metadata_df[self.metadata_df['any_uk'] == any_uk_filter].index)
                candidate_indices = candidate_indices.intersection(uk_filtered_indices)
                print(f"   Candidates after any_uk filter: {len(candidate_indices):,}")
            else:
                print(f"   ⚠️  Invalid any_uk filter format: {any_uk_filter} (must be boolean)")


        # --- Perform FAISS Search ---
        # If no candidates left after filtering, return empty results
        if not candidate_indices:
            print("No candidates found after applying filters.")
            return []

        # Get the actual embeddings for the candidate indices
        candidate_indices_list = sorted(list(candidate_indices)) # FAISS requires sorted indices for sub-indexing
        # It's more efficient to create a sub-index if filtering significantly reduces candidates
        if len(candidate_indices_list) < self.index.ntotal * 0.1 and len(candidate_indices_list) >= k: # Threshold for creating sub-index
             print(f"   Creating temporary sub-index for {len(candidate_indices_list):,} candidates...")
             try:
                 candidate_embeddings = self.index.reconstruct_n(len(candidate_indices_list), np.array(candidate_indices_list))
                 temp_index = faiss.IndexFlatIP(self.index.d)
                 temp_index.add(candidate_embeddings)
                 # Search the temporary index
                 D_temp, I_temp = temp_index.search(query_embedding, k=min(k, len(candidate_indices_list)))
                 # Map temporary indices back to original metadata_df indices
                 original_metadata_indices = [candidate_indices_list[i] for i in I_temp[0]]
                 scores = D_temp[0]
             except Exception as e:
                  print(f"   ⚠️  Error during temporary sub-index search: {e}. Falling back to main index search.")
                  # Fallback to main index search if sub-indexing fails
                  D_main, I_main = self.index.search(query_embedding, k=max(k * 10, 100)) # Search deeper in main index
                  # Filter results based on candidate indices
                  filtered_results = [(score, index) for score, index in zip(D_main[0], I_main[0]) if index in candidate_indices]
                  # Sort by score and take top k
                  filtered_results.sort(key=lambda x: x[0], reverse=True)
                  top_k_filtered = filtered_results[:k]
                  scores = np.array([score for score, index in top_k_filtered])
                  original_metadata_indices = [index for score, index in top_k_filtered]

        else:
            # If filtering didn't reduce candidates much, search the main index and filter results
            print("   Searching the main index and filtering results...")
            D_main, I_main = self.index.search(query_embedding, k=max(k * 10, 100)) # Search deeper in main index

            # Filter results based on candidate indices
            filtered_results = [(score, index) for score, index in zip(D_main[0], I_main[0]) if index in candidate_indices]

            # Sort by score and take top k
            filtered_results.sort(key=lambda x: x[0], reverse=True)
            top_k_filtered = filtered_results[:k]

            scores = np.array([score for score, index in top_k_filtered])
            original_metadata_indices = [index for score, index in top_k_filtered]


        print(f"   FAISS search completed. Found {len(original_metadata_indices)} results.")


        # --- Format Results ---
        results = []
        # Retrieve full paper details from the *master_df* using metadata indices
        # Use the helper function get_paper_details_by_metadata_index
        for i, original_metadata_index in enumerate(original_metadata_indices):
            paper_details = self.get_paper_details_by_metadata_index(original_metadata_index)
            if paper_details:
                # Add similarity score
                paper_details['similarity_score'] = float(scores[i])
                paper_details['metadata_index'] = int(original_metadata_index) # Index in the metadata/embeddings df
                results.append(paper_details)
            # else: Warning handled inside get_paper_details_by_metadata_index


        # Sort results by similarity score in descending order (already sorted if using sub-index/main index filtering, but good practice)
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)

        if print_results:
          print("\n--- Search Results Sample ---")
          if results:
              for i, res in enumerate(results[:min(5, len(results))]):
                  print(f"{i+1}. Title: {res.get('title', 'N/A')}")
                  print(f"   Score: {res.get('similarity_score', 0):.3f}")
                  print(f"   PMID: {res.get('pmid', 'N/A')}")
                  print(f"   Year: {res.get('year', 'N/A')}")
                  themes_display = ', '.join(ensure_list(res.get('themes', []))[:5])
                  print(f"   Themes: {themes_display}{'...' if len(ensure_list(res.get('themes', []))) > 5 else ''}")
                  institutions_display = ', '.join(ensure_list(res.get('institutions', []))[:5])
                  print(f"   Institutions: {institutions_display}{'...' if len(ensure_list(res.get('institutions', []))) > 5 else ''}")
                  authors_display = ', '.join(ensure_list(res.get('authors', []))[:5])
                  print(f"   Authors: {authors_display}{'...' if len(ensure_list(res.get('authors', []))) > 5 else ''}")
                  print(f"   Journal: {res.get('journal', 'N/A')}")
                  print("-" * 20)
          else:
              print("No results found.")


        return results

    # --- Getter Methods leveraging Lookup Dictionaries ---

    def get_paper_details_by_pmid(self, pmid):
        """Retrieves full details for a paper given its PMID from the master_df."""
        # This function is already implemented and used internally, making it public here
        if self.master_df.empty:
             print("Master DataFrame not loaded.")
             return None
        pmid_str = str(pmid).strip()
        if pmid_str in self.master_pmid_to_index:
            master_df_index = self.master_pmid_to_index[pmid_str]
            paper_details = self.master_df.iloc[master_df_index].to_dict()
            # Ensure list columns are lists
            for col in ['authors', 'institutions', 'themes']:
                 if col in paper_details:
                      paper_details[col] = ensure_list(paper_details[col])
            # Ensure boolean 'any_uk'
            if 'any_uk' in paper_details:
                 paper_details['any_uk'] = bool(paper_details['any_uk'])
            # Ensure year is nullable integer
            if 'year' in paper_details and pd.notna(paper_details['year']):
                 paper_details['year'] = int(paper_details['year'])
            else:
                 paper_details['year'] = None
            # Include journal information
            paper_details['journal'] = paper_details.get('journal', 'N/A')

            return paper_details
        else:
            print(f"Paper with PMID '{pmid}' not found in master DataFrame.")
            return None


    def get_papers_by_author(self, author_name):
        """Retrieves full details for papers by a specific author."""
        if 'author_to_papers' in self.lookups:
            author_name_lower = str(author_name).strip().lower()
            matching_authors = [key for key in self.author_to_papers.keys() if key.lower() == author_name_lower]
            if matching_authors:
                paper_indices = set()
                for matching_author in matching_authors:
                    paper_indices.update(self.author_to_papers.get(matching_author, []))
                # Retrieve full details from master_df using the helper
                results = [self.get_paper_details_by_metadata_index(i) for i in sorted(list(paper_indices))]
                return [res for res in results if res is not None] # Filter out any None results
            else:
                print(f"No papers found for author '{author_name}'.")
                return []
        else:
            print("Author to papers lookup not available.")
            return []

    def get_authors_by_institution(self, institution_name):
        """Retrieves authors affiliated with a specific institution."""
        if 'institution_to_authors' in self.lookups:
            institution_name_lower = str(institution_name).strip().lower()
            matching_institutions = [key for key in self.institution_to_authors.keys() if key.lower() == institution_name_lower]
            if matching_institutions:
                authors_set = set()
                for matching_institution in matching_institutions:
                     authors_set.update(self.institution_to_authors.get(matching_institution, []))
                return sorted(list(authors_set))
            else:
                print(f"No authors found for institution '{institution_name}'.")
                return []
        else:
            print("Institution to authors lookup not available.")
            return []

    def get_papers_by_year(self, year):
        """Retrieves full details for papers published in a specific year."""
        if 'year_to_papers' in self.lookups and pd.notna(year):
            try:
                year_int = int(year)
                if year_int in self.year_to_papers:
                    paper_indices = self.year_to_papers[year_int]
                    # Retrieve full details from master_df using the helper
                    results = [self.get_paper_details_by_metadata_index(i) for i in sorted(list(paper_indices))]
                    return [res for res in results if res is not None]
                else:
                    print(f"No papers found for year {year_int}.")
                    return []
            except ValueError:
                print(f"Invalid year format: {year}")
                return []
        else:
            print("Year to papers lookup not available or invalid year.")
            return []

    def get_papers_by_journal(self, journal_name):
         """Retrieves full details for papers published in a specific journal."""
         if 'journal_to_papers' in self.lookups:
             journal_name_lower = str(journal_name).strip().lower()
             matching_journals = [key for key in self.journal_to_papers.keys() if key.lower() == journal_name_lower]
             if matching_journals:
                 paper_indices = set()
                 for matching_journal in matching_journals:
                      paper_indices.update(self.journal_to_papers.get(matching_journal, []))
                 # Retrieve full details from master_df using the helper
                 results = [self.get_paper_details_by_metadata_index(i) for i in sorted(list(paper_indices))]
                 return [res for res in results if res is not None]
             else:
                 print(f"No papers found for journal '{journal_name}'.")
                 return []
         else:
             print("Journal to papers lookup not available.")
             return []

    def get_authors_by_theme(self, theme_name):
        """Retrieves authors who have published on a specific theme."""
        if 'theme_to_authors' in self.lookups:
            theme_name_lower = str(theme_name).strip().lower()
            matching_themes = [key for key in self.theme_to_authors.keys() if key.lower() == theme_name_lower]
            if matching_themes:
                authors_set = set()
                for matching_theme in matching_themes:
                    authors_set.update(self.theme_to_authors.get(matching_theme, []))
                return sorted(list(authors_set))
            else:
                print(f"No authors found for theme '{theme_name}'.")
                return []
        else:
            print("Theme to authors lookup not available.")
            return []

    def get_papers_by_theme(self, theme_name):
        """Retrieves full details for papers associated with a specific theme."""
        if 'theme_to_papers' in self.lookups:
            theme_name_lower = str(theme_name).strip().lower()
            matching_themes = [key for key in self.theme_to_papers.keys() if key.lower() == theme_name_lower]
            if matching_themes:
                paper_indices = set()
                for matching_theme in matching_themes:
                    paper_indices.update(self.theme_to_papers.get(matching_theme, []))
                # Retrieve full details from master_df using the helper
                results = [self.get_paper_details_by_metadata_index(i) for i in sorted(list(paper_indices))]
                return [res for res in results if res is not None]
            else:
                print(f"No papers found for theme '{theme_name}'.")
                return []
        else:
            print("Theme to papers lookup not available.")
            return []

    # Add more methods for other lookups like get_papers_by_institution, get_papers_by_city etc. following the pattern



# --- Step 4: Instantiate the Themes RAG System ---
# Make sure the paths defined at the top of the cell are correct for your setup

print("\nInstantiating ThemesRAGSystem_Revised...")
themes_rag_system = ThemesRAGSystem_Revised(
    master_data_path=master_data_path,
    metadata_path=metadata_path,
    faiss_index_path=faiss_index_path,
    lookups_path=lookups_path
)

print("\nTHEMES RAG System is ready for use.")
print("You can now use the 'themes_rag_system' object to perform searches and lookups.")
# Example:
# search_results = themes_rag_system.semantic_search("CRISPR applications in medicine", k=5, year_filter=(2020, 2024))
# papers_by_author = themes_rag_system.get_papers_by_author("Smith J")