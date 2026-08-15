import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple, Optional
import json
from collections import defaultdict

class EnhancedElixirRAG:
    """
    Enhanced RAG system that integrates your improved theme classification
    with semantic search and LLM reasoning capabilities.
    """
    
    def __init__(self, 
                 docs_df_path: str,
                 docs_embeddings_path: str,
                 theme_classified_path: str,
                 faiss_index_path: str):
        """
        Initialize the enhanced RAG system with your improved theme data.
        
        Args:
            docs_df_path: Path to original document metadata
            docs_embeddings_path: Path to BioBERT document embeddings
            theme_classified_path: Path to your v2 theme-classified data
            faiss_index_path: Path to FAISS index
        """
        print("Loading enhanced RAG components...")
        
        # Load document data
        self.docs_df = pd.read_parquet(docs_df_path)
        self.docs_X = np.load(docs_embeddings_path).astype("float32")
        
        # Load your enhanced theme classifications
        self.docs_with_themes = pd.read_parquet(theme_classified_path)
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Create theme-based indices for faster filtering
        self._build_theme_indices()
        
        # Create author and institution indices
        self._build_entity_indices()
        
        print(f"Loaded {len(self.docs_df)} papers with enhanced theme classification")
        
    def _build_theme_indices(self):
        """Build reverse indices for theme-based filtering."""
        self.theme_to_papers = defaultdict(list)
        self.paper_to_themes = {}
        
        for idx, row in self.docs_with_themes.iterrows():
            pmid = row['pmid']
            themes = row['themes_pred'] if isinstance(row['themes_pred'], list) else []
            
            self.paper_to_themes[pmid] = themes
            for theme in themes:
                self.theme_to_papers[theme].append(idx)
        
        print(f"Built theme indices for {len(self.theme_to_papers)} themes")
        
    def _build_entity_indices(self):
        """Build indices for authors and institutions."""
        self.author_to_papers = defaultdict(list)
        self.institution_to_papers = defaultdict(list)
        
        for idx, row in self.docs_with_themes.iterrows():
            # Authors (if available)
            if 'authors' in row and pd.notna(row['authors']):
                authors = row['authors'] if isinstance(row['authors'], list) else []
                for author in authors:
                    self.author_to_papers[author].append(idx)
            
            # Extract institutions from affiliations (simplified)
            if 'affiliation' in row and pd.notna(row['affiliation']):
                # Basic institution extraction - you can enhance this
                affiliation = str(row['affiliation']).lower()
                if 'cambridge' in affiliation:
                    self.institution_to_papers['Cambridge'].append(idx)
                elif 'oxford' in affiliation:
                    self.institution_to_papers['Oxford'].append(idx)
                elif 'imperial' in affiliation:
                    self.institution_to_papers['Imperial'].append(idx)
                elif 'edinburgh' in affiliation:
                    self.institution_to_papers['Edinburgh'].append(idx)
                elif 'manchester' in affiliation:
                    self.institution_to_papers['Manchester'].append(idx)
                elif 'ucl' in affiliation or 'university college london' in affiliation:
                    self.institution_to_papers['UCL'].append(idx)
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 10,
                       theme_filter: Optional[List[str]] = None,
                       institution_filter: Optional[List[str]] = None,
                       year_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Enhanced semantic search with theme and entity filtering.
        
        Args:
            query: Natural language query
            k: Number of results to return
            theme_filter: Filter by specific ELIXIR themes
            institution_filter: Filter by institutions
            year_range: Filter by publication year range (start, end)
        """
        # Get query embedding (you'll need to implement embed_query function)
        query_embedding = self._embed_query(query)
        
        # Determine candidate papers based on filters
        candidate_indices = self._get_candidate_indices(
            theme_filter, institution_filter, year_range
        )
        
        if candidate_indices is not None:
            # Search within filtered candidates
            candidate_embeddings = self.docs_X[candidate_indices]
            
            # Create temporary FAISS index for candidates
            temp_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
            temp_index.add(candidate_embeddings)
            
            # Search
            scores, indices = temp_index.search(query_embedding.reshape(1, -1), k)
            
            # Map back to original indices
            original_indices = [candidate_indices[i] for i in indices[0]]
            scores = scores[0]
        else:
            # Search all documents
            scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
            original_indices = indices[0]
            scores = scores[0]
        
        # Format results with enhanced metadata
        results = []
        for idx, score in zip(original_indices, scores):
            if idx < len(self.docs_with_themes):
                row = self.docs_with_themes.iloc[idx]
                
                result = {
                    'pmid': row['pmid'],
                    'title': row.get('title', 'No title'),
                    'journal': row.get('journal', 'Unknown journal'),
                    'year': row.get('year', 'Unknown year'),
                    'abstract': row.get('abstract', 'No abstract'),
                    'themes': row.get('themes_pred', []),
                    'theme_scores': row.get('themes_score', []),
                    'authors': row.get('authors', []),
                    'affiliation': row.get('affiliation', 'No affiliation'),
                    'similarity_score': float(score),
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/"
                }
                results.append(result)
        
        return results
    
    def _get_candidate_indices(self, 
                              theme_filter: Optional[List[str]], 
                              institution_filter: Optional[List[str]],
                              year_range: Optional[Tuple[int, int]]) -> Optional[List[int]]:
        """Get candidate paper indices based on filters."""
        candidates = None
        
        # Theme filtering
        if theme_filter:
            theme_candidates = set()
            for theme in theme_filter:
                if theme in self.theme_to_papers:
                    theme_candidates.update(self.theme_to_papers[theme])
            candidates = theme_candidates
        
        # Institution filtering
        if institution_filter:
            inst_candidates = set()
            for institution in institution_filter:
                if institution in self.institution_to_papers:
                    inst_candidates.update(self.institution_to_papers[institution])
            
            if candidates is not None:
                candidates = candidates.intersection(inst_candidates)
            else:
                candidates = inst_candidates
        
        # Year filtering
        if year_range:
            start_year, end_year = year_range
            year_mask = (self.docs_with_themes['year'] >= start_year) & \
                       (self.docs_with_themes['year'] <= end_year)
            year_candidates = set(self.docs_with_themes[year_mask].index)
            
            if candidates is not None:
                candidates = candidates.intersection(year_candidates)
            else:
                candidates = year_candidates
        
        return list(candidates) if candidates is not None else None
    
    def analyze_collaboration_gaps(self, theme: str, min_similarity: float = 0.8) -> List[Dict]:
        """
        Find researchers who should be collaborating but aren't.
        Uses your enhanced theme classification.
        """
        if theme not in self.theme_to_papers:
            return []
        
        theme_paper_indices = self.theme_to_papers[theme]
        theme_papers = self.docs_with_themes.iloc[theme_paper_indices]
        
        # Group by institution
        inst_researchers = defaultdict(list)
        for _, paper in theme_papers.iterrows():
            if 'authors' in paper and pd.notna(paper['authors']):
                authors = paper['authors'] if isinstance(paper['authors'], list) else []
                affiliation = str(paper.get('affiliation', '')).lower()
                
                # Extract institution (simplified)
                institution = 'Unknown'
                if 'cambridge' in affiliation:
                    institution = 'Cambridge'
                elif 'oxford' in affiliation:
                    institution = 'Oxford'
                elif 'imperial' in affiliation:
                    institution = 'Imperial'
                elif 'edinburgh' in affiliation:
                    institution = 'Edinburgh'
                
                for author in authors:
                    inst_researchers[institution].append({
                        'name': author,
                        'pmid': paper['pmid'],
                        'title': paper.get('title', '')
                    })
        
        # Find potential collaborations between institutions
        collaborations = []
        institutions = list(inst_researchers.keys())
        
        for i, inst1 in enumerate(institutions):
            for inst2 in institutions[i+1:]:
                if inst1 != inst2 and inst1 != 'Unknown' and inst2 != 'Unknown':
                    # Check if these institutions have collaborated
                    # (This is simplified - you'd want more sophisticated analysis)
                    
                    researchers1 = inst_researchers[inst1]
                    researchers2 = inst_researchers[inst2]
                    
                    if len(researchers1) > 0 and len(researchers2) > 0:
                        collaborations.append({
                            'institution1': inst1,
                            'institution2': inst2,
                            'researchers1': researchers1[:3],  # Sample
                            'researchers2': researchers2[:3],  # Sample
                            'theme': theme,
                            'potential_synergy': self._calculate_synergy(researchers1, researchers2)
                        })
        
        return collaborations
    
    def get_theme_experts(self, theme: str, min_papers: int = 3) -> List[Dict]:
        """
        Identify experts in a specific theme using your enhanced classification.
        """
        if theme not in self.theme_to_papers:
            return []
        
        theme_papers = self.docs_with_themes.iloc[self.theme_to_papers[theme]]
        
        # Count papers per author
        author_counts = defaultdict(lambda: {'count': 0, 'papers': [], 'recent_year': 0})
        
        for _, paper in theme_papers.iterrows():
            if 'authors' in paper and pd.notna(paper['authors']):
                authors = paper['authors'] if isinstance(paper['authors'], list) else []
                year = paper.get('year', 0)
                
                for author in authors:
                    author_counts[author]['count'] += 1
                    author_counts[author]['papers'].append({
                        'pmid': paper['pmid'],
                        'title': paper.get('title', ''),
                        'year': year
                    })
                    author_counts[author]['recent_year'] = max(
                        author_counts[author]['recent_year'], year
                    )
        
        # Filter and rank experts
        experts = []
        for author, data in author_counts.items():
            if data['count'] >= min_papers:
                experts.append({
                    'name': author,
                    'paper_count': data['count'],
                    'recent_papers': sorted(data['papers'], 
                                          key=lambda x: x['year'], reverse=True)[:5],
                    'most_recent_year': data['recent_year'],
                    'theme': theme
                })
        
        # Sort by paper count and recency
        experts.sort(key=lambda x: (x['paper_count'], x['most_recent_year']), reverse=True)
        
        return experts
    
    def analyze_publication_trends(self, theme: str, years: List[int] = None) -> Dict:
        """
        Analyze publication trends for a theme over time.
        """
        if years is None:
            years = list(range(2019, 2025))
        
        if theme not in self.theme_to_papers:
            return {'error': f'Theme {theme} not found'}
        
        theme_papers = self.docs_with_themes.iloc[self.theme_to_papers[theme]]
        
        # Count papers by year
        year_counts = {}
        journal_counts = defaultdict(int)
        
        for _, paper in theme_papers.iterrows():
            year = paper.get('year')
            if year in years:
                year_counts[year] = year_counts.get(year, 0) + 1
                
                journal = paper.get('journal', 'Unknown')
                journal_counts[journal] += 1
        
        # Top journals
        top_journals = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'theme': theme,
            'total_papers': len(theme_papers),
            'year_distribution': year_counts,
            'top_journals': top_journals,
            'trend_analysis': self._calculate_trend(year_counts, years)
        }
    
    def query_with_context(self, query: str, max_context_papers: int = 5) -> Dict:
        """
        Enhanced query function that provides rich context for LLM reasoning.
        """
        # Determine query type and filters
        query_lower = query.lower()
        theme_filter = None
        institution_filter = None
        
        # Extract themes from query
        for theme in self.theme_to_papers.keys():
            if theme.lower() in query_lower:
                theme_filter = [theme]
                break
        
        # Extract institutions from query
        institutions = ['Cambridge', 'Oxford', 'Imperial', 'Edinburgh', 'Manchester', 'UCL']
        for inst in institutions:
            if inst.lower() in query_lower:
                institution_filter = [inst]
                break
        
        # Perform semantic search
        results = self.semantic_search(
            query, 
            k=max_context_papers,
            theme_filter=theme_filter,
            institution_filter=institution_filter
        )
        
        # Add analysis based on query type
        analysis = {}
        
        if 'expert' in query_lower or 'who' in query_lower:
            if theme_filter:
                analysis['experts'] = self.get_theme_experts(theme_filter[0])
        
        if 'collaboration' in query_lower or 'working together' in query_lower:
            if theme_filter:
                analysis['collaboration_gaps'] = self.analyze_collaboration_gaps(theme_filter[0])
        
        if 'trend' in query_lower or 'over time' in query_lower:
            if theme_filter:
                analysis['trends'] = self.analyze_publication_trends(theme_filter[0])
        
        return {
            'query': query,
            'semantic_results': results,
            'analysis': analysis,
            'filters_applied': {
                'themes': theme_filter,
                'institutions': institution_filter
            }
        }
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed query using BioBERT. You'll need to implement this with your embedding function.
        """
        # Placeholder - implement with your actual embedding function
        # return embed_query(query)  # Your BioBERT function
        raise NotImplementedError("Implement with your BioBERT embedding function")
    
    def _calculate_synergy(self, researchers1: List, researchers2: List) -> float:
        """Calculate potential synergy between researcher groups."""
        # Simplified synergy calculation
        return min(len(researchers1), len(researchers2)) / max(len(researchers1), len(researchers2))
    
    def _calculate_trend(self, year_counts: Dict, years: List[int]) -> str:
        """Calculate publication trend."""
        if len(year_counts) < 2:
            return "insufficient_data"
        
        sorted_years = sorted(year_counts.keys())
        if len(sorted_years) >= 2:
            early_avg = np.mean([year_counts.get(y, 0) for y in sorted_years[:len(sorted_years)//2]])
            late_avg = np.mean([year_counts.get(y, 0) for y in sorted_years[len(sorted_years)//2:]])
            
            if late_avg > early_avg * 1.2:
                return "increasing"
            elif late_avg < early_avg * 0.8:
                return "decreasing"
            else:
                return "stable"
        
        return "unknown"

# Usage example:
def setup_enhanced_rag():
    """
    Setup function to initialize your enhanced RAG system.
    """
    rag = EnhancedElixirRAG(
        docs_df_path="/path/to/seeds_meta_plus_authors.parquet",
        docs_embeddings_path="/path/to/seeds_embeddings.npy", 
        theme_classified_path="/path/to/seeds_with_themes_diagnosed_v2.parquet",
        faiss_index_path="/path/to/seeds_faiss.index"
    )
    
    return rag

# Example queries:
def example_queries(rag):
    """
    Example of how to use the enhanced RAG system.
    """
    
    # 1. Find Galaxy experts
    galaxy_query = "Who in the UK community is a Galaxy expert?"
    results = rag.query_with_context(galaxy_query)
    
    # 2. Collaboration gaps in microbiome research
    collab_query = "Are there Cambridge and Oxford researchers who should be collaborating on microbiome research?"
    collab_results = rag.query_with_context(collab_query)
    
    # 3. LLM research trends
    llm_query = "Who's now writing papers about large language models in biomedicine?"
    llm_results = rag.semantic_search(llm_query, k=10)
    
    return results, collab_results, llm_results
