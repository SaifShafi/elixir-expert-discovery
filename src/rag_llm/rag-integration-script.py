"""
Integration script to connect your enhanced theme classification with RAG.
This bridges your existing code with the enhanced RAG system.
"""

import numpy as np
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from collections import defaultdict

# Import the Enhanced RAG class from the previous artifact
# You would normally import this from the enhanced-rag-integration file
# from enhanced_rag_integration import EnhancedElixirRAG

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
    
    def semantic_search(self, query: str, k: int = 10, **kwargs):
        """Basic semantic search - implement the full version from the other artifact."""
        # This is a simplified version - use the full implementation from enhanced-rag-integration
        query_embedding = self._embed_query(query)
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.docs_with_themes):
                row = self.docs_with_themes.iloc[idx]
                results.append({
                    'pmid': row['pmid'],
                    'title': row.get('title', 'No title'),
                    'themes': row.get('themes_pred', []),
                    'similarity_score': float(score)
                })
        return results
    
    def query_with_context(self, query: str):
        """Basic query with context - implement the full version."""
        return {'semantic_results': self.semantic_search(query), 'analysis': {}}
    
    def get_theme_experts(self, theme: str, min_papers: int = 3):
        """Get experts for a theme - implement the full version."""
        return []
    
    def analyze_collaboration_gaps(self, theme: str):
        """Analyze collaboration gaps - implement the full version.""" 
        return []
    
    def analyze_publication_trends(self, theme: str):
        """Analyze publication trends - implement the full version."""
        return {}
    
    def _embed_query(self, query: str):
        """Placeholder for query embedding."""
        raise NotImplementedError("This will be implemented by the integrator")

class ElixirRAGIntegrator:
    """
    Integrates your existing BioBERT embedding function with the enhanced RAG system.
    """
    
    def __init__(self, model_name="dmis-lab/biobert-v1.1"):
        """Initialize BioBERT for query embedding."""
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a query using BioBERT - matches your existing embedding approach.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Normalize (to match your document embeddings)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        return embedding.astype("float32").flatten()

def setup_complete_rag_system(base_path: str):
    """
    Complete setup function that connects everything together.
    
    Args:
        base_path: Path to your outputs directory
    """
    
    # Paths based on your project structure
    docs_df_path = f"{base_path}/seeds_approach/seeds_biobert_embeddings_outputs/seeds_meta_plus_authors.parquet"
    docs_embeddings_path = f"{base_path}/seeds_approach/seeds_biobert_embeddings_outputs/seeds_embeddings.npy"
    theme_classified_path = f"{base_path}/seeds_approach/seeds_theme_rebuild_and_tag_outputs/seeds_with_themes_diagnosed_v2.parquet"
    faiss_index_path = f"{base_path}/seeds_approach/seeds_biobert_embeddings_outputs/seeds_faiss.index"
    
    # Initialize the BioBERT integrator
    print("Setting up BioBERT integration...")
    integrator = ElixirRAGIntegrator()
    
    # Create the enhanced RAG system
    print("Setting up enhanced RAG system...")
    
    class ConnectedElixirRAG(EnhancedElixirRAG):
        """Enhanced RAG with BioBERT integration."""
        
        def __init__(self, integrator, *args, **kwargs):
            self.integrator = integrator
            super().__init__(*args, **kwargs)
        
        def _embed_query(self, query: str) -> np.ndarray:
            """Use the BioBERT integrator for query embedding."""
            return self.integrator.embed_query(query)
    
    # Initialize connected RAG
    rag = ConnectedElixirRAG(
        integrator=integrator,
        docs_df_path=docs_df_path,
        docs_embeddings_path=docs_embeddings_path,
        theme_classified_path=theme_classified_path,
        faiss_index_path=faiss_index_path
    )
    
    return rag

def answer_research_questions(rag):
    """
    Function to answer your specific research questions using the enhanced RAG.
    """
    
    questions_and_answers = {}
    
    # 1. Galaxy experts
    print("🔍 Finding Galaxy experts...")
    galaxy_results = rag.query_with_context("Who in the UK community is a Galaxy expert?")
    questions_and_answers["galaxy_experts"] = galaxy_results
    
    # 2. Collaboration gaps
    print("🔍 Analyzing collaboration gaps...")
    collab_results = rag.analyze_collaboration_gaps("Microbiome", min_similarity=0.8)
    questions_and_answers["collaboration_gaps"] = collab_results
    
    # 3. LLM research
    print("🔍 Finding LLM research...")
    llm_results = rag.semantic_search("large language models biomedicine AI", k=10)
    questions_and_answers["llm_research"] = llm_results
    
    # 4. Quantum research
    print("🔍 Finding quantum research...")
    quantum_results = rag.semantic_search("quantum computing biology quantum biology", k=10)
    questions_and_answers["quantum_research"] = quantum_results
    
    # 5. Human gut microbiome
    print("🔍 Finding human gut microbiome research...")
    gut_results = rag.semantic_search("human gut microbiome", k=15, theme_filter=["Microbiome"])
    questions_and_answers["gut_microbiome"] = gut_results
    
    # 6. Nature papers (last 5 years)
    print("🔍 Finding recent Nature papers...")
    nature_results = rag.semantic_search("", k=50, year_range=(2019, 2024))
    nature_papers = [r for r in nature_results if 'nature' in r['journal'].lower()]
    questions_and_answers["nature_papers"] = nature_papers[:20]
    
    # 7. Geographic analysis
    print("🔍 Analyzing geographic distribution...")
    geographic_analysis = {}
    for institution in ['Cambridge', 'Oxford', 'Edinburgh', 'Manchester', 'Imperial', 'UCL']:
        inst_papers = len(rag.institution_to_papers.get(institution, []))
        geographic_analysis[institution] = inst_papers
    questions_and_answers["geographic_distribution"] = geographic_analysis
    
    # 8. Theme experts for each community
    print("🔍 Finding experts for each theme...")
    theme_experts = {}
    for theme in ['Galaxy', 'Microbiome', 'Proteomics', 'Cancer Data', '3D-BioInfo']:
        experts = rag.get_theme_experts(theme, min_papers=3)
        theme_experts[theme] = experts[:10]  # Top 10 experts
    questions_and_answers["theme_experts"] = theme_experts
    
    # 9. Publication trends
    print("🔍 Analyzing publication trends...")
    trends = {}
    for theme in ['Galaxy', 'Microbiome', 'Single-Cell Omics', 'Proteomics']:
        trend_analysis = rag.analyze_publication_trends(theme)
        trends[theme] = trend_analysis
    questions_and_answers["publication_trends"] = trends
    
    return questions_and_answers

def generate_insights_report(questions_and_answers: dict) -> str:
    """
    Generate a comprehensive insights report from the analysis.
    """
    
    report = "# ELIXIR-UK Research Analysis Report\n\n"
    
    # Galaxy experts
    if "galaxy_experts" in questions_and_answers:
        experts = questions_and_answers["galaxy_experts"].get("analysis", {}).get("experts", [])
        report += "## Galaxy Community Experts\n\n"
        for expert in experts[:5]:
            report += f"- **{expert['name']}**: {expert['paper_count']} papers, most recent: {expert['most_recent_year']}\n"
        report += "\n"
    
    # Geographic distribution
    if "geographic_distribution" in questions_and_answers:
        geo = questions_and_answers["geographic_distribution"]
        report += "## Geographic Distribution\n\n"
        sorted_geo = sorted(geo.items(), key=lambda x: x[1], reverse=True)
        for inst, count in sorted_geo:
            percentage = (count / sum(geo.values())) * 100 if sum(geo.values()) > 0 else 0
            report += f"- **{inst}**: {count} papers ({percentage:.1f}%)\n"
        report += "\n"
    
    # Nature papers
    if "nature_papers" in questions_and_answers:
        nature = questions_and_answers["nature_papers"]
        report += f"## High-Impact Publications (Nature journals)\n\n"
        report += f"Found {len(nature)} Nature papers in the last 5 years:\n\n"
        for paper in nature[:5]:
            report += f"- **{paper['title'][:80]}...** ({paper['year']})\n"
            report += f"  - Journal: {paper['journal']}\n"
            report += f"  - Themes: {', '.join(paper['themes'])}\n\n"
    
    # Collaboration gaps
    if "collaboration_gaps" in questions_and_answers:
        collabs = questions_and_answers["collaboration_gaps"]
        report += "## Potential Collaboration Opportunities\n\n"
        for collab in collabs[:3]:
            report += f"- **{collab['institution1']} ↔ {collab['institution2']}** in {collab['theme']}\n"
            report += f"  - Synergy score: {collab['potential_synergy']:.2f}\n\n"
    
    # Publication trends
    if "publication_trends" in questions_and_answers:
        trends = questions_and_answers["publication_trends"]
        report += "## Publication Trends (2019-2024)\n\n"
        for theme, trend_data in trends.items():
            report += f"### {theme}\n"
            report += f"- Total papers: {trend_data['total_papers']}\n"
            report += f"- Trend: {trend_data['trend_analysis']}\n"
            if trend_data['top_journals']:
                report += f"- Top journal: {trend_data['top_journals'][0][0]} ({trend_data['top_journals'][0][1]} papers)\n"
            report += "\n"
    
    return report

# Main execution function
def run_complete_analysis(base_path: str):
    """
    Run the complete analysis pipeline.
    """
    
    print("🚀 Starting complete ELIXIR-UK analysis...")
    
    # Setup RAG system
    rag = setup_complete_rag_system(base_path)
    
    # Answer research questions
    print("\n📊 Answering research questions...")
    results = answer_research_questions(rag)
    
    # Generate report
    print("\n📝 Generating insights report...")
    report = generate_insights_report(results)
    
    # Save results
    import json
    output_path = f"{base_path}/analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    report_path = f"{base_path}/analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Analysis complete!")
    print(f"📄 Results saved to: {output_path}")
    print(f"📄 Report saved to: {report_path}")
    
    return rag, results, report

# Example usage and testing functions
def test_specific_queries(rag):
    """
    Test specific queries mentioned in your requirements.
    """
    
    test_queries = [
        "Who should I be asking to work with on cancer data?",
        "Are there people who sped up publishing or slowed down?", 
        "Who's working on human data and privacy?",
        "What's the gendered demographic of each community?",
        "Is anything coming out of Scotland?",
        "Give me the top 5 journals published in for microbiome research",
        "Which researchers changed fields recently?",
        "Who in the Galaxy community should be collaborating but aren't?"
    ]
    
    results = {}
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = rag.query_with_context(query)
            results[query] = result
            
            # Print summary
            semantic_results = result.get('semantic_results', [])
            analysis = result.get('analysis', {})
            
            print(f"   Found {len(semantic_results)} semantic matches")
            if 'experts' in analysis:
                print(f"   Found {len(analysis['experts'])} experts")
            if 'collaboration_gaps' in analysis:
                print(f"   Found {len(analysis['collaboration_gaps'])} collaboration opportunities")
            if 'trends' in analysis:
                print(f"   Trend analysis: {analysis['trends'].get('trend_analysis', 'N/A')}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[query] = {'error': str(e)}
    
    return results

def analyze_community_demographics(rag):
    """
    Analyze demographics across ELIXIR communities.
    """
    
    demographics = {}
    
    # Gender analysis (simplified - you'd need a proper name-to-gender mapping)
    gender_patterns = {
        'likely_female': ['sarah', 'emily', 'jennifer', 'lisa', 'maria', 'anna', 'claire', 'helen', 'jane', 'carol'],
        'likely_male': ['john', 'david', 'michael', 'james', 'robert', 'mark', 'paul', 'andrew', 'simon', 'peter']
    }
    
    for theme in rag.theme_to_papers.keys():
        if len(rag.theme_to_papers[theme]) > 50:  # Only analyze themes with sufficient data
            
            theme_papers = rag.docs_with_themes.iloc[rag.theme_to_papers[theme]]
            
            # Gender analysis
            male_count, female_count, unknown_count = 0, 0, 0
            
            # Geographic analysis
            geo_counts = {'Scotland': 0, 'England': 0, 'Wales': 0, 'Northern Ireland': 0, 'Unknown': 0}
            
            # Institution analysis
            inst_counts = {}
            
            for _, paper in theme_papers.iterrows():
                # Gender analysis
                if 'authors' in paper and pd.notna(paper['authors']):
                    authors = paper['authors'] if isinstance(paper['authors'], list) else []
                    for author in authors:
                        first_name = author.split()[0].lower() if author else ''
                        if any(name in first_name for name in gender_patterns['likely_female']):
                            female_count += 1
                        elif any(name in first_name for name in gender_patterns['likely_male']):
                            male_count += 1
                        else:
                            unknown_count += 1
                
                # Geographic analysis
                if 'affiliation' in paper and pd.notna(paper['affiliation']):
                    affiliation = str(paper['affiliation']).lower()
                    if any(word in affiliation for word in ['edinburgh', 'glasgow', 'stirling', 'aberdeen', 'dundee']):
                        geo_counts['Scotland'] += 1
                    elif any(word in affiliation for word in ['cambridge', 'oxford', 'london', 'manchester', 'birmingham']):
                        geo_counts['England'] += 1
                    elif any(word in affiliation for word in ['cardiff', 'swansea', 'bangor']):
                        geo_counts['Wales'] += 1
                    else:
                        geo_counts['Unknown'] += 1
                
                # Institution analysis
                if 'affiliation' in paper and pd.notna(paper['affiliation']):
                    # Extract main institution name (simplified)
                    affiliation = str(paper['affiliation'])
                    for inst in ['Cambridge', 'Oxford', 'Imperial', 'Edinburgh', 'Manchester', 'UCL', 'King\'s']:
                        if inst.lower() in affiliation.lower():
                            inst_counts[inst] = inst_counts.get(inst, 0) + 1
                            break
            
            total_authors = male_count + female_count + unknown_count
            total_geo = sum(geo_counts.values())
            
            demographics[theme] = {
                'total_papers': len(theme_papers),
                'gender': {
                    'male_pct': (male_count / total_authors * 100) if total_authors > 0 else 0,
                    'female_pct': (female_count / total_authors * 100) if total_authors > 0 else 0,
                    'unknown_pct': (unknown_count / total_authors * 100) if total_authors > 0 else 0
                },
                'geography': {
                    'scotland_pct': (geo_counts['Scotland'] / total_geo * 100) if total_geo > 0 else 0,
                    'england_pct': (geo_counts['England'] / total_geo * 100) if total_geo > 0 else 0,
                    'wales_pct': (geo_counts['Wales'] / total_geo * 100) if total_geo > 0 else 0
                },
                'top_institutions': sorted(inst_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            }
    
    return demographics

def find_field_changers(rag, years_back: int = 3):
    """
    Identify researchers who have changed fields/topics.
    """
    
    current_year = 2024
    cutoff_year = current_year - years_back
    
    # Group papers by author and time period
    author_themes = {}
    
    for _, paper in rag.docs_with_themes.iterrows():
        if 'authors' in paper and pd.notna(paper['authors']) and 'year' in paper and pd.notna(paper['year']):
            authors = paper['authors'] if isinstance(paper['authors'], list) else []
            year = paper['year']
            themes = paper.get('themes_pred', [])
            
            for author in authors:
                if author not in author_themes:
                    author_themes[author] = {'early': set(), 'recent': set(), 'papers': []}
                
                author_themes[author]['papers'].append({
                    'year': year,
                    'themes': themes,
                    'title': paper.get('title', ''),
                    'pmid': paper.get('pmid')
                })
                
                if year >= cutoff_year:
                    author_themes[author]['recent'].update(themes)
                else:
                    author_themes[author]['early'].update(themes)
    
    # Find authors who changed themes
    field_changers = []
    
    for author, data in author_themes.items():
        if len(data['early']) > 0 and len(data['recent']) > 0:
            # Calculate theme overlap
            overlap = len(data['early'].intersection(data['recent']))
            total_themes = len(data['early'].union(data['recent']))
            
            if total_themes > 0:
                overlap_ratio = overlap / total_themes
                
                # Consider someone a "field changer" if less than 50% theme overlap
                if overlap_ratio < 0.5 and len(data['papers']) >= 5:
                    field_changers.append({
                        'author': author,
                        'early_themes': list(data['early']),
                        'recent_themes': list(data['recent']),
                        'overlap_ratio': overlap_ratio,
                        'total_papers': len(data['papers']),
                        'transition': f"{', '.join(data['early'])} → {', '.join(data['recent'])}"
                    })
    
    # Sort by most dramatic changes (lowest overlap ratio)
    field_changers.sort(key=lambda x: x['overlap_ratio'])
    
    return field_changers

# Complete workflow function
def complete_elixir_analysis_workflow(base_path: str):
    """
    Complete workflow that runs all analyses and generates comprehensive outputs.
    """
    
    print("🚀 Starting Complete ELIXIR-UK Analysis Workflow")
    print("="*60)
    
    # Step 1: Setup RAG system
    print("\n📋 Step 1: Setting up Enhanced RAG System...")
    rag = setup_complete_rag_system(base_path)
    
    # Step 2: Run main analysis
    print("\n📋 Step 2: Running Main Research Questions Analysis...")
    main_results = answer_research_questions(rag)
    
    # Step 3: Test specific queries
    print("\n📋 Step 3: Testing Specific Query Types...")
    query_results = test_specific_queries(rag)
    
    # Step 4: Demographic analysis
    print("\n📋 Step 4: Analyzing Community Demographics...")
    demographics = analyze_community_demographics(rag)
    
    # Step 5: Find field changers
    print("\n📋 Step 5: Identifying Field Changers...")
    field_changers = find_field_changers(rag)
    
    # Step 6: Generate comprehensive report
    print("\n📋 Step 6: Generating Comprehensive Report...")
    
    # Combine all results
    comprehensive_results = {
        'main_analysis': main_results,
        'query_tests': query_results,
        'demographics': demographics,
        'field_changers': field_changers[:20],  # Top 20 field changers
        'metadata': {
            'total_papers': len(rag.docs_with_themes),
            'total_themes': len(rag.theme_to_papers),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    # Generate detailed report
    report = generate_comprehensive_report(comprehensive_results)
    
    # Save everything
    results_path = f"{base_path}/comprehensive_analysis_results.json"
    report_path = f"{base_path}/comprehensive_analysis_report.md"
    
    with open(results_path, 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Complete Analysis Finished!")
    print(f"📊 Results: {results_path}")
    print(f"📄 Report: {report_path}")
    print(f"🔍 Analyzed {len(rag.docs_with_themes)} papers across {len(rag.theme_to_papers)} themes")
    
    return rag, comprehensive_results

def generate_comprehensive_report(results: dict) -> str:
    """
    Generate a comprehensive markdown report from all analyses.
    """
    
    report = """# Comprehensive ELIXIR-UK Research Analysis Report

## Executive Summary

This report provides a comprehensive analysis of UK research aligned with ELIXIR communities, based on enhanced theme classification using BioBERT embeddings and semantic similarity analysis.

"""
    
    # Add metadata
    metadata = results.get('metadata', {})
    report += f"""### Analysis Scope
- **Total Papers Analyzed**: {metadata.get('total_papers', 'N/A'):,}
- **ELIXIR Communities**: {metadata.get('total_themes', 'N/A')}
- **Analysis Date**: {metadata.get('analysis_date', 'N/A')}

---

"""
    
    # Demographics section
    demographics = results.get('demographics', {})
    if demographics:
        report += "## Community Demographics\n\n"
        
        report += "### Gender Distribution by Community\n\n"
        for theme, demo in demographics.items():
            if demo['total_papers'] > 100:  # Only show major communities
                gender = demo['gender']
                report += f"**{theme}** ({demo['total_papers']} papers):\n"
                report += f"- Female: {gender['female_pct']:.1f}%\n"
                report += f"- Male: {gender['male_pct']:.1f}%\n"
                report += f"- Unknown: {gender['unknown_pct']:.1f}%\n\n"
        
        report += "### Geographic Distribution\n\n"
        scotland_themes = [(theme, demo['geography']['scotland_pct']) 
                          for theme, demo in demographics.items() 
                          if demo['geography']['scotland_pct'] > 10]
        scotland_themes.sort(key=lambda x: x[1], reverse=True)
        
        report += "**Themes with Strong Scottish Representation (>10%):**\n\n"
        for theme, pct in scotland_themes[:5]:
            report += f"- {theme}: {pct:.1f}%\n"
        
        report += "\n---\n\n"
    
    # Field changers section
    field_changers = results.get('field_changers', [])
    if field_changers:
        report += "## Researchers Who Changed Fields\n\n"
        report += "Researchers who have significantly shifted their research themes in recent years:\n\n"
        
        for changer in field_changers[:10]:
            report += f"**{changer['author']}** ({changer['total_papers']} papers)\n"
            report += f"- Transition: {changer['transition']}\n"
            report += f"- Theme overlap: {changer['overlap_ratio']:.1%}\n\n"
        
        report += "\n---\n\n"
    
    # Main analysis results
    main_analysis = results.get('main_analysis', {})
    if main_analysis:
        
        # Geographic distribution
        geo_dist = main_analysis.get('geographic_distribution', {})
        if geo_dist:
            report += "## Institutional Distribution\n\n"
            total_papers = sum(geo_dist.values())
            for inst, count in sorted(geo_dist.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_papers * 100) if total_papers > 0 else 0
                report += f"- **{inst}**: {count:,} papers ({pct:.1f}%)\n"
            report += "\n"
        
        # Nature papers
        nature_papers = main_analysis.get('nature_papers', [])
        if nature_papers:
            report += f"## High-Impact Publications\n\n"
            report += f"Recent Nature journal publications ({len(nature_papers)} found):\n\n"
            for paper in nature_papers[:10]:
                report += f"**{paper['title'][:100]}...** ({paper['year']})\n"
                report += f"- Journal: {paper['journal']}\n"
                report += f"- Themes: {', '.join(paper['themes'][:3])}\n"
                report += f"- [PubMed]({paper['url']})\n\n"
        
        # Publication trends
        trends = main_analysis.get('publication_trends', {})
        if trends:
            report += "## Publication Trends (2019-2024)\n\n"
            for theme, trend_data in trends.items():
                report += f"### {theme}\n"
                report += f"- **Total Papers**: {trend_data['total_papers']:,}\n"
                report += f"- **Trend**: {trend_data['trend_analysis'].title()}\n"
                
                if trend_data.get('top_journals'):
                    top_journal = trend_data['top_journals'][0]
                    report += f"- **Top Journal**: {top_journal[0]} ({top_journal[1]} papers)\n"
                
                year_dist = trend_data.get('year_distribution', {})
                if year_dist:
                    recent_years = [str(year) for year in sorted(year_dist.keys())[-3:]]
                    report += f"- **Recent Years**: {', '.join(recent_years)}\n"
                
                report += "\n"
    
    report += "\n---\n\n"
    report += "## Methodology\n\n"
    report += """This analysis used:
1. **Enhanced Theme Classification**: BioBERT embeddings with improved theme descriptions
2. **Principal Component Removal**: Debiasing to prevent theme collapse
3. **Percentile-based Thresholds**: 97th percentile for theme assignment
4. **Semantic Search**: FAISS indexing for fast similarity search
5. **Multi-modal Analysis**: Combining text analysis with metadata

The system can answer complex questions about collaboration patterns, expertise mapping, demographic distributions, and research trends across the UK ELIXIR research landscape.
"""
    
    return report

# Usage example:
if __name__ == "__main__":
    # Run the complete analysis
    BASE_PATH = "/content/drive/MyDrive/outputs"
    
    rag, results = complete_elixir_analysis_workflow(BASE_PATH)
    
    print("\n🎉 Analysis complete! You can now use the RAG system to answer questions like:")
    print("- rag.query_with_context('Who should be collaborating on microbiome research?')")
    print("- rag.get_theme_experts('Galaxy', min_papers=3)")
    print("- rag.analyze_collaboration_gaps('Proteomics')")
    print("- rag.semantic_search('quantum computing biology', k=10)")