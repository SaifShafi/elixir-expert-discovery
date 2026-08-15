"""
Quick start script to immediately test your enhanced RAG system with the theme classification improvements.
This works with your existing data structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json

def quick_analysis_of_enhanced_themes(theme_classified_path: str):
    """
    Quick analysis of your enhanced theme classification results.
    
    Args:
        theme_classified_path: Path to your seeds_with_themes_diagnosed_v2.parquet file
    """
    
    print("🔍 Loading enhanced theme classification data...")
    df = pd.read_parquet(theme_classified_path)
    
    print(f"📊 Loaded {len(df)} papers with enhanced theme classification")
    print(f"📊 Columns available: {list(df.columns)}")
    
    # Basic statistics
    print("\n" + "="*60)
    print("ENHANCED THEME CLASSIFICATION ANALYSIS")
    print("="*60)
    
    # Theme distribution
    theme_counts = Counter()
    themes_per_paper = []
    
    for themes in df['themes_pred']:
        if isinstance(themes, list):
            theme_counts.update(themes)
            themes_per_paper.append(len(themes))
        else:
            themes_per_paper.append(0)
    
    print(f"\n📈 THEME DISTRIBUTION:")
    print(f"   Average themes per paper: {np.mean(themes_per_paper):.2f}")
    print(f"   Max themes per paper: {max(themes_per_paper) if themes_per_paper else 0}")
    
    print(f"\n🏆 TOP 15 THEMES:")
    for i, (theme, count) in enumerate(theme_counts.most_common(15), 1):
        percentage = (count / len(df)) * 100
        print(f"   {i:2d}. {theme:<35} {count:>6,} papers ({percentage:5.1f}%)")
    
    return df, theme_counts

def answer_specific_questions(df: pd.DataFrame, theme_counts: Counter):
    """
    Answer the specific questions you mentioned using the enhanced data.
    """
    
    answers = {}
    
    print("\n" + "="*60)
    print("ANSWERING YOUR SPECIFIC RESEARCH QUESTIONS")
    print("="*60)
    
    # 1. Galaxy experts
    print(f"\n🌌 GALAXY COMMUNITY ANALYSIS:")
    galaxy_papers = df[df['themes_pred'].apply(lambda x: 'Galaxy' in x if isinstance(x, list) else False)]
    print(f"   Total Galaxy papers: {len(galaxy_papers)}")
    
    if len(galaxy_papers) > 0:
        # Sample Galaxy papers
        print(f"   Sample Galaxy papers:")
        for _, paper in galaxy_papers.head(3).iterrows():
            title = str(paper.get('title', 'No title'))[:80]
            year = paper.get('year', 'Unknown')
            print(f"     • [{paper['pmid']}] {title}... ({year})")
        
        answers['galaxy'] = {
            'total_papers': len(galaxy_papers),
            'sample_papers': galaxy_papers[['pmid', 'title', 'year']].head(5).to_dict('records')
        }
    
    # 2. Microbiome research
    print(f"\n🦠 MICROBIOME COMMUNITY ANALYSIS:")
    microbiome_papers = df[df['themes_pred'].apply(lambda x: 'Microbiome' in x if isinstance(x, list) else False)]
    print(f"   Total Microbiome papers: {len(microbiome_papers)}")
    
    if len(microbiome_papers) > 0:
        # Look for human gut specifically
        gut_papers = microbiome_papers[
            microbiome_papers['title'].str.contains('gut|intestin|colon', case=False, na=False) |
            microbiome_papers['abstract'].str.contains('gut|intestin|colon', case=False, na=False)
        ]
        print(f"   Human gut microbiome papers: {len(gut_papers)}")
        
        answers['microbiome'] = {
            'total_papers': len(microbiome_papers),
            'gut_papers': len(gut_papers)
        }
    
    # 3. Publication trends by year
    print(f"\n📊 PUBLICATION TRENDS ANALYSIS:")
    
    # Recent years analysis
    recent_years = [2020, 2021, 2022, 2023, 2024]
    yearly_theme_counts = {}
    
    for year in recent_years:
        year_papers = df[df['year'] == year]
        year_theme_counts = Counter()
        
        for themes in year_papers['themes_pred']:
            if isinstance(themes, list):
                year_theme_counts.update(themes)
        
        yearly_theme_counts[year] = year_theme_counts
        print(f"   {year}: {len(year_papers)} papers")
    
    answers['trends'] = yearly_theme_counts
    
    # 4. Cross-institutional analysis (simplified)
    print(f"\n🏛️  INSTITUTIONAL ANALYSIS:")
    
    institutions = ['Cambridge', 'Oxford', 'Imperial', 'Edinburgh', 'Manchester', 'UCL']
    inst_counts = {}
    
    for inst in institutions:
        inst_papers = df[df['affiliation'].str.contains(inst, case=False, na=False)]
        inst_counts[inst] = len(inst_papers)
        print(f"   {inst}: {len(inst_papers)} papers")
    
    answers['institutions'] = inst_counts
    
    # 5. High-impact journals
    print(f"\n📚 HIGH-IMPACT JOURNAL ANALYSIS:")
    
    nature_journals = df[df['journal'].str.contains('nature', case=False, na=False)]
    science_journals = df[df['journal'].str.contains('science', case=False, na=False)]
    cell_journals = df[df['journal'].str.contains('cell', case=False, na=False)]
    
    print(f"   Nature journals: {len(nature_journals)} papers")
    print(f"   Science journals: {len(science_journals)} papers")
    print(f"   Cell journals: {len(cell_journals)} papers")
    
    # Recent high-impact papers (last 3 years)
    recent_high_impact = df[
        (df['year'] >= 2022) & 
        (df['journal'].str.contains('nature|science|cell', case=False, na=False))
    ]
    
    print(f"   Recent high-impact (2022+): {len(recent_high_impact)} papers")
    
    answers['high_impact'] = {
        'nature': len(nature_journals),
        'science': len(science_journals), 
        'cell': len(cell_journals),
        'recent_total': len(recent_high_impact)
    }
    
    # 6. Top journals by theme
    print(f"\n📖 TOP JOURNALS BY THEME:")
    
    for theme in ['Galaxy', 'Microbiome', 'Proteomics', 'Cancer Data']:
        theme_papers = df[df['themes_pred'].apply(lambda x: theme in x if isinstance(x, list) else False)]
        if len(theme_papers) > 0:
            top_journals = theme_papers['journal'].value_counts().head(3)
            print(f"   {theme}:")
            for journal, count in top_journals.items():
                print(f"     • {journal}: {count} papers")
    
    # 7. LLM and AI research
    print(f"\n🤖 AI/LLM RESEARCH ANALYSIS:")
    
    ai_keywords = ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 
                   'language model', 'transformer', 'bert', 'gpt', 'llm']
    
    ai_papers = df[
        df['title'].str.contains('|'.join(ai_keywords), case=False, na=False) |
        df['abstract'].str.contains('|'.join(ai_keywords), case=False, na=False)
    ]
    
    print(f"   AI/ML papers: {len(ai_papers)} papers")
    
    # Recent AI papers
    recent_ai = ai_papers[ai_papers['year'] >= 2022]
    print(f"   Recent AI papers (2022+): {len(recent_ai)} papers")
    
    answers['ai_research'] = {
        'total': len(ai_papers),
        'recent': len(recent_ai)
    }
    
    # 8. Quantum research
    print(f"\n⚛️  QUANTUM RESEARCH ANALYSIS:")
    
    quantum_papers = df[
        df['title'].str.contains('quantum', case=False, na=False) |
        df['abstract'].str.contains('quantum', case=False, na=False)
    ]
    
    print(f"   Quantum papers: {len(quantum_papers)} papers")
    
    if len(quantum_papers) > 0:
        print(f"   Sample quantum papers:")
        for _, paper in quantum_papers.head(3).iterrows():
            title = str(paper.get('title', 'No title'))[:80]
            year = paper.get('year', 'Unknown')
            print(f"     • [{paper['pmid']}] {title}... ({year})")
    
    answers['quantum'] = {
        'total': len(quantum_papers),
        'sample_papers': quantum_papers[['pmid', 'title', 'year']].head(3).to_dict('records')
    }
    
    return answers

def find_collaboration_opportunities(df: pd.DataFrame):
    """
    Identify potential collaboration opportunities between institutions.
    """
    
    print(f"\n🤝 COLLABORATION OPPORTUNITY ANALYSIS:")
    
    # Group papers by institution and theme
    inst_theme_papers = defaultdict(lambda: defaultdict(list))
    
    institutions = ['Cambridge', 'Oxford', 'Imperial', 'Edinburgh', 'Manchester', 'UCL']
    
    for _, paper in df.iterrows():
        # Identify institution
        affiliation = str(paper.get('affiliation', '')).lower()
        institution = None
        
        for inst in institutions:
            if inst.lower() in affiliation:
                institution = inst
                break
        
        if institution and isinstance(paper['themes_pred'], list):
            for theme in paper['themes_pred']:
                inst_theme_papers[institution][theme].append({
                    'pmid': paper['pmid'],
                    'title': paper.get('title', ''),
                    'year': paper.get('year', 0)
                })
    
    # Find themes where multiple institutions are active but might not be collaborating
    collaboration_opportunities = []
    
    major_themes = ['Microbiome', 'Proteomics', 'Cancer Data', 'Single-Cell Omics', 'Metabolomics']
    
    for theme in major_themes:
        active_institutions = []
        
        for inst in institutions:
            if theme in inst_theme_papers[inst] and len(inst_theme_papers[inst][theme]) >= 5:
                active_institutions.append({
                    'institution': inst,
                    'paper_count': len(inst_theme_papers[inst][theme]),
                    'recent_papers': [p for p in inst_theme_papers[inst][theme] if p['year'] >= 2022]
                })
        
        if len(active_institutions) >= 2:
            collaboration_opportunities.append({
                'theme': theme,
                'institutions': active_institutions,
                'total_papers': sum(inst['paper_count'] for inst in active_institutions)
            })
    
    print(f"   Found {len(collaboration_opportunities)} collaboration opportunities:")
    
    for opp in collaboration_opportunities:
        print(f"   • {opp['theme']}: {len(opp['institutions'])} active institutions ({opp['total_papers']} papers)")
        for inst in opp['institutions']:
            recent_count = len(inst['recent_papers'])
            print(f"     - {inst['institution']}: {inst['paper_count']} papers ({recent_count} recent)")
    
    return collaboration_opportunities

def create_theme_network_analysis(df: pd.DataFrame):
    """
    Analyze theme co-occurrence patterns.
    """
    
    print(f"\n🕸️  THEME NETWORK ANALYSIS:")
    
    # Create theme co-occurrence matrix
    theme_pairs = Counter()
    
    for themes in df['themes_pred']:
        if isinstance(themes, list) and len(themes) > 1:
            # Count all pairs of themes in the same paper
            for i, theme1 in enumerate(themes):
                for theme2 in themes[i+1:]:
                    pair = tuple(sorted([theme1, theme2]))
                    theme_pairs[pair] += 1
    
    print(f"   Most common theme pairs:")
    for i, ((theme1, theme2), count) in enumerate(theme_pairs.most_common(10), 1):
        print(f"   {i:2d}. {theme1} + {theme2}: {count} papers")
    
    return theme_pairs

def generate_recommendations(answers: dict, collaboration_opportunities: list):
    """
    Generate actionable recommendations based on the analysis.
    """
    
    print(f"\n" + "="*60)
    print("ACTIONABLE RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    # 1. Galaxy community recommendations
    if answers.get('galaxy', {}).get('total_papers', 0) > 0:
        recommendations.append({
            'category': 'Galaxy Community',
            'recommendation': f"With {answers['galaxy']['total_papers']} Galaxy papers identified, consider organizing a UK Galaxy workshop to connect researchers.",
            'priority': 'High',
            'next_steps': ['Contact Galaxy paper authors', 'Organize virtual meetup', 'Create collaboration matrix']
        })
    
    # 2. AI/LLM research recommendations  
    if answers.get('ai_research', {}).get('recent', 0) > 10:
        recommendations.append({
            'category': 'AI/LLM Research',
            'recommendation': f"Strong recent AI activity ({answers['ai_research']['recent']} papers since 2022) suggests opportunity for UK ELIXIR AI working group.",
            'priority': 'High',
            'next_steps': ['Map AI researchers', 'Identify common challenges', 'Propose AI infrastructure needs']
        })
    
    # 3. Collaboration recommendations
    if collaboration_opportunities:
        top_opportunity = max(collaboration_opportunities, key=lambda x: x['total_papers'])
        recommendations.append({
            'category': 'Cross-Institutional Collaboration',
            'recommendation': f"Strong opportunity in {top_opportunity['theme']} with {len(top_opportunity['institutions'])} active institutions.",
            'priority': 'Medium',
            'next_steps': ['Facilitate introductions', 'Organize theme-specific workshop', 'Identify shared resources']
        })
    
    # 4. Quantum research recommendations
    if answers.get('quantum', {}).get('total', 0) > 0:
        recommendations.append({
            'category': 'Quantum Biology',
            'recommendation': f"Emerging quantum research ({answers['quantum']['total']} papers) may need dedicated community support.",
            'priority': 'Low',
            'next_steps': ['Monitor growth', 'Connect with quantum computing initiatives', 'Assess infrastructure needs']
        })
    
    print(f"   Generated {len(recommendations)} recommendations:\n")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. **{rec['category']}** (Priority: {rec['priority']})")
        print(f"      {rec['recommendation']}")
        print(f"      Next steps: {', '.join(rec['next_steps'])}\n")
    
    return recommendations

def export_results(answers: dict, collaboration_opportunities: list, recommendations: list, output_path: str):
    """
    Export all results to JSON for further analysis.
    """
    
    results = {
        'analysis_summary': answers,
        'collaboration_opportunities': collaboration_opportunities,
        'recommendations': recommendations,
        'metadata': {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_themes_analyzed': len(answers.get('trends', {}).get(2024, {})) if answers.get('trends') else 0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Results exported to: {output_path}")
    
    return results

def main_quick_analysis(theme_classified_path: str, output_dir: str = "."):
    """
    Run the complete quick analysis pipeline.
    """
    
    print("🚀 Starting Quick Analysis of Enhanced Theme Classification")
    print("="*70)
    
    # Step 1: Load and analyze theme distribution
    df, theme_counts = quick_analysis_of_enhanced_themes(theme_classified_path)
    
    # Step 2: Answer specific research questions
    answers = answer_specific_questions(df, theme_counts)
    
    # Step 3: Find collaboration opportunities
    collaboration_opportunities = find_collaboration_opportunities(df)
    
    # Step 4: Theme network analysis
    theme_pairs = create_theme_network_analysis(df)
    
    # Step 5: Generate recommendations
    recommendations = generate_recommendations(answers, collaboration_opportunities)
    
    # Step 6: Export results
    output_path = f"{output_dir}/quick_analysis_results.json"
    results = export_results(answers, collaboration_opportunities, recommendations, output_path)
    
    print(f"\n✅ Quick Analysis Complete!")
    print(f"📊 Analyzed {len(df)} papers across {len(theme_counts)} themes")
    print(f"🎯 Generated {len(recommendations)} actionable recommendations")
    
    return df, results

# Example usage
if __name__ == "__main__":
    # Update this path to your actual file location
    THEME_CLASSIFIED_PATH = "/content/drive/MyDrive/outputs/seeds_approach/seeds_theme_rebuild_and_tag_outputs/seeds_with_themes_diagnosed_v2.parquet"
    OUTPUT_DIR = "/content/drive/MyDrive/outputs"
    
    # Run the analysis
    df, results = main_quick_analysis(THEME_CLASSIFIED_PATH, OUTPUT_DIR)
    
    print("\n🎉 You can now explore the data further:")
    print("- df: DataFrame with enhanced theme classifications")
    print("- results: Dictionary with all analysis results")
    print("\nNext steps:")
    print("1. Review the collaboration opportunities")
    print("2. Contact researchers in high-priority themes")
    print("3. Set up the full RAG system for interactive queries")

def quick_query_examples(df: pd.DataFrame):
    """
    Show examples of how to query the enhanced theme data manually.
    """
    
    print("\n" + "="*60)
    print("MANUAL QUERY EXAMPLES")
    print("="*60)
    
    # Example 1: Find all papers in a specific theme
    print("\n📝 Example 1: Find all Galaxy papers")
    galaxy_papers = df[df['themes_pred'].apply(lambda x: 'Galaxy' in x if isinstance(x, list) else False)]
    print(f"   Found {len(galaxy_papers)} Galaxy papers")
    
    # Example 2: Find papers with multiple themes
    print("\n📝 Example 2: Papers with multiple themes")
    multi_theme = df[df['themes_pred'].apply(lambda x: len(x) > 1 if isinstance(x, list) else False)]
    print(f"   Found {len(multi_theme)} papers with multiple themes")
    
    # Example 3: Find recent papers in specific institution
    print("\n📝 Example 3: Recent Cambridge papers")
    cambridge_recent = df[
        (df['affiliation'].str.contains('cambridge', case=False, na=False)) &
        (df['year'] >= 2023)
    ]
    print(f"   Found {len(cambridge_recent)} recent Cambridge papers")
    
    # Example 4: Search by keywords in title
    print("\n📝 Example 4: Papers about 'machine learning'")
    ml_papers = df[df['title'].str.contains('machine learning', case=False, na=False)]
    print(f"   Found {len(ml_papers)} machine learning papers")
    
    # Example 5: Theme overlap analysis
    print("\n📝 Example 5: Papers that combine Microbiome + Metabolomics")
    combined_theme = df[
        df['themes_pred'].apply(
            lambda x: isinstance(x, list) and 'Microbiome' in x and 'Metabolomics' in x
        )
    ]
    print(f"   Found {len(combined_theme)} papers combining these themes")
    
    return {
        'galaxy_papers': len(galaxy_papers),
        'multi_theme': len(multi_theme),
        'cambridge_recent': len(cambridge_recent),
        'ml_papers': len(ml_papers),
        'combined_theme': len(combined_theme)
    }