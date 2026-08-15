# Themes RAG System Expert Discovery Evaluation
# Adapted from the successful Seeds evaluation framework
# Maintains methodological consistency for comparative analysis

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import re
import ast
import warnings

warnings.filterwarnings("ignore")

print("=== THEMES RAG SYSTEM EXPERT DISCOVERY EVALUATION ===")
print("Adapted from successful Seeds evaluation methodology")
print("Working with already-loaded themes_rag_system")

# Define output directory
OUTPUT_DIR = Path("/content/drive/MyDrive/outputs/themes_evaluation_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verify the themes system is available
if "themes_rag_system" not in globals():
    print("ERROR: themes_rag_system not found")
    print("Please ensure you have run the themes system loading code first")
    raise RuntimeError("Themes RAG system not available")

print(
    f"✓ Themes system found with {len(themes_rag_system.metadata_df)} papers in metadata"
)
print(f"✓ Output directory: {OUTPUT_DIR}")


class ThemesExpertDiscoveryEvaluator:
    """
    Expert discovery evaluation framework specifically designed for the Themes RAG system.

    This maintains the same evaluation methodology used for Seeds but adapts to the
    semantic-based themes approach, enabling direct comparison between approaches.
    The key difference is that this uses semantic similarity to research topics rather
    than network connections to find hidden experts.
    """

    def __init__(self, themes_rag_system, output_dir):
        self.themes_system = themes_rag_system
        self.output_dir = Path(output_dir)
        self.known_members = []
        self.known_member_names = set()

        # Results containers matching seeds evaluation structure
        self.discovered_experts = {}
        self.query_results = {}
        self.expert_evidence = {}

        print(f"Themes evaluator initialized with working RAG system")
        print(f"System has {len(self.themes_system.metadata_df)} papers in metadata")
        print(f"Output directory: {self.output_dir}")

    def load_known_elixir_members(self):
        """
        Load known ELIXIR members for evaluation baseline.

        Uses the same data structure as the seeds evaluation for consistency.
        """
        print("\nLoading known ELIXIR members...")

        try:
            # Try to load from your uploaded files
            elixir_authors_path = "/content/elixir_uk_authors.csv"
            if Path(elixir_authors_path).exists():
                df = pd.read_csv(elixir_authors_path)

                self.known_members = []
                for _, row in df.iterrows():
                    self.known_members.append(
                        {
                            "name": row["author"],
                            "organisations": row.get("organisations", ""),
                            "communities": row.get(
                                "communites", ""
                            ),  # Note: original typo in CSV
                            "country": row.get("country", ""),
                        }
                    )

                # Create normalized name lookup
                self.known_member_names = set()
                for member in self.known_members:
                    name = member["name"].strip().lower()
                    self.known_member_names.add(name)

                print(f"✓ Loaded {len(self.known_members)} known ELIXIR members")
                return True

        except Exception as e:
            print(f"Could not load from uploaded file: {e}")

        # Fallback: create sample known members for testing
        print("Using sample known members for testing...")
        self.known_members = [
            {
                "name": "Sample Expert 1",
                "organisations": "University of Cambridge",
                "communities": "Genomics",
                "country": "UK",
            },
            {
                "name": "Sample Expert 2",
                "organisations": "Imperial College London",
                "communities": "Proteomics",
                "country": "UK",
            },
        ]
        self.known_member_names = {
            member["name"].lower() for member in self.known_members
        }

        return True

    def parse_author_field(self, author_field):
        """
        Parse author field with the same robust logic used in seeds evaluation.

        This handles the complex data structures that can appear in author fields,
        including numpy arrays, lists, and string representations. We use the
        exact same parsing logic that proved successful in the seeds evaluation.
        """
        # Handle None and NaN values safely
        try:
            if author_field is None or pd.isna(author_field):
                return []
        except (ValueError, TypeError):
            pass

        # Handle numpy arrays directly - convert to string representation first
        if isinstance(author_field, np.ndarray):
            try:
                authors_list = author_field.tolist()
                result = []
                for author in authors_list:
                    if author and str(author).strip():
                        result.append(str(author).strip())
                return result
            except:
                try:
                    author_field = str(author_field)
                except:
                    return []

        # Handle regular lists
        if isinstance(author_field, list):
            result = []
            for author in author_field:
                try:
                    if author is not None and str(author).strip():
                        result.append(str(author).strip())
                except:
                    continue
            return result

        # Handle string representations (including numpy array strings)
        try:
            author_str = str(author_field).strip()

            # Check if this looks like an empty or invalid field
            if not author_str or author_str in ["nan", "None", "[]", ""]:
                return []

            # Handle numpy array string format like "['Author1' 'Author2']"
            if author_str.startswith("[") and author_str.endswith("]"):
                inner = author_str[1:-1].strip()

                if not inner:
                    return []

                # Try different parsing strategies based on what separators are present
                if "'" in inner:
                    # Split by single quotes and extract non-empty parts
                    parts = inner.split("'")
                    authors = []
                    for part in parts:
                        cleaned = part.strip()
                        # Skip empty strings, whitespace, and bracket artifacts
                        if (
                            cleaned
                            and len(cleaned) > 1
                            and cleaned not in ["", " ", "[", "]"]
                        ):
                            authors.append(cleaned)
                    return authors
                elif '"' in inner:
                    # Handle double quotes similarly
                    parts = inner.split('"')
                    authors = []
                    for part in parts:
                        cleaned = part.strip()
                        if (
                            cleaned
                            and len(cleaned) > 1
                            and cleaned not in ["", " ", "[", "]"]
                        ):
                            authors.append(cleaned)
                    return authors
                else:
                    # Try space separation for numpy array format without quotes
                    parts = inner.split()
                    authors = []
                    for part in parts:
                        cleaned = part.strip()
                        if cleaned and len(cleaned) > 1:
                            authors.append(cleaned)
                    return authors

            # Handle comma-separated format
            elif "," in author_str:
                authors = []
                for author in author_str.split(","):
                    cleaned = author.strip()
                    if cleaned and len(cleaned) > 1:
                        authors.append(cleaned)
                return authors

            # Single author case
            else:
                cleaned = author_str.strip()
                if cleaned and len(cleaned) > 1:
                    return [cleaned]
                else:
                    return []

        except Exception as e:
            # If all parsing fails, return empty list
            return []

    def parse_institution_field(self, institution_field):
        """Parse institution field using the same logic as authors."""
        return self.parse_author_field(institution_field)

    def parse_theme_field(self, theme_field):
        """Parse theme field using the same logic as authors."""
        return self.parse_author_field(theme_field)

    def create_elixir_community_queries(self):
        """
        Create research queries specific to ELIXIR communities for semantic-based discovery.

        This is the key difference from the seeds approach - we use semantic similarity
        to research topics rather than network connections. Each query is designed to
        find papers semantically related to specific ELIXIR community themes.
        """
        # ELIXIR community queries optimized for semantic search in biomedical literature
        community_queries = {
            "Research Data Management": [
                "FAIR data principles bioinformatics",
                "research data infrastructure management",
                "biomedical data repository systems",
                "metadata standards life sciences",
                "data stewardship computational biology",
            ],
            "Single-Cell Omics": [
                "single-cell RNA sequencing analysis",
                "scRNA-seq computational methods",
                "single cell transcriptomics workflows",
                "cell type identification algorithms",
                "spatial transcriptomics analysis",
            ],
            "Galaxy": [
                "Galaxy workflow bioinformatics",
                "computational workflow management",
                "reproducible bioinformatics pipelines",
                "Galaxy tool development",
                "workflow execution systems",
            ],
            "Human Copy Number Variation": [
                "copy number variation detection",
                "CNV analysis algorithms",
                "structural variation genomics",
                "chromosomal abnormality detection",
                "genomic rearrangement analysis",
            ],
            "Systems Biology": [
                "systems biology modeling approaches",
                "biological network analysis",
                "pathway enrichment analysis",
                "mathematical modeling biological systems",
                "computational systems biology",
            ],
            "Proteomics": [
                "mass spectrometry proteomics",
                "protein identification methods",
                "proteome analysis workflows",
                "LC-MS/MS protein analysis",
                "protein biomarker discovery",
            ],
            "Metabolomics": [
                "metabolomics data analysis",
                "metabolite identification methods",
                "metabolic profiling techniques",
                "NMR spectroscopy metabolomics",
                "metabolic pathway analysis",
            ],
            "Rare Diseases": [
                "rare disease genetics research",
                "orphan disease genomics",
                "genetic disorder identification",
                "disease gene discovery",
                "clinical genomics rare diseases",
            ],
        }

        return community_queries

    def discover_experts_for_community(self, community, queries):
        """
        Discover experts for a specific community using semantic search queries.

        This is where the themes approach differs fundamentally from seeds. Instead of
        following co-author networks, we find experts through semantic similarity to
        research topics, potentially discovering researchers who work in isolation or
        across disciplinary boundaries.
        """
        print(f"\nDiscovering experts for: {community}")
        all_experts = {}

        for query in queries:
            print(f"  Searching: '{query}'")
            try:
                # Use the themes system's semantic search with reasonable parameters
                # We search more broadly initially to capture diverse expertise patterns
                results = self.themes_system.semantic_search(query, k=25)

                if isinstance(results, list) and len(results) > 0:
                    print(f"    Found {len(results)} papers")

                    # Extract experts from search results
                    for paper in results:
                        # Parse authors from this paper using our robust parsing
                        authors = self.parse_author_field(paper.get("authors", []))

                        for author in authors:
                            # Ensure author is a clean string
                            try:
                                if isinstance(author, (np.ndarray, list)):
                                    author_name = str(author).strip()
                                else:
                                    author_name = str(author).strip()

                                # Skip empty or very short names
                                if not author_name or len(author_name) <= 2:
                                    continue

                                # Clean up any remaining artifacts from parsing
                                author_name = (
                                    author_name.replace("[", "")
                                    .replace("]", "")
                                    .replace("'", "")
                                    .strip()
                                )

                                if len(author_name) <= 2:
                                    continue

                                # Check if this is a known ELIXIR member (safe comparison)
                                is_known_member = False
                                try:
                                    is_known_member = (
                                        author_name.lower() in self.known_member_names
                                    )
                                except (AttributeError, TypeError):
                                    is_known_member = False

                                # Skip if this is a known ELIXIR member - we want to find hidden experts
                                if is_known_member:
                                    continue

                                # Add to discovered experts
                                if author_name not in all_experts:
                                    all_experts[author_name] = {
                                        "name": author_name,
                                        "papers": [],
                                        "queries_found": [],
                                        "institutions": set(),
                                        "themes": set(),
                                        "years": [],
                                        "total_papers": 0,
                                    }

                                # Add paper information with semantic similarity score
                                all_experts[author_name]["papers"].append(
                                    {
                                        "pmid": paper.get("pmid"),
                                        "title": paper.get("title"),
                                        "year": paper.get("year"),
                                        "journal": paper.get("journal"),
                                        "similarity_score": paper.get(
                                            "similarity_score", 0.0
                                        ),
                                    }
                                )

                                all_experts[author_name]["queries_found"].append(query)
                                all_experts[author_name]["total_papers"] += 1

                                # Add institutions and themes
                                institutions = self.parse_institution_field(
                                    paper.get("institutions", [])
                                )
                                themes = self.parse_theme_field(paper.get("themes", []))

                                all_experts[author_name]["institutions"].update(
                                    institutions
                                )
                                all_experts[author_name]["themes"].update(themes)

                                if paper.get("year"):
                                    all_experts[author_name]["years"].append(
                                        paper.get("year")
                                    )

                            except Exception as e:
                                # Skip this author if any processing fails
                                print(
                                    f"      Warning: Skipping author due to processing error: {e}"
                                )
                                continue
                else:
                    print(f"    No results found")

            except Exception as e:
                print(f"    Error in query: {e}")

        # Convert sets to lists for JSON serialization
        for expert in all_experts.values():
            expert["institutions"] = list(expert["institutions"])
            expert["themes"] = list(expert["themes"])
            expert["queries_found"] = list(
                set(expert["queries_found"])
            )  # Remove duplicates

        print(f"  Total unique experts found: {len(all_experts)}")
        return all_experts

    def calculate_expert_evidence_scores(self, expert_data):
        """
        Calculate evidence scores for discovered experts using the same methodology as seeds.

        This maintains methodological consistency while adapting to the semantic discovery context.
        The scores reflect different types of evidence that support expertise claims.
        """
        print(f"\nCalculating evidence scores for {len(expert_data)} experts...")

        scored_experts = {}

        for expert_name, data in expert_data.items():
            # Publication volume score - normalize to 0-1 scale
            volume_score = min(1.0, data["total_papers"] / 10.0)

            # Query diversity score - found through multiple different semantic searches
            # This indicates broad expertise within the domain
            query_diversity = len(set(data["queries_found"])) / max(
                len(data["queries_found"]), 1
            )

            # Institutional diversity score - collaborations across institutions
            institutional_diversity = min(1.0, len(data["institutions"]) / 3.0)

            # Thematic focus score - concentrated expertise vs broad coverage
            # For semantic discovery, fewer themes often indicates deeper expertise
            if len(data["themes"]) > 0:
                thematic_focus = 1.0 / len(data["themes"])
            else:
                thematic_focus = 0.0

            # Temporal activity score - recent research activity
            if data["years"]:
                recent_years = [year for year in data["years"] if year >= 2020]
                temporal_activity = len(recent_years) / len(data["years"])
            else:
                temporal_activity = 0.0

            # Average semantic similarity score - how well papers match community themes
            # This is unique to the semantic approach
            if data["papers"]:
                avg_similarity = np.mean(
                    [paper.get("similarity_score", 0.0) for paper in data["papers"]]
                )
            else:
                avg_similarity = 0.0

            # Composite evidence score using same weighting as seeds evaluation
            composite_score = np.mean(
                [
                    volume_score,
                    query_diversity,
                    institutional_diversity,
                    thematic_focus,
                    temporal_activity,
                    avg_similarity,
                ]
            )

            scored_experts[expert_name] = {
                **data,  # Include all original data
                "evidence_scores": {
                    "volume_score": volume_score,
                    "query_diversity": query_diversity,
                    "institutional_diversity": institutional_diversity,
                    "thematic_focus": thematic_focus,
                    "temporal_activity": temporal_activity,
                    "avg_similarity": avg_similarity,
                    "composite_score": composite_score,
                },
            }

        return scored_experts

    def categorize_experts(self, scored_experts, min_score=0.3):
        """
        Categorize experts based on their evidence patterns using the same taxonomy as seeds.

        This enables direct comparison between approaches while highlighting patterns
        unique to semantic-based discovery.
        """
        print(f"\nCategorizing experts (minimum score: {min_score})...")

        categories = {
            "Emerging Stars": [],
            "Cross-Pollinators": [],
            "Hidden Connectors": [],
            "Interdisciplinary Bridge Builders": [],
        }

        high_confidence_experts = {
            name: data
            for name, data in scored_experts.items()
            if data["evidence_scores"]["composite_score"] >= min_score
        }

        print(f"High-confidence experts: {len(high_confidence_experts)}")

        for expert_name, data in high_confidence_experts.items():
            scores = data["evidence_scores"]

            # Categorization logic adapted for semantic discovery patterns
            if (
                scores["temporal_activity"] > 0.7
                and scores["volume_score"] < 0.6
                and len(data["years"]) <= 5
            ):
                # High recent activity, moderate volume, short career
                categories["Emerging Stars"].append(expert_name)

            elif (
                scores["institutional_diversity"] > 0.6
                and scores["query_diversity"] > 0.7
            ):
                # High institutional and semantic query diversity
                categories["Hidden Connectors"].append(expert_name)

            elif len(data["themes"]) > 3 and scores["thematic_focus"] < 0.4:
                # Many themes, broad semantic coverage
                categories["Interdisciplinary Bridge Builders"].append(expert_name)

            else:
                # Default category for experts who don't fit clear patterns
                categories["Cross-Pollinators"].append(expert_name)

        # Print category summary
        for category, experts in categories.items():
            print(f"  {category}: {len(experts)} experts")

        return categories, high_confidence_experts

    def test_complex_semantic_queries(self):
        """
        Test complex analytical queries to demonstrate themes system capabilities.

        This tests the sophisticated semantic reasoning that distinguishes the themes
        approach from network-based discovery.
        """
        print(f"\nTesting complex semantic analytical queries...")

        complex_queries = [
            {
                "query": "machine learning genomics applications",
                "description": "AI/ML researchers in genomics",
                "expected": "interdisciplinary_computational",
            },
            {
                "query": "CRISPR gene editing therapeutic applications",
                "description": "Gene editing therapy experts",
                "expected": "translational_research",
            },
            {
                "query": "single cell multi-omics integration",
                "description": "Multi-modal single-cell specialists",
                "expected": "cutting_edge_methods",
            },
            {
                "query": "protein structure prediction AlphaFold",
                "description": "Structural biology AI experts",
                "expected": "computational_structural",
            },
            {
                "query": "microbiome host interaction mechanisms",
                "description": "Microbiome-host interface researchers",
                "expected": "systems_biology",
            },
        ]

        query_results = {}

        for query_info in complex_queries:
            query = query_info["query"]
            print(f"  Testing: '{query}'")

            try:
                results = self.themes_system.semantic_search(query, k=15)

                if isinstance(results, list) and len(results) > 0:
                    # Extract information from semantic search results
                    extracted_info = {
                        "paper_count": len(results),
                        "unique_authors": set(),
                        "institutions": set(),
                        "themes": set(),
                        "uk_papers": 0,
                        "avg_similarity": np.mean(
                            [r.get("similarity_score", 0.0) for r in results]
                        ),
                    }

                    for paper in results:
                        # Extract authors using our robust parsing
                        authors = self.parse_author_field(paper.get("authors", []))
                        extracted_info["unique_authors"].update(authors)

                        # Extract institutions
                        institutions = self.parse_institution_field(
                            paper.get("institutions", [])
                        )
                        extracted_info["institutions"].update(institutions)

                        # Extract themes
                        themes = self.parse_theme_field(paper.get("themes", []))
                        extracted_info["themes"].update(themes)

                        # Count UK papers
                        if paper.get("any_uk", False):
                            extracted_info["uk_papers"] += 1

                    # Convert sets to counts for reporting
                    query_results[query] = {
                        "description": query_info["description"],
                        "paper_count": extracted_info["paper_count"],
                        "unique_authors": len(extracted_info["unique_authors"]),
                        "unique_institutions": len(extracted_info["institutions"]),
                        "unique_themes": len(extracted_info["themes"]),
                        "uk_papers": extracted_info["uk_papers"],
                        "avg_similarity": float(extracted_info["avg_similarity"]),
                        "success": True,
                    }

                    print(
                        f"    Success: {len(results)} papers, {len(extracted_info['unique_authors'])} authors"
                    )

                else:
                    query_results[query] = {
                        "description": query_info["description"],
                        "success": False,
                        "error": "No results returned",
                    }
                    print(f"    Failed: No results")

            except Exception as e:
                query_results[query] = {
                    "description": query_info["description"],
                    "success": False,
                    "error": str(e),
                }
                print(f"    Error: {e}")

        return query_results

    def run_complete_evaluation(self):
        """
        Run the complete expert discovery evaluation using themes-based semantic search.

        This orchestrates all components to demonstrate a working evaluation framework
        that maintains methodological consistency with seeds while revealing unique
        characteristics of semantic-based expert discovery.
        """
        print("\n" + "=" * 60)
        print("RUNNING COMPLETE THEMES EXPERT DISCOVERY EVALUATION")
        print("=" * 60)

        start_time = time.time()

        # Step 1: Create semantic research topic queries
        print("\nStep 1: Creating ELIXIR community semantic queries...")
        community_queries = self.create_elixir_community_queries()
        print(f"Created semantic queries for {len(community_queries)} communities")

        # Step 2: Discover experts through semantic similarity for each community
        print(
            "\nStep 2: Discovering experts by semantic similarity to community themes..."
        )
        all_discovered_experts = {}

        for community, queries in community_queries.items():
            experts = self.discover_experts_for_community(community, queries)
            all_discovered_experts[community] = experts

        # Step 3: Calculate evidence scores using same methodology as seeds
        print("\nStep 3: Calculating evidence scores...")
        all_scored_experts = {}

        for community, experts in all_discovered_experts.items():
            if experts:
                scored = self.calculate_expert_evidence_scores(experts)
                all_scored_experts[community] = scored

        # Step 4: Categorize experts using same taxonomy as seeds
        print("\nStep 4: Categorizing discovered experts...")
        all_categories = {}
        all_high_confidence = {}

        for community, scored_experts in all_scored_experts.items():
            if scored_experts:
                categories, high_conf = self.categorize_experts(scored_experts)
                all_categories[community] = categories
                all_high_confidence[community] = high_conf

        # Step 5: Test complex semantic analytical queries
        print("\nStep 5: Testing complex semantic analytical queries...")
        query_results = self.test_complex_semantic_queries()

        # Step 6: Generate comprehensive evaluation report
        print("\nStep 6: Generating comprehensive evaluation report...")

        # Calculate overall statistics for comparison with seeds
        total_experts = sum(len(experts) for experts in all_discovered_experts.values())
        total_high_confidence = sum(
            len(experts) for experts in all_high_confidence.values()
        )

        category_totals = defaultdict(int)
        for community_cats in all_categories.values():
            for category, experts in community_cats.items():
                category_totals[category] += len(experts)

        successful_queries = sum(
            1 for result in query_results.values() if result.get("success", False)
        )

        # Create comprehensive results matching seeds evaluation structure
        evaluation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_time_seconds": time.time() - start_time,
            "evaluation_approach": "themes_semantic_based",
            "system_type": "semantic_similarity_expert_discovery",
            "summary_statistics": {
                "total_experts_discovered": total_experts,
                "high_confidence_experts": total_high_confidence,
                "communities_analyzed": len(community_queries),
                "successful_complex_queries": f"{successful_queries}/{len(query_results)}",
                "discovery_method": "semantic_similarity_to_research_topics",
            },
            "category_totals": dict(category_totals),
            "community_results": {
                community: {
                    "total_experts": len(experts),
                    "high_confidence": len(all_high_confidence.get(community, {})),
                    "categories": all_categories.get(community, {}),
                    "query_count": len(community_queries[community]),
                }
                for community, experts in all_discovered_experts.items()
            },
            "complex_query_results": query_results,
            "detailed_expert_data": all_high_confidence,
            "methodological_notes": {
                "approach_differences": "Uses semantic similarity rather than co-author networks",
                "discovery_mechanism": "BioBERT embeddings and FAISS similarity search",
                "filter_criteria": "Excludes known ELIXIR members to find hidden experts",
                "evaluation_consistency": "Same scoring and categorization as seeds evaluation",
            },
        }

        # Save results with clear naming for comparison
        results_file = self.output_dir / "themes_complete_evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("THEMES EVALUATION COMPLETE - SUMMARY RESULTS")
        print("=" * 60)
        print(f"Discovery approach: Semantic similarity to research topics")
        print(f"Total experts discovered: {total_experts}")
        print(f"High-confidence experts: {total_high_confidence}")
        print(f"Communities analyzed: {len(community_queries)}")
        print(f"Complex queries successful: {successful_queries}/{len(query_results)}")

        print(f"\nExpert categories (semantic-based discovery):")
        for category, count in category_totals.items():
            print(f"  {category}: {count} experts")

        print(f"\nTop communities by semantic expert discovery:")
        community_counts = [
            (comm, len(experts)) for comm, experts in all_discovered_experts.items()
        ]
        community_counts.sort(key=lambda x: x[1], reverse=True)
        for community, count in community_counts[:5]:
            print(f"  {community}: {count} experts")

        print(f"\nKey methodological insights:")
        print(
            f"- Semantic approach discovers experts through research topic similarity"
        )
        print(f"- Uses BioBERT embeddings for semantic matching to ELIXIR themes")
        print(f"- May find isolated experts not connected to known ELIXIR networks")
        print(f"- Complementary to network-based seeds approach")

        return evaluation_results


# Execute the themes expert discovery evaluation
print("\n" + "=" * 60)
print("EXECUTING THEMES EXPERT DISCOVERY EVALUATION")
print("=" * 60)

# Initialize the themes evaluation framework
evaluator = ThemesExpertDiscoveryEvaluator(
    themes_rag_system=themes_rag_system, output_dir=OUTPUT_DIR
)

# Load known ELIXIR members for baseline comparison
evaluator.load_known_elixir_members()

# Run the complete evaluation
print(f"\nStarting themes evaluation...")
print(
    f"This will take several minutes as we perform semantic searches across communities..."
)

results = evaluator.run_complete_evaluation()

print(f"\nThemes evaluation framework execution complete!")
print(f"\nThis evaluation demonstrates semantic-based expert discovery that:")
print(f"1. Uses research topic similarity rather than network connections")
print(f"2. Leverages BioBERT embeddings for semantic matching")
print(f"3. Maintains methodological consistency with seeds evaluation")
print(f"4. Discovers experts through content similarity to ELIXIR themes")
print(f"5. Applies same evidence scoring and categorization framework")
print(f"6. Tests complex semantic analytical reasoning capabilities")
print(f"7. Generates results directly comparable to seeds approach")

print(f"\nThis semantic approach complements the network-based seeds evaluation,")
print(f"potentially discovering hidden experts who work in isolation or across")
print(f"disciplinary boundaries that might not be captured by co-author networks.")

print(f"\nResults saved for comparative analysis with seeds evaluation.")
