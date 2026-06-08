import json
import argparse
import os

# ArXiv broad category mapping based on the official arXiv taxonomy.
# Maps the primary category code to a human-readable broad domain name.
# Reference: https://arxiv.org/category_taxonomy
ARXIV_BROAD_CATEGORY_MAP = {
    # Astrophysics (astro-ph, astro-ph.*)
    "astro-ph": "Astrophysics",
    # Condensed Matter (cond-mat.*)
    "cond-mat": "Condensed Matter",
    # Computer Science (cs.*)
    "cs": "Computer Science",
    # Economics (econ.*)
    "econ": "Economics",
    # Electrical Engineering and Systems Science (eess.*)
    "eess": "Electrical Engineering",
    # Mathematics (math.*, math-ph)
    "math": "Mathematics",
    "math-ph": "Mathematics",
    # Physics (physics.*, gr-qc, hep-*, nucl-*, quant-ph)
    "physics": "Physics",
    "gr-qc": "Physics",
    "hep-ph": "Physics",
    "hep-th": "Physics",
    "hep-ex": "Physics",
    "hep-lat": "Physics",
    "nucl-th": "Physics",
    "nucl-ex": "Physics",
    "quant-ph": "Physics",
    # Quantitative Biology (q-bio.*)
    "q-bio": "Quantitative Biology",
    # Quantitative Finance (q-fin.*)
    "q-fin": "Quantitative Finance",
    # Statistics (stat.*)
    "stat": "Statistics",
    # Nonlinear Sciences (nlin.*)
    "nlin": "Nonlinear Sciences",
}


def map_to_broad_category(raw_categories: str) -> str:
    """
    Map arXiv category string to a broad domain name.
    
    Takes the first space-separated field as the primary category,
    then matches it against ARXIV_BROAD_CATEGORY_MAP using:
      1. Exact match (e.g., 'hep-ph' -> 'Physics')
      2. Prefix before '.' (e.g., 'cs.AI' -> prefix 'cs' -> 'Computer Science')
      3. Prefix before '-' for sub-fields (e.g., 'hep-ph' is already exact)
    
    Returns the broad domain name, or 'Other' if no match found.
    """
    fields = raw_categories.strip().split()
    if not fields:
        return "Other"
    
    primary = fields[0]  # Take the first category as primary
    
    # 1. Exact match
    if primary in ARXIV_BROAD_CATEGORY_MAP:
        return ARXIV_BROAD_CATEGORY_MAP[primary]
    
    # 2. Match by prefix before '.'
    prefix = primary.split(".")[0]
    if prefix in ARXIV_BROAD_CATEGORY_MAP:
        return ARXIV_BROAD_CATEGORY_MAP[prefix]
    
    # 3. Fallback: try prefix before '-' (e.g., unknown hep-xxx)
    prefix_dash = primary.split("-")[0]
    # Check if any key starts with this prefix
    for key, value in ARXIV_BROAD_CATEGORY_MAP.items():
        if key.startswith(prefix_dash + "-") or key == prefix_dash:
            return value
    
    return "Other"


def process_arxiv(input_path, output_path, limit=10000, category_filter=None):
    """
    Process the large ArXiv metadata JSON Lines file and save a formatted subset.
    
    Args:
        input_path: Path to the raw arxiv-metadata-oai-snapshot.json
        output_path: Path to write the processed JSON file
        limit: Max number of papers to extract (to avoid memory issues during loading)
        category_filter: Optional string (e.g. 'cs' or 'cs.AI') to filter papers
    """
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    if category_filter:
        print(f"Filtering categories by prefix: {category_filter}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    processed_data = []
    count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if limit and count >= limit:
                break
                
            try:
                item = json.loads(line)
                
                # Extract original fields
                paper_id = item.get("id", "").strip()
                title = item.get("title", "").replace("\n", " ").strip()
                # Clean multiple whitespaces in title
                title = " ".join(title.split())
                
                abstract = item.get("abstract", "").replace("\n", " ").strip()
                abstract = " ".join(abstract.split())
                
                categories = item.get("categories", "").strip()
                
                # Check category filter (e.g., 'cs.AI' or startswith 'cs.')
                if category_filter:
                    paper_categories = categories.split()
                    matched = False
                    for cat in paper_categories:
                        if cat.startswith(category_filter):
                            matched = True
                            break
                    if not matched:
                        continue
                
                # Create combined text for clustering (Title + Abstract)
                combined_text = f"{title}. {abstract}"
                
                # Map the raw arXiv category to a broad domain name
                broad_category = map_to_broad_category(categories)
                
                # Format to support loader.py
                # loader.py expects a key 'text' or 'content' for the main text,
                # and treats all other keys as aspects.
                record = {
                    "text": combined_text,
                    "id": paper_id,
                    "title": title,
                    "abstract": abstract,
                    "category": broad_category  # Mapped to broad domain name
                }
                
                processed_data.append(record)
                count += 1
                
                if count % 1000 == 0:
                    print(f"Processed {count} records...")
                    
            except json.JSONDecodeError:
                continue
                
    # Write as a standard JSON array
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(processed_data, out_f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed and saved {len(processed_data)} records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw ArXiv metadata JSON Lines into a formatted JSON for C3.")
    parser.add_argument("--input", "-i", default="data/arxiv-metadata-oai-snapshot.json", help="Path to raw ArXiv JSON Lines file.")
    parser.add_argument("--output", "-o", default="data/arxiv_processed.json", help="Output path for processed JSON.")
    parser.add_argument("--limit", "-l", type=int, default=10000, help="Max number of papers to extract. Set to 0 for unlimited.")
    parser.add_argument("--category", "-c", default=None, help="Prefix filter for categories (e.g. 'cs' for Computer Science).")
    
    args = parser.parse_args()
    
    # Handle limit = 0 (unlimited)
    limit_val = None if args.limit <= 0 else args.limit
    
    process_arxiv(args.input, args.output, limit=limit_val, category_filter=args.category)
