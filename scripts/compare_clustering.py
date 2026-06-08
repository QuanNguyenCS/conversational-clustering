import os
import sys
import json
import time
import re
import argparse
import warnings
import numpy as np
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

# Ensure standard output uses UTF-8 to prevent errors on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure the root project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import C3 modules
from src.embeddings.local_embeddings import SentenceTransformerEmbeddings
from src.pipeline import ConversationalClusteringPipeline, parse_json_robustly
from src.agent.cloud_agent import GitHubModelsAgent
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc

# Ignore metric-learn and FutureWarnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="metric_learn")
warnings.filterwarnings("ignore", category=FutureWarning)

# Detailed dataset prompts (user intents) and their cluster count mappings
DATASET_PROMPTS = {
    "arxiv_fine": "Group these scientific papers and articles by their specific academic research sub-category or sub-field. We prefer a highly fine-grained and detailed granularity of categories, separating narrow scientific topics rather than broad academic disciplines. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "banking77": "Group these banking customer service queries purely by their Temporal Focus (the temporal orientation of the query) into exactly {n_clusters} clusters. The allowable clusters are strictly: (1) past (referencing past transactions, historical statements, past actions, or already executed payments/orders), (2) present (dealing with current ongoing states, active configuration attempts, immediate problems, or urgent requests happening right now), and (3) future (pertaining to hypothetical scenarios, general/upcoming timelines, planned actions, or future expectations). YOU MUST FOLLOW THE EXACT NUMBER OF CLUSTERS ({n_clusters}) STRICTLY.",
    "clinc": "Group these user queries by their specific task intent for a smart home assistant. We expect a highly fine-grained granularity of categories, where each precise voice command task is grouped into its own distinct class. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "clinc_domain": "Group these smart assistant user queries by their broad functional application domain. We prefer a coarse-grained granularity of categories that maps each query to a high-level operational domain rather than narrow individual intents. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "few_event": "Group these sentences based on the main event type or action described in the text. We require a fine-grained granularity that categorizes texts by the specific type of event taking place. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "few_nerd_nat": "Group these entity descriptions and noun phrases by their specific named entity semantic type or category. We prefer a fine-grained granularity that distinguishes specific entity types rather than grouping them into broad categories. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "few_rel_nat": "Group these sentences by the specific semantic relation type being described between entities. We require a fine-grained granularity that clusters sentences according to the exact nature of the relationship. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "go_emotion": "Group these text snippets by the primary human emotion or sentiment expressed. We require a fine-grained granularity that separates subtle differences in emotional states rather than grouping them into simple positive or negative sentiments. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "massive_intent": "Group these smart assistant voice commands by their specific, fine-grained action intent. We require a highly specific granularity of categories, separating closely related actions into individual classes. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "massive_scenario": "Group these smart assistant voice commands by their broad functional scenario or target domain. We prefer a coarse-grained granularity that aggregates multiple specific intents into high-level scenarios. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "mtop_domain": "Group these user queries by their broad functional application domain. We expect a coarse-grained granularity where queries are partitioned into high-level operational modules. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "mtop_intent": "Group these voice commands by their specific action intent. We require a highly fine-grained granularity of categories, capturing the exact operation requested. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "reddit": "Group these posts by the specific Subreddit online community topic source they originate from. We require a fine-grained granularity that maps each post to its exact discussion board topic. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "stackexchange": "Group these forum questions by the specific StackExchange Q&A community forum category they were posted on. We expect a fine-grained granularity that partitions questions by their specific technical or academic discipline. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY."
}

def load_jsonl_dataset(filepath: str) -> Tuple[List[str], List[str]]:
    """Load JSONL dataset files."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    texts = []
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    item = json.loads(line_str)
                    text = item.get("input") or item.get("text")
                    label = item.get("label") or item.get("category")
                    if text is not None and label is not None:
                        texts.append(str(text))
                        labels.append(str(label))
                except json.JSONDecodeError:
                    continue
    if not texts:
        raise ValueError(f"No records found in {filepath}.")
    return texts, labels

def compute_confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    """Calculate mean, standard deviation, and 95% CI margin of error."""
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0, 0.0
    std = np.std(values, ddof=1)
    
    if n == 2:
        t_critical = 12.706
    elif n == 3:
        t_critical = 4.303
    elif n == 4:
        t_critical = 3.182
    elif n == 5:
        t_critical = 2.776
    else:
        t_critical = 1.96
        
    margin = t_critical * (std / np.sqrt(n))
    return float(mean), float(std), float(margin)

def format_metric(stat: dict, is_time: bool = False, is_token: bool = False) -> str:
    mean = stat["mean"]
    ci = stat["ci"]
    if is_token:
        return f"{int(round(mean))} ± {int(round(ci))}"
    elif is_time:
        return f"{mean:.3f}s ± {ci:.3f}s"
    else:
        return f"{mean:.4f} ± {ci:.4f}"

def run_traditional_kmeans(embeddings: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, float]:
    """Run standard K-Means on unit-normalized embeddings."""
    from sklearn.cluster import KMeans
    t0 = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(embeddings)
    elapsed = time.time() - t0
    return kmeans.labels_, elapsed

def main():
    parser = argparse.ArgumentParser(description="Compare traditional KMeans, C3 pipeline, and LLM-as-a-clustering-engine.")
    parser.add_argument(
        "--datasets", 
        type=str, 
        default="go_emotion", 
        help="Comma-separated list of datasets to run or 'all'."
    )
    parser.add_argument(
        "--size", 
        type=str, 
        choices=["small", "large", "smallcopy","300new_aspect", "300new_aspect_time"], 
        default="small", 
        help="Dataset size to use: small or large."
    )
    parser.add_argument(
        "--num_docs", 
        type=int, 
        default=0, 
        help="Number of documents to evaluate (0 for entire dataset)."
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=1, 
        help="Number of runs per dataset to average the results."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini", 
        help="Model name to use with GitHubModelsAgent."
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Base random state for reproducibility."
    )
    parser.add_argument(
        "--use_itml", 
        action="store_true", 
        help="Enable Information-Theoretic Metric Learning (ITML) in the C3 pipeline."
    )
    parser.add_argument(
        "--pca_dim", 
        type=int, 
        default=None, 
        help="PCA dimensionality reduction target for embeddings (e.g. 50). If not specified, keeps raw embedding dimensions."
    )
    
    args = parser.parse_args()
    
    print("=======================================================================")
    print("         CLUSTERING METHODS COMPARISON BENCHMARK (C3 VS BASELINES)      ")
    print("=======================================================================\n")
    
    # 1. Parse datasets list
    if args.datasets.strip().lower() == "all":
        datasets_to_run = list(DATASET_PROMPTS.keys())
    else:
        datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]
        
    print(f"Configured datasets: {datasets_to_run}")
    print(f"Runs to average: {args.runs}")
    print(f"Agent mode: REAL LLM ({args.model})")
    print(f"Dataset Scale: {args.size}")
    if args.num_docs > 0:
        print(f"Document limit: First {args.num_docs} documents")
    else:
        print("Document limit: Entire dataset size\n")
        
    # Initialize embeddings
    print("Initializing SentenceTransformer embeddings model...")
    emb_provider = SentenceTransformerEmbeddings()
    print("Embeddings model loaded.\n")
    
    github_token = os.getenv("GITHUB_TOKEN", "")
    comparison_results = []
    
    for dataset_name in datasets_to_run:
        if dataset_name not in DATASET_PROMPTS:
            print(f"[WARNING] Skipping unsupported dataset '{dataset_name}'")
            continue
            
        print(f"\nEvaluating Dataset: {dataset_name.upper()}")
        print("-" * 60)
        
        filepath = f"data/datasets/{dataset_name}/{args.size}.jsonl"
        try:
            full_texts, full_labels = load_jsonl_dataset(filepath)
        except Exception as e:
            print(f" -> [ERROR] Failed to load dataset: {e}")
            continue
            
        total_docs = len(full_texts)
        if args.num_docs > 0 and args.num_docs < total_docs:
            texts = full_texts[:args.num_docs]
            labels = full_labels[:args.num_docs]
        else:
            texts = full_texts
            labels = full_labels
            
        N = len(texts)
        unique_labels = sorted(list(set(labels)))
        n_clusters = len(unique_labels)
        print(f"Loaded {N} documents for comparison (Unique GT categories: {n_clusters})")
        
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        true_codes = [label_map[lbl] for lbl in labels]
        
        # Resolve user intent prompt template dynamically
        prompt_tmpl = DATASET_PROMPTS[dataset_name]
        if dataset_name == "banking77":
            if args.size == "300new_aspect_time":
                prompt_tmpl = "Group these banking customer service queries purely by their Temporal Focus (the temporal orientation of the query) into exactly {n_clusters} clusters. The allowable clusters are strictly: (1) past (referencing past transactions, historical statements, past actions, or already executed payments/orders), (2) present (dealing with current ongoing states, active configuration attempts, immediate problems, or urgent requests happening right now), and (3) future (pertaining to hypothetical scenarios, general/upcoming timelines, planned actions, or future expectations). YOU MUST FOLLOW THE EXACT NUMBER OF CLUSTERS ({n_clusters}) STRICTLY."
            elif args.size == "300new_aspect":
                prompt_tmpl = "Group these banking customer service queries purely by their Action Intent into exactly {n_clusters} clusters. The allowable clusters are strictly: (1) Information Inquiry (general informational questions, looking up policies, rates, supported currencies, or app features without performing actions or disputing issues), (2) Complaint & Problem (reporting errors, failed transactions, delayed shipments, unexpected fees, or complaining about issues), and (3) Urgent / Action Request (direct requests for active operations, changes, or immediate assistance). YOU MUST FOLLOW THE EXACT NUMBER OF CLUSTERS ({n_clusters}) STRICTLY."
            else:
                prompt_tmpl = "Group these banking customer service queries by their specific fine-grained customer intent or topic (e.g., card arrival, card linking, exchange rate, automatic top up). The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY."
        
        user_intent = prompt_tmpl.format(n_clusters=n_clusters)
        
        # Build embeddings of selected subset
        subset_embeddings = emb_provider.embed_texts(texts)
        if args.pca_dim is not None:
            from sklearn.decomposition import PCA
            n_components = min(args.pca_dim, len(texts), subset_embeddings.shape[1])
            pca = PCA(n_components=n_components, random_state=args.random_state)
            subset_embeddings = pca.fit_transform(subset_embeddings)
            print(f"Applied PCA dimension reduction to benchmark embeddings: {subset_embeddings.shape}")
        
        # We will average over multiple runs
        metrics_accum = {
            "kmeans": {"ari": [], "nmi": [], "acc": [], "time": []},
            "c3": {"ari": [], "nmi": [], "acc": [], "time": [], "input_tokens": [], "output_tokens": []},
            "pure_llm": {"ari": [], "nmi": [], "acc": [], "time": [], "input_tokens": [], "output_tokens": []}
        }
        
        for run_id in range(args.runs):
            seed = args.random_state + run_id
            print(f"   * Run {run_id+1}/{args.runs} (Seed: {seed})...")
            
            # --- METHOD 1: Traditional KMeans ---
            kmeans_labels, kmeans_time = run_traditional_kmeans(subset_embeddings, n_clusters, seed)
            metrics_accum["kmeans"]["ari"].append(compute_ari(true_codes, kmeans_labels))
            metrics_accum["kmeans"]["nmi"].append(compute_nmi(true_codes, kmeans_labels))
            metrics_accum["kmeans"]["acc"].append(compute_acc(true_codes, kmeans_labels))
            metrics_accum["kmeans"]["time"].append(kmeans_time)
            
            # --- METHOD 2: C3 Pipeline ---
            c3_agent = GitHubModelsAgent(api_key=github_token, model_name=args.model, verbose=False)
                
            num_samples = max(15, int(N * 0.2)) # 20% sampling
            c3_pipeline = ConversationalClusteringPipeline(
                embedding_provider=emb_provider,
                agent=c3_agent,
                num_samples=num_samples,
                batch_size=100,
                random_state=seed,
                user_intent=user_intent,
                n_clusters=n_clusters,
                use_itml=args.use_itml,
                pca_dim=args.pca_dim,
                clustering_params={
                    "w": 30.0
                }
            )
            c3_pipeline.set_data(texts)
            
            t_c3_start = time.time()
            try:
                c3_pipeline.run_initial_clustering(bypass_qa=True)
                c3_time = time.time() - t_c3_start
                
                metrics_accum["c3"]["ari"].append(compute_ari(true_codes, c3_pipeline.labels))
                metrics_accum["c3"]["nmi"].append(compute_nmi(true_codes, c3_pipeline.labels))
                metrics_accum["c3"]["acc"].append(compute_acc(true_codes, c3_pipeline.labels))
                metrics_accum["c3"]["time"].append(c3_time)
                
                # Retrieve tokens
                metrics_accum["c3"]["input_tokens"].append(c3_agent.input_tokens)
                metrics_accum["c3"]["output_tokens"].append(c3_agent.output_tokens)
            except Exception as e:
                print(f"     [ERROR] C3 run failed: {e}")
                
            # --- METHOD 3: Pure LLM as Clustering Engine ---
            pure_agent = GitHubModelsAgent(api_key=github_token, model_name=args.model, verbose=False)
                
            t_pure_start = time.time()
            try:
                # Step 1: Candidate Taxonomy Synthesis on representative subset (10% sample)
                # Determine sample size
                n_samples = max(15, int(N * 0.1))
                
                # FPS logic
                pipeline_for_fps = ConversationalClusteringPipeline(
                    embedding_provider=emb_provider,
                    num_samples=n_samples,
                    random_state=seed
                )
                pipeline_for_fps.set_data(texts)
                fps_indices = pipeline_for_fps.sampled_indices
                    
                doc_titles_str = ""
                for idx in fps_indices:
                    snippet = texts[idx][:120].strip() + "..."
                    doc_titles_str += f"[Doc ID: {idx}] {snippet}\n"
                    
                user_intent = user_intent
                step1_system_prompt = f"""You are an expert taxonomy designer. The user's ultimate goal is: {user_intent}

You are given a list of documents represented by their summaries.
Create a draft taxonomy (candidate registry of categories) that covers these documents under the user's intent.
The number of categories should be determined naturally and dynamically based on the semantic variation in the data and the user's intent.
Provide a clear description for each category.
All draft taxonomy category names and descriptions MUST be written strictly in English.

You must return ONLY a valid JSON object matching this schema:
{{
  "candidate_registry": {{
     "Cluster_1": "Description of the first category",
     "Cluster_2": "Description of the second category",
     ...
  }}
}}"""
                step1_user_prompt = f"Here are the documents:\n\n{doc_titles_str}"
                
                response_text = pure_agent.generate_text(step1_system_prompt, step1_user_prompt)
                parsed = parse_json_robustly(response_text)
                candidate_registry = parsed.get("candidate_registry", {})
                
                # Step 2: Guided Classification on all documents in mini-batches
                global_registry = candidate_registry or {}
                assignments = {}
                batch_size = 100
                
                for start_idx in range(0, N, batch_size):
                    batch_indices = list(range(start_idx, min(N, start_idx + batch_size)))
                    
                    docs_list_str = ""
                    for idx in batch_indices:
                        text_content = texts[idx]
                        docs_list_str += f"[Doc ID: {idx}] {text_content}\n\n"
                        
                    registry_str = json.dumps(global_registry, indent=2, ensure_ascii=False)
                    step2_system_prompt = f"""You are an expert data classification assistant. The user's ultimate goal is: {user_intent}

CANDIDATE CLUSTERS REGISTRY:
{registry_str}

INSTRUCTIONS:
Classify the following text documents into the existing candidate clusters in the registry.
- Assign each document to the most appropriate Cluster ID.
- If a document absolutely does not fit any of the existing candidate categories, you are allowed to create a new Cluster_ID (e.g. Cluster_X) with a description and assign the document to it.
All descriptions of new fallback categories MUST be written strictly in English.

You must return ONLY a valid JSON object matching this schema:
{{
  "new_clusters": {{"Cluster_X": "Description of the new fallback category (only if needed)"}},
  "assignments": {{"doc_id_1": "Cluster_1", "doc_id_2": "Cluster_2"}}
}}"""
                    step2_user_prompt = f"Here are the documents for this batch:\n\n{docs_list_str}"
                    
                    response_text = pure_agent.generate_text(step2_system_prompt, step2_user_prompt)
                    output_json = parse_json_robustly(response_text)
                    
                    new_clusters = {}
                    for k, v in output_json.items():
                        if k.lower().replace("_", "") == "newclusters":
                            new_clusters = v
                            break
                            
                    batch_assignments = {}
                    for k, v in output_json.items():
                        if k.lower() == "assignments":
                            batch_assignments = v
                            break
                            
                    for cid, desc in new_clusters.items():
                        global_registry[cid] = desc
                        
                    for doc_id_str, cid in batch_assignments.items():
                        nums = re.findall(r'\d+', doc_id_str)
                        if nums:
                            abs_idx = int(nums[0])
                            if abs_idx < N:
                                assignments[abs_idx] = cid
                                
                # Fill missing
                for i in range(N):
                    if i not in assignments:
                        assignments[i] = "Cluster_unknown"
                        
                # Map to integer label codes
                unique_assigned = sorted(list(set(assignments.values())))
                assign_map = {cid: idx for idx, cid in enumerate(unique_assigned)}
                pure_labels = [assign_map[assignments[i]] for i in range(N)]
                
                pure_time = time.time() - t_pure_start
                metrics_accum["pure_llm"]["ari"].append(compute_ari(true_codes, pure_labels))
                metrics_accum["pure_llm"]["nmi"].append(compute_nmi(true_codes, pure_labels))
                metrics_accum["pure_llm"]["acc"].append(compute_acc(true_codes, pure_labels))
                metrics_accum["pure_llm"]["time"].append(pure_time)
                
                # Retrieve tokens
                metrics_accum["pure_llm"]["input_tokens"].append(pure_agent.input_tokens)
                metrics_accum["pure_llm"]["output_tokens"].append(pure_agent.output_tokens)
            except Exception as e:
                print(f"     [ERROR] Pure LLM run failed: {e}")
                
        # Calculate means and CIs
        def get_stat(values):
            mean, std, ci = compute_confidence_interval(values) if values else (0.0, 0.0, 0.0)
            return {"mean": mean, "std": std, "ci": ci, "raw": values}

        summary = {
            "dataset": dataset_name,
            "docs_count": N,
            "gt_k": n_clusters,
            "kmeans": {
                "ari": get_stat(metrics_accum["kmeans"]["ari"]),
                "nmi": get_stat(metrics_accum["kmeans"]["nmi"]),
                "acc": get_stat(metrics_accum["kmeans"]["acc"]),
                "time": get_stat(metrics_accum["kmeans"]["time"]),
            },
            "c3": {
                "ari": get_stat(metrics_accum["c3"]["ari"]),
                "nmi": get_stat(metrics_accum["c3"]["nmi"]),
                "acc": get_stat(metrics_accum["c3"]["acc"]),
                "time": get_stat(metrics_accum["c3"]["time"]),
                "input_tokens": get_stat(metrics_accum["c3"]["input_tokens"]),
                "output_tokens": get_stat(metrics_accum["c3"]["output_tokens"]),
            },
            "pure_llm": {
                "ari": get_stat(metrics_accum["pure_llm"]["ari"]),
                "nmi": get_stat(metrics_accum["pure_llm"]["nmi"]),
                "acc": get_stat(metrics_accum["pure_llm"]["acc"]),
                "time": get_stat(metrics_accum["pure_llm"]["time"]),
                "input_tokens": get_stat(metrics_accum["pure_llm"]["input_tokens"]),
                "output_tokens": get_stat(metrics_accum["pure_llm"]["output_tokens"]),
            }
        }
        
        comparison_results.append(summary)
        
        # Display individual dataset report
        print(f"\nResults for {dataset_name.upper()} ({N} documents):")
        print(f"| Method | ARI (Mean ± 95% CI) | NMI (Mean ± 95% CI) | ACC (Mean ± 95% CI) | Time (Mean ± 95% CI) | Input Tokens (Mean ± 95% CI) | Output Tokens (Mean ± 95% CI) |")
        print(f"|---|---|---|---|---|---|---|")
        
        km = summary['kmeans']
        c3 = summary['c3']
        pl = summary['pure_llm']
        
        print(f"| Traditional KMeans | {format_metric(km['ari'])} | {format_metric(km['nmi'])} | {format_metric(km['acc'])} | {format_metric(km['time'], is_time=True)} | - | - |")
        print(f"| C3 (Our System)   | {format_metric(c3['ari'])} | {format_metric(c3['nmi'])} | {format_metric(c3['acc'])} | {format_metric(c3['time'], is_time=True)} | {format_metric(c3['input_tokens'], is_token=True)} | {format_metric(c3['output_tokens'], is_token=True)} |")
        print(f"| Pure LLM Cluster  | {format_metric(pl['ari'])} | {format_metric(pl['nmi'])} | {format_metric(pl['acc'])} | {format_metric(pl['time'], is_time=True)} | {format_metric(pl['input_tokens'], is_token=True)} | {format_metric(pl['output_tokens'], is_token=True)} |")
        
    # Final Table Report
    print("\n" + "=" * 140)
    print("                              FINAL CLUSTERING METHOD COMPARISON TABLE")
    print("=" * 140)
    print("| Dataset | Method | ARI (Mean ± 95% CI) | NMI (Mean ± 95% CI) | ACC (Mean ± 95% CI) | Time (Mean ± 95% CI) | Input Tokens (Mean ± 95% CI) | Output Tokens (Mean ± 95% CI) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in comparison_results:
        km = r['kmeans']
        c3 = r['c3']
        pl = r['pure_llm']
        print(f"| {r['dataset']} ({r['docs_count']} docs) | KMeans | {format_metric(km['ari'])} | {format_metric(km['nmi'])} | {format_metric(km['acc'])} | {format_metric(km['time'], is_time=True)} | - | - |")
        print(f"| | C3 (Our System) | {format_metric(c3['ari'])} | {format_metric(c3['nmi'])} | {format_metric(c3['acc'])} | {format_metric(c3['time'], is_time=True)} | {format_metric(c3['input_tokens'], is_token=True)} | {format_metric(c3['output_tokens'], is_token=True)} |")
        print(f"| | Pure LLM Cluster | {format_metric(pl['ari'])} | {format_metric(pl['nmi'])} | {format_metric(pl['acc'])} | {format_metric(pl['time'], is_time=True)} | {format_metric(pl['input_tokens'], is_token=True)} | {format_metric(pl['output_tokens'], is_token=True)} |")
        print("|---|---|---|---|---|---|---|---|")
    print("=" * 140 + "\n")
    
    # Save to JSON
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "compare_clustering_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "size": args.size,
            "runs": args.runs,
            "mock": False,
            "model": args.model,
            "num_docs": args.num_docs,
            "results": comparison_results
        }, f, indent=2, ensure_ascii=False)
    print(f"Comparison raw results saved successfully to: {json_path}")

if __name__ == "__main__":
    main()
