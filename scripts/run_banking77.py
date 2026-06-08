import os
import sys
import json
import argparse
import warnings
import numpy as np
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

# Ensure the root project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Reconfigure stdout/stderr to use UTF-8 on Windows systems
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


# Import C3 modules
from src.embeddings.local_embeddings import SentenceTransformerEmbeddings
from src.pipeline import ConversationalClusteringPipeline
from src.agent.cloud_agent import GitHubModelsAgent
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc

# Ignore metric-learn and FutureWarnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="metric_learn")
warnings.filterwarnings("ignore", category=FutureWarning)

# Detailed dataset prompts (user intents) aligned with their respective ground truth aspects
DATASET_PROMPTS = {
    "arxiv_fine": "Group these scientific papers and articles by their specific academic research sub-category or sub-field. We prefer a highly fine-grained and detailed granularity of categories, separating narrow scientific topics rather than broad academic disciplines. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
    "banking77": "Group these banking customer service queries by their specific, fine-grained service query intent. We require a highly specific and fine-grained categorization where each distinct banking action or issue forms its own cluster. The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY.",
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
    """
    Load a JSON Lines dataset (e.g. banking77 or massive_intent).
    Extracts 'input'/'text' field as text and 'label'/'category' field as the category label.
    """
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
        raise ValueError(f"No valid records found in {filepath}. Make sure it has 'input'/'text' and 'label'/'category' fields.")
        
    return texts, labels

def compute_confidence_interval(values: List[float]) -> Tuple[float, float, float]:
    """
    Calculate the mean, standard deviation, and 95% confidence interval margin of error
    using the Student's t-distribution critical values for small sample sizes.
    """
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0, 0.0
    std = np.std(values, ddof=1)
    
    # Student's t critical values for 95% confidence (two-tailed)
    # for df = n - 1
    if n == 2:
        t_critical = 12.706
    elif n == 3:
        t_critical = 4.303
    elif n == 4:
        t_critical = 3.182
    elif n == 5:
        t_critical = 2.776
    else:
        t_critical = 1.96  # Normal approximation fallback
        
    margin = t_critical * (std / np.sqrt(n))
    return float(mean), float(std), float(margin)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Conversational Clustering across multiple datasets without Q&A.")
    parser.add_argument(
        "--datasets", 
        type=str, 
        default="all", 
        help="Comma-separated list of datasets to run (e.g., 'banking77,arxiv_fine') or 'all' for all 14 datasets."
    )
    parser.add_argument(
        "--size", 
        type=str, 
        choices=["small", "large"], 
        default="small", 
        help="Dataset size to use: small or large (default: small)."
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=3, 
        help="Number of runs per dataset to calculate the confidence interval (default: 3)."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini", 
        help="Model name to use with GitHubModelsAgent (default: gpt-4o-mini)."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=300, 
        help="Number of samples to draw via Farthest Point Sampling (default: 200)."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100, 
        help="Discovery batch size for label discovery (default: 100)."
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Base random state for reproducibility (each run uses random_state + run_index)."
    )
    
    args = parser.parse_args()
    
    print("=======================================================================")
    print("      C3 MULTI-DATASET BATCH EVALUATION (BYPASSING Q&A INTERVIEWS)      ")
    print("=======================================================================\n")
    
    # 1. Parse datasets list
    if args.datasets.strip().lower() == "all":
        datasets_to_run = list(DATASET_PROMPTS.keys())
    else:
        datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]
        
    print(f"Configured to run {len(datasets_to_run)} datasets: {datasets_to_run}")
    print(f"Number of runs per dataset: {args.runs}")
    print(f"Clustering constraints model: {args.model}")
    print(f"FPS Sample count: {args.num_samples}")
    print(f"Mini-batch size: {args.batch_size}")
    print(f"Dataset Scale: {args.size}\n")
    
    # 2. Initialize Embeddings
    print("Initializing SentenceTransformer embeddings model...")
    emb_provider = SentenceTransformerEmbeddings()
    print(" -> Embeddings model loaded successfully.\n")
    
    # Configure LLM token
    github_token = os.getenv("GITHUB_TOKEN", "")
    
    summary_results = []
    
    # Loop over all datasets
    for dataset_idx, dataset_name in enumerate(datasets_to_run):
        if dataset_name not in DATASET_PROMPTS:
            print(f"[WARNING] Dataset '{dataset_name}' not supported. Skipping.")
            continue
            
        print(f"\n[{dataset_idx + 1}/{len(datasets_to_run)}] PROCESSING DATASET: {dataset_name.upper()}")
        print("-" * 50)
        
        # Determine dataset path
        filepath = f"data/datasets/{dataset_name}/{args.size}.jsonl"
        
        try:
            texts, labels = load_jsonl_dataset(filepath)
        except Exception as e:
            print(f" -> [ERROR] Failed to load dataset '{dataset_name}': {e}. Skipping.")
            continue
            
        unique_labels = sorted(list(set(labels)))
        n_clusters = len(unique_labels)
        print(f" -> Loaded {len(texts)} documents with {n_clusters} ground truth categories.")
        
        # Map labels to integer codes
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        true_codes = [label_map[lbl] for lbl in labels]
        
        run_aris = []
        run_nmis = []
        run_accs = []
        discovered_ks = []
        
        # Perform runs for confidence interval calculation
        for run_id in range(args.runs):
            seed = args.random_state + run_id
            print(f"   * Run {run_id + 1}/{args.runs} (Random Seed: {seed})...")
            
            # Instantiating Agent
            agent = GitHubModelsAgent(api_key=github_token, model_name=args.model, verbose=False)
                
            # Initialize Pipeline
            pipeline = ConversationalClusteringPipeline(
                embedding_provider=emb_provider,
                agent=agent,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                random_state=seed,
                user_intent=DATASET_PROMPTS[dataset_name].format(n_clusters=n_clusters),
                use_qa_phase=False,
                n_clusters=n_clusters,
                use_itml=True,
                pca_dim=50,
                cl_distance_threshold_ratio=1.5,
                clustering_params={'w':30}
            )
            pipeline.set_data(texts)
            
            # Run clustering from scratch with Q&A bypassed
            try:
                pipeline.run_initial_clustering(bypass_qa=True)
                
                # Compute metrics
                ari = compute_ari(true_codes, pipeline.labels)
                nmi = compute_nmi(true_codes, pipeline.labels)
                acc = compute_acc(true_codes, pipeline.labels)
                
                run_aris.append(ari)
                run_nmis.append(nmi)
                run_accs.append(acc)
                discovered_ks.append(pipeline.n_clusters)
                
                print(f"     Metrics: ARI={ari:.4f}, NMI={nmi:.4f}, ACC={acc:.4f}, Clusters Discovered={pipeline.n_clusters}")
            except Exception as e:
                print(f"     [ERROR] Run {run_id + 1} failed: {e}")
                
        if not run_aris:
            print(f" -> [ERROR] All runs failed for dataset '{dataset_name}'.")
            continue
            
        # Compute confidence interval statistics
        ari_mean, ari_std, ari_ci = compute_confidence_interval(run_aris)
        nmi_mean, nmi_std, nmi_ci = compute_confidence_interval(run_nmis)
        acc_mean, acc_std, acc_ci = compute_confidence_interval(run_accs)
        mean_k = int(round(np.mean(discovered_ks)))
        
        print(f" -> Results Summary for {dataset_name}:")
        print(f"    ARI: {ari_mean:.4f} ± {ari_ci:.4f}")
        print(f"    NMI: {nmi_mean:.4f} ± {nmi_ci:.4f}")
        print(f"    ACC: {acc_mean:.4f} ± {acc_ci:.4f}")
        
        summary_results.append({
            "dataset": dataset_name,
            "total_docs": len(texts),
            "ground_truth_k": n_clusters,
            "discovered_k": mean_k,
            "ari_mean": ari_mean,
            "ari_ci": ari_ci,
            "nmi_mean": nmi_mean,
            "nmi_ci": nmi_ci,
            "acc_mean": acc_mean,
            "acc_ci": acc_ci,
            "raw_metrics": {
                "ari": run_aris,
                "nmi": run_nmis,
                "acc": run_accs,
                "discovered_k": discovered_ks
            }
        })
        
    # 3. Print Final Summary Markdown Table
    print("\n" + "=" * 90)
    print("                             FINAL EVALUATION SUMMARY TABLE")
    print("=" * 90)
    print("| Dataset | Size | GT K | Discovered K (Mean) | ARI (Mean ± 95% CI) | NMI (Mean ± 95% CI) | ACC (Mean ± 95% CI) |")
    print("|---|---|---|---|---|---|---|")
    for r in summary_results:
        print(f"| {r['dataset']} | {r['total_docs']} | {r['ground_truth_k']} | {r['discovered_k']} | {r['ari_mean']:.4f} ± {r['ari_ci']:.4f} | {r['nmi_mean']:.4f} ± {r['nmi_ci']:.4f} | {r['acc_mean']:.4f} ± {r['acc_ci']:.4f} |")
    print("=" * 90 + "\n")
    
    # 4. Save results to output JSON file
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_results.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "size": args.size,
            "runs": args.runs,
            "model": args.model,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "datasets": summary_results
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Detailed raw results saved to: {output_path}")

if __name__ == "__main__":
    main()
