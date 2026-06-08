import os
import sys
import json
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
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
from src.pipeline import ConversationalClusteringPipeline
from src.agent.cloud_agent import GitHubModelsAgent
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc

# Ignore metric-learn and FutureWarnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="metric_learn")
warnings.filterwarnings("ignore", category=FutureWarning)

# Detailed dataset prompts (user intents) and their cluster count mappings
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate conversational clustering sensitivity under varying sample sizes.")
    parser.add_argument(
        "--datasets", 
        type=str, 
        default="go_emotion", 
        help="Comma-separated list of datasets to run or 'all'."
    )
    parser.add_argument(
        "--size", 
        type=str, 
        choices=["small", "large","smallcopy"], 
        default="small", 
        help="Dataset size to use: small or large."
    )
    parser.add_argument(
        "--runs", 
        type=int, 
        default=3, 
        help="Number of runs per sample size to compute confidence intervals."
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
    
    args = parser.parse_args()
    
    print("=======================================================================")
    print("               SAMPLE SIZE SENSITIVITY EVALUATION                      ")
    print("=======================================================================\n")
    
    # 1. Parse datasets list
    if args.datasets.strip().lower() == "all":
        datasets_to_run = list(DATASET_PROMPTS.keys())
    else:
        datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]
        
    print(f"Configured datasets: {datasets_to_run}")
    print(f"Sample size percentages: 10%, 20%, 30%")
    print(f"Runs per configuration: {args.runs}")
    print(f"Agent mode: REAL LLM ({args.model})")
    print(f"Dataset Scale: {args.size}\n")
    
    # Initialize embeddings
    print("Initializing SentenceTransformer embeddings...")
    emb_provider = SentenceTransformerEmbeddings()
    print("Embeddings model loaded.\n")
    
    github_token = os.getenv("GITHUB_TOKEN", "")
    
    percentages = [20, 30]
    
    results = {}
    
    for dataset_name in datasets_to_run:
        if dataset_name not in DATASET_PROMPTS:
            print(f"[WARNING] Skipping unsupported dataset '{dataset_name}'")
            continue
            
        print(f"\nProcessing Dataset: {dataset_name.upper()}")
        print("=" * 60)
        
        filepath = f"data/datasets/{dataset_name}/{args.size}.jsonl"
        try:
            texts, labels = load_jsonl_dataset(filepath)
        except Exception as e:
            print(f" -> [ERROR] Failed to load dataset: {e}")
            continue
            
        unique_labels = sorted(list(set(labels)))
        n_clusters = len(unique_labels)
        total_docs = len(texts)
        print(f"Loaded {total_docs} docs, ground truth categories: {n_clusters}")
        
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        true_codes = [label_map[lbl] for lbl in labels]
        
        results[dataset_name] = {}
        
        for pct in percentages:
            # Determine target samples
            num_samples = int(round((pct / 100.0) * total_docs))
            # Enforce reasonable minimum/maximum bounds
            num_samples = max(10, min(total_docs, num_samples))
            print(f"\n -> Evaluating {pct}% sample size ({num_samples} samples)...")
            
            run_aris = []
            run_nmis = []
            run_accs = []
            run_times = []
            run_input_tokens = []
            run_output_tokens = []
            
            for run_id in range(args.runs):
                seed = args.random_state + run_id
                
                # Instantiating Agent
                agent = GitHubModelsAgent(api_key=github_token, model_name=args.model, verbose=False)
                    
                pipeline = ConversationalClusteringPipeline(
                    embedding_provider=emb_provider,
                    agent=agent,
                    num_samples=num_samples,
                    batch_size=100,
                    random_state=seed,
                    user_intent=DATASET_PROMPTS[dataset_name].format(n_clusters=n_clusters),
                    use_qa_phase=False,
                    n_clusters=n_clusters,
                    clustering_params={"w": 30},
                    use_itml=True,
                )
                pipeline.set_data(texts)
                
                try:
                    import time
                    start_time = time.time()
                    pipeline.run_initial_clustering(bypass_qa=True)
                    elapsed_time = time.time() - start_time
                    
                    ari = compute_ari(true_codes, pipeline.labels)
                    nmi = compute_nmi(true_codes, pipeline.labels)
                    acc = compute_acc(true_codes, pipeline.labels)
                    
                    run_aris.append(ari)
                    run_nmis.append(nmi)
                    run_accs.append(acc)
                    run_times.append(elapsed_time)
                    run_input_tokens.append(agent.input_tokens)
                    run_output_tokens.append(agent.output_tokens)
                except Exception as e:
                    print(f"    * Run {run_id+1} failed: {e}")
                    
            if not run_aris:
                print(f"    [WARNING] All runs failed for {pct}%")
                continue
                
            ari_mean, ari_std, ari_ci = compute_confidence_interval(run_aris)
            nmi_mean, nmi_std, nmi_ci = compute_confidence_interval(run_nmis)
            acc_mean, acc_std, acc_ci = compute_confidence_interval(run_accs)
            time_mean, time_std, time_ci = compute_confidence_interval(run_times)
            input_tokens_mean, input_tokens_std, input_tokens_ci = compute_confidence_interval(run_input_tokens)
            output_tokens_mean, output_tokens_std, output_tokens_ci = compute_confidence_interval(run_output_tokens)
            
            print(f"    Result: ARI={ari_mean:.4f}±{ari_ci:.4f} | NMI={nmi_mean:.4f}±{nmi_ci:.4f} | ACC={acc_mean:.4f}±{acc_ci:.4f} | Time={time_mean:.2f}s±{time_ci:.2f}s | Input Tokens={input_tokens_mean:.1f}±{input_tokens_ci:.1f} | Output Tokens={output_tokens_mean:.1f}±{output_tokens_ci:.1f}")
            
            results[dataset_name][pct] = {
                "num_samples": num_samples,
                "ari_mean": ari_mean, "ari_ci": ari_ci,
                "nmi_mean": nmi_mean, "nmi_ci": nmi_ci,
                "acc_mean": acc_mean, "acc_ci": acc_ci,
                "time_mean": time_mean, "time_ci": time_ci,
                "input_tokens_mean": input_tokens_mean, "input_tokens_ci": input_tokens_ci,
                "output_tokens_mean": output_tokens_mean, "output_tokens_ci": output_tokens_ci,
                "raw_ari": run_aris,
                "raw_nmi": run_nmis,
                "raw_acc": run_accs,
                "raw_time": run_times,
                "raw_input_tokens": run_input_tokens,
                "raw_output_tokens": run_output_tokens
            }

    # 2. Print Summary Table
    print("\n" + "=" * 120)
    print("                 SAMPLE SIZE SENSITIVITY SUMMARY TABLE")
    print("=" * 120)
    print("| Dataset | Sample % | Sample Size | ARI (Mean ± 95% CI) | NMI (Mean ± 95% CI) | ACC (Mean ± 95% CI) | Time (Mean ± 95% CI) | Input Tokens (Mean ± 95% CI) | Output Tokens (Mean ± 95% CI) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for dataset_name, pct_data in results.items():
        for pct in percentages:
            if pct in pct_data:
                d = pct_data[pct]
                print(f"| {dataset_name} | {pct}% | {d['num_samples']} | {d['ari_mean']:.4f} ± {d['ari_ci']:.4f} | {d['nmi_mean']:.4f} ± {d['nmi_ci']:.4f} | {d['acc_mean']:.4f} ± {d['acc_ci']:.4f} | {d['time_mean']:.2f}s ± {d['time_ci']:.2f}s | {d['input_tokens_mean']:.1f} ± {d['input_tokens_ci']:.1f} | {d['output_tokens_mean']:.1f} ± {d['output_tokens_ci']:.1f} |")
    print("=" * 120 + "\n")
    
    # 3. Plot Line Charts
    print("Generating sensitivity plots...")
    num_datasets = len(results)
    if num_datasets == 0:
        print("No results to plot.")
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metrics_info = [
        ("ari_mean", "ari_ci", "Adjusted Rand Index (ARI)", axes[0]),
        ("nmi_mean", "nmi_ci", "Normalized Mutual Info (NMI)", axes[1]),
        ("acc_mean", "acc_ci", "Unsupervised Accuracy (ACC)", axes[2])
    ]
    
    # Sleek palette and styling parameters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for metric_key, ci_key, metric_title, ax in metrics_info:
        for idx, (dataset_name, pct_data) in enumerate(results.items()):
            x = []
            y = []
            ci = []
            for pct in percentages:
                if pct in pct_data:
                    x.append(pct)
                    y.append(pct_data[pct][metric_key])
                    ci.append(pct_data[pct][ci_key])
            
            if not x:
                continue
                
            x = np.array(x)
            y = np.array(y)
            ci = np.array(ci)
            
            color = colors[idx % len(colors)]
            line, = ax.plot(x, y, marker='o', linewidth=2, label=dataset_name, color=color)
            ax.fill_between(x, y - ci, y + ci, color=line.get_color(), alpha=0.15)
            
        ax.set_title(metric_title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Sample Size Percentage (%)", fontsize=10)
        ax.set_ylabel("Metric Score", fontsize=10)
        ax.set_xticks(percentages)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax == axes[0]:
            ax.legend(loc="best", frameon=True, facecolor='white', framealpha=0.9)
            
    plt.tight_layout()
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "sample_size_sensitivity.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Line plot saved successfully to: {plot_path}")
    
    # 4. Save raw results to JSON
    json_path = os.path.join(output_dir, "sample_size_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "size": args.size,
            "runs": args.runs,
            "mock": False,
            "model": args.model,
            "percentages": percentages,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Raw results saved to: {json_path}")

if __name__ == "__main__":
    main()
