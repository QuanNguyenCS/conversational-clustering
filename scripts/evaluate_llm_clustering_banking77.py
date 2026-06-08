import os
import sys
import re
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import List, Tuple

load_dotenv()

# Ensure standard output/error use UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure the root project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset.loader import load_dataset
from src.embeddings.local_embeddings import SentenceTransformerEmbeddings
from src.pipeline import ConversationalClusteringPipeline
from src.agent.cloud_agent import GitHubModelsAgent, OpenAIAgent, GeminiAgent
from src.agent.local_agent import OllamaLocalAgent
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class PrecomputedEmbeddings:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    def embed_texts(self, texts):
        # Always return the precomputed embeddings for the entire dataset
        return self.embeddings

def load_jsonl(filepath, limit=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    texts, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    item = json.loads(line_str)
                    texts.append(item["input"])
                    labels.append(item["label"])
                    if limit and len(texts) == limit:
                        break
                except Exception:
                    continue
    return texts, labels

def run_initial_discovery(user_intent, agent, raw_embeddings, texts, num_samples, batch_size, random_state):
    """
    Runs only Phase 2 (FPS) and Phase 3 (LLM Label Discovery) of the pipeline.
    """
    pre_provider = PrecomputedEmbeddings(raw_embeddings)
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=pre_provider,
        agent=agent,
        num_samples=num_samples,
        batch_size=batch_size,
        random_state=random_state,
        user_intent=user_intent,
        use_qa_phase=False
    )
    pipeline.set_data(texts)
    # Generate FPS sampled indices
    pipeline.sampled_indices = pipeline._farthest_point_sampling(pipeline.embeddings, num_samples)
    pipeline.sampled_texts = [pipeline.texts[idx] for idx in pipeline.sampled_indices]
    
    # Run Phase 3 Label Discovery
    pipeline._run_label_discovery(pipeline.sampled_indices, is_refinement=False)
    return pipeline.sampled_indices, pipeline.final_assignments, pipeline.global_registry

def sample_cluster_docs_fps(cluster_doc_indices, embeddings, num_samples=20, random_state=42):
    """
    Selects up to num_samples from cluster_doc_indices using Farthest Point Sampling (FPS).
    """
    if len(cluster_doc_indices) <= num_samples:
        return cluster_doc_indices
        
    sub_embs = embeddings[cluster_doc_indices]
    n = len(cluster_doc_indices)
    distances = np.full(n, np.inf)
    sampled_local_indices = []
    
    # Start with first document
    idx = 0
    sampled_local_indices.append(idx)
    
    for _ in range(1, num_samples):
        diff = sub_embs - sub_embs[idx]
        new_distances = np.sum(diff**2, axis=1)
        distances = np.minimum(distances, new_distances)
        idx = np.argmax(distances)
        sampled_local_indices.append(idx)
        
    return [cluster_doc_indices[i] for i in sampled_local_indices]

def parse_json_robustly(text: str) -> dict:
    content = text.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

def evaluate_cluster_consistency(cluster_id, cluster_desc, doc_tuples, aspect_name, user_intent, judge):
    """
    Invokes the LLM Judge to evaluate the coherence of a single cluster.
    """
    if not doc_tuples:
        return []
        
    docs_str = ""
    for doc_id, text in doc_tuples:
        docs_str += f"--- Document ID: {doc_id} ---\n{text}\n\n"
        
    system_prompt = f"""You are an expert academic evaluator checking the quality of document clustering.
The clustering aspect is: "{aspect_name}"
The user's intent was: "{user_intent}"
The cluster description is: "{cluster_desc}" (ID: {cluster_id})

You are provided with a list of documents that were assigned to this cluster.
For each document, determine if it belongs to this cluster according to the cluster description and the main clustering intent.
- Answer 'true' (correct) if the document fits the description.
- Answer 'false' (incorrect) if the document does not fit or belongs to a different topic/theme.

You must respond ONLY with a valid JSON object matching this schema:
{{
  "evaluations": [
    {{
      "doc_id": <document ID as integer>,
      "correct": <true or false>,
      "reason": "<short explanation>"
    }},
    ...
  ]
}}
Do not write any markdown code blocks or additional text."""

    user_prompt = f"Please evaluate these documents:\n\n{docs_str}"
    
    try:
        response_text = judge.generate_text(system_prompt, user_prompt)
        parsed = parse_json_robustly(response_text)
        return parsed.get("evaluations", [])
    except Exception as e:
        print(f"Error judging cluster {cluster_id}: {e}")
        return [{"doc_id": doc_id, "correct": False, "reason": f"Error: {e}"} for doc_id, _ in doc_tuples]

def get_cluster_desc(cid, registry):
    if cid in registry:
        return registry[cid]
    # Suffix mapping: e.g. if cid is "Cluster_1" or "1", map to the 1st key in registry
    nums = re.findall(r'\d+', str(cid))
    if nums:
        idx = int(nums[0]) - 1
        keys = list(registry.keys())
        if 0 <= idx < len(keys):
            return registry[keys[idx]]
    # Fallback to the first key in registry if we can't map
    keys = list(registry.keys())
    return registry[keys[0]] if keys else "Unknown Cluster"

def compute_confidence_interval(values: List[float]) -> Tuple[float, float, float]:
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
    else:
        t_critical = 1.96
        
    ci_margin = t_critical * (std / np.sqrt(n))
    return mean, std, ci_margin

def get_agent(model_name, token):
    if token:
        return GitHubModelsAgent(api_key=token, model_name=model_name, verbose=False)
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAIAgent(model_name=model_name, verbose=False)
    elif os.getenv("GEMINI_API_KEY"):
        gemini_model = "gemini-1.5-pro" if "4o" in model_name and "mini" not in model_name else "gemini-1.5-flash"
        return GeminiAgent(model_name=gemini_model, verbose=False)
    else:
        return OllamaLocalAgent(base_url="http://localhost:11434", model_name="qwen2.5-coder:latest", verbose=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM-as-a-clusterer (Phase 3 Label Discovery) on 3 aspects of Banking77.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Generator model name (default: gpt-4o-mini).")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Judge model name (default: gpt-4o).")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of samples to draw/evaluate (default: 300).")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for label discovery (default: 100).")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per aspect to calculate confidence intervals (default: 3).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="Output", help="Directory to save JSON and chart outputs.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load datasets
    print("Loading datasets...")
    # Paths for Banking77 aspects
    path_topic = "data/datasets/banking77/small.jsonl"
    path_intent = "data/datasets/banking77/300new_aspect.jsonl"
    path_time = "data/datasets/banking77/300new_aspect_time.jsonl"
    
    texts_topic, labels_topic = load_jsonl(path_topic, limit=args.num_samples)
    texts_intent, labels_intent = load_jsonl(path_intent)
    texts_time, labels_time = load_jsonl(path_time)
    
    # Sanity checks
    assert len(texts_topic) == args.num_samples, f"Topic dataset has {len(texts_topic)} rows, expected {args.num_samples}."
    assert len(texts_intent) == args.num_samples, f"Intent dataset has {len(texts_intent)} rows, expected {args.num_samples}."
    assert len(texts_time) == args.num_samples, f"Time dataset has {len(texts_time)} rows, expected {args.num_samples}."
    
    print(f"Successfully loaded {args.num_samples} texts for each of the 3 aspects.")
    
    # 2. Configure agents
    github_token = os.getenv("GITHUB_TOKEN", "")
    generator = get_agent(args.model, github_token)
    judge = get_agent(args.judge_model, github_token)
    
    print(f"Generator Agent: {generator.__class__.__name__} ({args.model})")
    print(f"Judge Agent: {judge.__class__.__name__} ({args.judge_model})")
    
    # Initialize embedding model
    print("Initializing SentenceTransformer embeddings model...")
    emb_provider = SentenceTransformerEmbeddings()
    
    # 4. Set up Aspect evaluation prompts and definitions
    aspects = {
        "Topic/Category": {
            "texts": texts_topic,
            "labels": labels_topic,
            "prompt_tmpl": "Group these banking customer service queries by their specific fine-grained customer intent or topic (e.g., card arrival, card linking, exchange rate, automatic top up). The EXACT NUMBER OF CLUSTERS IS {n_clusters} AND YOU HAVE TO FOLLOW IT STRICTLY."
        },
        "Action Intent": {
            "texts": texts_intent,
            "labels": labels_intent,
            "prompt_tmpl": "Group these banking customer service queries purely by their Action Intent into exactly {n_clusters} clusters. The allowable clusters are strictly: (1) Information Inquiry (general informational questions, looking up policies, rates, supported currencies, or app features without performing actions or disputing issues), (2) Complaint & Problem (reporting errors, failed transactions, delayed shipments, unexpected fees, or complaining about issues), and (3) Urgent / Action Request (direct requests for active operations, changes, or immediate assistance). YOU MUST FOLLOW THE EXACT NUMBER OF CLUSTERS ({n_clusters}) STRICTLY."
        },
        "Temporal Focus": {
            "texts": texts_time,
            "labels": labels_time,
            "prompt_tmpl": "Group these banking customer service queries purely by their Temporal Focus (the temporal orientation of the query) into exactly {n_clusters} clusters. The allowable clusters are strictly: (1) past (referencing past transactions, historical statements, past actions, or already executed payments/orders), (2) present (dealing with current ongoing states, active configuration attempts, immediate problems, or urgent requests happening right now), and (3) future (pertaining to hypothetical scenarios, general/upcoming timelines, planned actions, or future expectations). YOU MUST FOLLOW THE EXACT NUMBER OF CLUSTERS ({n_clusters}) STRICTLY."
        }
    }
    
    results = {}
    
    # Iterate over the aspects
    for aspect_name, config in aspects.items():
        print("\n" + "="*80)
        print(f"EVALUATING ASPECT: {aspect_name} over {args.runs} runs")
        print("="*80)
        
        aspect_texts = config["texts"]
        y_true_labels = config["labels"]
        
        # Compute embeddings for this specific aspect's texts
        print(f"Computing embeddings for {aspect_name} aspect...")
        raw_embeddings = emb_provider.embed_texts(aspect_texts)
        
        unique_gt = sorted(list(set(y_true_labels)))
        n_clusters_gt = len(unique_gt)
        print(f"Ground Truth labels: {n_clusters_gt} unique categories.")
        
        user_intent = config["prompt_tmpl"].format(n_clusters=n_clusters_gt)
        
        run_aris = []
        run_nmis = []
        run_accs = []
        run_purities = []
        run_ks = []
        
        # Loop over runs
        for run_id in range(args.runs):
            seed = args.random_state + run_id
            print(f"\n--- Run {run_id + 1}/{args.runs} (Random Seed: {seed}) ---")
            
            # Run Phase 3 Discovery
            print("Running initial LLM discovery phase...")
            sampled_indices, assignments, registry = run_initial_discovery(
                user_intent=user_intent,
                agent=generator,
                raw_embeddings=raw_embeddings,
                texts=aspect_texts,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                random_state=seed
            )
            
            print(f"Discovered {len(registry)} clusters.")
            for cid, desc in registry.items():
                print(f"  - {cid}: {desc}")
                
            # Ground Truth Metrics inside the subset (which has N=300 samples)
            label_map = {lbl: i for i, lbl in enumerate(unique_gt)}
            y_true = np.array([label_map[y_true_labels[idx]] for idx in sampled_indices])
            
            assigned_labels = list(assignments.values())
            if not assigned_labels:
                assigned_labels = sorted(list(registry.keys()))
            unique_pred = sorted(list(set(assigned_labels)))
            pred_map = {c: i for i, c in enumerate(unique_pred)}
            y_pred = np.array([pred_map[assignments.get(idx, unique_pred[0])] for idx in sampled_indices])
            
            ari_val = compute_ari(y_true, y_pred)
            nmi_val = compute_nmi(y_true, y_pred)
            acc_val = compute_acc(y_true, y_pred)
            
            print(f"Subset Metrics: ARI={ari_val:.4f}, NMI={nmi_val:.4f}, ACC={acc_val:.4f}")
            
            run_aris.append(ari_val)
            run_nmis.append(nmi_val)
            run_accs.append(acc_val)
            run_ks.append(len(registry))
            
            # LLM as a Judge Coherence check
            print("Running LLM-as-a-Judge Coherence checks...")
            cluster_purities = {}
            for cid in unique_pred:
                cluster_docs = [idx for idx in sampled_indices if assignments.get(idx) == cid]
                if not cluster_docs:
                    continue
                
                desc = get_cluster_desc(cid, registry)
                sampled_docs = sample_cluster_docs_fps(cluster_docs, raw_embeddings, num_samples=20, random_state=seed)
                doc_tuples = [(idx, aspect_texts[idx]) for idx in sampled_docs]
                
                evals = evaluate_cluster_consistency(
                    cluster_id=cid,
                    cluster_desc=desc,
                    doc_tuples=doc_tuples,
                    aspect_name=aspect_name,
                    user_intent=user_intent,
                    judge=judge
                )
                
                correct = sum(1 for e in evals if e.get("correct") is True)
                total = len(evals)
                purity = correct / total if total > 0 else 0.0
                cluster_purities[cid] = purity
                
            macro_purity = np.mean(list(cluster_purities.values())) if cluster_purities else 0.0
            print(f"Macro-Averaged Judge Coherence Purity for Run: {macro_purity:.2%}")
            run_purities.append(macro_purity)
            
        # After runs, compute metrics mean and CI
        ari_mean, _, ari_ci = compute_confidence_interval(run_aris)
        nmi_mean, _, nmi_ci = compute_confidence_interval(run_nmis)
        acc_mean, _, acc_ci = compute_confidence_interval(run_accs)
        purity_mean, _, purity_ci = compute_confidence_interval(run_purities)
        mean_k = float(np.mean(run_ks))
        
        results[aspect_name] = {
            "ground_truth_k": n_clusters_gt,
            "discovered_k_mean": mean_k,
            "ari_mean": ari_mean,
            "ari_ci": ari_ci,
            "nmi_mean": nmi_mean,
            "nmi_ci": nmi_ci,
            "acc_mean": acc_mean,
            "acc_ci": acc_ci,
            "judge_purity_mean": purity_mean,
            "judge_purity_ci": purity_ci,
            "raw_metrics": {
                "ari": run_aris,
                "nmi": run_nmis,
                "acc": run_accs,
                "judge_purity": run_purities,
                "discovered_k": run_ks
            }
        }
        
    # 5. Print Summary Table
    print("\n" + "="*110)
    print("                                   EVALUATION RESULTS SUMMARY TABLE")
    print("="*110)
    print("| Aspect | GT K | Discovered K (Mean) | ARI (Mean ± 95% CI) | NMI (Mean ± 95% CI) | ACC (Mean ± 95% CI) | LLM Judge Purity (Mean ± 95% CI) |")
    print("|---|---|---|---|---|---|---|")
    for aspect_name, res in results.items():
        print(f"| {aspect_name} | {res['ground_truth_k']} | {res['discovered_k_mean']:.2f} | {res['ari_mean']:.4f} ± {res['ari_ci']:.4f} | {res['nmi_mean']:.4f} ± {res['nmi_ci']:.4f} | {res['acc_mean']:.4f} ± {res['acc_ci']:.4f} | {res['judge_purity_mean']:.2%} ± {res['judge_purity_ci']:.2%} |")
    print("="*110 + "\n")
    
    # 6. Save results to JSON
    json_path = os.path.join(args.output_dir, "banking77_multi_aspect_llm_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "judge_model": args.judge_model,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "runs": args.runs,
            "results": results
        }, f, indent=4)
    print(f"Detailed evaluation metrics saved to: {json_path}")
    
    # 7. Generate Visualization Chart
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    
    aspect_names = list(results.keys())
    
    # Left Subplot: Ground Truth Metrics (grouped bar chart)
    x = np.arange(len(aspect_names))
    width = 0.25
    
    aris = [results[a]["ari_mean"] for a in aspect_names]
    aris_ci = [results[a]["ari_ci"] for a in aspect_names]
    nmis = [results[a]["nmi_mean"] for a in aspect_names]
    nmis_ci = [results[a]["nmi_ci"] for a in aspect_names]
    accs = [results[a]["acc_mean"] for a in aspect_names]
    accs_ci = [results[a]["acc_ci"] for a in aspect_names]
    
    rects1 = ax1.bar(x - width, aris, width, yerr=aris_ci, label="ARI", color="#2563eb", edgecolor="none", capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#334155"}, zorder=3)
    rects2 = ax1.bar(x, nmis, width, yerr=nmis_ci, label="NMI", color="#7c3aed", edgecolor="none", capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#334155"}, zorder=3)
    rects3 = ax1.bar(x + width, accs, width, yerr=accs_ci, label="ACC", color="#10b981", edgecolor="none", capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#334155"}, zorder=3)
    
    ax1.set_facecolor('#ffffff')
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax1.set_ylabel("Metric Value")
    ax1.set_title("Ground-Truth Evaluation (Subset Metrics)", pad=10, fontweight="semibold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(aspect_names)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")
    
    # Add value labels
    for rects in [rects1, rects2, rects3]:
        for bar in rects:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1e293b"
            )
            
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#cbd5e1")
    ax1.spines["bottom"].set_color("#cbd5e1")
    ax1.tick_params(colors="#475569")
    
    # Right Subplot: LLM Judge Purity
    purities = [results[a]["judge_purity_mean"] for a in aspect_names]
    purities_ci = [results[a]["judge_purity_ci"] for a in aspect_names]
    colors = ["#2563eb", "#7c3aed", "#10b981"]
    
    rects_purity = ax2.bar(aspect_names, purities, yerr=purities_ci, color=colors, width=0.4, alpha=0.85, capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#334155"}, zorder=3)
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax2.set_ylabel("Macro-Averaged Coherence Purity")
    ax2.set_title("LLM-as-a-Judge Consistency Purity", pad=10, fontweight="semibold")
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bar in rects_purity:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.2%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#1e293b"
        )
        
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["left"].set_color("#cbd5e1")
    ax2.spines["bottom"].set_color("#cbd5e1")
    ax2.tick_params(colors="#475569")
    
    fig.suptitle(f"LLM-as-a-Clusterer Multi-Aspect Evaluation (Banking77 N = {args.num_samples})", y=0.97, fontweight="bold", fontsize=15, color="#0f172a")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    chart_path = os.path.join(args.output_dir, "banking77_multi_aspect_llm_eval.png")
    plt.savefig(chart_path, dpi=300, facecolor="#ffffff", edgecolor="none")
    print(f"Visualization chart saved to: {chart_path}")
    plt.close()
    
    print("\nEvaluation successfully completed!")

if __name__ == "__main__":
    main()
