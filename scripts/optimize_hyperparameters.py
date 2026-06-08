import os
import sys
import time
import re
import json
import warnings
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Ensure the root project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import C3 modules
from src.dataset.loader import load_dataset
from src.embeddings.local_embeddings import SentenceTransformerEmbeddings
from src.embeddings.base_embeddings import BaseEmbeddings
from src.pipeline import ConversationalClusteringPipeline
from src.agent.cloud_agent import GitHubModelsAgent
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc

# Configurable parameters
DATASET_PATH = "data/datasets/arxiv_fine/small.jsonl"

NUM_TRIALS = 100
ASPECT = "label"
RANDOM_STATE = 42

# GitHub Models API token fallback from notebooks
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

USER_INTENT = "Group academic paper abstracts by scientific category."

def load_jsonl_dataset(filepath: str):
    """Load JSONL dataset and return texts and labels."""
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
    return texts, labels

# Ignore metric-learn and FutureWarnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="metric_learn")
warnings.filterwarnings("ignore", category=FutureWarning)

class PrecomputedEmbeddings(BaseEmbeddings):
    """A dummy embedding provider that simply returns pre-calculated embeddings."""
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.embeddings

def main():
    print("======================================================================")
    print("         C3 HYPERPARAMETER OPTIMIZATION VIA BAYESIAN SEARCH           ")
    print("                  (USING REAL LLM CONSTRAINTS CACHED)                 ")
    print("======================================================================")
    
    # 1. Load dataset & labels
    print(f"Loading dataset from: {DATASET_PATH}...")
    try:
        texts, true_labels = load_jsonl_dataset(DATASET_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load JSONL dataset: {e}")
        sys.exit(1)
        
    # Convert labels to numeric codes
    unique_types = sorted(list(set(true_labels)))
    label_map = {t: i for i, t in enumerate(unique_types)}
    true_codes = np.array([label_map[l] for l in true_labels])
    n_clusters_expected = len(unique_types)
    print(f"Loaded {len(texts)} texts, expected clusters: {n_clusters_expected}")
    
    # 2. Compute embeddings once (using SentenceTransformers)
    print("Computing base embeddings using SentenceTransformers...")
    t_start = time.time()
    
    emb_provider_raw = SentenceTransformerEmbeddings()
    raw_embeddings = emb_provider_raw.embed_texts(texts)
    
    print(f"Embeddings generated in {time.time() - t_start:.2f}s. Shape: {raw_embeddings.shape}")
    
    # Setup the precomputed provider for pipeline
    emb_provider = PrecomputedEmbeddings(raw_embeddings)
    
    # 3. Initialize real GitHubModelsAgent to run constraint discovery once
    print(f"Initializing real GitHubModelsAgent (gpt-4o-mini) to gather constraints...")
    print(f"Token: {GITHUB_TOKEN[:10]}...{GITHUB_TOKEN[-4:]}")
    
    real_agent = GitHubModelsAgent(api_key=GITHUB_TOKEN, model_name="gpt-4o-mini", verbose=False)
    
    print("Running initial clustering pipeline to call LLM (100 samples in batches of 25)...")
    real_pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=real_agent,
        num_samples=300,
        batch_size=50,
        random_state=RANDOM_STATE,
        user_intent=USER_INTENT,
        use_qa_phase=False
    )
    
    t_llm_start = time.time()
    real_pipeline.set_data(texts)
    real_pipeline.run_initial_clustering(bypass_qa=True)
    
    # Cache the real constraints and assignments
    precomputed_sampled_indices = real_pipeline.sampled_indices
    precomputed_final_assignments = real_pipeline.final_assignments
    precomputed_global_registry = real_pipeline.global_registry
    precomputed_constraint_ledger = real_pipeline.constraint_ledger
    
    print(f"LLM Label Discovery finished in {time.time() - t_llm_start:.2f}s.")
    print(f"  - Discovered clusters: {len(precomputed_global_registry)}")
    print(f"  - Sampled indices: {len(precomputed_sampled_indices)}")
    print(f"  - must-link constraints: {len(real_pipeline.must_link)}")
    print(f"  - cannot-link constraints: {len(real_pipeline.cannot_link)}")
    print("Constraints cached. Beginning hyperparameter optimization trials using Optuna...")
    
    try:

        import optuna
    except ImportError:
        print("[Optuna Error] Optuna library is not installed. Please run: pip install optuna")
        sys.exit(1)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # 4. Define Optuna objective function
    def objective(trial):
        # PCKMeans parameters search space
        pck_max_iter = trial.suggest_int("pck_max_iter", 20, 300)
        pck_tol = trial.suggest_float("pck_tol", 1e-6, 1e-2, log=True)
        pck_w = trial.suggest_float("pck_w", 0.1, 100.0)
        
        clustering_params = {
            "max_iter": pck_max_iter,
            "tol": pck_tol,
            "w": pck_w
        }
        
        # Instantiate pipeline (No LLM agent needed for trials, ITML disabled, PCA fixed at 50)
        pipeline = ConversationalClusteringPipeline(
            embedding_provider=emb_provider,
            agent=None,
            num_samples=300,
            batch_size=50,
            random_state=RANDOM_STATE,
            user_intent=USER_INTENT,
            use_qa_phase=False,
            clustering_params=clustering_params
        )
        
        # Measure execution time
        t0 = time.time()
        
        try:
            pipeline.set_data(texts)
            # Inject precomputed LLM assignments and constraints (set_data calls reset() which clears them)
            pipeline.sampled_indices = precomputed_sampled_indices
            pipeline.final_assignments = precomputed_final_assignments
            pipeline.global_registry = precomputed_global_registry
            pipeline.constraint_ledger = precomputed_constraint_ledger
            
            # Run metric learning and clustering only
            pipeline._refit_and_cluster()
            
            exec_time = time.time() - t0
            ari = compute_ari(true_codes, pipeline.labels)
            nmi = compute_nmi(true_codes, pipeline.labels)
            acc = compute_acc(true_codes, pipeline.labels)
            trial.set_user_attr("acc", acc)
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return 0.0, 0.0, 999.0
            
        return ari, nmi, exec_time

    # 5. Run Study
    print(f"Running Optuna study with {NUM_TRIALS} trials (Multi-Objective: Maximize ARI & NMI, Minimize Time)...")
    study = optuna.create_study(directions=["maximize", "maximize", "minimize"])
    
    t_opt_start = time.time()
    study.optimize(objective, n_trials=NUM_TRIALS)
    print(f"Optimization completed in {time.time() - t_opt_start:.2f}s.")
    
    # 6. Print Results (Pareto Front)
    print("\n======================================================================")
    print("                    OPTIMIZATION RESULTS (PARETO FRONT)               ")
    print("======================================================================")
    
    best_trials = study.best_trials
    print(f"Found {len(best_trials)} Pareto-optimal configurations:\n")
    
    for idx, trial in enumerate(best_trials):
        ari, nmi, duration = trial.values
        acc = trial.user_attrs.get("acc", 0.0)
        params = trial.params
        print(f"--- Configuration #{idx + 1} ---")
        print(f"  [Objectives] ARI: {ari:.4f} | NMI: {nmi:.4f} | ACC: {acc:.4f} | Time: {duration:.3f}s")
        print(f"  [Parameters]")
        print(f"    - PCKMeans params: w={params['pck_w']:.2f}, max_iter={params['pck_max_iter']}, tol={params['pck_tol']:.6f}")
        print()

if __name__ == "__main__":
    import multiprocessing
    # Guard against Windows multiprocessing/joblib/loky spawning re-executing main()
    if multiprocessing.current_process().name == 'MainProcess' and multiprocessing.parent_process() is None:
        main()
