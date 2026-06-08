import os
import sys
import json
import requests
import numpy as np

# Add src to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset.loader import load_dataset
from src.embeddings.local_embeddings import SentenceTransformerEmbeddings
from src.pipeline import ConversationalClusteringPipeline
from src.agent.local_agent import OllamaLocalAgent
from src.evaluation.metrics import compute_ari, compute_nmi, compute_acc

def transform_dataset():
    """
    Step 1: Transform arxiv_processed_sample.json into arxiv_transformed_test.json
    - id = title
    - text = abstract
    - category = category
    """
    src_path = "data/arxiv_processed.json"
    dest_path = "data/arxiv_transformed_test.json"

    if not os.path.exists(src_path):
        print(f"[ERROR] Source dataset not found at {src_path}.")
        print("Please run: python src/dataset/process_arxiv.py --limit 20")
        sys.exit(1)
        
    print(f"Reading source data from: {src_path}")
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    transformed = []
    for item in data:
        transformed.append({
            "id": item.get("title", ""),
            "text": item.get("abstract", ""),
            "category": item.get("category", "")
        })
        
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully transformed and saved {len(transformed)} records to {dest_path}")
    return dest_path

def check_ollama(model_name="qwen2.5:7b", url="http://localhost:11434"):
    """
    Step 2: Check connection to local Ollama and check if model is pulled.
    """
    print(f"Checking Ollama service at {url}...")
    try:
        res = requests.get(f"{url}/api/tags", timeout=5)
        if res.status_code != 200:
            print(f"[ERROR] Ollama returned status code {res.status_code}")
            sys.exit(1)
            
        models = [m["name"] for m in res.json().get("models", [])]
        print(f" -> Ollama is running. Available models: {models}")
        
        # Check if the model is pulled
        has_model = False
        for m in models:
            if model_name in m or m.startswith("qwen2.5"):
                has_model = True
                model_name = m # use the exact local model name tag
                break
                
        if not has_model:
            print(f"[ERROR] Model '{model_name}' was not found in Ollama.")
            print(f"Please run 'ollama pull {model_name}' in your terminal first.")
            sys.exit(1)
            
        print(f" -> Model '{model_name}' verified successfully.")
        return model_name
    except Exception as e:
        print(f"[ERROR] Could not connect to local Ollama server at {url}.")
        print("Please install Ollama (https://ollama.com/) and make sure it is running.")
        sys.exit(1)

def run_demo():
    print("======================================================================")
    print("      C3 (Conversational Constrained Clustering) - E2E Interactive Demo")
    print("======================================================================\n")

    # 1. Transform dataset
    print("[1/5] Transforming ArXiv sample dataset...")
    dataset_path = transform_dataset()

    # 2. Verify Ollama
    print("\n[2/5] Verifying Ollama and model setup...")
    model_name = check_ollama(model_name="qwen2.5:7b")

    # 3. Load dataset
    print(f"\n[3/5] Loading transformed dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    texts = dataset.get_texts()
    
    # Extract ground truth labels for evaluation metrics
    true_labels = dataset.get_aspect_labels("category")
    unique_types = sorted(list(set(true_labels)))
    label_map = {t: i for i, t in enumerate(unique_types)}
    true_codes = [label_map[l] for l in true_labels]

    # 4. Initialize pipeline
    print("\n[4/5] Initializing C3 pipeline and computing sentence embeddings...")
    emb_provider = SentenceTransformerEmbeddings()
    agent = OllamaLocalAgent(base_url="http://localhost:11434", model_name=model_name)
    
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=10,
        random_state=42
    )
    pipeline.set_data(texts)

    # Run initial unconstrained clustering
    print(" -> Running initial clustering...")
    pipeline.run_initial_clustering()
    
    ari = compute_ari(true_codes, pipeline.labels)
    nmi = compute_nmi(true_codes, pipeline.labels)
    acc = compute_acc(true_codes, pipeline.labels)
    print(f"\n==================== INITIAL CLUSTERING STATE ====================")
    print(f"Number of Clusters (K): {pipeline.n_clusters}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"Clustering Accuracy (ACC): {acc:.4f}")
    
    keywords = pipeline.get_cluster_keywords(top_k=4)
    for l in range(pipeline.n_clusters):
        kw_str = ", ".join(keywords.get(l, []))
        print(f" - Cluster {l} (Keywords: {kw_str})")
        central_indices = pipeline.get_central_documents(top_n=1)[l]
        if central_indices:
            print(f"   Representative doc: \"{texts[central_indices[0]]}\"")

    # 5. Interactive Turns
    print("\n[5/5] Entering Interactive Conversational Loop...")
    
    # Turn 1: Global feedback
    default_feedback_1 = "Group these papers into two main categories: Physics/Astronomy versus Mathematics/Computer Science."
    print(f"\n--- Conversational Turn 1 ---")
    print(f"Suggested default feedback:")
    print(f"  \"{default_feedback_1}\"")
    feedback_1 = input("Enter your global feedback (or press Enter to use default): ").strip()
    if not feedback_1:
        feedback_1 = default_feedback_1
        
    print(f"\n -> Sending feedback: \"{feedback_1}\"")
    print(" -> Generating constraints and re-clustering...")
    pipeline.step(feedback_1)
    
    ari_1 = compute_ari(true_codes, pipeline.labels)
    nmi_1 = compute_nmi(true_codes, pipeline.labels)
    acc_1 = compute_acc(true_codes, pipeline.labels)
    print(f"\n==================== TURN 1 CLUSTERING STATE ====================")
    print(f"Number of Clusters (K): {pipeline.n_clusters}")
    print(f"Must-Link constraints: {len(pipeline.must_link)}")
    print(f"Cannot-Link constraints: {len(pipeline.cannot_link)}")
    print(f"Adjusted Rand Index (ARI): {ari_1:.4f} (Before: {ari:.4f})")
    print(f"Normalized Mutual Info (NMI): {nmi_1:.4f} (Before: {nmi:.4f})")
    print(f"Clustering Accuracy (ACC): {acc_1:.4f} (Before: {acc:.4f})")
    
    keywords_1 = pipeline.get_cluster_keywords(top_k=4)
    for l in range(pipeline.n_clusters):
        kw_str = ", ".join(keywords_1.get(l, []))
        print(f" - Cluster {l} (Keywords: {kw_str})")
        central_indices = pipeline.get_central_documents(top_n=1)[l]
        if central_indices:
            print(f"   Representative doc: \"{texts[central_indices[0]]}\"")

    # Turn 2: Local feedback
    default_feedback_2 = "Split the Physics/Astronomy cluster to separate astronomy papers from other physics topics."
    print(f"\n--- Conversational Turn 2 ---")
    print(f"Suggested default feedback:")
    print(f"  \"{default_feedback_2}\"")
    feedback_2 = input("Enter your local/refinement feedback (or press Enter to use default): ").strip()
    if not feedback_2:
        feedback_2 = default_feedback_2
        
    print(f"\n -> Sending feedback: \"{feedback_2}\"")
    print(" -> Processing local constraints and re-clustering...")
    pipeline.step(feedback_2)
    
    ari_2 = compute_ari(true_codes, pipeline.labels)
    nmi_2 = compute_nmi(true_codes, pipeline.labels)
    acc_2 = compute_acc(true_codes, pipeline.labels)
    print(f"\n==================== TURN 2 CLUSTERING STATE ====================")
    print(f"Number of Clusters (K): {pipeline.n_clusters}")
    print(f"Must-Link constraints: {len(pipeline.must_link)}")
    print(f"Cannot-Link constraints: {len(pipeline.cannot_link)}")
    print(f"Adjusted Rand Index (ARI): {ari_2:.4f} (Turn 1: {ari_1:.4f})")
    print(f"Normalized Mutual Info (NMI): {nmi_2:.4f} (Turn 1: {nmi_1:.4f})")
    print(f"Clustering Accuracy (ACC): {acc_2:.4f} (Turn 1: {acc_1:.4f})")
    
    keywords_2 = pipeline.get_cluster_keywords(top_k=4)
    for l in range(pipeline.n_clusters):
        kw_str = ", ".join(keywords_2.get(l, []))
        print(f" - Cluster {l} (Keywords: {kw_str})")
        central_indices = pipeline.get_central_documents(top_n=1)[l]
        if central_indices:
            print(f"   Representative doc: \"{texts[central_indices[0]]}\"")
            
    print("\n======================================================================")
    print("             Interactive Conversational Clustering E2E Completed!      ")
    print("======================================================================\n")

if __name__ == "__main__":
    run_demo()
