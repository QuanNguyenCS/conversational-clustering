import os
import json
import numpy as np
import matplotlib.pyplot as plt

# File paths for the three aspects
aspect_files = {
    "Topic / Category (K = 8)": "Output/compare_clustering_results_small.json",
    "Action Intent (K = 3)": "Output/compare_clustering_results_new_aspect.json",
    "Temporal Focus (K = 3)": "Output/compare_clustering_results_new_aspect_time.json"
}

# Methods to compare
methods = ["kmeans", "c3", "pure_llm"]
method_labels = {
    "kmeans": "K-Means (Baseline)",
    "c3": "C3 Framework (Ours)",
    "pure_llm": "LLM-as-a-clusterer"
}

# Curated high-contrast academic colors
colors = {
    "kmeans": "#64748b",    # Slate Muted Gray
    "c3": "#2563eb",        # Cyber Blue
    "pure_llm": "#7c3aed"   # Amethyst Purple
}

# Metrics to extract
metrics = ["ari", "nmi", "acc"]
metric_titles = {
    "ari": "Adjusted Rand Index (ARI)",
    "nmi": "Normalized Mutual Info (NMI)",
    "acc": "Clustering Accuracy (ACC)"
}

def extract_data():
    """Extract mean and CI values for each aspect, method, and metric."""
    data = {metric: {method: {"means": [], "cis": []} for method in methods} for metric in metrics}
    aspect_labels = list(aspect_files.keys())
    
    for aspect_name, file_path in aspect_files.items():
        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}. Using mock zero data.")
            for metric in metrics:
                for method in methods:
                    data[metric][method]["means"].append(0.0)
                    data[metric][method]["cis"].append(0.0)
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            res_json = json.load(f)
            
        # Extract first result block
        res_block = res_json["results"][0]
        
        for metric in metrics:
            for method in methods:
                method_data = res_block.get(method, {})
                metric_data = method_data.get(metric, {})
                mean = metric_data.get("mean", 0.0)
                ci = metric_data.get("ci", 0.0)
                
                data[metric][method]["means"].append(mean)
                data[metric][method]["cis"].append(ci)
                
    return aspect_labels, data

def main():
    aspect_labels, data = extract_data()
    
    # Set up matplotlib style for academic presentation
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 16
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=150)
    
    n_metrics = len(metrics)
    x = np.arange(n_metrics)  # Label locations (0, 1, 2 for ARI, NMI, ACC)
    width = 0.25             # Width of the bars
    metric_labels_x = ["ARI", "NMI", "ACC"]
    
    for a_idx, aspect_name in enumerate(aspect_labels):
        ax = axes[a_idx]
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
        
        # Plot bars for each method
        for m_idx, method in enumerate(methods):
            offset = (m_idx - 1) * width
            means = [data[metric][method]["means"][a_idx] for metric in metrics]
            cis = [data[metric][method]["cis"][a_idx] for metric in metrics]
            
            bars = ax.bar(
                x + offset, 
                means, 
                width, 
                yerr=cis, 
                label=method_labels[method],
                color=colors[method],
                edgecolor="none",
                capsize=5,
                error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "#334155"},
                zorder=3
            )
            
            # Add value labels on top of the bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:
                    ax.annotate(
                        f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),  # Offset label slightly
                        textcoords="offset points",
                        ha="center", 
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                        color="#1e293b"
                    )
                    
        ax.set_title(aspect_name, pad=15, fontweight="semibold", color="#0f172a")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels_x)
        ax.set_ylim(0, 1.05)
        
        # Clean spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.tick_params(colors="#475569")
        
        # Legend only on the middle subplot
        if a_idx == 1:
            ax.legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, -0.15),
                ncol=3, 
                frameon=True, 
                facecolor="#ffffff", 
                edgecolor="#e2e8f0"
            )
            
    fig.suptitle("Multi-Aspect Clustering Performance Comparison (Banking77 N = 300)", y=0.97, fontweight="bold", color="#0f172a")
    
    # Adjust layout to fit subtitle and legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.81)
    
    # Save files
    os.makedirs("Output", exist_ok=True)
    out_path = "Output/multi_aspect_comparison.png"
    plt.savefig(out_path, dpi=300, facecolor="#ffffff", edgecolor="none")
    print(f"[SUCCESS] Saved multi-aspect comparison chart to: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
