import os
import numpy as np
import matplotlib.pyplot as plt

# Hardcoded data from the sensitivity analysis tables
data = {
    "reddit": {
        "title": "Reddit (N=3217, K=50)",
        "samples": [10, 20, 30],
        "ari": {"means": [0.2280, 0.2068, 0.1972], "cis": [0.0101, 0.0129, 0.0141]},
        "nmi": {"means": [0.4648, 0.4424, 0.4389], "cis": [0.0180, 0.0032, 0.0386]},
        "acc": {"means": [0.3920, 0.3735, 0.3574], "cis": [0.0317, 0.0087, 0.0187]}
    },
    "arxiv_fine": {
        "title": "ArxivS2S (N=1000, K=13)",
        "samples": [10, 20, 30],
        "ari": {"means": [0.3221, 0.3023, 0.2898], "cis": [0.0554, 0.0216, 0.0535]},
        "nmi": {"means": [0.5620, 0.5457, 0.5313], "cis": [0.0508, 0.0189, 0.0537]},
        "acc": {"means": [0.4800, 0.4600, 0.4377], "cis": [0.1152, 0.0282, 0.0743]}
    },
    "banking77": {
        "title": "Banking77 (N=1000, K=25)",
        "samples": [10, 20, 30],
        "ari": {"means": [0.5912, 0.5711, 0.5725], "cis": [0.0597, 0.0609, 0.0537]},
        "nmi": {"means": [0.7970, 0.7839, 0.7833], "cis": [0.0344, 0.0161, 0.0066]},
        "acc": {"means": [0.6730, 0.6703, 0.6743], "cis": [0.0778, 0.0552, 0.0540]}
    }
}

# Aesthetic parameters
metrics = ["ari", "nmi", "acc"]
metric_labels = {
    "ari": "Adjusted Rand Index (ARI)",
    "nmi": "Normalized Mutual Info (NMI)",
    "acc": "Clustering Accuracy (ACC)"
}
colors = {
    "ari": "#2563eb",  # Cyber Blue
    "nmi": "#ea580c",  # Rust/Orange
    "acc": "#16a34a"   # Forest Green
}
markers = {
    "ari": "o",   # Circle
    "nmi": "s",   # Square
    "acc": "^"    # Triangle
}

def main():
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
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150)
    datasets = list(data.keys())
    
    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        ax.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
        
        ds_info = data[ds_name]
        x = ds_info["samples"]
        
        for metric in metrics:
            means = np.array(ds_info[metric]["means"])
            cis = np.array(ds_info[metric]["cis"])
            
            # Plot line
            ax.plot(
                x, 
                means, 
                label=metric_labels[metric], 
                color=colors[metric], 
                marker=markers[metric], 
                markersize=6, 
                linewidth=2.0, 
                zorder=3
            )
            
            # Draw shaded 95% CI region
            ax.fill_between(
                x, 
                means - cis, 
                means + cis, 
                color=colors[metric], 
                alpha=0.15, 
                zorder=2
            )
            
            # Add value labels for readability at each point
            for i, (val, ci) in enumerate(zip(means, cis)):
                ax.annotate(
                    f"{val:.3f}",
                    xy=(x[i], val),
                    xytext=(0, 8 if metric != "acc" else -15),  # Alternating offset to prevent overlaps
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=colors[metric],
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=colors[metric], alpha=0.85, lw=0.5),
                    zorder=4
                )
                
        ax.set_title(ds_info["title"], pad=15, fontweight="semibold", color="#0f172a")
        ax.set_xlabel("Sample Size Percentage", labelpad=10, color="#334155")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{pct}%" for pct in x])
        ax.set_ylim(0.1, 0.95)  # Scale to fit data beautifully
        
        # Clean spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.tick_params(colors="#475569")
        
        # Shared legend on the middle subplot
        if idx == 1:
            ax.legend(
                loc="upper center", 
                bbox_to_anchor=(0.5, -0.2),
                ncol=3, 
                frameon=True, 
                facecolor="#ffffff", 
                edgecolor="#e2e8f0"
            )
            
    fig.suptitle("Sample Size Sensitivity Analysis on PCKMeans Performance", y=0.97, fontweight="bold", color="#0f172a")
    
    # Adjust layout to fit subtitle and legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.81)
    
    # Save output plot
    os.makedirs("Output", exist_ok=True)
    out_path = "Output/sample_size_sensitivity_comparison.png"
    plt.savefig(out_path, dpi=300, facecolor="#ffffff", edgecolor="none")
    print(f"[SUCCESS] Saved sample size sensitivity chart to: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()
