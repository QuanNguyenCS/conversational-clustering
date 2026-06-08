import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    json_path = "Output/banking77_multi_aspect_llm_eval.json"
    
    if not os.path.exists(json_path):
        print(f"[ERROR] Evaluation JSON file not found at: {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    results = data.get("results", {})
    aspect_names = list(results.keys())
    
    # 1. Matplotlib academic style configurations
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    
    # 2. Left Subplot: Ground Truth Metrics (grouped bar chart)
    x = np.arange(len(aspect_names))
    width = 0.25
    
    aris = [results[a].get("ari_mean", results[a].get("ari", 0.0)) for a in aspect_names]
    aris_ci = [results[a].get("ari_ci", 0.0) for a in aspect_names]
    nmis = [results[a].get("nmi_mean", results[a].get("nmi", 0.0)) for a in aspect_names]
    nmis_ci = [results[a].get("nmi_ci", 0.0) for a in aspect_names]
    accs = [results[a].get("acc_mean", results[a].get("acc", 0.0)) for a in aspect_names]
    accs_ci = [results[a].get("acc_ci", 0.0) for a in aspect_names]
    
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
    
    # Add values on top of the bars
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
    
    # 3. Right Subplot: LLM Judge Purity
    purities = [results[a].get("judge_purity_mean", results[a].get("judge_purity", 0.0)) for a in aspect_names]
    purities_ci = [results[a].get("judge_purity_ci", 0.0) for a in aspect_names]
    colors = ["#2563eb", "#7c3aed", "#10b981"]
    
    rects_purity = ax2.bar(aspect_names, purities, yerr=purities_ci, color=colors, width=0.4, alpha=0.85, capsize=4, error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#334155"}, zorder=3)
    ax2.set_facecolor('#ffffff')
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
    ax2.set_ylabel("Macro-Averaged Coherence Purity")
    ax2.set_title("LLM-as-a-Judge Consistency Purity", pad=10, fontweight="semibold")
    ax2.set_ylim(0, 1.1)
    
    # Add values on top of the bars
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
    
    fig.suptitle(f"LLM-as-a-Clusterer Multi-Aspect Evaluation (Banking77 N = {data.get('num_samples', 300)})", y=0.97, fontweight="bold", fontsize=15, color="#0f172a")
    
    # Elegant spacing to prevent overlap with suptitle
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    chart_path = "Output/banking77_multi_aspect_llm_eval.png"
    plt.savefig(chart_path, dpi=300, facecolor="#ffffff", edgecolor="none")
    print(f"[SUCCESS] Saved multi-aspect LLM evaluation chart to: {chart_path}")
    plt.close()

if __name__ == "__main__":
    main()
