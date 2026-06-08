from typing import List, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def compute_ari(labels_true: List[int], labels_pred: List[int]) -> float:
    """Calculate the Adjusted Rand Index (ARI) between ground truth and predicted labels."""
    return float(adjusted_rand_score(labels_true, labels_pred))

def compute_nmi(labels_true: List[int], labels_pred: List[int]) -> float:
    """Calculate the Normalized Mutual Information (NMI) between ground truth and predicted labels."""
    return float(normalized_mutual_info_score(labels_true, labels_pred))

def compute_icu(num_c3_turns: int, num_pairwise_queries: int) -> float:
    """
    Calculate Interaction Cost Units (ICU).
    C3 conversational turn costs 10 ICU.
    Pairwise manual query costs 1 ICU.
    """
    return float(10 * num_c3_turns + 1 * num_pairwise_queries)

def compute_auac(icu_history: List[float], alignment_history: List[float]) -> float:
    """
    Compute the Area Under the Alignment Curve (AUAC).
    Integrates the alignment metric (e.g. ARI) over the accumulated ICU cost
    and normalizes by the maximum possible area (width * 1.0).
    
    Args:
        icu_history: List of accumulated ICU values (must be increasing).
        alignment_history: List of alignment metric values (e.g. ARI/NMI between 0 and 1).
        
    Returns:
        The normalized AUAC score between 0.0 and 1.0.
    """
    if len(icu_history) < 2:
        return 0.0
        
    # Sort by ICU to ensure proper integration order
    sorted_pairs = sorted(zip(icu_history, alignment_history))
    x = np.array([p[0] for p in sorted_pairs])
    y = np.array([p[1] for p in sorted_pairs])
    
    # Calculate area using trapezoidal rule
    if hasattr(np, 'trapezoid'):
        area = np.trapezoid(y, x)
    else:
        area = np.trapz(y, x)
    
    # Normalize by maximum possible area: width * 1.0 (assuming alignment metric is bounded by [0, 1])
    width = x[-1] - x[0]
    if width == 0:
        return 0.0
        
    return float(area / width)

def compute_acc(labels_true: List[int], labels_pred: List[int]) -> float:
    """
    Calculate the Clustering Accuracy (ACC) after Hungarian alignment.
    
    Args:
        labels_true: Ground truth labels.
        labels_pred: Predicted labels.
        
    Returns:
        The clustering accuracy score between 0.0 and 1.0.
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("The shape of ground truth and predicted labels must be the same.")
        
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    true_num = len(unique_true)
    pred_num = len(unique_pred)
    n_classes = max(true_num, pred_num)
    
    cost_matrix = np.zeros((n_classes, n_classes))
    for i, t_lbl in enumerate(unique_true):
        for j, p_lbl in enumerate(unique_pred):
            overlap = np.sum((labels_true == t_lbl) & (labels_pred == p_lbl))
            cost_matrix[i, j] = -overlap
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_overlap = 0
    for r, c in zip(row_ind, col_ind):
        if r < true_num and c < pred_num:
            matched_overlap += -cost_matrix[r, c]
            
    return float(matched_overlap / labels_true.shape[0])
