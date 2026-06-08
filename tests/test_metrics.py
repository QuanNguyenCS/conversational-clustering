import pytest
from src.evaluation.metrics import compute_ari, compute_nmi, compute_icu, compute_auac, compute_acc

def test_evaluation_metrics():
    labels_true = [0, 0, 1, 1, 2, 2]
    labels_pred = [0, 0, 1, 1, 2, 2]
    
    # Perfect clustering
    assert compute_ari(labels_true, labels_pred) == 1.0
    assert compute_nmi(labels_true, labels_pred) == 1.0
    
    # Imperfect clustering
    labels_bad = [0, 1, 2, 0, 1, 2]
    assert compute_ari(labels_true, labels_bad) < 0.5
    assert compute_nmi(labels_true, labels_bad) < 0.5

def test_icu_calculation():
    # 3 C3 turns (30 ICU) + 5 pairwise queries (5 ICU) = 35 ICU
    assert compute_icu(3, 5) == 35.0
    assert compute_icu(0, 0) == 0.0
    assert compute_icu(10, 0) == 100.0

def test_auac_calculation():
    # Setup curves
    # Alignment goes up over time
    icu_history = [0, 10, 20, 30]
    alignment_history = [0.2, 0.5, 0.8, 1.0]
    
    auac = compute_auac(icu_history, alignment_history)
    
    # Since alignment is positive and monotonically increasing, AUAC should be between 0.2 and 1.0
    assert 0.0 < auac <= 1.0
    
    # Single point or empty history should return 0.0
    assert compute_auac([0], [0.5]) == 0.0
    assert compute_auac([], []) == 0.0

def test_acc_calculation():
    # Perfect clustering
    labels_true = [0, 0, 1, 1, 2, 2]
    labels_pred = [0, 0, 1, 1, 2, 2]
    assert compute_acc(labels_true, labels_pred) == 1.0

    # Permuted labels (perfect alignment, but different labels)
    labels_perm = [1, 1, 2, 2, 0, 0]
    assert compute_acc(labels_true, labels_perm) == 1.0

    # Imperfect clustering (5/6 overlap)
    labels_imperfect = [0, 0, 1, 2, 2, 2]
    assert abs(compute_acc(labels_true, labels_imperfect) - 5/6) < 1e-6

    # Dimension mismatch
    with pytest.raises(ValueError):
        compute_acc([0, 1], [0, 1, 2])
