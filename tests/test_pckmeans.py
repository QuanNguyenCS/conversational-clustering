import numpy as np
import pytest
from src.clustering.pckmeans import PCKMeans

def test_constraint_propagation():
    pckm = PCKMeans(n_clusters=2)
    # Define 5 items
    # Must-link: (0, 1), (1, 2) -> (0, 2) implied
    # Cannot-link: (2, 3) -> (0, 3), (1, 3) implied
    must_link = [(0, 1), (1, 2)]
    cannot_link = [(2, 3)]
    
    ml_dict, cl_dict = pckm._propagate_constraints(5, must_link, cannot_link)
    
    # Check must-link propagation
    assert 1 in ml_dict[0]
    assert 2 in ml_dict[0]
    assert 0 in ml_dict[1]
    assert 2 in ml_dict[1]
    assert 0 in ml_dict[2]
    assert 1 in ml_dict[2]
    
    # Check cannot-link propagation
    assert 3 in cl_dict[2]
    assert 3 in cl_dict[1]
    assert 3 in cl_dict[0]
    assert 0 in cl_dict[3]
    assert 1 in cl_dict[3]
    assert 2 in cl_dict[3]

def test_pckmeans_clustering_enforcement():
    # 4 data points in 1D
    # X = [0.0, 1.0, 10.0, 11.0]
    # Without constraints, naturally clusters into {0, 1} and {10, 11}
    # With a cannot-link between 0 and 1, they must be separated!
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    
    # 1. No constraints
    pckm = PCKMeans(n_clusters=2, random_state=42)
    labels = pckm.fit_predict(X, must_link=[], cannot_link=[])
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    
    # 2. Cannot link 0 and 1. They should be forced into different clusters.
    pckm_cl = PCKMeans(n_clusters=2, w=100.0, random_state=42)
    labels_cl = pckm_cl.fit_predict(X, must_link=[], cannot_link=[(0, 1)])
    assert labels_cl[0] != labels_cl[1]
    
    # 3. Must link 0 and 2. They should be forced into the same cluster.
    pckm_ml = PCKMeans(n_clusters=2, w=100.0, random_state=42)
    labels_ml = pckm_ml.fit_predict(X, must_link=[(0, 2)], cannot_link=[])
    assert labels_ml[0] == labels_ml[2]

def test_cannot_link_distance_filtering():
    # 4 points in 1D space
    # Pairwise distances: d(0,1)=1.0, d(0,2)=10.0, d(0,3)=11.0, d(1,2)=9.0, d(1,3)=10.0, d(2,3)=1.0
    # Mean of all 6 pairwise distances = (1 + 10 + 11 + 9 + 10 + 1) / 6 = 7.0
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    
    pckm = PCKMeans(n_clusters=2, cl_distance_threshold_ratio=0.5, random_state=42)
    
    received_cannot_link = []
    original_propagate = pckm._propagate_constraints
    def mock_propagate(n_samples, must_link, cannot_link):
        received_cannot_link.extend(cannot_link)
        return original_propagate(n_samples, must_link, cannot_link)
    pckm._propagate_constraints = mock_propagate
    
    pckm.fit(X, must_link=[], cannot_link=[(0, 1), (0, 2)])
    
    # Pair (0, 1) has distance 1.0 <= 0.5 * 7.0 (3.5) -> Retained
    # Pair (0, 2) has distance 10.0 > 0.5 * 7.0 (3.5) -> Filtered
    assert (0, 1) in received_cannot_link
    assert (0, 2) not in received_cannot_link
