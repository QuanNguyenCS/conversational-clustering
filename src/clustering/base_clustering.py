from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class BaseConstrainedClustering(ABC):
    """Abstract base class for constrained clustering algorithms."""
    
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.labels_ = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, must_link: Optional[List[Tuple[int, int]]] = None, cannot_link: Optional[List[Tuple[int, int]]] = None):
        """
        Fit the model on the data X with constraints.
        
        Args:
            X: Data matrix of shape (n_samples, n_features).
            must_link: List of index pairs (i, j) that must belong to the same cluster.
            cannot_link: List of index pairs (i, j) that cannot belong to the same cluster.
        """
        pass
        
    def fit_predict(self, X: np.ndarray, must_link: Optional[List[Tuple[int, int]]] = None, cannot_link: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """
        Fit the model and return the cluster labels.
        
        Args:
            X: Data matrix.
            must_link: List of index pairs.
            cannot_link: List of index pairs.
            
        Returns:
            Cluster labels of shape (n_samples,).
        """
        self.fit(X, must_link, cannot_link)
        return self.labels_
