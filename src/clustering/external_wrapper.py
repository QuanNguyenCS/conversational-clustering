import numpy as np
from typing import List, Tuple, Optional
import warnings

from .base_clustering import BaseConstrainedClustering
from .pckmeans import PCKMeans

# Try to import external packages
try:
    from copkmeans.cop_kmeans import cop_kmeans  # type: ignore
    HAS_COPKMEANS = True
except ImportError:
    HAS_COPKMEANS = False

class WrapperConstrainedClustering(BaseConstrainedClustering):
    """
    A wrapper for constrained clustering that integrates third-party packages (e.g. copkmeans)
    and falls back to the robust custom PCKMeans implementation if they are not installed or fail.
    """
    
    def __init__(self, n_clusters: int, method: str = "pckmeans", **kwargs):
        """
        Initialize the wrapper.
        
        Args:
            n_clusters: Number of clusters.
            method: The method to use ('copkmeans' or 'pckmeans').
            **kwargs: Arguments passed to the underlying algorithm.
        """
        super().__init__(n_clusters)
        self.method = method.lower()
        self.kwargs = kwargs
        self.algorithm_ = None
        
    def fit(self, X: np.ndarray, must_link: Optional[List[Tuple[int, int]]] = None, cannot_link: Optional[List[Tuple[int, int]]] = None):
        """
        Fit PCKMeans clustering.
        """
        must_link = must_link or []
        cannot_link = cannot_link or []
        
        # Use custom PCKMeans
        pckm = PCKMeans(
            n_clusters=self.n_clusters,
            w=self.kwargs.get("w", None),
            max_iter=self.kwargs.get("max_iter", 100),
            tol=self.kwargs.get("tol", 1e-4),
            random_state=self.kwargs.get("random_state", None),
            cl_distance_threshold_ratio=self.kwargs.get("cl_distance_threshold_ratio", None)
        )
        pckm.fit(X, must_link, cannot_link)
        self.labels_ = pckm.labels_
        self.algorithm_ = pckm
        return self
