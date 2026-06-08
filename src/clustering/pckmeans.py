import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from .base_clustering import BaseConstrainedClustering

class PCKMeans(BaseConstrainedClustering):
    """
    Pairwise Constrained K-Means (PCK-Means) clustering algorithm.
    Incorporates must-link and cannot-link constraints with penalties.
    """
    
    def __init__(self, n_clusters: int, w: Optional[float] = None, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None, cl_distance_threshold_ratio: Optional[float] = None):
        """
        Initialize the PCK-Means algorithm.
        
        Args:
            n_clusters: Number of clusters.
            w: Penalty weight for constraint violations. If None, it will be computed dynamically.
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence (unused if using change detection).
            random_state: Seed for random number generator.
            cl_distance_threshold_ratio: Ratio of average pairwise distance to filter out cannot-links.
        """
        super().__init__(n_clusters)
        self.w = w
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cl_distance_threshold_ratio = cl_distance_threshold_ratio
        self.centroids_ = None
        
    def _propagate_constraints(self, n_samples: int, must_link: List[Tuple[int, int]], cannot_link: List[Tuple[int, int]]) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        """
        Perform constraint propagation using Disjoint Set Union (DSU) / Union-Find.
        - Must-links are transitive (if A-B and B-C are must-link, A-C is must-link).
        - Cannot-links are propagated (if A-B is must-link and B-C is cannot-link, A-C is cannot-link).
        
        Returns:
            ml_dict: Dict mapping each sample index to its set of must-linked indices.
            cl_dict: Dict mapping each sample index to its set of cannot-linked indices.
        """
        # Union-Find initialization
        parent = list(range(n_samples))
        
        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        # Union must-links
        for u, v in must_link:
            if u < n_samples and v < n_samples:
                union(u, v)
                
        # Group samples by component
        components = {}
        for i in range(n_samples):
            root = find(i)
            if root not in components:
                components[root] = set()
            components[root].add(i)
            
        # Build propagated must-links dictionary
        ml_dict = {i: set() for i in range(n_samples)}
        for root, members in components.items():
            for m in members:
                ml_dict[m] = members - {m}
                
        # Build propagated cannot-links dictionary
        cl_dict = {i: set() for i in range(n_samples)}
        root_map = {i: find(i) for i in range(n_samples)}
        
        for u, v in cannot_link:
            if u < n_samples and v < n_samples:
                root_u = root_map[u]
                root_v = root_map[v]
                if root_u == root_v:
                    # Contradiction: must-link and cannot-link both exist between components.
                    # Ignore or skip to prevent infinite cost
                    continue
                # All members of component U cannot be in the same cluster as members of component V
                for member_u in components[root_u]:
                    for member_v in components[root_v]:
                        cl_dict[member_u].add(member_v)
                        cl_dict[member_v].add(member_u)
                        
        return ml_dict, cl_dict

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using K-Means++."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        centroids = np.empty((self.n_clusters, n_features))
        
        # Pick the first centroid randomly from X
        centroids[0] = X[np.random.choice(n_samples)]
        
        for c in range(1, self.n_clusters):
            # Compute distance squared to the nearest already chosen centroid
            distances = np.min([np.sum((X - centroids[prev])**2, axis=1) for prev in range(c)], axis=0)
            if np.sum(distances) == 0:
                probabilities = np.ones(n_samples) / n_samples
            else:
                probabilities = distances / np.sum(distances)
            centroids[c] = X[np.random.choice(n_samples, p=probabilities)]
            
        return centroids

    def _estimate_average_pairwise_distance(self, X: np.ndarray, sample_size: int = 1000) -> float:
        """
        Estimate the average pairwise Euclidean distance of dataset X.
        For N <= sample_size, compute exact average pairwise distance.
        For N > sample_size, estimate using a random sample of points to scale efficiently.
        """
        n_samples = X.shape[0]
        if n_samples < 2:
            return 0.0
            
        from scipy.spatial.distance import pdist
        if n_samples <= sample_size:
            dists = pdist(X)
            return float(np.mean(dists)) if len(dists) > 0 else 0.0
        else:
            if self.random_state is not None:
                rng = np.random.default_rng(self.random_state)
            else:
                rng = np.random.default_rng()
            indices = rng.choice(n_samples, size=sample_size, replace=False)
            dists = pdist(X[indices])
            return float(np.mean(dists)) if len(dists) > 0 else 0.0

    def fit(self, X: np.ndarray, must_link: Optional[List[Tuple[int, int]]] = None, cannot_link: Optional[List[Tuple[int, int]]] = None):
        """
        Fit PCK-Means on data X with given pairwise constraints.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        must_link = must_link or []
        cannot_link = cannot_link or []
        
        # Filter out-of-bounds cannot-link pairs first
        cannot_link = [(u, v) for u, v in cannot_link if u < n_samples and v < n_samples]
        
        # Filter cannot-link pairs that are too far apart based on cl_distance_threshold_ratio
        if self.cl_distance_threshold_ratio is not None and len(cannot_link) > 0:
            avg_dist = self._estimate_average_pairwise_distance(X)
            threshold = self.cl_distance_threshold_ratio * avg_dist
            
            cannot_link_arr = np.array(cannot_link)
            u_coords = X[cannot_link_arr[:, 0]]
            v_coords = X[cannot_link_arr[:, 1]]
            dists = np.sqrt(np.sum((u_coords - v_coords)**2, axis=1))
            valid_mask = dists <= threshold
            cannot_link = [(int(pair[0]), int(pair[1])) for pair in cannot_link_arr[valid_mask]]
        
        # Determine penalty weight if not set
        if self.w is None:
            # Dynamic penalty weight based on average variance of features
            # Scale by 10.0 to ensure constraints are strongly prioritized over distance
            self.w = float(np.sum(np.var(X, axis=0))) * 10.0
            if self.w == 0:
                self.w = 1.0
                
        # Propagate constraints
        ml_dict, cl_dict = self._propagate_constraints(n_samples, must_link, cannot_link)
        
        # Initialize centroids and labels
        self.centroids_ = self._init_centroids(X)
        
        # Initial assignment based on Euclidean distance only
        labels = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            dists = np.sum((X[i] - self.centroids_)**2, axis=1)
            labels[i] = np.argmin(dists)
            
        # PCK-Means Iterative Optimization
        for iteration in range(self.max_iter):
            labels_changed = False
            
            # Assignment Step (sequential updates to ensure quick convergence)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)  # Shuffle points to avoid ordering bias
            
            for i in indices:
                best_label = labels[i]
                min_cost = np.inf
                
                # Compute costs for each cluster assignment
                for l in range(self.n_clusters):
                    # Squared Euclidean distance
                    dist_cost = np.sum((X[i] - self.centroids_[l])**2)
                    
                    # Constraint violations penalty
                    penalty = 0.0
                    
                    # Must-link violations
                    for j in ml_dict[i]:
                        if labels[j] != l:
                            penalty += self.w
                            
                    # Cannot-link violations
                    for j in cl_dict[i]:
                        if labels[j] == l:
                            penalty += self.w
                            
                    cost = dist_cost + penalty
                    if cost < min_cost:
                        min_cost = cost
                        best_label = l
                        
                if labels[i] != best_label:
                    labels[i] = best_label
                    labels_changed = True
                    
            # Update Step
            new_centroids = np.zeros_like(self.centroids_)
            for l in range(self.n_clusters):
                points = X[labels == l]
                if len(points) > 0:
                    new_centroids[l] = points.mean(axis=0)
                else:
                    # Empty cluster: re-initialize centroid to the point furthest from its current centroid
                    # to prevent collapsing to fewer clusters
                    dists = np.sum((X - self.centroids_[l])**2, axis=1)
                    new_centroids[l] = X[np.argmax(dists)]
                    
            # Check for convergence
            centroid_shift = np.sum((self.centroids_ - new_centroids)**2)
            self.centroids_ = new_centroids
            
            if not labels_changed or centroid_shift < self.tol:
                break
                
        self.labels_ = labels
        return self
