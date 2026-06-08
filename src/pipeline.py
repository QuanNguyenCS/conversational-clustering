import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_extraction.text import TfidfVectorizer

from .embeddings.base_embeddings import BaseEmbeddings
from .agent.base_agent import BaseLLMAgent

def parse_json_robustly(text: str) -> Dict:
    import json
    content = text.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}

class ConversationalClusteringPipeline:
    """
    Orchestrates the decoupled, K-agnostic C3 (Conversational Constrained Clustering) flow.
    Manages embedding projection, farthest point sampling, mini-batch label discovery,
    ITML distance metric learning, constrained agglomerative clustering, and ledger refinements.
    """
    
    def __init__(
        self,
        embedding_provider: BaseEmbeddings,
        agent: Optional[BaseLLMAgent] = None,
        qa_agent: Optional[BaseLLMAgent] = None,
        clustering_agent: Optional[BaseLLMAgent] = None,
        num_samples: int = 15,
        batch_size: int = 20,
        random_state: Optional[int] = 42,
        user_intent: str = "Group similar documents together by topic or theme.",
        use_qa_phase: bool = False,
        clustering_params: Optional[Dict[str, Any]] = None,
        n_clusters: Optional[int] = None,
        use_itml: bool = False,
        pca_dim: Optional[int] = None,
        cl_distance_threshold_ratio: Optional[float] = None
    ):
        self.embedding_provider = embedding_provider
        self.agent = agent
        self.qa_agent = qa_agent or agent
        self.clustering_agent = clustering_agent or agent
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.random_state = random_state
        self.use_qa_phase = use_qa_phase
        self.use_itml = use_itml
        self.pca_dim = pca_dim
        self.clustering_params = clustering_params or {}
        self.cl_distance_threshold_ratio = cl_distance_threshold_ratio or self.clustering_params.get("cl_distance_threshold_ratio", None)
        self.n_clusters_target = n_clusters
        self.n_clusters = n_clusters
        
        # State variables
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.raw_embeddings: Optional[np.ndarray] = None
        self.pca_coords: Optional[np.ndarray] = None
        
        self.user_intent = user_intent
        self.global_registry: Dict[str, str] = {}
        self.sampled_indices: List[int] = []
        self.sampled_texts: List[str] = []
        self.final_assignments: Dict[int, str] = {}
        self.constraint_ledger: Dict[Tuple[int, int], Dict[str, Any]] = {}
        
        self.turn: int = 0
        self.labels: Optional[np.ndarray] = None
        self.centroids: Optional[np.ndarray] = None
        
        # Q&A clarification states
        self.qa_active = False
        self.qa_turn = 0
        self.max_qa_turns = 5
        self.qa_type = "initial"
        self.pending_feedback = ""
        self.qa_awaiting_confirm = False
        self.qa_summary = None
        self.qa_memory: Dict[str, Any] = {
            "aspect": None,
            "granularity": None,
            "example_feedback": None
        }
        self.qa_example_docs: List[int] = []

    def set_data(self, texts: List[str]):
        """Set dataset and compute text embeddings."""
        self.texts = texts
        raw_embs = self.embedding_provider.embed_texts(texts)
        
        # Apply PCA dimensionality reduction if pca_dim is specified
        if getattr(self, "pca_dim", None) is not None:
            from sklearn.decomposition import PCA
            # Ensure n_components is not larger than number of samples or features
            n_components = min(self.pca_dim, len(texts), raw_embs.shape[1])
            pca = PCA(n_components=n_components, random_state=self.random_state)
            self.raw_embeddings = pca.fit_transform(raw_embs)
            print(f"Applied PCA dimension reduction: {raw_embs.shape[1]} -> {n_components}")
        else:
            self.raw_embeddings = raw_embs
            
        self.embeddings = self.raw_embeddings.copy()
            
        try:
            from sklearn.decomposition import PCA
            pca_2d = PCA(n_components=2, random_state=42)
            self.pca_coords = pca_2d.fit_transform(self.embeddings)
        except Exception:
            self.pca_coords = None
        self.reset()
        
    def reset(self):
        """Reset history, constraints, and sample selection."""
        self.global_registry = {}
        self.final_assignments = {}
        self.constraint_ledger = {}
        self.history = []
        self.labels = None
        self.centroids = None
        self.turn = 0
        self.qa_memory = {
            "aspect": None,
            "granularity": None,
            "example_feedback": None
        }
        self.qa_example_docs = []
        
        # Perform initial sampling once data is set
        if self.embeddings is not None and len(self.texts) > 0:
            N = len(self.texts)
            if self.num_samples != 15:
                n_samples = self.num_samples
            else:
                n_samples = max(30, min(int(0.02 * N), 150))
            n_samples = min(N, n_samples)
            self.sampled_indices = self._farthest_point_sampling(self.embeddings, n_samples)
            self.sampled_texts = [self.texts[idx] for idx in self.sampled_indices]

    def _farthest_point_sampling(self, X: np.ndarray, num_samples: int) -> List[int]:
        """
        Phase 2: Vectorized Farthest Point Sampling (FPS) algorithm.
        """
        N = X.shape[0]
        n = min(N, num_samples)
        if n == 0:
            return []
            
        distances = np.full(N, np.inf)
        sampled_indices = []
        
        # Select first index pseudo-randomly
        np.random.seed(self.random_state)
        idx = np.random.choice(N)
        sampled_indices.append(int(idx))
        
        for _ in range(1, n):
            diff = X - X[idx]
            new_distances = np.sum(diff**2, axis=1)
            distances = np.minimum(distances, new_distances)
            idx = np.argmax(distances)
            sampled_indices.append(int(idx))
            
        return sampled_indices

    def _run_label_discovery(self, indices_to_discover: List[int], is_refinement: bool = False, target_cluster_desc: str = "", refinement_feedback: str = ""):
        """
        Phase 3: Dynamic Sequential Label Discovery via mini-batching.
        """
        if is_refinement:
            # Standard single-step sequential label discovery for refinements
            batch_size = self.batch_size
            n_samples = len(indices_to_discover)
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices_to_discover[start_idx : start_idx + batch_size]
                
                docs_list_str = ""
                for idx in batch_indices:
                    text_content = self.texts[idx]
                    docs_list_str += f"[Doc ID: {idx}] {text_content}\n\n"
                    
                import json
                registry_str = json.dumps(self.global_registry, indent=2, ensure_ascii=False)
                
                system_prompt = f"""You are an expert data clustering assistant. The user's ultimate goal is: {self.user_intent}
We are refining a specific cluster described as: "{target_cluster_desc}".
"""
                if refinement_feedback:
                    system_prompt += f"User's specific refinement split instructions: \"{refinement_feedback}\"\n"
                    
                system_prompt += f"""
KNOWN CLUSTERS REGISTRY (DISCOVERED FROM PREVIOUS BATCHES):
{registry_str}

INSTRUCTIONS:
Categorize the following text documents. Assign them to an existing category name (the exact key from the registry) if they match the semantic description. If they belong to a new distinct topic, create a new category name as key with a brief description.
"""
                if refinement_feedback:
                    system_prompt += "Ensure you follow the user's specific refinement split instructions to partition these documents.\n"
                    
                system_prompt += f"""All category names and descriptions MUST be written strictly in English.
You must return ONLY a valid JSON object matching the schema:
{{
  "new_clusters": {{"Name of new category": "Description of the new category"}},
  "assignments": {{"doc_id_1": "Category name from registry"}}
}}"""
                
                user_prompt = f"Here are the documents for this batch:\n\n{docs_list_str}"
                response_text = self.clustering_agent.generate_text(system_prompt, user_prompt)
                output_json = parse_json_robustly(response_text)
                
                new_clusters = {}
                for k, v in output_json.items():
                    if k.lower().replace("_", "") == "newclusters":
                        new_clusters = v
                        break
                
                assignments = {}
                for k, v in output_json.items():
                    if k.lower() == "assignments":
                        assignments = v
                        break
                
                for cid, desc in new_clusters.items():
                    self.global_registry[cid] = desc
                    
                for doc_id_str, cid in assignments.items():
                    nums = re.findall(r'\d+', doc_id_str)
                    if nums:
                        abs_idx = int(nums[0])
                        if abs_idx < len(self.texts):
                            self.final_assignments[abs_idx] = cid
        else:
            # Strategy 4: Two-Step Guided Label Discovery
            # Step 1: Candidate Taxonomy Synthesis
            doc_titles_str = ""
            for idx in indices_to_discover:
                snippet = self.texts[idx][:120].strip() + "..."
                doc_titles_str += f"[Doc ID: {idx}] {snippet}\n"
                
            step1_system_prompt = f"""You are an expert taxonomy designer. The user's ultimate goal is: {self.user_intent}

You are given a list of documents represented by their summaries.
Create a draft taxonomy (candidate registry of categories) that covers these documents under the user's intent.
The number of categories should be determined naturally and dynamically based on the semantic variation in the data and the user's intent.
Provide a clear description for each category.
All draft taxonomy category names and descriptions MUST be written strictly in English.

You must return ONLY a valid JSON object matching this schema:
{{
  "candidate_registry": {{
     "Name of the first category": "Description of the first category",
     "Name of the second category": "Description of the second category",
     ...
  }}
}}"""
            
            step1_user_prompt = f"Here are the documents:\n\n{doc_titles_str}"
            
            try:
                response_text = self.clustering_agent.generate_text(step1_system_prompt, step1_user_prompt)
                parsed = parse_json_robustly(response_text)
                candidate_registry = parsed.get("candidate_registry", {})
            except Exception as e:
                import warnings
                warnings.warn(f"Step 1 taxonomy synthesis failed: {e}. Falling back to dynamic discovery.")
                candidate_registry = {}
                
            self.global_registry = candidate_registry or {}
            
            # Step 2: Guided Classification via sequential mini-batching
            batch_size = self.batch_size
            n_samples = len(indices_to_discover)
            
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices_to_discover[start_idx : start_idx + batch_size]
                
                docs_list_str = ""
                for idx in batch_indices:
                    text_content = self.texts[idx]
                    docs_list_str += f"[Doc ID: {idx}] {text_content}\n\n"
                    
                import json
                registry_str = json.dumps(self.global_registry, indent=2, ensure_ascii=False)
                
                system_prompt = f"""You are an expert data classification assistant. The user's ultimate goal is: {self.user_intent}

CANDIDATE CLUSTERS REGISTRY:
{registry_str}

INSTRUCTIONS:
Classify the following text documents into the existing candidate categories in the registry.
- Assign each document to the most appropriate category name (the exact key from the registry).
- If a document absolutely does not fit any of the existing candidate categories, you are allowed to create a new category name as key with its description in "new_clusters", and assign the document to it.
All category names and descriptions of new fallback categories MUST be written strictly in English.

You must return ONLY a valid JSON object matching this schema:
{{
  "new_clusters": {{"Name of new category": "Description of the new fallback category (only if needed)"}},
  "assignments": {{"doc_id_1": "Category name from registry", "doc_id_2": "Category name from registry"}}
}}"""
                
                user_prompt = f"Here are the documents for this batch:\n\n{docs_list_str}"
                response_text = self.clustering_agent.generate_text(system_prompt, user_prompt)
                output_json = parse_json_robustly(response_text)
                
                new_clusters = {}
                for k, v in output_json.items():
                    if k.lower().replace("_", "") == "newclusters":
                        new_clusters = v
                        break
                        
                assignments = {}
                for k, v in output_json.items():
                    if k.lower() == "assignments":
                        assignments = v
                        break
                        
                for cid, desc in new_clusters.items():
                    self.global_registry[cid] = desc
                    
                for doc_id_str, cid in assignments.items():
                    nums = re.findall(r'\d+', doc_id_str)
                    if nums:
                        abs_idx = int(nums[0])
                        if abs_idx < len(self.texts):
                            self.final_assignments[abs_idx] = cid

    def run_initial_clustering(self, bypass_qa: bool = False) -> Tuple[Optional[np.ndarray], Dict[int, List[str]], Dict[int, List[str]]]:
        """
        Run initial clustering loop from scratch.
        """
        if self.embeddings is None:
            raise ValueError("No data has been loaded. Call set_data first.")
        if self.qa_agent is None or self.clustering_agent is None:
            raise ValueError("LLM Agent is not set.")
            
        if self.use_qa_phase and not bypass_qa:
            self.start_initial_qa()
            return None, {}, {}
            
        # Reset state
        self.global_registry = {}
        self.final_assignments = {}
        self.constraint_ledger = {}
        self.itml_model = None
        self.turn = 0
        
        # Phase 2: FPS
        N = len(self.texts)
        if self.num_samples != 15:
            n_samples = self.num_samples
        else:
            n_samples = max(30, min(int(0.02 * N), 150))
        n_samples = min(N, n_samples)
        self.sampled_indices = self._farthest_point_sampling(self.embeddings, n_samples)
        self.sampled_texts = [self.texts[idx] for idx in self.sampled_indices]
        
        # Phase 3: Label Discovery
        self._run_label_discovery(self.sampled_indices, is_refinement=False)
        
        # Phase 4: Constraint Extraction
        from itertools import combinations
        for idx_i, idx_j in combinations(self.sampled_indices, 2):
            lbl_i = self.final_assignments.get(idx_i)
            lbl_j = self.final_assignments.get(idx_j)
            if lbl_i is not None and lbl_j is not None:
                c_type = "Must-Link" if lbl_i == lbl_j else "Cannot-Link"
                pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                self.constraint_ledger[pair] = {"type": c_type, "turn": self.turn}
                
        # Phase 5 & 6: Clustering
        self._refit_and_cluster()
        
        summaries = self.get_cluster_keywords()
        central_docs = self.get_central_documents()
        return self.labels, summaries, central_docs

    def _estimate_average_pairwise_distance(self, X: np.ndarray, sample_size: int = 1000) -> float:
        """
        Estimate the average pairwise Euclidean distance of dataset X.
        For N <= sample_size, compute exact average pairwise distance.
        For N > sample_size, estimate using a random sample of points to scale efficiently.
        """
        n_samples = X.shape[0]
        if n_samples < 2:
            return 0.0
            
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

    def _refit_and_cluster(self):
        """
        Run PCKMeans clustering with Must-Link and Cannot-Link constraints directly on embeddings.
        """
        if self.n_clusters_target is not None:
            K_ref = self.n_clusters_target
        else:
            K_ref = len(self.global_registry)
            
        if K_ref < 2:
            raise ValueError(f"LLM discovered fewer than 2 clusters (K_ref={K_ref}). Please adjust your settings.")
            
        # Filter cannot-link constraints that are too far apart before retrieving ml and cl
        if self.cl_distance_threshold_ratio is not None:
            cl_pairs = [pair for pair, info in self.constraint_ledger.items() if info["type"] == "Cannot-Link"]
            if len(cl_pairs) > 0:
                avg_dist = self._estimate_average_pairwise_distance(self.raw_embeddings)
                threshold = self.cl_distance_threshold_ratio * avg_dist
                
                to_delete = []
                for u, v in cl_pairs:
                    dist = np.linalg.norm(self.raw_embeddings[u] - self.raw_embeddings[v])
                    if dist > threshold:
                        to_delete.append((u, v))
                
                if len(to_delete) > 0:
                    print(f"[Distance-Based Filtering] Deleting {len(to_delete)} cannot-link pairs exceeding distance threshold ({threshold:.4f})")
                    for pair in to_delete:
                        del self.constraint_ledger[pair]
                        
        ml = self.must_link
        cl = self.cannot_link
        
        if self.use_itml and (len(ml) > 0 or len(cl) > 0):
            from metric_learn import ITML
            
            pairs = []
            y = []
            for i, j in ml:
                pairs.append([self.raw_embeddings[i], self.raw_embeddings[j]])
                y.append(1)
            for i, j in cl:
                pairs.append([self.raw_embeddings[i], self.raw_embeddings[j]])
                y.append(-1)
                
            pairs = np.array(pairs)
            y = np.array(y)
            
            self.itml_model = ITML(random_state=self.random_state)
            try:
                self.itml_model.fit(pairs, y)
                self.embeddings = self.itml_model.transform(self.raw_embeddings)
                print(f"ITML metric learning completed. Embedding space transformed. ML: {len(ml)}, CL: {len(cl)}")
            except Exception as e:
                print(f"[WARNING] ITML fitting failed: {e}. Falling back to raw embeddings.")
                self.embeddings = self.raw_embeddings.copy()
        else:
            self.embeddings = self.raw_embeddings.copy()

        from .clustering.external_wrapper import WrapperConstrainedClustering
        clustering_args = {
            "n_clusters": K_ref,
            "method": "pckmeans",
            "random_state": self.random_state
        }
        clustering_args.update(self.clustering_params)
        clustering_algo = WrapperConstrainedClustering(**clustering_args)
        
        # Cluster on transformed/raw embeddings
        clustering_algo.fit(self.embeddings, must_link=self.must_link, cannot_link=self.cannot_link)
        self.labels = clustering_algo.labels_
        self.n_clusters = K_ref
        
        # Update Centroids
        centroids_list = []
        for l in range(self.n_clusters):
            cluster_pts = self.embeddings[self.labels == l]
            if len(cluster_pts) > 0:
                centroids_list.append(cluster_pts.mean(axis=0))
            else:
                centroids_list.append(np.zeros(self.embeddings.shape[1]))
        self.centroids = np.array(centroids_list)

    def step(self, user_feedback: str) -> Tuple[Optional[np.ndarray], Dict[int, List[str]], Dict[int, List[str]]]:
        """
        Execute one step of the conversational flow (either Q&A turn or updating intent and re-clustering).
        """
        if self.embeddings is None:
            raise ValueError("No data has been loaded. Call set_data first.")
        if self.qa_agent is None or self.clustering_agent is None:
            raise ValueError("LLM Agent is not set.")
            
        if self.use_qa_phase and self.qa_active:
            if user_feedback.strip().lower() == "start over":
                self.start_initial_qa()
                return None, {}, {}
                
            self.qa_turn += 1
            self.history.append({"role": "user", "content": user_feedback})
            
            if self.qa_turn > self.max_qa_turns:
                self._finish_qa_and_cluster()
            else:
                self._run_qa_turn()
                
            if not self.qa_active:
                if getattr(self, "qa_awaiting_confirm", False):
                    return None, {}, {}
                return self.labels, self.get_cluster_keywords(), self.get_central_documents()
            else:
                return None, {}, {}
        else:
            self.turn += 1
            self.history.append({"role": "user", "content": user_feedback})
            
            # Synthesize updated intent from previous intent and new user feedback
            system_prompt = f"""You are an expert data analysis assistant.
The user previously defined their clustering intent as: "{self.user_intent}"
They have now provided additional instructions or feedback: "{user_feedback}"

INSTRUCTIONS:
Synthesize the new instructions with the previous intent to form a single, cohesive, updated clustering intent description.
Ensure it details the aspect of clustering, desired granularity, and noise/ambiguity strategy, integrating all new details requested by the user.
Write the updated intent strictly in English.

You must return ONLY the updated intent description, nothing else."""
            try:
                updated_intent = self.qa_agent.generate_text(system_prompt, "Please synthesize the updated intent.")
                self.user_intent = updated_intent.strip()
            except Exception:
                self.user_intent = f"{self.user_intent} Additional requirement: {user_feedback}"
                
            # Run initial clustering from scratch under the updated intent
            self.run_initial_clustering(bypass_qa=True)
            
            greeting = f"I have updated your clustering intent to: **\"{self.user_intent}\"** and re-clustered the data from scratch."
            natural_text = self._generate_natural_summary(greeting)
            
            image = self._generate_pca_plot_image()
            keywords = self.get_cluster_keywords(top_k=6)
            central_docs = self.get_central_docs_strings(top_n=3)
            
            self.history.append({
                "role": "assistant",
                "content": natural_text,
                "image": image,
                "labels": list(self.labels) if self.labels is not None else None,
                "cluster_details": {
                    "keywords": keywords,
                    "central_docs": central_docs
                },
                "type": "clustering_result"
            })
            return self.labels, keywords, self.get_central_documents()

    def start_initial_qa(self):
        """Start the initial Q&A clarification phase."""
        if self.embeddings is None:
            raise ValueError("No data has been loaded. Call set_data first.")
        if self.qa_agent is None or self.clustering_agent is None:
            raise ValueError("LLM Agent is not set.")
            
        self.qa_active = True
        self.qa_turn = 1
        self.qa_type = "initial"
        self.history = []
        self.qa_memory = {
            "aspect": None,
            "granularity": None,
            "example_feedback": None
        }
        
        raw_pca_plot = self._generate_raw_pca_plot_image()
        
        # Take exactly up to 20 representative documents from the FPS sampled indices
        qa_indices = self.sampled_indices[:20]
        self.qa_example_docs = qa_indices
        sampled_texts = [self.texts[idx] for idx in qa_indices]
        
        sampled_docs = [{"id": int(idx), "text": str(text)} for idx, text in zip(qa_indices, sampled_texts)]
        
        docs_list_str = ""
        for idx, text in zip(qa_indices, sampled_texts):
            docs_list_str += f"[Doc ID: {idx}] {text[:300]}...\n\n"
            
        system_prompt = f"""You are a helpful and professional data analysis assistant.
Your task is to guide the user in defining their clustering criteria.

Here are 20 representative documents from the dataset:
{docs_list_str}

INSTRUCTIONS:
1. Summarize these documents briefly (2-3 sentences max).
2. Ask the user what primary aspect or dimension they want to cluster these documents by.
3. At the end of your response, provide 3 specific, detailed aspect suggestions based on the documents. Suggestions MUST be complete context-rich options (15-25 words each, e.g., 'Group the scientific papers by their specific application domains, such as medical versus robotics' rather than just 'By Topic') and formatted EXACTLY as:
[SUGGESTIONS] Detailed Suggestion 1 | Detailed Suggestion 2 | Detailed Suggestion 3 [/SUGGESTIONS]
All parts of your response MUST be written strictly in English."""
        
        try:
            response = self.qa_agent.generate_text(system_prompt, "Please analyze the sample and suggest aspects.")
            content = response.strip()
            
            # Parse suggestions from tags if present
            suggestions = ["By Topic", "By Sentiment", "By Methodology"]
            sugg_match = re.search(r'\[SUGGESTIONS\](.*?)\[/SUGGESTIONS\]', content, re.DOTALL)
            if sugg_match:
                sugg_str = sugg_match.group(1).strip()
                suggestions = [s.strip() for s in sugg_str.split('|') if s.strip()]
                content = content.replace(sugg_match.group(0), "").strip()
        except Exception as e:
            content = f"I have loaded the dataset. What aspect would you like to cluster these documents by?"
            suggestions = ["By Topic", "By Sentiment", "By Methodology"]
            
        self.history.append({
            "role": "assistant",
            "content": content,
            "suggestions": suggestions,
            "raw_pca_plot": raw_pca_plot,
            "sampled_docs": sampled_docs,
            "type": "qa_start"
        })

    def _get_current_qa_history_str(self) -> str:
        start_idx = 0
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            if msg.get("role") == "assistant" and msg.get("type") in ["qa_start", "qa_refinement"]:
                start_idx = i
                break
        
        qa_msgs = self.history[start_idx:]
        history_str = ""
        for m in qa_msgs:
            role_label = "User" if m["role"] == "user" else "Assistant"
            content = m.get("content", "")
            history_str += f"{role_label}: {content}\n"
        return history_str

    def _run_qa_turn(self):
        history_str = self._get_current_qa_history_str()
        
        if self.qa_type == "initial":
            sampled_texts_str = ""
            for idx in self.qa_example_docs:
                sampled_texts_str += f"[Doc ID: {idx}] {self.texts[idx][:250]}...\n\n"
                
            system_prompt = f"""You are a helpful and professional data clustering assistant.
We are clarifying the user's clustering intent through a conversational dialogue.

To establish a solid clustering setup, you must clarify/collect the following checklist items:
1. **Aspect**: What primary dimension/aspect are we grouping by (e.g. topic, methodology, target audience)?
2. **Granularity**: Do they prefer coarse/broad clusters or fine-grained/detailed sub-themes?
3. **Example preference**: Ask the user how to group a specific pair of documents from the 20 samples below to verify their criteria.

Here are the 20 representative documents from the dataset to select examples from:
{sampled_texts_str}

CONVERSATION HISTORY OF THIS Q&A PHASE:
{history_str}

INSTRUCTIONS:
- Review the history carefully. Identify which checklist items (Aspect, Granularity, Example preference) have been successfully resolved.
- Speak naturally as an analytical, curious assistant. Focus on digging deeper into the user's responses to understand their exact logic in depth.
- IMPORTANT: Ask exactly ONE clarifying question at a time. Do NOT ask multiple questions in a single turn, and do NOT combine multiple checklist items.
- Focus on the next unresolved item:
  1. Aspect: If the user provides a high-level aspect (e.g. "by topic"), show curiosity and dig deeper. Probe them on what that means specifically for the 20 representative documents below. Ask follow-up questions to understand their sorting logic in depth.
  2. Granularity: If Aspect is settled, ask about Granularity (coarse vs. fine-grained) and prompt them for the level of specificity they want.
  3. Example preference: Once Aspect and Granularity are clarified, dedicate a turn to validating their rules on a boundary pair of documents. Select two specific documents from the 20 samples below that might test the boundaries of their logic. Present them and ask exactly: "Should Document A and Document B belong to the same cluster or separate clusters? Please explain why."
- At the end of your response, provide exactly 3 detailed suggestion options (15-25 words each) representing choices/replies the user might select for the current question only. Suggestions must be complete, context-rich options. Format exactly as:
  [SUGGESTIONS] Option 1 | Option 2 | Option 3 [/SUGGESTIONS]
- When you are confident you have gathered all necessary criteria (Aspect, Granularity, and Example preference), write a clear summary of the resolved intent and end your response exactly with:
  [CONFIRMED] Summary of the final clustering intent description

Write your response strictly in English."""
        else:
            import json
            registry_str = json.dumps(self.global_registry, indent=2, ensure_ascii=False)
            keywords = self.get_cluster_keywords(top_k=5)
            central_docs = self.get_central_docs_strings(top_n=2)
            
            cluster_summary_str = ""
            for l in sorted(keywords.keys()):
                kw = ", ".join(keywords.get(l, []))
                docs = "\n  - ".join(central_docs.get(l, []))
                cluster_summary_str += f"Cluster {l} (Keywords: {kw}):\n  - {docs}\n\n"
                
            system_prompt = f"""You are an expert data analysis assistant.
The user wanted to refine the clustering with original feedback: "{self.pending_feedback}"
 
Here are the current cluster partitions:
{cluster_summary_str}
 
CONVERSATION HISTORY OF THIS CLARIFICATION PHASE:
{history_str}
 
INSTRUCTIONS:
Evaluate if the user's refinement intent is now clear and specific enough to apply constraints.
- If it is clear and specific, write a summary of the clarified refinement intent and end your response exactly with:
  [CONFIRMED] Summary of the final refinement intent
- If you need more clarification, ask a clarifying question.
- At the end of your response, provide 3 suggestions formatted exactly as:
  [SUGGESTIONS] Option 1 | Option 2 | Option 3 [/SUGGESTIONS]

All fields, questions, suggestions, and intents MUST be written strictly in English."""

        try:
            response = self.qa_agent.generate_text(system_prompt, "Please continue the Q&A clarification turn.")
            content = response.strip()
            
            # Parse suggestions if present
            suggestions = ["By Topic", "By Sentiment", "By Methodology"]
            sugg_match = re.search(r'\[SUGGESTIONS\](.*?)\[/SUGGESTIONS\]', content, re.DOTALL)
            if sugg_match:
                sugg_str = sugg_match.group(1).strip()
                suggestions = [s.strip() for s in sugg_str.split('|') if s.strip()]
                # Remove suggestions block from display content
                content = content.replace(sugg_match.group(0), "").strip()
                
            # Parse confirmation if present
            if "[CONFIRMED]" in content:
                parts = content.split("[CONFIRMED]")
                display_content = parts[0].strip()
                final_intent = parts[1].strip()
                
                # If final_intent is empty, fallback to the display content
                if not final_intent:
                    final_intent = display_content
                
                self._finish_qa_and_cluster(final_intent, display_content)
            else:
                self.history.append({
                    "role": "assistant",
                    "content": content,
                    "suggestions": suggestions,
                    "type": f"qa_{self.qa_type}_turn"
                })
        except Exception as e:
            self.history.append({
                "role": "assistant",
                "content": f"Bot encountered an error: {e}. Could you clarify what aspect you'd like to cluster these documents by?",
                "suggestions": ["By Topic", "By Sentiment", "By Methodology"],
                "type": f"qa_{self.qa_type}_turn"
            })

    def _finish_qa_and_cluster(self, final_intent: Optional[str] = None, summary_text: Optional[str] = None):
        if final_intent is None:
            final_intent = self.compile_final_intent()
            
        self.user_intent = final_intent
        self.qa_active = False
        
        self.qa_awaiting_confirm = True
        if summary_text is None:
            summary_text = f"""### 📋 Clustering Summary
- **Aspect/Intent:** {final_intent}

Please click **Confirm & Run** to execute the clustering, or **Modify Request** to make edits."""
        
        self.qa_summary = summary_text
        self.history.append({
            "role": "assistant",
            "content": summary_text,
            "type": "qa_confirm"
        })

    def confirm_and_run(self):
        """Execute the initial clustering after user confirmation."""
        self.qa_awaiting_confirm = False
        self.qa_active = False
        
        # Run actual clustering
        self.run_initial_clustering(bypass_qa=True)
        
        greeting = f"I have clarified your intent: **\"{self.user_intent}\"**."
        natural_text = self._generate_natural_summary(greeting)
        
        image = self._generate_pca_plot_image()
        keywords = self.get_cluster_keywords(top_k=6)
        central_docs = self.get_central_docs_strings(top_n=3)
        
        self.history.append({
            "role": "assistant",
            "content": natural_text,
            "image": image,
            "labels": list(self.labels) if self.labels is not None else None,
            "cluster_details": {
                "keywords": keywords,
                "central_docs": central_docs
            },
            "type": "clustering_result"
        })

    def modify_qa_request(self):
        """Rollback to let the user modify their clustering request."""
        self.qa_awaiting_confirm = False
        self.qa_active = True
        # Decrement Q&A turn so they aren't immediately forced out
        self.qa_turn = max(1, self.qa_turn - 1)
        self.history.append({
            "role": "assistant",
            "content": "Sure, what would you like to modify about the clustering intent?",
            "suggestions": ["Change aspect/intent", "Start over"],
            "type": "qa_modify"
        })

    def compile_final_intent(self) -> str:
        history_str = self._get_current_qa_history_str()
        system_prompt = f"""You are an expert data analysis assistant.
Based on the following conversation history, write a concise summary (1-2 sentences) of the final clustering/refinement intent of the user.
The summary MUST be written strictly in English.

CONVERSATION HISTORY:
{history_str}

You must return ONLY the summary text, nothing else."""
        try:
            final_intent = self.qa_agent.generate_text(system_prompt, "Please summarize the final intent.")
            return final_intent.strip()
        except Exception:
            user_msgs = [m["content"] for m in self.history if m["role"] == "user"]
            return user_msgs[-1] if user_msgs else "Group similar documents."

    def _generate_natural_summary(self, title_msg: str) -> str:
        """
        Ask the Q&A Agent to write a natural, professional summary of the current clusters.
        """
        if self.qa_agent is None:
            return title_msg
            
        keywords = self.get_cluster_keywords(top_k=6)
        central_docs = self.get_central_docs_strings(top_n=3)
        
        clusters_desc = ""
        for l in sorted(keywords.keys()):
            kw_str = ", ".join(keywords.get(l, []))
            docs_str = "\n  - ".join([f'"{d}"' for d in central_docs.get(l, [])])
            clusters_desc += f"Cluster {l}:\n- Keywords: {kw_str}\n- Representative Documents:\n  - {docs_str}\n\n"
            
        system_prompt = f"""You are a helpful and professional data analysis assistant.
The user's clustering intent is: "{self.user_intent}"
We have successfully completed the clustering execution (producing {self.n_clusters} clusters).

Here is the details of the discovered clusters (keywords and representative document medoids):
{clusters_desc}

INSTRUCTIONS:
Write a natural, friendly, and concise assistant summary in English explaining that the clustering/refinement has been completed.
- Present the overview of the discovered categories and briefly comment on their quality or relationship to the user's intent.
- Do NOT list the documents or keywords in full detail (as they will be displayed in structured tables and interactive plots below). Keep the text conversational, clear, and professional.
- Ask the user if they want to make any further refinements (e.g., split a specific cluster or regroup).
- Return ONLY the natural text response, nothing else."""

        try:
            return self.qa_agent.generate_text(system_prompt, title_msg)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to generate natural summary: {e}. Falling back to default.")
            return title_msg
    def _generate_pca_plot_image(self) -> str:
        import io
        import base64
        from scipy.spatial import ConvexHull
        from scipy.interpolate import splprep, splev
        import matplotlib.patches as patches
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        
        ax.grid(True, linestyle=':', alpha=0.15, color='#94a3b8')
        ax.set_facecolor('#0a0e17')
        fig.patch.set_facecolor('#080c14')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#1e293b')
        ax.spines['bottom'].set_color('#1e293b')
        ax.tick_params(colors='#64748b')
        
        colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#EC4899', '#EAB308']
        
        n_clusters = self.n_clusters
        labels = self.labels
        sampled_indices = set(self.sampled_indices)
        pca_coords = self.pca_coords
        
        if pca_coords is None:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                pca_coords = pca.fit_transform(self.embeddings)
                self.pca_coords = pca_coords
            except Exception:
                return ""
                
        x_span = pca_coords[:, 0].max() - pca_coords[:, 0].min()
        y_span = pca_coords[:, 1].max() - pca_coords[:, 1].min()
        span = max(x_span, y_span, 1.0)
        
        r = 0.08 * span
        padding = 0.12 * span
        
        for l in range(n_clusters):
            cluster_points = pca_coords[labels == l]
            if len(cluster_points) == 0:
                continue
                
            color = colors[l % len(colors)]
            
            if len(cluster_points) == 1:
                x, y = cluster_points[0]
                circle = patches.Circle((x, y), r, edgecolor=color, facecolor=color, alpha=0.15, linewidth=2, linestyle='-', zorder=1)
                ax.add_patch(circle)
            elif len(cluster_points) == 2:
                p1, p2 = cluster_points[0], cluster_points[1]
                center = (p1 + p2) / 2
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                dist = np.sqrt(dx**2 + dy**2)
                angle = np.degrees(np.arctan2(dy, dx))
                
                ellipse = patches.Ellipse(center, width=dist + padding, height=padding, angle=angle,
                                          edgecolor=color, facecolor=color, alpha=0.15, linewidth=2, linestyle='-', zorder=1)
                ax.add_patch(ellipse)
            else:
                try:
                    hull = ConvexHull(cluster_points)
                    hull_pts = cluster_points[hull.vertices]
                    centroid = np.mean(hull_pts, axis=0)
                    expanded_pts = centroid + 1.25 * (hull_pts - centroid)
                    
                    x = list(expanded_pts[:, 0])
                    y = list(expanded_pts[:, 1])
                    x.append(x[0])
                    y.append(y[0])
                    
                    tck, u = splprep([x, y], s=0, per=True)
                    u_new = np.linspace(0, 1, 100)
                    x_smooth, y_smooth = splev(u_new, tck)
                    
                    ax.fill(x_smooth, y_smooth, color=color, alpha=0.15, zorder=1)
                    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, linestyle='-', zorder=2)
                except Exception:
                    try:
                        hull = ConvexHull(cluster_points)
                        hull_pts = cluster_points[hull.vertices]
                        centroid = np.mean(hull_pts, axis=0)
                        expanded_pts = centroid + 1.25 * (hull_pts - centroid)
                        polygon = patches.Polygon(expanded_pts, closed=True, edgecolor=color, facecolor=color, alpha=0.15, linewidth=2.5, zorder=1)
                        ax.add_patch(polygon)
                    except Exception:
                        pass
                        
        for l in range(n_clusters):
            color = colors[l % len(colors)]
            cluster_indices = np.where(labels == l)[0]
            
            rep_indices = [idx for idx in cluster_indices if idx in sampled_indices]
            reg_indices = [idx for idx in cluster_indices if idx not in sampled_indices]
            
            if reg_indices:
                reg_points = pca_coords[reg_indices]
                ax.scatter(
                    reg_points[:, 0], reg_points[:, 1],
                    color=color, s=40, alpha=0.85, edgecolors='none',
                    label=f'Cluster {l} ({len(cluster_indices)} pts)', zorder=3
                )
                
            if rep_indices:
                rep_points = pca_coords[rep_indices]
                ax.scatter(
                    rep_points[:, 0], rep_points[:, 1],
                    color=color, s=120, alpha=1.0, edgecolors='#080c14', linewidths=1.5,
                    marker='o', zorder=4
                )
                
        ax.set_xlabel("PCA Component 1", fontsize=9, color='#64748b')
        ax.set_ylabel("PCA Component 2", fontsize=9, color='#64748b')
        
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, facecolor='#0a0e17', edgecolor='#1e293b', fontsize=8, labelcolor='#e2e8f0')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def _generate_raw_pca_plot_image(self) -> str:
        import io
        import base64
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.grid(True, linestyle=':', alpha=0.15, color='#94a3b8')
        ax.set_facecolor('#0a0e17')
        fig.patch.set_facecolor('#080c14')
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#1e293b')
        ax.spines['bottom'].set_color('#1e293b')
        ax.tick_params(colors='#64748b')
        
        pca_coords = self.pca_coords
        if pca_coords is None:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                pca_coords = pca.fit_transform(self.embeddings)
                self.pca_coords = pca_coords
            except Exception:
                return ""
        
        # Plot all points in a neutral slate color
        ax.scatter(
            pca_coords[:, 0], pca_coords[:, 1],
            color='#475569', s=40, alpha=0.6, edgecolors='none',
            label='Documents'
        )
        
        # Highlight the sampled indices
        sampled_indices = self.sampled_indices
        if sampled_indices:
            sampled_coords = pca_coords[sampled_indices]
            ax.scatter(
                sampled_coords[:, 0], sampled_coords[:, 1],
                color='#6366f1', s=120, alpha=1.0, edgecolors='#080c14', linewidths=1.5,
                label='Sampled (FPS) Documents'
            )
            
        ax.set_xlabel("PCA Component 1", fontsize=9, color='#64748b')
        ax.set_ylabel("PCA Component 2", fontsize=9, color='#64748b')
        ax.legend(loc='best', frameon=True, facecolor='#0a0e17', edgecolor='#1e293b', fontsize=8, labelcolor='#e2e8f0')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    def get_cluster_keywords(self, top_k: int = 5) -> Dict[int, List[str]]:
        """Compute top keywords for each cluster using TF-IDF."""
        summaries = {}
        unique_labels = np.unique(self.labels)
        
        if len(unique_labels) <= 1:
            for l in range(self.n_clusters):
                cluster_texts = [self.texts[idx] for idx in range(len(self.texts)) if self.labels[idx] == l]
                words = re.findall(r'\b\w{3,}\b', " ".join(cluster_texts).lower())
                common = [w for w, _ in Counter(words).most_common(top_k)]
                summaries[l] = common
            return summaries
            
        cluster_docs = []
        for l in range(self.n_clusters):
            cluster_texts = [self.texts[idx] for idx in range(len(self.texts)) if self.labels[idx] == l]
            if not cluster_texts:
                cluster_docs.append("")
            else:
                cluster_docs.append(" ".join(cluster_texts))
                
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
            tfidf_matrix = vectorizer.fit_transform(cluster_docs)
            feature_names = vectorizer.get_feature_names_out()
            
            for l in range(self.n_clusters):
                if not cluster_docs[l]:
                    summaries[l] = []
                    continue
                row = tfidf_matrix.getrow(l).toarray()[0]
                top_indices = np.argsort(row)[::-1][:top_k]
                summaries[l] = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        except Exception:
            for l in range(self.n_clusters):
                cluster_texts = [self.texts[idx] for idx in range(len(self.texts)) if self.labels[idx] == l]
                words = re.findall(r'\b\w{3,}\b', " ".join(cluster_texts).lower())
                common = [w for w, _ in Counter(words).most_common(top_k)]
                summaries[l] = common
                
        return summaries

    def get_central_documents(self, top_n: int = 3) -> Dict[int, List[int]]:
        """Get indices of documents closest to centroids in each cluster."""
        central_indices = {}
        for l in range(self.n_clusters):
            indices = np.where(self.labels == l)[0]
            if len(indices) == 0:
                central_indices[l] = []
                continue
            dists = np.sum((self.embeddings[indices] - self.centroids[l])**2, axis=1)
            sorted_sub_indices = np.argsort(dists)[:top_n]
            central_indices[l] = [int(indices[idx]) for idx in sorted_sub_indices]
        return central_indices

    def get_central_docs_strings(self, top_n: int = 3) -> Dict[int, List[str]]:
        """Get text content of central documents."""
        indices_dict = self.get_central_documents(top_n)
        return {l: [self.texts[idx] for idx in idxs] for l, idxs in indices_dict.items()}

    @property
    def must_link(self) -> List[Tuple[int, int]]:
        return [pair for pair, info in self.constraint_ledger.items() if info["type"] == "Must-Link"]
        
    @property
    def cannot_link(self) -> List[Tuple[int, int]]:
        return [pair for pair, info in self.constraint_ledger.items() if info["type"] == "Cannot-Link"]
