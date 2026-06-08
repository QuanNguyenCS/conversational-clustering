import numpy as np
import pytest
from src.embeddings.base_embeddings import BaseEmbeddings
from src.agent.base_agent import BaseLLMAgent
from src.pipeline import ConversationalClusteringPipeline

class MockEmbeddings(BaseEmbeddings):
    def embed_texts(self, texts):
        # Generate simple deterministic embeddings for mock testing
        # 4 dimensions
        return np.array([[len(t), len(t) * 1.5, 0.0, 1.0] for t in texts])

class MockLLMAgent(BaseLLMAgent):
    def generate_constraints(self, history, sampled_data):
        return {
            "must_link": [[0, 1]],
            "cannot_link": [[0, 2]]
        }
        
    def generate_text(self, system_prompt, user_prompt):
        import re
        import json
        
        # 1. Label Discovery & Refinement (which expects new_clusters and assignments)
        if "candidate_registry" in system_prompt:
            return json.dumps({
                "candidate_registry": {
                    "Cluster_0": "First group",
                    "Cluster_1": "Second group"
                }
            }, ensure_ascii=False)

        if "merge" in system_prompt.lower() or "merge" in user_prompt.lower():
            match_k = re.search(r'exactly\s+(\d+)\s+categories', system_prompt)
            target_k = int(match_k.group(1)) if match_k else 2
            
            original_cids = list(set(re.findall(r'\"(Cluster_[^\"]+)\"', system_prompt)))
            if not original_cids:
                original_cids = list(set(re.findall(r'\'(Cluster_[^\']+)\'', system_prompt)))
            if not original_cids:
                original_cids = ["Cluster_0", "Cluster_1"]
                
            merged_clusters = {}
            mapping = {}
            for idx, cid in enumerate(original_cids):
                merged_id = f"Merged_Cluster_{idx % target_k + 1}"
                mapping[cid] = merged_id
                merged_clusters[merged_id] = f"Merged category containing {cid}"
                
            return json.dumps({
                "merged_clusters": merged_clusters,
                "mapping": mapping
            }, ensure_ascii=False)

        if "new_clusters" in system_prompt and "assignments" in system_prompt:
            doc_ids = [int(num) for num in re.findall(r'\[Doc ID:\s*(\d+)\]', user_prompt)]
            assignments = {}
            new_clusters = {}
            for i, idx in enumerate(doc_ids):
                if i < 2:
                    assignments[f"doc_id_{idx}"] = "Cluster_0"
                    new_clusters["Cluster_0"] = "First group"
                else:
                    assignments[f"doc_id_{idx}"] = f"Cluster_{i}"
                    new_clusters[f"Cluster_{i}"] = f"Group {i}"
            return json.dumps({
                "new_clusters": new_clusters,
                "assignments": assignments
            }, ensure_ascii=False)

        # 2. Q&A Clarification Phase
        if "clustering criteria" in system_prompt or "data analysis assistant" in system_prompt:
            return "Here is a summary of the 20 documents. What aspect would you like to group by?\n[SUGGESTIONS] Group by Academic Subject | Group by Research Method | Group by Application [/SUGGESTIONS]"
            
        if "data clustering assistant" in system_prompt:
            return "I understand your preferences. Here is the confirmation summary.\n[CONFIRMED] Group by Academic Subject"

        # Default fallback
        return "Please clarify your criteria.\n[SUGGESTIONS] By Topic | By Sentiment | By Methodology [/SUGGESTIONS]"


def test_pipeline_integration():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor.",
        "The battery life is terrible on this generic brand.",
        "The sound system is muffled and tinny.",
        "Audio is fantastic and high definition.",
        "Great speakers with deep bass response."
    ]
    
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    # sample size 4
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    
    pipeline.set_data(texts)
    
    # Verify data setup and sampling
    assert len(pipeline.sampled_indices) == 4
    assert len(pipeline.sampled_texts) == 4
    
    # Run initial unconstrained clustering
    labels, keywords, central_docs = pipeline.run_initial_clustering()
    assert len(labels) == len(texts)
    discovered_k = pipeline.n_clusters
    assert len(keywords) == discovered_k
    assert len(central_docs) == discovered_k
    
    # Execute one feedback turn
    labels_turn, keywords_turn, central_docs_turn = pipeline.step("Group the reviews by Sentiment.")
    
    # Verify constraints were updated and mapped
    assert len(pipeline.must_link) >= 1
    assert len(pipeline.cannot_link) >= 1
    
    # The constraints should map to the full indices corresponding to sampled_indices[0], sampled_indices[1], etc.
    expected_ml_pair = (min(pipeline.sampled_indices[0], pipeline.sampled_indices[1]), max(pipeline.sampled_indices[0], pipeline.sampled_indices[1]))
    assert expected_ml_pair in pipeline.must_link
    
    # Verify pipeline state saved history
    assert len(pipeline.history) == 2  # User feedback, then assistant response
    assert pipeline.history[0]["role"] == "user"
    assert pipeline.history[1]["role"] == "assistant"

def test_pipeline_constrained_modes():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor.",
        "The battery life is terrible on this generic brand.",
        "The sound system is muffled and tinny.",
        "Audio is fantastic and high definition.",
        "Great speakers with deep bass response."
    ]
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    pipeline.set_data(texts)
    
    # Run initial clustering
    labels, keywords, central_docs = pipeline.run_initial_clustering()
    assert len(labels) == len(texts)
    # Mock agent discovers 4 clusters during initial discovery (Cluster_0, Cluster_1 from registry, plus Cluster_2, Cluster_3 fallbacks)
    assert pipeline.n_clusters == 4
    
    # Test error if K_ref < 2 by manually clearing global_registry
    pipeline.global_registry = {}
    with pytest.raises(ValueError, match="LLM discovered fewer than 2 clusters"):
        pipeline._refit_and_cluster()

def test_granularity_and_confirmation_flow():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor.",
        "The battery life is terrible on this generic brand.",
        "The sound system is muffled and tinny.",
        "Audio is fantastic and high definition.",
        "Great speakers with deep bass response."
    ]
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        use_qa_phase=True,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    pipeline.set_data(texts)
    
    # 1. Start Q&A: should trigger qa_active
    pipeline.start_initial_qa()
    assert pipeline.qa_active is True
    assert pipeline.qa_turn == 1
    assert len(pipeline.history) == 1
    assert pipeline.history[0]["type"] == "qa_start"
    
    # 2. User submits aspect/intent -> goes directly to confirm state since granularity is removed
    pipeline.step("Group by Academic Subject")
    assert pipeline.qa_active is False
    assert pipeline.qa_awaiting_confirm is True
    assert pipeline.user_intent == "Group by Academic Subject"
    assert pipeline.history[-1]["type"] == "qa_confirm"
    
    # 3. User modifies request -> back to active Q&A state
    pipeline.modify_qa_request()
    assert pipeline.qa_active is True
    assert pipeline.qa_awaiting_confirm is False
    assert pipeline.qa_turn == 1
    assert pipeline.history[-1]["type"] == "qa_modify"
    
    # 4. User submits modified selection -> back to confirm state
    pipeline.step("Group by Academic Subject")
    assert pipeline.qa_active is False
    assert pipeline.qa_awaiting_confirm is True
    
    # 5. User clicks Confirm & Run -> run clustering
    pipeline.confirm_and_run()
    assert pipeline.qa_awaiting_confirm is False
    assert pipeline.qa_active is False
    assert pipeline.labels is not None
    assert len(pipeline.labels) == len(texts)
    assert pipeline.history[-1]["type"] == "clustering_result"


def test_pipeline_custom_params():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor.",
        "The battery life is terrible on this generic brand.",
        "The sound system is muffled and tinny.",
        "Audio is fantastic and high definition.",
        "Great speakers with deep bass response."
    ]
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    # Initialize with custom clustering_params
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        use_itml=True,
        pca_dim=150,
        clustering_params={"max_iter": 50, "tol": 1e-3, "w": 25}
    )
    
    pipeline.set_data(texts)
    
    # Verify that the custom parameters are stored
    assert pipeline.clustering_params == {"max_iter": 50, "tol": 1e-3}
    
    # Verify execution runs successfully with these custom parameters
    labels, keywords, central_docs = pipeline.run_initial_clustering()
    assert len(labels) == len(texts)

def test_pipeline_explicit_k():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor.",
        "The battery life is terrible on this generic brand.",
        "The sound system is muffled and tinny.",
        "Audio is fantastic and high definition.",
        "Great speakers with deep bass response."
    ]
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    # Initialize with n_clusters = 5
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        n_clusters=5,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    pipeline.set_data(texts)
    
    labels, keywords, central_docs = pipeline.run_initial_clustering()
    assert len(labels) == len(texts)
    assert pipeline.n_clusters == 5
    assert len(keywords) == 5
    assert len(central_docs) == 5

def test_strategy4_discovery():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor.",
        "The battery life is terrible on this generic brand.",
        "The sound system is muffled and tinny.",
        "Audio is fantastic and high definition.",
        "Great speakers with deep bass response."
    ]
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    # Initialize pipeline
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    
    pipeline.set_data(texts)
    labels, keywords, central_docs = pipeline.run_initial_clustering()
    
    # Verify execution ran successfully
    assert len(labels) == len(texts)
    assert len(pipeline.global_registry) >= 2
    for idx in pipeline.sampled_indices:
        assert pipeline.final_assignments[idx] in pipeline.global_registry


def test_qa_checklist_memory():
    class MemoryMockAgent(BaseLLMAgent):
        def __init__(self):
            self.turn = 0
            
        def generate_constraints(self, history, sampled_data):
            return {"must_link": [], "cannot_link": []}
            
        def generate_text(self, system_prompt, user_prompt):
            import json
            if "data clustering assistant" in system_prompt:
                self.turn += 1
                if self.turn == 1:
                    return "What is the desired granularity?\n[SUGGESTIONS] Fine-grained | Coarse-grained | Let AI decide [/SUGGESTIONS]"
                else:
                    return "Understood. I will proceed with clustering.\n[CONFIRMED] Group academic paper abstracts by category with coarse granularity and dedicated outliers."
            elif "data analysis assistant" in system_prompt:
                return "I have analyzed the sample. What aspect would you like to group by?\n[SUGGESTIONS] Academic Category | Topic | Sentiment [/SUGGESTIONS]"
            elif "candidate_registry" in system_prompt:
                return json.dumps({
                    "candidate_registry": {
                        "Cluster_0": "First group",
                        "Cluster_1": "Second group"
                    }
                })
            elif "new_clusters" in system_prompt:
                return json.dumps({
                    "new_clusters": {},
                    "assignments": {"doc_id_0": "Cluster_0", "doc_id_1": "Cluster_1"}
                })
            return "{}"

    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor."
    ]
    emb_provider = MockEmbeddings()
    agent = MemoryMockAgent()
    
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=2,
        random_state=42,
        use_qa_phase=True,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    pipeline.set_data(texts)
    
    pipeline.start_initial_qa()
    assert pipeline.qa_active is True
    
    # Run Q&A Step 1
    pipeline.step("Let's group by Academic Category")
    assert pipeline.qa_active is True
    
    # Run Q&A Step 2 -> Confirmed state
    pipeline.step("Coarse granularity please")
    assert pipeline.qa_active is False
    assert pipeline.qa_awaiting_confirm is True
    assert pipeline.user_intent == "Group academic paper abstracts by category with coarse granularity and dedicated outliers."

def test_pipeline_cl_distance_filtering():
    texts = [
        "Apple makes great iphones and tablets.",
        "I love the battery life of Apple devices.",
        "Samsung android phones are very powerful.",
        "Google Pixel has a superb camera sensor."
    ]
    emb_provider = MockEmbeddings()
    agent = MockLLMAgent()
    
    # With a very small cl_distance_threshold_ratio, the cannot-link constraint should be filtered out
    pipeline = ConversationalClusteringPipeline(
        embedding_provider=emb_provider,
        agent=agent,
        num_samples=4,
        random_state=42,
        cl_distance_threshold_ratio=0.001,
        use_itml=True,
        pca_dim=150,
        clustering_params={"w": 25}
    )
    pipeline.set_data(texts)
    pipeline.run_initial_clustering()
    
    # Manually populate constraint ledger
    pipeline.constraint_ledger[(0, 2)] = {"type": "Cannot-Link", "turn": 0}
    pipeline.constraint_ledger[(0, 1)] = {"type": "Must-Link", "turn": 0}
    
    pipeline._refit_and_cluster()
    
    # The cannot-link constraint should have been deleted, must-link should remain
    assert (0, 2) not in pipeline.constraint_ledger
    assert (0, 1) in pipeline.constraint_ledger

