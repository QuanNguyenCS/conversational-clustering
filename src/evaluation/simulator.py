from typing import List, Dict, Any, Optional
from ..agent.base_agent import BaseLLMAgent

class UserSimulator:
    """
    Simulates a human user by inspecting the difference between current clusters
    and the ground truth, and generating natural language instructions for the LLM agent.
    """
    
    def __init__(self, agent: BaseLLMAgent):
        """
        Initialize the simulator.
        
        Args:
            agent: The LLM agent instance used to generate the simulated natural language response.
        """
        self.agent = agent
        
    def generate_feedback(
        self,
        aspect: str,
        sampled_texts: List[str],
        sampled_ground_truth_labels: List[Any],
        current_clustering_labels: List[int],
        cluster_keywords: Optional[Dict[int, List[str]]] = None,
        central_docs: Optional[Dict[int, List[str]]] = None
    ) -> str:
        """
        Generate feedback based on discrepancy between ground-truth and current clustering.
        
        Args:
            aspect: The aspect target (e.g. 'Sentiment', 'Subject Domain').
            sampled_texts: Texts of the sampled subset.
            sampled_ground_truth_labels: Ground truth labels for the aspect of the sampled subset.
            current_clustering_labels: Current cluster assignments for the sampled subset.
            cluster_keywords: Top terms/keywords for each cluster.
            central_docs: Central document texts for each cluster.
            
        Returns:
            A string containing the natural language feedback.
        """
        cluster_keywords = cluster_keywords or {}
        central_docs = central_docs or {}
        
        # Build the instruction for the simulator LLM
        system_prompt = (
            "You play the role of a human user interacting with the C3 conversational document clustering system.\n"
            "Your goal is to help the system cluster the entire dataset according to a specific target aspect.\n"
            "You will receive the target clustering aspect, the ground truth labels of the sampled documents, "
            "the current assignments of the sampled documents, and the top keywords/representative texts of each cluster.\n"
            "Your task is to write a short, natural, and direct feedback message (1-3 sentences) in English "
            "to guide the system to adjust the current clustering to match the ground truth aspect.\n\n"
            "Good examples of feedback:\n"
            "- 'Please separate positive and negative reviews since they are currently mixed in Cluster 0.'\n"
            "- 'Cluster 1 and Cluster 2 both discuss computer science topics, please merge them.'\n"
            "- 'I want to classify based on Sentiment. The current clustering is based on topic, please re-cluster based on customer satisfaction.'\n\n"
            "NOTE: The feedback must be natural as if chatting directly. Do not explain your reasoning, do not use markdown, and only return the raw text."
        )
        
        # Build prompt describing current discrepancy
        user_prompt = (
            f"Target Aspect: {aspect}\n\n"
            "List of sampled documents, ground truth, and current cluster:\n"
        )
        
        for idx, (text, gt, label) in enumerate(zip(sampled_texts, sampled_ground_truth_labels, current_clustering_labels)):
            # Truncate text for prompt cleanliness if long
            short_text = text[:150] + "..." if len(text) > 150 else text
            user_prompt += (
                f"Document [{idx}]: \"{short_text}\"\n"
                f"  - Ground Truth: {gt}\n"
                f"  - Current Cluster: {label}\n\n"
            )
            
        user_prompt += "Summary of current clusters:\n"
        n_clusters = len(set(current_clustering_labels))
        for l in range(n_clusters):
            keywords = ", ".join(cluster_keywords.get(l, []))
            docs = "\n  * ".join([d[:100] + "..." if len(d) > 100 else d for d in central_docs.get(l, [])])
            user_prompt += f"- Cluster {l} (Keywords: {keywords}):\n  * {docs}\n"
            
        user_prompt += (
            "\nCompare the distribution of 'Current Cluster' and 'Ground Truth' for the sampled documents under the requested aspect.\n"
            "Write a feedback message to the system to improve the current clustering."
        )
        
        feedback = self.agent.generate_text(system_prompt, user_prompt)
        return feedback.strip()
