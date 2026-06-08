from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLMAgent(ABC):
    """Abstract base class defining the methods that all agents must implement."""
    
    @abstractmethod
    def generate_constraints(self, history: List[Dict[str, str]], sampled_data: List[str]) -> Dict:
        """
        Accept chat history and sampled data, and return must-link and cannot-link constraints.
        """
        pass

    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send system prompt and user prompt to LLM and return raw response text.
        """ 
        pass
