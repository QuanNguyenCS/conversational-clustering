from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbeddings(ABC):
    """Abstract base class for all embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text documents into a dense matrix of shape (num_texts, embedding_dim).
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            A numpy array of shape (num_texts, embedding_dim) containing the embeddings.
        """
        pass
