from typing import List
import numpy as np
import warnings
from .base_embeddings import BaseEmbeddings

class SentenceTransformerEmbeddings(BaseEmbeddings):
    """
    Embedding provider using the local sentence-transformers library.
    Falls back gracefully to TF-IDF + TruncatedSVD if sentence-transformers is unavailable.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: The name of the Hugging Face SentenceTransformer model to load.
        """
        self.model_name = model_name
        self.use_fallback = False
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            warnings.warn(
                f"Failed to load sentence-transformers ({e}). "
                "Falling back to TF-IDF + TruncatedSVD embeddings."
            )
            self.use_fallback = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.svd = TruncatedSVD(n_components=4, random_state=42)
            self.is_fit = False
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using sentence-transformers, or TF-IDF fallback if enabled.
        
        Args:
            texts: List of strings.
            
        Returns:
            A numpy array of shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.empty((0, 0))
            
        if self.use_fallback:
            try:
                if not self.is_fit:
                    tfidf = self.vectorizer.fit_transform(texts)
                    # Adjust SVD components if there are too few samples/features
                    n_comps = min(self.svd.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
                    if n_comps > 0:
                        from sklearn.decomposition import TruncatedSVD
                        self.svd = TruncatedSVD(n_components=n_comps, random_state=42)
                        embs = self.svd.fit_transform(tfidf)
                    else:
                        embs = tfidf.toarray()
                    self.is_fit = True
                else:
                    tfidf = self.vectorizer.transform(texts)
                    if hasattr(self, 'svd') and hasattr(self.svd, 'components_'):
                        embs = self.svd.transform(tfidf)
                    else:
                        embs = tfidf.toarray()
            except Exception as e:
                warnings.warn(f"Fallback embedding generation failed: {e}. Returning random embeddings.")
                np.random.seed(42)
                embs = np.random.randn(len(texts), 16)
        else:
            # Standard sentence-transformers execution
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            
        # Perform L2 Normalization on the vectors to optimize for Cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms


class OllamaEmbeddings(BaseEmbeddings):
    """
    Embedding provider using local Ollama service's `/api/embeddings` or `/api/embed` endpoint.
    """
    
    def __init__(self, model_name: str = "embeddinggemma:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama embedding client.
        
        Args:
            model_name: The name of the embedding model pulled in Ollama.
            base_url: The URL of the local Ollama instance.
        """
        self.base_url = base_url
        self.model_name = self._resolve_model(base_url, model_name)
        
    def _resolve_model(self, base_url: str, model_name: str) -> str:
        try:
            import requests
            res = requests.get(f"{base_url}/api/tags", timeout=5)
            if res.status_code == 200:
                models = [m["name"] for m in res.json().get("models", [])]
                # 1. Exact match
                if model_name in models:
                    return model_name
                # 2. Try finding any model with "embed" in its name
                for m in models:
                    if "embed" in m.lower():
                        print(f"[Ollama] Embedding model '{model_name}' not found. Resolving to available embedding model '{m}'.")
                        return m
                # 3. Match tag prefix or sub-string
                for m in models:
                    if model_name.split(":")[0] in m:
                        print(f"[Ollama] Embedding model '{model_name}' not found. Resolving to sibling model '{m}'.")
                        return m
                # 4. Fall back to first available model if any
                if models:
                    print(f"[Ollama] Embedding model '{model_name}' not found. Falling back to available model '{models[0]}'.")
                    return models[0]
        except Exception as e:
            print(f"[Ollama Warning] Failed to query tags to resolve embedding model: {e}")
        return model_name
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Ollama's embeddings endpoint.
        
        Args:
            texts: List of strings.
            
        Returns:
            A numpy array of shape (num_texts, embedding_dim).
        """
        if not texts:
            return np.empty((0, 0))
            
        import requests
        
        # Try batch /api/embed first for performance
        try:
            res = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=30
            )
            if res.status_code == 200:
                embs = res.json().get("embeddings")
                if embs and len(embs) == len(texts):
                    embs = np.array(embs)
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    return embs / norms
        except Exception:
            pass
            
        # Fallback to single text /api/embeddings endpoint if batch fails
        embeddings = []
        for text in texts:
            try:
                res = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=10
                )
                if res.status_code == 200:
                    emb = res.json().get("embedding")
                    if emb:
                        embeddings.append(emb)
                    else:
                        raise ValueError(f"No embedding found in response: {res.json()}")
                else:
                    raise RuntimeError(f"Ollama returned status code {res.status_code}: {res.text}")
            except Exception as e:
                # Try single /api/embed as last resort
                try:
                    res = requests.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model_name, "input": [text]},
                        timeout=10
                    )
                    if res.status_code == 200:
                        embs = res.json().get("embeddings")
                        if embs and len(embs) > 0:
                            embeddings.append(embs[0])
                        else:
                            raise ValueError(f"No embedding found in response: {res.json()}")
                    else:
                        raise RuntimeError(f"Ollama returned status code {res.status_code}: {res.text}")
                except Exception as inner_e:
                    raise RuntimeError(f"Ollama embedding request failed for model '{self.model_name}': {e} -> {inner_e}")
                    
        embs = np.array(embeddings)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

