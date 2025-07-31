from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for list of texts"""
        if not texts:
            return np.array([])
            
        # Process in batches for memory efficiency
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )[0]