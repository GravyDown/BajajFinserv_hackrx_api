import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

class VectorStoreService:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []
        self.chunk_embeddings = None
        
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """Build FAISS index from embeddings"""
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
            
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.index.add(embeddings)
        
        self.chunks = chunks
        self.chunk_embeddings = embeddings
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for k most similar chunks"""
        if self.index is None:
            raise ValueError("Index not built yet")
            
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        # Return chunks with scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
                
        return results
    
    def save_index(self, path: str):
        """Save index and metadata to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save metadata
        metadata = {
            'chunks': self.chunks,
            'embeddings': self.chunk_embeddings,
            'embedding_dim': self.embedding_dim
        }
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(metadata, f)
            
    def load_index(self, path: str):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load metadata
        with open(f"{path}.meta", 'rb') as f:
            metadata = pickle.load(f)
            
        self.chunks = metadata['chunks']
        self.chunk_embeddings = metadata['embeddings']
        self.embedding_dim = metadata['embedding_dim']