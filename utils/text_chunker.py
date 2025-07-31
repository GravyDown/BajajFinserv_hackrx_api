from typing import List, Dict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into chunks with metadata"""
        # Clean text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self.splitter.split_text(text)
        
        # Add metadata
        chunk_data = []
        for idx, chunk in enumerate(chunks):
            chunk_data.append({
                'id': idx,
                'text': chunk,
                'length': len(chunk),
                'start_char': text.find(chunk),
                'end_char': text.find(chunk) + len(chunk)
            })
            
        return chunk_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might break processing
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()