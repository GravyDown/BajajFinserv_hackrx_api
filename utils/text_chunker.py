from typing import List, Dict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Split text into chunks with metadata (tries semantic, falls back to char-based)"""
        text = self._clean_text(text)
        # Try semantic chunking first
        semantic_chunks = self._split_by_headings(text)
        chunk_data = []
        idx = 0
        for section_title, section_text in semantic_chunks:
            # Further split large sections using char-based chunking
            if len(section_text) > self.chunk_size:
                sub_chunks = self.splitter.split_text(section_text)
                for sub_chunk in sub_chunks:
                    chunk_data.append({
                        'id': idx,
                        'text': sub_chunk,
                        'section': section_title,
                        'length': len(sub_chunk),
                        'start_char': section_text.find(sub_chunk),
                        'end_char': section_text.find(sub_chunk) + len(sub_chunk)
                    })
                    idx += 1
            else:
                chunk_data.append({
                    'id': idx,
                    'text': section_text,
                    'section': section_title,
                    'length': len(section_text),
                    'start_char': 0,
                    'end_char': len(section_text)
                })
                idx += 1
        return chunk_data

    def _split_by_headings(self, text: str) -> List[tuple]:
        """
        Splits text by common headings (Section, Chapter, Article, etc.).
        Returns list of (heading, section_text) tuples.
        """
        # Regex for common headings (customize as needed)
        pattern = r'((?:Section|Chapter|Article|Clause|Part|Schedule)[\s\d\w\.\-:]+)'
        splits = [m for m in re.finditer(pattern, text, re.IGNORECASE)]
        if not splits:
            # No headings found, treat whole text as one chunk
            return [("Full Document", text)]
        chunks = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i+1].start() if i+1 < len(splits) else len(text)
            heading = match.group(1).strip()
            section_text = text[start:end].strip()
            chunks.append((heading, section_text))
        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('’', "'").replace('‘', "'")
        return text.strip()