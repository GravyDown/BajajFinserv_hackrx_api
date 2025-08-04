from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    UNKNOWN = "unknown"

class HackRxRequest(BaseModel):
    documents: str = Field(..., description="Direct URL to document (PDF, DOCX, etc.)")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer from document")


class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to each question")
    
    
# Legacy models for backward compatibility (if needed)
class DocumentInput(BaseModel):
    url: str = Field(..., description="Direct URL or Google Drive link to document")
    document_type: Optional[DocumentType] = Field(None, description="Optional document type hint")

class QuestionInput(BaseModel):
    questions: List[str] = Field(..., description="List of questions to answer")

class Answer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    
class DetailedHackRxResponse(BaseModel):
    """Detailed response format with additional metadata"""
    document_url: str
    total_chunks: int
    answers: List[Answer]