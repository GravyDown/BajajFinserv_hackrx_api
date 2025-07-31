from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EMAIL = "email"
    UNKNOWN = "unknown"

class DocumentInput(BaseModel):
    url: str = Field(..., description="Direct URL or Google Drive link to document")
    document_type: Optional[DocumentType] = Field(None, description="Optional document type hint")

class QuestionInput(BaseModel):
    questions: List[str] = Field(..., description="List of questions to answer")

class HackRxRequest(BaseModel):
    document: DocumentInput
    questions: List[str] = Field(..., min_items=1, description="Questions to answer from document")
    
class Answer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    # source_chunks: List[str] = Field(default_factory=list, description="Relevant text chunks used")
    
class HackRxResponse(BaseModel):
    document_url: str
    total_chunks: int
    answers: List[Answer]
    processing_time: float
    # model_used: str