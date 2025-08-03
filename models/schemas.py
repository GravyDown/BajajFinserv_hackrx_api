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
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A20Z",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?",
                    "Does this policy cover maternity expenses, and what are the conditions?",
                    "What is the waiting period for cataract surgery?",
                    "Are the medical expenses for an organ donor covered under this policy?",
                    "What is the No Claim Discount (NCD) offered in this policy?",
                    "Is there a benefit for preventive health check-ups?",
                    "How does the policy define a 'Hospital'?",
                    "What is the extent of coverage for AYUSH treatments?",
                    "Are there any sub-limits on room rent and ICU charges for Plan A?"
                ]
            }
        }

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