from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Dict
import logging
import os

from models.schemas import HackRxRequest, HackRxResponse
from services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize processor
processor = DocumentProcessor()

@router.post("/run", response_model=HackRxResponse)
async def process_document(request: Request) -> HackRxResponse:
    """
    Process document and answer questions
    
    - Accepts document URL (direct or Google Drive)
    - Processes list of questions
    - Returns structured answers with confidence scores
    """
    # 🔐 API Key Authentication (non-invasive addition)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    expected_token = os.getenv("HACKRX_API_KEY")

    if token != expected_token:
        raise HTTPException(status_code=403, detail="Invalid API key")

    try:
        body = await request.json()
        request_model = HackRxRequest(**body)

        logger.info(f"Received request for document: {request_model.document.url}")
        logger.info(f"Number of questions: {len(request_model.questions)}")

        if not request_model.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        response = await processor.process_request(request_model)

        logger.info(f"Successfully processed request in {response.processing_time:.2f}s")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "HackRx Document Q&A API"}
