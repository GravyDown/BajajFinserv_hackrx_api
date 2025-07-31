from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv

from api.routes import router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create FastAPI app
app = FastAPI(
    title="HackRx Document Q&A API",
    description="Intelligent document-based question-answering system for legal, insurance, and compliance use cases",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/hackrx", tags=["document-qa"])

@app.get("/")
async def root():
    return {
        "message": "HackRx Document Q&A API",
        "endpoints": {
            "process": "/hackrx/run",
            "health": "/hackrx/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )