import time
import asyncio
import os
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.llm_service import LLMService
from utils.document_parser import DocumentParser
from utils.text_chunker import TextChunker
from models.schemas import HackRxRequest, HackRxResponse

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        
    async def process_request(self, request: HackRxRequest) -> HackRxResponse:
        """Main processing pipeline"""
        start_time = time.time()
        file_path = None
        
        try:
            # Step 1: Download and parse document (run in thread pool to avoid blocking)
            logger.info(f"Downloading document from: {request.documents}")  # Changed from request.document.url
            loop = asyncio.get_event_loop()
            
            # Run download and parse as a single atomic operation
            file_path, doc_type, text = await loop.run_in_executor(
                None, self._download_and_parse, request.documents  # Changed from request.document.url
            )
            
            if not text or len(text.strip()) < 10:
                raise ValueError("Document appears to be empty or invalid")
                
            # Step 2: Chunk the document
            logger.info("Chunking document")
            chunks = await loop.run_in_executor(
                None, self.chunker.chunk_text, text
            )
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings (run in thread pool)
            logger.info("Generating embeddings")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = await loop.run_in_executor(
                None, self.embedding_service.embed_texts, chunk_texts
            )
            
            # Step 4: Build vector store
            logger.info("Building vector index")
            vector_store = VectorStoreService(self.embedding_service.embedding_dim)
            await loop.run_in_executor(
                None, vector_store.build_index, embeddings, chunks
            )
            
            # Step 5: Process questions
            logger.info(f"Processing {len(request.questions)} questions")
            answers = await self._process_questions(
                request.questions,
                vector_store,
                chunks
            )
            
            processing_time = time.time() - start_time
            
            # Return simplified response format
            return HackRxResponse(
                answers=answers  # Just return list of answer strings
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
        finally:
            # Ensure cleanup runs even on error
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.parser.cleanup)
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
    
    def _download_and_parse(self, url: str) -> tuple:
        """Atomic download and parse operation to avoid race conditions"""
        try:
            # Download file
            file_path, doc_type = self.parser.download_file(url)
            
            # Verify file exists and has content
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Downloaded file not found at: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"Downloaded file is empty: {file_path}")
            
            logger.info(f"File downloaded successfully: {file_path} ({file_size} bytes)")
            logger.info(f"Parsing {doc_type} document")
            
            # Parse document
            text = self.parser.parse_document(file_path, doc_type)
            
            return file_path, doc_type, text
            
        except Exception as e:
            logger.error(f"Error in download_and_parse: {e}")
            # If file was created but parsing failed, try to clean it up
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            raise
            
    async def _process_questions(
        self,
        questions: List[str],
        vector_store: VectorStoreService,
        chunks: List[Dict]
    ) -> List[str]:  # Changed return type to List[str]
        """Process multiple questions concurrently"""
        # Use asyncio.gather for concurrent async processing
        tasks = []
        for question in questions:
            task = self._answer_single_question(question, vector_store)
            tasks.append(task)
        
        # Process all questions concurrently
        try:
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions that occurred
            processed_answers = []
            for i, result in enumerate(answers):
                if isinstance(result, Exception):
                    logger.error(f"Error answering question '{questions[i]}': {result}")
                    processed_answers.append(f"Error processing question: {str(result)}")
                else:
                    processed_answers.append(result)  # result is now just a string
            
            return processed_answers
            
        except Exception as e:
            logger.error(f"Error in concurrent question processing: {e}")
            # Fallback to sequential processing
            return await self._process_questions_sequential(questions, vector_store)
    
    async def _process_questions_sequential(
        self,
        questions: List[str],
        vector_store: VectorStoreService
    ) -> List[str]:  # Changed return type to List[str]
        """Fallback sequential processing"""
        answers = []
        for question in questions:
            try:
                answer = await self._answer_single_question(question, vector_store)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                answers.append(f"Error processing question: {str(e)}")
        return answers
        
    async def _answer_single_question(
        self,
        question: str,
        vector_store: VectorStoreService
    ) -> str:  # Changed return type to str
        """Answer a single question using RAG"""
        loop = asyncio.get_event_loop()
        
        # Find relevant chunks (run in thread pool)
        query_embedding = await loop.run_in_executor(
            None, self.embedding_service.embed_query, question
        )
        relevant_chunks = await loop.run_in_executor(
            None, vector_store.search, query_embedding, 5
        )
        
        # Prepare context from relevant chunks
        context_parts = []
        
        for chunk, score in relevant_chunks:
            if score > 0.3:  # Relevance threshold
                context_parts.append(chunk['text'])
                
        context = "\n\n".join(context_parts)
        
        # Generate answer (run in thread pool if LLM calls are blocking)
        if context:
            result = await loop.run_in_executor(
                None, self.llm_service.answer_question, question, context
            )
            answer_text = result["answer"]
            # We're only returning the answer text now, not the full Answer object
            return answer_text
        else:
            return "No relevant information found in the document to answer this question."