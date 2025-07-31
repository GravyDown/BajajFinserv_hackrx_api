import time
import asyncio
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.embedding_service import EmbeddingService
from services.vector_store_service import VectorStoreService
from services.llm_service import LLMService
from utils.document_parser import DocumentParser
from utils.text_chunker import TextChunker
from models.schemas import HackRxRequest, HackRxResponse, Answer

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
        
        try:
            # Step 1: Download and parse document (run in thread pool to avoid blocking)
            logger.info(f"Downloading document from: {request.document.url}")
            loop = asyncio.get_event_loop()
            
            # Run blocking I/O operations in thread pool
            file_path, doc_type = await loop.run_in_executor(
                None, self.parser.download_file, request.document.url
            )
            
            logger.info(f"Parsing {doc_type} document")
            text = await loop.run_in_executor(
                None, self.parser.parse_document, file_path, doc_type
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
            
            # Clean up
            await loop.run_in_executor(None, self.parser.cleanup)
            
            processing_time = time.time() - start_time
            
            return HackRxResponse(
                document_url=request.document.url,
                total_chunks=len(chunks),
                answers=answers,
                processing_time=processing_time,
                # model_used=answers[0].model_used if answers else "none"
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            # Ensure cleanup runs even on error
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.parser.cleanup)
            raise
            
    async def _process_questions(
        self,
        questions: List[str],
        vector_store: VectorStoreService,
        chunks: List[Dict]
    ) -> List[Answer]:
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
                    processed_answers.append(Answer(
                        question=questions[i],
                        answer=f"Error processing question: {str(result)}",
                        confidence=0.0,
                        # source_chunks=[],
                        model_used="none"
                    ))
                else:
                    processed_answers.append(result)
            
            return processed_answers
            
        except Exception as e:
            logger.error(f"Error in concurrent question processing: {e}")
            # Fallback to sequential processing
            return await self._process_questions_sequential(questions, vector_store)
    
    async def _process_questions_sequential(
        self,
        questions: List[str],
        vector_store: VectorStoreService
    ) -> List[Answer]:
        """Fallback sequential processing"""
        answers = []
        for question in questions:
            try:
                answer = await self._answer_single_question(question, vector_store)
                answers.append(answer)
            except Exception as e:
                logger.error(f"Error answering question '{question}': {e}")
                answers.append(Answer(
                    question=question,
                    answer=f"Error processing question: {str(e)}",
                    confidence=0.0,
                    # source_chunks=[],
                    model_used="none"
                ))
        return answers
        
    async def _answer_single_question(
        self,
        question: str,
        vector_store: VectorStoreService
    ) -> Answer:
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
        # source_chunks = []
        
        for chunk, score in relevant_chunks:
            if score > 0.3:  # Relevance threshold
                context_parts.append(chunk['text'])
                # source_chunks.append(chunk['text'][:200] + "...")  # First 200 chars
                
        context = "\n\n".join(context_parts)
        
        # Generate answer (run in thread pool if LLM calls are blocking)
        if context:
            result = await loop.run_in_executor(
                None, self.llm_service.answer_question, question, context
            )
            answer_text = result["answer"]
            model_used = result["model_used"]
            confidence = await loop.run_in_executor(
                None, self.llm_service.calculate_confidence, answer_text, context, question
            )
        else:
            answer_text = "No relevant information found in the document to answer this question."
            model_used = "none"
            confidence = 0.1
            
        return Answer(
            question=question,
            answer=answer_text,
            confidence=confidence,
            # source_chunks=source_chunks[:3],  # Top 3 chunks
            model_used=model_used
        )