import os
from typing import List, Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.primary_llm = None
        self.fallback_llms = []
        self._initialize_llms()
        
    def _initialize_llms(self):
        """Initialize LLMs with fallback options"""
        # Try Google Gemini first
        # print("Google API Key:", os.getenv("GOOGLE_API_KEY"))
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.3,
                    max_output_tokens=1024
                )
                logger.info("Initialized Google Gemini as primary LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                
        # Try Groq as fallback
        if os.getenv("GROQ_API_KEY"):
            try:
                groq_llm = ChatGroq(
                    model="mixtral-8x7b-32768",
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.3,
                    max_tokens=1024
                )
                if self.primary_llm is None:
                    self.primary_llm = groq_llm
                else:
                    self.fallback_llms.append(groq_llm)
                logger.info("Initialized Groq LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
                
        # HuggingFace as final fallback (works offline)
        try:
            hf_llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                model_kwargs={"temperature": 0.3, "max_length": 512}
            )
            self.fallback_llms.append(hf_llm)
            logger.info("Initialized HuggingFace model as fallback")
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace model: {e}")
            
        if self.primary_llm is None and not self.fallback_llms:
            raise ValueError("No LLM models could be initialized. Please check API keys.")
            
    def _create_qa_prompt(self) -> PromptTemplate:
        """Create prompt template for Q&A"""
        template = """You are an expert assistant specializing in legal, insurance, and compliance matters. 
        Based on the following context from the document, provide a clear, accurate, and comprehensive answer to the question.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        - Answer based ONLY on the provided context
        - If the context doesn't contain enough information, say so clearly
        - Be precise and cite specific parts of the context when relevant
        - For legal/compliance questions, be extra careful about accuracy
        
        Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def answer_question(self, question: str, context: str, model_preference: Optional[str] = None) -> Dict:
        """Generate answer using available LLMs with fallback"""
        prompt = self._create_qa_prompt()
        
        # Try primary LLM first
        if self.primary_llm:
            try:
                chain = LLMChain(llm=self.primary_llm, prompt=prompt)
                answer = chain.run(context=context, question=question)
                return {
                    "answer": answer.strip(),
                    "model_used": self.primary_llm.__class__.__name__
                }
            except Exception as e:
                logger.warning(f"Primary LLM failed: {e}")
                
        # Try fallback LLMs
        for llm in self.fallback_llms:
            try:
                chain = LLMChain(llm=llm, prompt=prompt)
                answer = chain.run(context=context, question=question)
                return {
                    "answer": answer.strip(),
                    "model_used": llm.__class__.__name__
                }
            except Exception as e:
                logger.warning(f"Fallback LLM {llm.__class__.__name__} failed: {e}")
                continue
                
        # If all LLMs fail, return error
        return {
            "answer": "Unable to generate answer due to LLM service unavailability.",
            "model_used": "none"
        }
        
    def calculate_confidence(self, answer: str, context: str, question: str) -> float:
        """Calculate confidence score for the answer"""
        # Simple heuristic-based confidence scoring
        confidence = 0.5  # Base confidence
        
        # Check if answer indicates uncertainty
        uncertainty_phrases = [
            "not mentioned", "doesn't contain", "no information",
            "unclear", "cannot determine", "not specified"
        ]
        for phrase in uncertainty_phrases:
            if phrase.lower() in answer.lower():
                confidence -= 0.3
                
        # Check if answer is substantive
        if len(answer.split()) > 20:
            confidence += 0.2
            
        # Check if answer contains specific references
        if any(word in answer.lower() for word in ["according to", "states that", "mentions"]):
            confidence += 0.1
            
        # Ensure confidence is between 0 and 1
        return max(0.1, min(1.0, confidence))