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
import re

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
        if os.getenv("GOOGLE_API_KEY"):
            try:
                self.primary_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.3,
                    max_output_tokens=1024
                )
                logger.info("Initialized Google Gemini as primary LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                
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

    def _clean_answer(self, answer: str) -> str:
        answer = re.sub(r'\n\s*\n', ' ', answer)
        answer = answer.replace('\n', ' ')
        answer = re.sub(r'\s+', ' ', answer)
        return answer.strip()
    
    def _create_qa_prompt(self) -> PromptTemplate:
        template = """You are an expert assistant specializing in legal, insurance, and compliance matters.
        Based on the following context from the document, provide a clear, accurate, and comprehensive answer to the question.
        Example Q&A Format:
        Q: What is the waiting period for pre-existing diseases?
        A: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

        Q: Is maternity covered under this policy?
        A: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.

        Q: Are expenses for cataract surgery covered?
        A: Yes, but only after a waiting period of two (2) years from the policy inception date.
        Context:
        {context}

        Question: {question}

        Instructions:
        - Answer based ONLY on the provided context
        - If the context doesn't contain enough information, say so clearly
        - Be precise and cite specific parts of the context when relevant
        - For legal/compliance questions, be extra careful about accuracy
        - Be precise and try to return answers in under 3-4 sentences.
        - Start with Yes/No if applicable
        - Try to cite facts, numbers and figures from the context wherever possible.
        Answer:"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def _select_relevant_chunks(self, chunks: List[Dict], question: str, max_tokens: int = 3000) -> List[str]:
        """
        Select the most relevant chunks for the question using simple keyword scoring.
        Returns a list of chunk texts, up to max_tokens in total.
        """
        question_words = set(re.findall(r'\w+', question.lower()))
        scored_chunks = []
        for chunk in chunks:
            chunk_text = chunk['text'].lower()
            score = sum(1 for word in question_words if word in chunk_text)
            scored_chunks.append((score, chunk['text']))
        # Sort by score descending
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        # Select top chunks until max_tokens is reached
        selected = []
        total_tokens = 0
        for score, text in scored_chunks:
            chunk_tokens = len(text.split())
            if total_tokens + chunk_tokens > max_tokens:
                break
            selected.append(text)
            total_tokens += chunk_tokens
        return selected

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def answer_question(self, question: str, chunks: List[Dict], model_preference: Optional[str] = None) -> Dict:
        """
        Generate answer using available LLMs with fallback.
        Chunks: list of dicts with 'text' key (from your chunker).
        """
        prompt = self._create_qa_prompt()
        # Select relevant chunks for the question
        selected_chunks = self._select_relevant_chunks(chunks, question, max_tokens=3000)
        context = "\n\n".join(selected_chunks)
        # Try primary LLM first
        if self.primary_llm:
            try:
                chain = LLMChain(llm=self.primary_llm, prompt=prompt)
                answer = chain.run(context=context, question=question)
                cleaned_answer = self._clean_answer(answer)
                return {
                    "answer": cleaned_answer,
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