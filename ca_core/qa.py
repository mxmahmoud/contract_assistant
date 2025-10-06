# ca_core/qa.py
"""
Question answering module for contract analysis.

This module provides QA chains and entity extraction capabilities.
"""
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Any
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever

from utility.config import settings

logger = logging.getLogger(__name__)


# Entity-based question routing (DISABLED BY DEFAULT)
# Kept for backward compatibility - can be re-enabled via ENABLE_ENTITY_ROUTING=true
ENTITY_QUESTION_RULES = [
    {
        "keywords": ["who", "party", "parties", "involved"],
        "labels": ["Party", "ORG", "PERSON"],
        "answer_template": "Based on the extracted entities, the involved parties are:\n\n* {}",
        "not_found_message": "I could not identify the parties from the extracted entities."
    },
    {
        "keywords": ["date", "when", "what date"],
        "labels": ["Date", "Agreement Date"],
        "answer_template": "Based on the extracted entities, the following dates were found:\n\n* {}",
        "not_found_message": "I could not identify any specific dates from the extracted entities."
    }
]

def answer_from_entities(prompt: str, entities: List[Dict[str, Any]]) -> Optional[str]:
    """
    Check if question can be answered from extracted entities.
    
    DISABLED BY DEFAULT: Returns None to force full RAG pipeline.
    Entities are displayed in sidebar, so full RAG provides better answers.
    
    To re-enable: Set ENABLE_ENTITY_ROUTING=true in .env
    """
    if not settings.ENABLE_ENTITY_ROUTING:
        return None  # Use full RAG for all questions
    
    # Entity routing enabled - check for matching patterns
    prompt_lower = prompt.lower()

    for rule in ENTITY_QUESTION_RULES:
        if any(keyword in prompt_lower for keyword in rule["keywords"]):
            found_values = [
                f'{e["value"]} (found on page {e["page"]})' if e["label"] in ["Date"] else e["value"]
                for e in entities if e["label"] in rule["labels"]
            ]
            
            if found_values:
                formatted_values = "\n* ".join(found_values)
                return rule["answer_template"].format(formatted_values)
            else:
                return rule["not_found_message"]

    return None


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Initializes and returns a cached LLM client instance.
    This ensures the client is created only once per session.
    """
    logger.info(f"Initializing LLM client for model={settings.llm_model}")
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


QA_SYSTEM_PROMPT = """You are a meticulous AI contract assistant. Your task is to answer questions based ONLY on the provided contract excerpts. Do not use any external knowledge.

Your response must be in three parts:
1.  **Direct Answer:** Provide a clear and concise answer to the user's question.
2.  **Supporting Excerpts:** Quote 1-3 relevant excerpts from the contract that justify your answer. Each quote must be clearly marked and include its page number, like this: "'...relevant text from contract...'".
3.  **Ambiguities:** If the contract does not contain the information or is unclear, state that explicitly. Do not invent answers.

Context:
{context}

Question:
{question}
"""

def get_qa_chain(retriever: VectorStoreRetriever) -> RetrievalQA:
    """
    Initializes and returns the main RetrievalQA chain.

    Args:
        retriever: The configured vector store retriever instance

    Returns:
        A RetrievalQA chain ready for querying
    """
    logger.info(f"Initializing QA chain with model={settings.llm_model}")
    
    llm = get_llm()

    qa_prompt = PromptTemplate(
        template=QA_SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": qa_prompt}
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
        verbose=True,  # Enable verbose logging to see retrieved documents
    )
    
    logger.info("QA chain initialized successfully")
    return qa_chain