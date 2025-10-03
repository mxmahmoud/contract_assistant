# ca_core/qa.py
"""
Question answering module for contract analysis.

This module provides QA chains and entity extraction capabilities.
"""
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import PydanticOutputParser

from utility.config import settings

logger = logging.getLogger(__name__)


# Data-driven router for entity-based questions
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
    Answers questions that can be directly addressed by extracted entities
    using a data-driven ruleset.
    """
    prompt_lower = prompt.lower()

    for rule in ENTITY_QUESTION_RULES:
        # Check if any keyword is in the prompt
        if any(keyword in prompt_lower for keyword in rule["keywords"]):
            
            # Find all matching entities based on labels
            found_values = [
                f'{e["value"]} (found on page {e["page"]})' if e["label"] in ["Date"] else e["value"]
                for e in entities if e["label"] in rule["labels"]
            ]
            
            if found_values:
                # Format the answer using the rule's template
                formatted_values = "\n* ".join(found_values)
                return rule["answer_template"].format(formatted_values)
            else:
                # Return the specific not-found message for that rule
                return rule["not_found_message"]

    return None


QA_SYSTEM_PROMPT = """You are a meticulous AI contract assistant. Your task is to answer questions based ONLY on the provided contract excerpts. Do not use any external knowledge.

Your response must be in three parts:
1.  **Direct Answer:** Provide a clear and concise answer to the user's question.
2.  **Supporting Excerpts:** Quote 1-3 relevant excerpts from the contract that justify your answer. Each quote must be clearly marked and include its page number, like this: "[Page X]: '...relevant text from contract...'".
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
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

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
    )
    
    logger.info("QA chain initialized successfully")
    return qa_chain


class KeyEntities(BaseModel):
    """Represents the key entities extracted from a contract."""
    parties: List[str] = Field(description="The names of all companies, organizations, or individuals party to the agreement.")
    agreement_date: str = Field(description="The effective date of the agreement. If not found, return 'Not Found'.")
    jurisdiction: str = Field(description="The governing law or jurisdiction for the agreement. If not found, return 'Not Found'.")
    contract_type: str = Field(description="The type of contract (e.g., NDA, Service Agreement, Employment Contract). If not found, return 'Not Found'.")
    termination_date: str = Field(description="The termination or expiration date of the agreement. If not found, return 'Not Found'.")
    monetary_amounts: List[str] = Field(description="Any monetary amounts, fees, or financial terms mentioned in the contract.")
    key_obligations: List[str] = Field(description="Key obligations, duties, or responsibilities mentioned in the contract.")


KEY_ENTITIES_EXTRACTION_PROMPT = """
You are an expert AI trained to analyze legal contracts. Your task is to extract comprehensive key entities from the provided text of a contract.

From the 'Contract Page Text' below, extract the following information:
1. Parties: All companies, organizations, or individuals party to the agreement
2. Agreement Date: The effective date of the agreement
3. Jurisdiction: The governing law or jurisdiction for the agreement
4. Contract Type: The type of contract (e.g., NDA, Service Agreement, Employment Contract)
5. Termination Date: The termination or expiration date of the agreement
6. Monetary Amounts: Any monetary amounts, fees, or financial terms mentioned
7. Key Obligations: Key obligations, duties, or responsibilities mentioned

You must respond with ONLY a valid JSON object. Do not add any other text before or after the JSON object.

{format_instructions}

Contract Page Text:
-------------------
{contract_text}
-------------------
"""


def extract_key_entities(text: str) -> dict:
    """
    Uses the LLM to extract key entities from contract text.

    Args:
        text: The text content to extract entities from

    Returns:
        A dictionary containing the extracted entities
    """
    logger.info("Extracting key entities using LLM")
    logger.debug(f"Text length: {len(text)} characters")
    
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    
    parser = PydanticOutputParser(pydantic_object=KeyEntities)

    prompt = PromptTemplate(
        template=KEY_ENTITIES_EXTRACTION_PROMPT,
        input_variables=["contract_text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    
    try:
        response = chain.invoke({"contract_text": text})
        result = response.dict()
        logger.info(f"Successfully extracted entities: {len(result.get('parties', []))} parties, "
                   f"type: {result.get('contract_type', 'unknown')}")
        return result
    except Exception as e:
        logger.error(f"Failed to extract entities from LLM response: {e}", exc_info=True)
        # Fallback to a default error structure
        return {
            "parties": ["Extraction Failed"],
            "agreement_date": "Extraction Failed",
            "jurisdiction": "Extraction Failed",
            "contract_type": "Extraction Failed",
            "termination_date": "Extraction Failed",
            "monetary_amounts": ["Extraction Failed"],
            "key_obligations": ["Extraction Failed"],
        }