# ca_core/qa.py
import logging
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStoreRetriever

from utility.config import settings
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
import streamlit as st

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

@st.cache_resource
def get_qa_chain(_retriever: VectorStoreRetriever, cache_key: str) -> RetrievalQA:
    """
    Initializes and returns the main RetrievalQA chain.
    Caches the chain per contract to avoid re-initialization.

    Args:
        _retriever: The configured vector store retriever instance. 
        cache_key: Deterministic key (e.g., contract_id) to scope the cache per contract.

    Returns:
        A RetrievalQA chain ready for querying.
    """
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
        retriever=_retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    
    logger.info("QA chain initialized.")
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


@st.cache_data
def extract_key_entities(text: str) -> dict:
    """
    Uses the LLM to extract key entities (parties, date, jurisdiction) from the first page of a contract.

    Args:
        text: The text content of the first page of the contract.

    Returns:
        A dictionary containing the extracted entities.
    """
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
        return response.dict()
    except Exception as e:
        logger.error(f"Could not parse entities from LLM response: {e}")
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