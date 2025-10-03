# ca_core/entities.py
"""
This module handles the extraction of key entities from contract text using an LLM.
It defines the data structure for key entities and provides the extraction function.
"""
import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from ca_core.qa import get_llm

logger = logging.getLogger(__name__)


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
    
    llm = get_llm()
    
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
