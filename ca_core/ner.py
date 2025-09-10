# ca_core/ner.py
import re
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_entities(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts key entities from the document text using regex patterns.
    This is a lightweight fallback for basic entity extraction.

    Args:
        pages_data: A list of dicts, each with page text and metadata.

    Returns:
        A list of dictionaries, each representing an extracted entity.
    """
    all_entities = []
    
    # Define regex patterns for common contract entities
    patterns = {
        "Monetary Value": re.compile(r'\$[\d,]+(?:\.\d{2})?\b|\b\d+\s*(?:USD|dollars|Dollars)\b', re.IGNORECASE),
        "Date": re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
        "Email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "Phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
    }

    for page in pages_data:
        text = page["text"]
        page_number = page["metadata"]["page_number"]
        
        # Extract entities using regex patterns
        for label, pattern in patterns.items():
            for match in pattern.finditer(text):
                all_entities.append({
                    "value": match.group(0).strip(),
                    "label": label,
                    "page": page_number
                })

    # --- Deduplication ---
    unique_entities = []
    seen = set()
    for entity in all_entities:
        # Create a unique key for each entity based on value and label
        entity_key = (entity['value'], entity['label'])
        if entity_key not in seen:
            unique_entities.append(entity)
            seen.add(entity_key)

    logger.info(f"Extracted {len(unique_entities)} unique entities using regex patterns.")
    return unique_entities


if __name__ == "__main__":
    from utility.config import BASE_DIR
    from ca_core import extraction

    pdf_path = BASE_DIR / "data" / "examples" / "Annex-1-NDA.pdf"
    pages_data = extraction.extract_text_from_pdf(pdf_path, strategy="pypdf2")
    
    logger.info("=== Regex-based Entity Extraction ===")
    entities = extract_entities(pages_data)
    logger.info(f"Extracted {len(entities)} entities:")
    for entity in entities[:10]:  # Show first 10
        logger.info(f"  {entity['label']}: {entity['value']} (page {entity['page']})")


# =============================================================================
# SPAcY NER IMPLEMENTATION (COMMENTED OUT - FULLY FUNCTIONAL)
# =============================================================================
# This section contains the complete spaCy-based NER implementation that was
# previously used. It's preserved here for demonstration purposes and can be
# easily restored if needed.
#
# To use this implementation:
# 1. Install spaCy: poetry install spacy["cuda12x"]
# 2. Download model: python -m spacy download xx_ent_wiki_sm
# 3. Uncomment the code below and comment out the regex-based implementation
# =============================================================================

# import spacy
# 
# MODEL_NAME = "xx_ent_wiki_sm"
# try:
#     nlp = spacy.load(MODEL_NAME)
# except OSError:
#     print(f"Downloading '{MODEL_NAME}' model...")
#     from spacy.cli import download
#     download(MODEL_NAME)
#     nlp = spacy.load(MODEL_NAME)
# 
# def extract_entities_spacy(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Extracts key entities from the document text using spaCy NER.
#     This is the original spaCy-based implementation.
# 
#     Args:
#         pages_data: A list of dicts, each with page text and metadata.
# 
#     Returns:
#         A list of dictionaries, each representing an extracted entity.
#     """
#     all_entities = []
#     
#     # Define regex for monetary values (spaCy doesn't handle these well)
#     money_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?\b|\b\d+\s*(?:USD|dollars|Dollars)\b', re.IGNORECASE)
# 
#     for page in pages_data:
#         text = page["text"]
#         page_number = page["metadata"]["page_number"]
#         
#         # --- spaCy Entity Extraction ---
#         doc = nlp(text)
#         for ent in doc.ents:
#             if ent.label_ in ["ORG", "PERSON"]:
#                 # Filter generic aliases using POS patterns (language-agnostic)
#                 if not _is_generic_alias(ent):
#                     all_entities.append({
#                         "value": ent.text.strip(),
#                         "label": "Party",
#                         "page": page_number
#                     })
#             elif ent.label_ == "DATE":
#                 all_entities.append({
#                     "value": ent.text.strip(),
#                     "label": "Date",
#                     "page": page_number
#                 })
# 
#         # --- Regex for Monetary Values ---
#         for match in money_pattern.finditer(text):
#             all_entities.append({
#                 "value": match.group(0).strip(),
#                 "label": "Monetary Value",
#                 "page": page_number
#             })
# 
#     # --- Deduplication ---
#     unique_entities = []
#     seen = set()
#     for entity in all_entities:
#         # Create a unique key for each entity based on value and label
#         entity_key = (entity['value'], entity['label'])
#         if entity_key not in seen:
#             unique_entities.append(entity)
#             seen.add(entity_key)
# 
#     logger.info(f"Extracted {len(unique_entities)} unique entities using spaCy NER.")
# 
#     # Print comparison with LLM key entities (first page) for debugging
#     try:
#         from ca_core import qa as _qa
#         llm_keys = {}
#         for page in pages_data:
#             first_page_text = page["text"] if pages_data else ""
#             llm_keys.update(_qa.extract_key_entities(first_page_text))
#         logger.debug("=== LLM key entities (all pages) ===")
#         logger.debug(json.dumps(llm_keys, indent=2, ensure_ascii=False))
#         logger.debug("=== spaCy NER entities (unique) ===")
#         logger.debug(json.dumps(unique_entities, indent=2, ensure_ascii=False))
#     except Exception as e:
#         logger.warning(f"Could not print LLM vs spaCy comparison: {e}")
# 
#     return unique_entities
# 
# 
# def _is_generic_alias(span) -> bool:
#     """
#     Heuristic filter for generic aliases like "the company", "the recipient"
#     using Universal POS tags (language-agnostic).
# 
#     Rules:
#     - Single-token common nouns/pronouns are treated as generic
#     - Spans starting with a determiner (DET) where the next token is not PROPN
#     - Spans with no PROPN and all lowercase text
#     """
#     try:
#         if not span or len(span) == 0:
#             return True
# 
#         # Single-token common nouns/pronouns
#         if len(span) == 1 and span[0].pos_ in ("NOUN", "PRON"):
#             return True
# 
#         # Starts with determiner and next is not a proper noun
#         if span[0].pos_ == "DET" and (len(span) == 1 or span[1].pos_ != "PROPN"):
#             return True
# 
#         # No proper nouns and all lowercase
#         if not any(t.pos_ == "PROPN" for t in span) and span.text.islower():
#             return True
#     except Exception:
#         return False
# 
#     return False
