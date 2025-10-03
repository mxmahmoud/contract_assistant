# ca_core/ner.py
"""
Named Entity Recognition module for contract documents.

This module provides lightweight regex-based entity extraction.
For advanced NER, see ca_core/ner_spacy_archived.py
"""
import re
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

# Note: For advanced spaCy-based NER, see ca_core/ner_spacy_archived.py
