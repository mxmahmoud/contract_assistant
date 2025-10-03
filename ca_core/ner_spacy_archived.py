# ca_core/ner_spacy_archived.py
"""
ARCHIVED: spaCy NER Implementation

This is the original spaCy-based NER implementation that was replaced with
a lightweight regex-based approach to reduce dependencies.

To use this implementation:
1. Add to pyproject.toml: spacy = "^3.8.0"
2. Install: poetry install
3. Download model: python -m spacy download xx_ent_wiki_sm
4. Import and use: from ca_core.ner_spacy_archived import extract_entities_spacy
"""

import re
import json
import logging
from typing import List, Dict, Any
import spacy

logger = logging.getLogger(__name__)

MODEL_NAME = "xx_ent_wiki_sm"

try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    print(f"Downloading '{MODEL_NAME}' model...")
    from spacy.cli import download
    download(MODEL_NAME)
    nlp = spacy.load(MODEL_NAME)


def extract_entities_spacy(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts key entities from the document text using spaCy NER.
    This is the original spaCy-based implementation.

    Args:
        pages_data: A list of dicts, each with page text and metadata.

    Returns:
        A list of dictionaries, each representing an extracted entity.
    """
    all_entities = []
    
    # Define regex for monetary values (spaCy doesn't handle these well)
    money_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?\b|\b\d+\s*(?:USD|dollars|Dollars)\b', re.IGNORECASE)

    for page in pages_data:
        text = page["text"]
        page_number = page["metadata"]["page_number"]
        
        # --- spaCy Entity Extraction ---
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"]:
                # Filter generic aliases using POS patterns (language-agnostic)
                if not _is_generic_alias(ent):
                    all_entities.append({
                        "value": ent.text.strip(),
                        "label": "Party",
                        "page": page_number
                    })
            elif ent.label_ == "DATE":
                all_entities.append({
                    "value": ent.text.strip(),
                    "label": "Date",
                    "page": page_number
                })

        # --- Regex for Monetary Values ---
        for match in money_pattern.finditer(text):
            all_entities.append({
                "value": match.group(0).strip(),
                "label": "Monetary Value",
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

    logger.info(f"Extracted {len(unique_entities)} unique entities using spaCy NER.")

    # Print comparison with LLM key entities (first page) for debugging
    try:
        from ca_core import qa as _qa
        llm_keys = {}
        for page in pages_data:
            first_page_text = page["text"] if pages_data else ""
            llm_keys.update(_qa.extract_key_entities(first_page_text))
        logger.debug("=== LLM key entities (all pages) ===")
        logger.debug(json.dumps(llm_keys, indent=2, ensure_ascii=False))
        logger.debug("=== spaCy NER entities (unique) ===")
        logger.debug(json.dumps(unique_entities, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"Could not print LLM vs spaCy comparison: {e}")

    return unique_entities


def _is_generic_alias(span) -> bool:
    """
    Heuristic filter for generic aliases like "the company", "the recipient"
    using Universal POS tags (language-agnostic).

    Rules:
    - Single-token common nouns/pronouns are treated as generic
    - Spans starting with a determiner (DET) where the next token is not PROPN
    - Spans with no PROPN and all lowercase text
    """
    try:
        if not span or len(span) == 0:
            return True

        # Single-token common nouns/pronouns
        if len(span) == 1 and span[0].pos_ in ("NOUN", "PRON"):
            return True

        # Starts with determiner and next is not a proper noun
        if span[0].pos_ == "DET" and (len(span) == 1 or span[1].pos_ != "PROPN"):
            return True

        # No proper nouns and all lowercase
        if not any(t.pos_ == "PROPN" for t in span) and span.text.islower():
            return True
    except Exception:
        return False

    return False

