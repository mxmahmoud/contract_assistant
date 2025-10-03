# ca_core/ner.py
"""
Unified Named Entity Recognition for contract documents.

Strategy is selected via settings.NER_STRATEGY: 'llm' (default) or 'spacy'.
In LLM mode, we extract key entities from every page and aggregate results.
In spaCy mode, we use the archived spaCy pipeline (with monetary regex) across pages.
"""
import logging
from typing import List, Dict, Any

from utility.config import settings, NERStrategy
from utility.caching import cache_data

logger = logging.getLogger(__name__)


# Mapping from LLM key-entities schema to display labels
LLM_ENTITY_LABELS: Dict[str, str] = {
    "parties": "Party",
    "agreement_date": "Agreement Date",
    "jurisdiction": "Jurisdiction",
    "contract_type": "Contract Type",
    "termination_date": "Termination Date",
    "monetary_amounts": "Monetary Amount",
    "key_obligations": "Key Obligation",
}


@cache_data(ttl=settings.DOCUMENT_PROCESSING_CACHE_TTL)
def _extract_key_entities_cached(page_text: str) -> Dict[str, Any]:
    """Cached LLM-based key entity extraction for a single page of text."""
    from ca_core import qa  # Local import to avoid heavy import at module import time
    return qa.extract_key_entities(page_text)


def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for e in entities:
        key = (e.get("label"), e.get("value"), e.get("page"))
        if key not in seen:
            unique.append(e)
            seen.add(key)
    return unique


def extract_entities(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract entities across all pages using the configured strategy.

    Args:
        pages_data: List of dicts, each containing 'text' and 'metadata'.

    Returns:
        List of entity dicts with keys: 'label', 'value', 'page'.
    """
    if not pages_data:
        logger.warning("No pages provided to NER extractor")
        return []

    try:
        if settings.NER_STRATEGY == NERStrategy.SPACY:
            from ca_core.ner_spacy import extract_entities_spacy
            entities = extract_entities_spacy(pages_data)
            return _dedupe_entities(entities)

        # Default: LLM per page, aggregate
        aggregated: List[Dict[str, Any]] = []
        for page in pages_data:
            page_text = page.get("text", "")
            page_number = page.get("metadata", {}).get("page_number")
            if not page_text:
                continue
            key_entities = _extract_key_entities_cached(page_text)
            for key, label in LLM_ENTITY_LABELS.items():
                values = key_entities.get(key)
                if not values or values == "Not Found" or values == ["Extraction Failed"]:
                    continue
                if isinstance(values, list):
                    for value in values:
                        if value != "Extraction Failed":
                            aggregated.append({
                                "label": label,
                                "value": value,
                                "page": page_number,
                            })
                else:
                    aggregated.append({
                        "label": label,
                        "value": values,
                        "page": page_number,
                    })

        unique = _dedupe_entities(aggregated)
        logger.info(
            "Extracted %d entities via %s",
            len(unique),
            settings.NER_STRATEGY.value,
        )
        return unique

    except Exception as e:
        logger.error("Entity extraction failed: %s", e, exc_info=True)
        return []


if __name__ == "__main__":
    from utility.config import BASE_DIR
    from ca_core import extraction

    pdf_path = BASE_DIR / "data" / "examples" / "Annex-1-NDA.pdf"
    pages_data = extraction.extract_text_from_pdf(str(pdf_path), strategy="pypdf2")
    entities = extract_entities(pages_data)
    logger.info("Extracted %d entities (sample): %s", len(entities), entities[:5])
