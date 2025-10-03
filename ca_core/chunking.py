# ca_core/chunking.py
"""
Document chunking module for contract processing.

This module handles splitting documents into chunks suitable for embedding.
"""
import hashlib
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utility.config import settings

logger = logging.getLogger(__name__)


def chunk_document(
    doc_contents: List[Dict[str, Any]], 
    contract_id: str, 
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Splits document text into manageable chunks for embedding.

    Args:
        doc_contents: A list of dicts, each containing text and metadata per page.
        contract_id: A unique identifier for the contract document.
        chunk_size: The target size of each chunk in characters. Defaults to settings.
        chunk_overlap: The overlap between chunks in characters. Defaults to settings.

    Returns:
        A list of LangChain Document objects, ready for embedding.
    """
    
    # Resolve chunking parameters if not provided
    if chunk_size is None or chunk_overlap is None:
        chunk_size, chunk_overlap = settings.resolve_chunking_params()
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    all_chunks = []
    for doc in doc_contents:
        page_text = doc["text"]
        page_metadata = doc["metadata"]
        
        chunks = text_splitter.create_documents([page_text])
        
        for i, chunk in enumerate(chunks):
            # Enrich chunk metadata
            chunk.metadata["contract_id"] = contract_id
            chunk.metadata["page_number"] = page_metadata.get("page_number")
            chunk.metadata["section"] = page_metadata.get("section")
            start_index = chunk.metadata.get("start_index", i)
            deterministic_id = hashlib.sha256(f"{contract_id}:{page_metadata.get('page_number')}:{start_index}".encode("utf-8")).hexdigest()
            chunk.metadata["chunk_id"] = deterministic_id
            all_chunks.append(chunk)

    logger.info(f"Split document '{contract_id}' into {len(all_chunks)} chunks.")
    return all_chunks