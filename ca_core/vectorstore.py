# ca_core/vectorstore.py
"""
Vector store management for contract embeddings.

This module provides vector store operations without UI dependencies.
Caching is handled by the UI layer when needed.
"""
import chromadb
import logging
from typing import List, Optional
from functools import lru_cache
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from utility.config import settings
from ca_core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """Initializes and returns a persistent ChromaDB client."""
    logger.info(f"Initializing ChromaDB client at {settings.CHROMA_PERSIST_DIR}")
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


@lru_cache(maxsize=1)
def get_embedding_function() -> OpenAIEmbeddings:
    """Initializes and returns the embedding function based on settings.
    
    Returns:
        OpenAIEmbeddings instance configured with current settings
        
    Raises:
        VectorStoreError: If embedding function initialization fails
    """
    try:
        return OpenAIEmbeddings(
            model=settings.embeddings_model,
            openai_api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_kwargs={"encoding_format": "float"},
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedding function: {e}", exc_info=True)
        raise VectorStoreError(f"Failed to initialize embedding function: {e}") from e

@lru_cache(maxsize=1)
def get_chroma_db() -> Chroma:
    """
    Initializes and returns a Chroma instance, reusing client and embedding function.
    """
    logger.info("Initializing Chroma vector store")
    client = get_chroma_client()
    embedding_function = get_embedding_function()
    
    db = Chroma(
        client=client,
        collection_name="contract_assistant_collection",
        embedding_function=embedding_function,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    logger.info("Chroma vector store initialized successfully")
    return db

def get_vector_store_retriever(contract_id: Optional[str] = None) -> VectorStoreRetriever:
    """
    Initializes and returns a ChromaDB vector store retriever.

    Args:
        contract_id: The ID of the contract to filter by (optional)

    Returns:
        A retriever configured for similarity search and optional filtering
    """
    logger.debug(f"Creating vector store retriever for contract_id={contract_id}")
    db = get_chroma_db()
    
    search_kwargs = {"k": settings.RETRIEVER_K}
    if contract_id:
        # ChromaDB filter syntax: use explicit equality operator
        search_kwargs["filter"] = {"contract_id": {"$eq": contract_id}}
        logger.info(f"Retriever configured with filter: contract_id={contract_id}, k={settings.RETRIEVER_K}")
    else:
        logger.info(f"Retriever configured without filter, k={settings.RETRIEVER_K}")
        
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    return retriever


def add_chunks_to_vector_store(chunks: List[Document]):
    """
    Adds document chunks to the persistent ChromaDB vector store.
    
    Args:
        chunks: A list of LangChain Document objects.
        
    Raises:
        VectorStoreError: If chunks cannot be added to the vector store
    """
    if not chunks:
        logger.warning("No chunks provided to add to vector store")
        return
        
    try:
        db = get_chroma_db()

        # Use a stable document id per chunk to enable upsert and prevent duplicates
        ids = [doc.metadata.get("chunk_id") or doc.page_content[:32] for doc in chunks]

        documents = [d.page_content for d in chunks]
        metadatas = [d.metadata for d in chunks]

        # Upsert ensures re-processing the same PDF won't duplicate entries
        try:
            # Get embedding function from our cache
            embedding_fn = get_embedding_function()
            embeddings = embedding_fn.embed_documents(documents)
            db._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info(f"Upserted {len(chunks)} chunks to ChromaDB using private API with precomputed embeddings")
        except Exception as e:
            logger.warning(f"Private API upsert failed, falling back to add_documents: {e}")
            # Fallback if private API changes; public API computes embeddings via the wrapper
            db.add_documents(documents=chunks, ids=ids)
            logger.info(f"Added {len(chunks)} chunks to ChromaDB using public API")

        logger.info(f"Successfully persisted {len(chunks)} chunks to ChromaDB")
        
    except Exception as e:
        error_msg = f"Failed to add chunks to vector store: {e}"
        logger.error(error_msg, exc_info=True)
        raise VectorStoreError(error_msg) from e


def delete_contract_from_vector_store(contract_id: str) -> int:
    """
    Delete all chunks for a specific contract from the vector store.
    
    Args:
        contract_id: The ID of the contract to delete
        
    Returns:
        Number of chunks deleted
        
    Raises:
        VectorStoreError: If deletion fails
    """
    try:
        db = get_chroma_db()
        
        # Query for all chunks with this contract_id
        results = db._collection.get(
            where={"contract_id": {"$eq": contract_id}}
        )
        
        if not results or not results.get("ids"):
            logger.info(f"No chunks found for contract_id={contract_id}")
            return 0
        
        chunk_ids = results["ids"]
        db._collection.delete(ids=chunk_ids)
        
        logger.info(f"Deleted {len(chunk_ids)} chunks for contract_id={contract_id}")
        return len(chunk_ids)
        
    except Exception as e:
        error_msg = f"Failed to delete chunks for contract {contract_id}: {e}"
        logger.error(error_msg, exc_info=True)
        raise VectorStoreError(error_msg) from e