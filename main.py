import streamlit as st
import uuid
import logging
from typing import Any, Dict, List
import time

# Set the page to be wide
st.set_page_config(layout="wide", page_title="Intelligent Contract Assistant")

# Import core modules AFTER setting page config
from ca_core import registry, extraction, chunking, vectorstore, qa, ner
from ca_core.feedback import log_feedback
from ca_core.exceptions import ValidationError
from utility.model_loader import check_ollama_status, check_tei_status
from utility.utility import save_flattened_pdf, get_hash_of_file
from utility.session_state import (
    get_session_state, 
    update_session_state, 
    reset_contract_state, 
    add_message
)
from utility.config import settings
from utility.caching import cache_data

logger = logging.getLogger(__name__)
logger.info("Starting Contract Assistant application")

 

# Initialize session state FIRST (before any other operations)
def initialize_session_state():
    """Initialize session state using the centralized state management."""
    state = get_session_state()
    logger.debug("Session state initialized")

# Call it immediately at module level
initialize_session_state()

@cache_data(ttl=settings.DOCUMENT_PROCESSING_CACHE_TTL)
def extract_pages_cached(pdf_path: str, strategy: str, max_pages: int):
    """Cached wrapper for pure PDF text extraction."""
    return extraction.extract_text_from_pdf(
        pdf_path,
        strategy=strategy,
        max_pages=max_pages,
    )


@cache_data(ttl=settings.DOCUMENT_PROCESSING_CACHE_TTL)
def extract_key_entities_cached(contract_text: str) -> Dict[str, Any]:
    """Cached wrapper for LLM key-entity extraction on first page text."""
    return qa.extract_key_entities(contract_text)

def process_document(uploaded_file):
    """
    Handles the ingestion and processing of an uploaded PDF.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Raises:
        ValidationError: If PDF validation fails
    """
    logger.info(f"Processing document: {uploaded_file.name}")
    
    file_bytes = uploaded_file.getvalue()
    
    # Validate file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_PDF_SIZE_MB:
        error_msg = f"PDF size {size_mb:.2f}MB exceeds maximum of {settings.MAX_PDF_SIZE_MB}MB"
        logger.error(error_msg)
        st.error(error_msg)
        return
    
    logger.debug(f"PDF size: {size_mb:.2f}MB")
    
    contract_id = get_hash_of_file(file_bytes)
    logger.info(f"Generated contract ID: {contract_id}")
    
    # Check if contract already exists
    try:
        if registry.contract_exists(contract_id):
            logger.info(f"Contract {contract_id} already exists, loading from cache")
            with st.spinner("Loading document from cache..."):
                load_existing_contract(contract_id)
                
                # Get contract metadata for display
                meta = registry.load_contract_meta(contract_id)
                if meta:
                    st.success(f"✅ Document '{meta.get('original_filename', 'Unknown')}' loaded successfully!")
                else:
                    st.success("✅ Document loaded successfully!")
            return
    except Exception as e:
        logger.error(f"Error during duplicate check: {e}", exc_info=True)
        st.warning(f"Could not check for duplicates, processing as new document...")
    
    # New document - proceed with full processing
    with st.spinner("Processing document... This may take a moment."):
        
        update_session_state(contract_id=contract_id)
        logger.info(f"New contract detected, proceeding with processing")

        # Save the uploaded file to its permanent location first
        contract_dir = registry._contract_dir(contract_id)
        registry._ensure_dir(contract_dir)
        pdf_path = registry._pdf_path(contract_id, uploaded_file.name)
        save_flattened_pdf(file_bytes, pdf_path)
        logger.debug(f"Saved PDF to {pdf_path}")


        # Core Processing Pipeline
        # Ingestion
        # Use OCR if checkbox is checked, otherwise use pypdf
        state = get_session_state()
        extraction_strategy = "paddleocr" if state.use_ocr else "pypdf2"
        logger.info(f"Extracting text using strategy: {extraction_strategy}")
        
        doc_pages = extract_pages_cached(
            str(pdf_path),
            extraction_strategy,
            settings.MAX_PDF_PAGES,
        )
        
        logger.info(f"Extracted {len(doc_pages)} pages from PDF")

        # Unified Entity Extraction delegated to ner module
        if doc_pages:
            extracted_entities = ner.extract_entities(doc_pages)
            update_session_state(entities=extracted_entities)
            logger.info(f"Extracted {len(extracted_entities)} entities via {settings.NER_STRATEGY.value}")
        else:
            update_session_state(entities=[])
            logger.warning("No pages extracted from PDF")

        # Chunking
        logger.info("Starting document chunking")
        chunks = chunking.chunk_document(doc_pages, contract_id)
        logger.info(f"Created {len(chunks)} chunks")

        # Vector Store
        logger.info("Adding chunks to vector store")
        vectorstore.add_chunks_to_vector_store(chunks)

        # QA Chain Initialization (cache per contract)
        logger.info("Initializing QA chain")
        retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
        qa_chain = qa.get_qa_chain(retriever)
        state = get_session_state()
        state.qa_chains_by_contract[contract_id] = qa_chain
        update_session_state(qa_chain=qa_chain)

        # Persist the uploaded PDF and metadata/entities for later reuse
        state = get_session_state()
        logger.info("Saving contract metadata and entities")
        registry.save_contract(
            contract_id=contract_id,
            original_filename=uploaded_file.name,
            uploaded_bytes=file_bytes,
            num_pages=len(doc_pages),
            entities=state.entities,
        )

        # Clear previous chat messages
        reset_contract_state()
        st.success("Document processed successfully! You can now ask questions.")
        logger.info(f"Document processing complete for contract {contract_id}")

def load_existing_contract(contract_id: str):
    """
    Load retriever and entities for an already ingested contract.
    
    Args:
        contract_id: The contract ID to load
    """
    logger.info(f"Loading existing contract: {contract_id}")
    
    # Load entities from registry
    entities = registry.load_contract_entities(contract_id)
    logger.debug(f"Loaded {len(entities)} entities")
    
    # Initialize or reuse QA chain from cache
    state = get_session_state()
    qa_chain = state.qa_chains_by_contract.get(contract_id)
    if qa_chain is None:
        logger.info("No cached QA chain for contract, creating new one")
        retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
        qa_chain = qa.get_qa_chain(retriever)
        state.qa_chains_by_contract[contract_id] = qa_chain
    else:
        logger.info("Reusing cached QA chain for contract")
    
    # Update session state
    update_session_state(
        contract_id=contract_id,
        entities=entities,
        qa_chain=qa_chain
    )
    
    # Reset chat history for new contract
    reset_contract_state()
    logger.info(f"Successfully loaded contract {contract_id}")


def render_chat_history():
    """Renders the chat history and feedback buttons in reverse chronological order."""
    state = get_session_state()
    
    for i in range(len(state.messages) - 1, 0, -2):
        # We assume messages are always in user, assistant order.
        assistant_msg = state.messages[i]
        user_msg = state.messages[i-1]

        # Display user's message
        with st.chat_message(user_msg["role"]):
            st.markdown(user_msg["content"])

        # Display assistant's message with feedback buttons
        with st.chat_message(assistant_msg["role"]):
            st.markdown(assistant_msg["content"])

            # Display sources if available
            source_documents = assistant_msg.get("source_documents", [])
            if source_documents:
                with st.expander("View Sources"):
                    for doc in source_documents:
                        st.info(f"Page {doc.metadata.get('page_number', 'N/A')}:")
                        st.text(doc.page_content)

            if "id" in assistant_msg:
                feedback_key_base = assistant_msg["id"]
                disable_buttons = assistant_msg.get("feedback_submitted", False)

                col1, col2, _ = st.columns([1, 1, 10])
                with col1:
                    st.button("👍", key=f"like_{feedback_key_base}", on_click=log_feedback, args=(
                        assistant_msg["id"], "positive", assistant_msg["question"], assistant_msg["content"], assistant_msg.get("sources", "")
                    ), disabled=disable_buttons)
                with col2:
                    st.button("👎", key=f"dislike_{feedback_key_base}", on_click=log_feedback, args=(
                        assistant_msg["id"], "negative", assistant_msg["question"], assistant_msg["content"], assistant_msg.get("sources", "")
                    ), disabled=disable_buttons)


state = get_session_state()
if not state.app_initialized:
    logger.info("First run for this session, performing initial setup (non-blocking)...")

    st.info("🔄 Initializing application, please wait...")
    st.write(
        "Connecting to backend services. This may take a moment on first startup as models are downloaded."
    )

    ollama_ready = check_ollama_status()
    tei_ready = check_tei_status()

    ollama_status = "✅ Ready" if ollama_ready else "⏳ Downloading/Loading..."
    tei_status = "✅ Ready" if tei_ready else "⏳ Starting..."

    st.status(
        f"\n- **LLM Service (Ollama):** {ollama_status}\n- **Embedding Service (TEI):** {tei_status}\n",
        state="complete" if (ollama_ready and tei_ready) else "running",
        expanded=True,
    )

    if ollama_ready and tei_ready:
        update_session_state(app_initialized=True)
        logger.info("All services ready. Proceeding to app UI.")
        st.rerun()
    else:
        time.sleep(5)
        st.rerun()

st.title("📄 Contract Assistant")

# Sidebar: list existing contracts + entity display
with st.sidebar:
    st.header("Contracts")
    contracts = registry.list_contracts()
    if contracts:
        options = {f"{c['original_filename']} ({c['contract_id'][:8]})": c["contract_id"] for c in contracts}
        selected = st.selectbox("Select a contract", options=list(options.keys()), key="contract_select")
        picked_id = options[selected]

        # Auto-load when selection changes
        state = get_session_state()
        if state.contract_id != picked_id:
            load_existing_contract(picked_id)
            update_session_state(last_sidebar_selection=picked_id)
            st.rerun()
    else:
        st.info("No ingested contracts yet.")

    st.divider()
    st.header("Key Entities")
    state = get_session_state()
    if state.entities:
        st.dataframe(state.entities, width='stretch')
    else:
        st.caption("Entities will appear after processing or loading a contract.")


# Main column for file upload and chat
main_col, _ = st.columns([2, 1])

with main_col:
    st.header("Upload Your Contract")
    
    # OCR checkbox
    state = get_session_state()
    new_use_ocr = st.checkbox(
        "Use OCR", 
        value=state.use_ocr,
        help="Check this box to use OCR (PaddleOCR) for text extraction. Uncheck to use PyPDF for faster processing of text-based PDFs."
    )
    if new_use_ocr != state.use_ocr:
        update_session_state(use_ocr=new_use_ocr)
    
    uploaded_file = st.file_uploader(
        "Upload a PDF contract to begin analysis.", type="pdf"
    )
    
    if uploaded_file is not None:
        state = get_session_state()
        new_contract_id = get_hash_of_file(uploaded_file.getvalue())
        if state.last_processed_id != new_contract_id:
            update_session_state(last_processed_id=new_contract_id)
            try:
                process_document(uploaded_file)
            except Exception as e:
                logger.error("Unhandled error during document processing: %s", e, exc_info=True)
                st.error(
                    "An unexpected error occurred while processing the document. "
                    "Check the application logs for details."
                )
                reset_contract_state()
                update_session_state(
                    last_processed_id=None,
                    contract_id=None,
                    qa_chain=None,
                    entities=[],
                )
            st.rerun()
    
    state = get_session_state()
    if state.qa_chain:
        st.header("Ask a Question")

        # Chat input at the top
        if prompt := st.chat_input("What would you like to know about the contract?"):
            state = get_session_state()
            add_message("user", prompt)
            
            with st.spinner("Thinking..."):
                response_content = ""
                source_documents_for_log = ""
                source_documents = []

                # Route question to entity-based answer if possible
                entity_answer = qa.answer_from_entities(prompt, state.entities)
                if entity_answer:
                    response_content = entity_answer
                else:
                    response = state.qa_chain.invoke({"query": prompt})
                    response_content = response["result"]
                    
                    # Prepare source documents for logging and display
                    source_documents = response.get("source_documents", [])
                    source_documents_for_log = "\n---\n".join([doc.page_content for doc in source_documents])
                
                # Append the full assistant message with context for feedback
                add_message(
                    "assistant",
                    response_content,
                    id=str(uuid.uuid4()),
                    question=prompt,
                    sources=source_documents_for_log,
                    source_documents=source_documents
                )
            
            # Rerun to display the new message and its feedback buttons immediately
            st.rerun()

        # Display chat history
        render_chat_history()
        
    else:
        st.info("Please upload or load a contract to enable the chat.")