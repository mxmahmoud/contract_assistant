import streamlit as st
import uuid
import logging
from typing import Any, Dict, List

# Set the page to be wide
st.set_page_config(layout="wide", page_title="Intelligent Contract Assistant")

# Import core modules AFTER setting page config
from ca_core import registry, extraction, chunking, vectorstore, qa, ner
from ca_core.feedback import log_feedback
from ca_core.exceptions import ValidationError
from utility.model_loader import ensure_model_is_loaded, ensure_tei_is_ready
from utility.utility import save_flattened_pdf, get_hash_of_file
from utility.session_state import (
    get_session_state, 
    update_session_state, 
    reset_contract_state, 
    add_message,
    mark_feedback_submitted
)
from utility.caching import cache_data
from utility.config import settings

logger = logging.getLogger(__name__)

logger.info("Starting Contract Assistant application")

# Initialize session state FIRST (before any other operations)
def initialize_session_state():
    """Initialize session state using the centralized state management."""
    state = get_session_state()
    logger.debug("Session state initialized")

# Call it immediately at module level
initialize_session_state()

# Model Loading - Now safe to check session state
# Ensures the required LLM model is downloaded and available if local mode is selected.
ensure_model_is_loaded()
ensure_tei_is_ready()


@cache_data(ttl=settings.DOCUMENT_PROCESSING_CACHE_TTL)
def process_document(uploaded_file):
    """
    Handles the ingestion and processing of an uploaded PDF.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Raises:
        ValidationError: If PDF validation fails
    """
    logger.info(f"Processing document: {uploaded_file.name}")
    
    with st.spinner("Processing document... This may take a moment."):
        file_bytes = uploaded_file.getvalue()
        
        # Validate file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_PDF_SIZE_MB:
            error_msg = f"PDF size {size_mb:.2f}MB exceeds maximum of {settings.MAX_PDF_SIZE_MB}MB"
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        logger.debug(f"PDF size: {size_mb:.2f}MB")
        
        contract_id = get_hash_of_file(file_bytes)
        update_session_state(contract_id=contract_id)
        logger.info(f"Generated contract ID: {contract_id}")

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
        
        doc_pages = extraction.extract_text_from_pdf(
            str(pdf_path), 
            strategy=extraction_strategy,
            max_pages=settings.MAX_PDF_PAGES
        )
        
        logger.info(f"Extracted {len(doc_pages)} pages from PDF")

        # Key Entity Extraction (run on all pages)
        if doc_pages:
            # Extract entities from all pages using NER
            all_entities = ner.extract_entities(doc_pages)
            
            # Also extract key entities from first page using LLM
            first_page_text = doc_pages[0]["text"]
            key_entities_data = qa.extract_key_entities(first_page_text)
            
            # Transform for display (data-driven approach)
            entity_mapping = {
                "parties": "Party",
                "agreement_date": "Agreement Date",
                "jurisdiction": "Jurisdiction",
                "contract_type": "Contract Type",
                "termination_date": "Termination Date",
                "monetary_amounts": "Monetary Amount",
                "key_obligations": "Key Obligation"
            }
            display_entities = []
            for key, label in entity_mapping.items():
                values = key_entities_data.get(key)
                if values and values != "Not Found" and values != ["Extraction Failed"]:
                    # Handle both lists (like parties) and single strings
                    if isinstance(values, list):
                        for value in values:
                            if value != "Extraction Failed":
                                display_entities.append({"label": label, "value": value, "page": 1})
                    else:
                        display_entities.append({"label": label, "value": values, "page": 1})
            
            # Add NER-extracted entities from all pages
            for entity in all_entities:
                display_entities.append({
                    "label": entity["label"],
                    "value": entity["value"],
                    "page": entity["page"]
                })
            
            update_session_state(entities=display_entities)
            logger.info(f"Extracted {len(display_entities)} total entities")
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

        # QA Chain Initialization
        logger.info("Initializing QA chain")
        retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
        qa_chain = qa.get_qa_chain(retriever)
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
        # Rerun to refresh the sidebar entities table immediately
        st.rerun()


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
    
    # Initialize retriever and QA chain
    retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
    qa_chain = qa.get_qa_chain(retriever)
    
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


# UI Layout
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
        if state.last_sidebar_selection != picked_id:
            load_existing_contract(picked_id)
            update_session_state(last_sidebar_selection=picked_id)
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
            # Set last_processed_id first to avoid rerun loops masking chat input
            update_session_state(last_processed_id=new_contract_id)
            process_document(uploaded_file)
    
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