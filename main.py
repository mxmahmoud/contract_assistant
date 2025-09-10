import streamlit as st
import tempfile
import hashlib
import uuid
from typing import Any, Dict, List, Optional

# Set the page to be wide
st.set_page_config(layout="wide", page_title="Intelligent Contract Assistant")

# Import core modules AFTER setting page config
from ca_core import registry, extraction, chunking, vectorstore, qa, ner
from ca_core.feedback import log_feedback
from utility.model_loader import ensure_model_is_loaded, ensure_tei_is_ready
from utility.utility import save_flattened_pdf, get_hash_of_file

# Model Loading
# Ensures the required LLM model is downloaded and available if local mode is selected.

ensure_model_is_loaded()
ensure_tei_is_ready()




def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if "contract_id" not in st.session_state:
        st.session_state.contract_id = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "entities" not in st.session_state:
        st.session_state.entities = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_processed_id" not in st.session_state:
        st.session_state.last_processed_id = None
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = set()
    if "use_ocr" not in st.session_state:
        st.session_state.use_ocr = False

@st.cache_data(ttl=3600)
def process_document(uploaded_file):
    """Handles the ingestion and processing of an uploaded PDF."""
    with st.spinner("Processing document... This may take a moment."):
        file_bytes = uploaded_file.getvalue()
        contract_id = get_hash_of_file(file_bytes)
        st.session_state.contract_id = contract_id

        # Save the uploaded file to its permanent location first
        contract_dir = registry._contract_dir(contract_id)
        registry._ensure_dir(contract_dir)
        pdf_path = registry._pdf_path(contract_id, uploaded_file.name)
        save_flattened_pdf(file_bytes, pdf_path)


        # Core Processing Pipeline
        # Ingestion
        # Use OCR if checkbox is checked, otherwise use pypdf
        if st.session_state.use_ocr:
            doc_pages = extraction.extract_text_from_pdf(str(pdf_path), strategy="paddleocr")
        else:
            doc_pages = extraction.extract_text_from_pdf(str(pdf_path), strategy="pypdf2")

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
            
            st.session_state.entities = display_entities
        else:
            st.session_state.entities = []

        # Chunking
        chunks = chunking.chunk_document(doc_pages, contract_id)

        # Vector Store
        vectorstore.add_chunks_to_vector_store(chunks)

        # QA Chain Initialization (cache scoped by contract)
        retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
        st.session_state.qa_chain = qa.get_qa_chain(retriever, cache_key=contract_id)

        # Persist the uploaded PDF and metadata/entities for later reuse
        registry.save_contract(
            contract_id=contract_id,
            original_filename=uploaded_file.name,
            uploaded_bytes=file_bytes,
            num_pages=len(doc_pages),
            entities=st.session_state.entities,
        )

        # Clear previous chat messages
        st.session_state.messages = []
        st.success("Document processed successfully! You can now ask questions.")
        # Rerun to refresh the sidebar entities table immediately
        st.rerun()


def load_existing_contract(contract_id: str):
    """Load retriever and entities for an already ingested contract."""
    st.session_state.contract_id = contract_id
    st.session_state.entities = registry.load_contract_entities(contract_id)
    retriever = vectorstore.get_vector_store_retriever(contract_id=contract_id)
    st.session_state.qa_chain = qa.get_qa_chain(retriever, cache_key=contract_id)
    st.session_state.messages = []


def render_chat_history():
    """Renders the chat history and feedback buttons in reverse chronological order."""
    for i in range(len(st.session_state.messages) - 1, 0, -2):
        # We assume messages are always in user, assistant order.
        assistant_msg = st.session_state.messages[i]
        user_msg = st.session_state.messages[i-1]

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
initialize_session_state()

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
        if st.session_state.get("_last_sidebar_selection") != picked_id:
            load_existing_contract(picked_id)
            st.session_state["_last_sidebar_selection"] = picked_id
    else:
        st.info("No ingested contracts yet.")

    st.divider()
    st.header("Key Entities")
    if st.session_state.entities:
        st.dataframe(st.session_state.entities, width='stretch')
    else:
        st.caption("Entities will appear after processing or loading a contract.")


# Main column for file upload and chat
main_col, _ = st.columns([2, 1])

with main_col:
    st.header("Upload Your Contract")
    
    # OCR checkbox
    st.session_state.use_ocr = st.checkbox(
        "Use OCR", 
        value=st.session_state.use_ocr,
        help="Check this box to use OCR (PaddleOCR) for text extraction. Uncheck to use PyPDF for faster processing of text-based PDFs."
    )
    
    uploaded_file = st.file_uploader(
        "Upload a PDF contract to begin analysis.", type="pdf"
    )
    
    if uploaded_file is not None:
        new_contract_id = get_hash_of_file(uploaded_file.getvalue())
        if st.session_state.last_processed_id != new_contract_id:
            # Set last_processed_id first to avoid rerun loops masking chat input
            st.session_state.last_processed_id = new_contract_id
            process_document(uploaded_file)
    
    if st.session_state.qa_chain:
        st.header("Ask a Question")

        # Chat input at the top
        if prompt := st.chat_input("What would you like to know about the contract?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response_content = ""
                source_documents_for_log = ""
                source_documents = []

                # Route question to entity-based answer if possible
                entity_answer = qa.answer_from_entities(prompt, st.session_state.entities)
                if entity_answer:
                    response_content = entity_answer
                else:
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    response_content = response["result"]
                    
                    # Prepare source documents for logging and display
                    source_documents = response.get("source_documents", [])
                    source_documents_for_log = "\n---\n".join([doc.page_content for doc in source_documents])
                
                # Append the full assistant message with context for feedback
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "id": str(uuid.uuid4()),
                    "question": prompt,
                    "sources": source_documents_for_log,
                    "source_documents": source_documents # Pass documents for rendering
                }
                st.session_state.messages.append(assistant_message)
            
            # Rerun to display the new message and its feedback buttons immediately
            st.rerun()

        # Display chat history
        render_chat_history()
        
    else:
        st.info("Please upload or load a contract to enable the chat.")