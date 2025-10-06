# utility/session_state.py
"""
Session state management for Streamlit application.

This module provides type-safe session state management using dataclasses.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import streamlit as st
from langchain.chains import RetrievalQA


@dataclass
class SessionState:
    """Type-safe session state for the Streamlit application."""
    
    contract_id: Optional[str] = None
    qa_chain: Optional[RetrievalQA] = None
    qa_chains_by_contract: Dict[str, RetrievalQA] = field(default_factory=dict)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    last_processed_id: Optional[str] = None
    feedback_submitted: set = field(default_factory=set)
    use_ocr: bool = False
    reprocess_existing: bool = False  # Force reprocessing of existing contracts
    
    # Service readiness flags
    ollama_model_ready: bool = False
    tei_ready: bool = False
    
    # Application lifecycle
    app_initialized: bool = False
    
    # Last selected contract from sidebar
    last_sidebar_selection: Optional[str] = None


def get_session_state() -> SessionState:
    """
    Get the current session state, initializing if necessary.
    
    Returns:
        SessionState object with current values
    """
    if "_app_state" not in st.session_state:
        st.session_state._app_state = SessionState()
    
    return st.session_state._app_state


def update_session_state(**kwargs) -> None:
    """
    Update session state fields.
    
    Args:
        **kwargs: Fields to update on the SessionState object
    """
    state = get_session_state()
    for key, value in kwargs.items():
        if hasattr(state, key):
            setattr(state, key, value)
        else:
            raise AttributeError(f"SessionState has no attribute '{key}'")


def reset_contract_state() -> None:
    """Reset contract-specific state when loading a new contract."""
    update_session_state(
        messages=[],
        feedback_submitted=set()
    )


def add_message(role: str, content: str, **metadata) -> None:
    """
    Add a message to the chat history.
    
    Args:
        role: Message role (user or assistant)
        content: Message content
        **metadata: Additional metadata (e.g., id, question, sources)
    """
    state = get_session_state()
    message = {
        "role": role,
        "content": content,
        **metadata
    }
    state.messages.append(message)


def mark_feedback_submitted(message_id: str) -> None:
    """Mark feedback as submitted for a message."""
    state = get_session_state()
    for msg in state.messages:
        if msg.get("id") == message_id:
            msg["feedback_submitted"] = True
            break


def clear_session_state() -> None:
    """
    Clear the entire session state and re-initializes it.
    """
    st.session_state._app_state = SessionState()
    st.rerun()


def handle_ocr_change():
    """
    Callback to handle the change in the OCR checkbox.
    This function is called when the checkbox value changes.
    """
    state = get_session_state()
    new_value = st.session_state.ocr_checkbox 
    
    if state.use_ocr != new_value:
        update_session_state(use_ocr=new_value)
        
        # Log the change
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"OCR checkbox changed to {new_value}")
        
        # Show a toast message to the user
        if new_value:
            st.toast("OCR mode enabled. Processing may be slower.", icon="📄")
        else:
            st.toast("OCR mode disabled. Using faster text extraction.", icon="⚡️")


def handle_reprocess_change():
    """
    Callback to handle the change in the reprocess checkbox.
    This function is called when the checkbox value changes.
    """
    state = get_session_state()
    new_value = st.session_state.reprocess_checkbox
    
    if state.reprocess_existing != new_value:
        update_session_state(reprocess_existing=new_value)
        
        # Log the change
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Reprocess existing contracts checkbox changed to {new_value}")
        
        # Show a toast message to the user
        if new_value:
            st.toast("Will reprocess existing contracts", icon="🔄")
        else:
            st.toast("Will load existing contracts from cache", icon="💾")

