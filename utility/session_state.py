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

