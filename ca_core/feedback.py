import logging
from pathlib import Path
import streamlit as st

# Feedback Logging Setup
# Ensure the data directory exists
Path("data/feedback").mkdir(parents=True, exist_ok=True)

# Create a specific logger for feedback
feedback_logger = logging.getLogger("feedback")
feedback_logger.setLevel(logging.INFO)

# Create a file handler which logs even info messages
fh = logging.FileHandler("data/feedback/feedback.log")
fh.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)

# Add the handlers to the logger
if not feedback_logger.handlers:
    feedback_logger.addHandler(fh)

def log_feedback(message_id: str, feedback_type: str, question: str, answer: str, context: str):
    """
    Logs user feedback to a file.
    
    Args:
        message_id: Unique message identifier
        feedback_type: 'positive' or 'negative'
        question: User's question
        answer: Assistant's answer
        context: Source context/documents
    """
    feedback_logger.info(
        f"FEEDBACK: message_id='{message_id}', type='{feedback_type}', "
        f"question='{question}', answer='{answer[:100]}...', context_length={len(context)}"
    )
    
    # Import here to avoid circular dependency
    from utility.session_state import mark_feedback_submitted
    mark_feedback_submitted(message_id)
    
    st.toast("Feedback submitted! Thank you.", icon="✅")
