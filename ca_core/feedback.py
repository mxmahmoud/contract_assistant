import logging
from pathlib import Path
import streamlit as st

# Feedback Logging Setup
# Ensure the data directory exists
Path("data").mkdir(exist_ok=True)

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
    """Logs user feedback to a file."""
    feedback_logger.info(f"FEEDBACK: message_id='{message_id}', type='{feedback_type}', question='{question}', answer='{answer[:100]}...', context='{context}'")
    # Update session state to reflect that feedback was given
    for msg in st.session_state.messages:
        if msg.get("id") == message_id:
            msg["feedback_submitted"] = True
            break
    st.toast(f"Feedback submitted! Thank you.", icon="✅")
