# utility/model_loader.py
"""
Model loading and service health checking for local LLM services.
"""
import requests
import time
import logging
import streamlit as st
from utility.config import settings
from utility.session_state import get_session_state, update_session_state
from ca_core.exceptions import ModelLoadingError, ServiceUnavailableError

logger = logging.getLogger(__name__)

def _normalize_ollama_model_name(model_name: str) -> str:
    """
    Normalize model name for direct Ollama API calls.
    Accepts values like 'ollama/llama3.2:3b' or 'llama3.2:3b' and returns 'llama3.2:3b'.
    """
    if model_name.startswith("ollama/"):
        return model_name.split("/", 1)[1]
    return model_name


def ensure_model_is_loaded():
    """
    Wait for the required Ollama model to be available.
    The Ollama container's entry script is responsible for pulling the model on startup.
    
    Raises:
        ModelLoadingError: If the model cannot be loaded
        ServiceUnavailableError: If Ollama service is not available
    """
    if not settings.is_local_mode:
        logger.info("Skipping model check, not in local mode.")
        return

    # Avoid repeating the check/messages on every rerun
    state = get_session_state()
    if state.ollama_model_ready:
        return

    model_name = _normalize_ollama_model_name(settings.LOCAL_LLM_MODEL)
    ollama_api_url = settings.ollama_url + "/api"

    try:
        logger.info(f"Checking for local model '{model_name}' at {ollama_api_url}...")

        # First quick check
        response = requests.get(f"{ollama_api_url}/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        if any(m.get("name") == model_name for m in models):
            logger.info(f"Model '{model_name}' found locally.")
            update_session_state(ollama_model_ready=True)
            st.toast(f"LLM model '{model_name}' is available.", icon="✅")
            return

        # Wait loop while the Ollama container pulls the model
        status_placeholder = st.empty()
        with status_placeholder.container():
            st.info(f"Waiting for LLM model '{model_name}' to become available...")
            with st.spinner("Connecting to Ollama and waiting for model preparation..."):
                timeout_seconds = settings.OLLAMA_STARTUP_TIMEOUT
                start_time = time.time()
                while time.time() - start_time < timeout_seconds:
                    try:
                        resp = requests.get(f"{ollama_api_url}/tags", timeout=5)
                        if resp.status_code == 200:
                            available = any(m.get("name") == model_name for m in resp.json().get("models", []))
                            if available:
                                logger.info("Ollama model is ready.")
                                status_placeholder.empty()
                                st.toast("LLM model is ready.", icon="✅")
                                update_session_state(ollama_model_ready=True)
                                st.rerun()
                                return
                    except requests.exceptions.RequestException as e:
                        logger.debug(f"Ollama health check failed: {e}")
                    time.sleep(5)

        error_message = (
            f"Timed out waiting for model '{model_name}' at {settings.ollama_url}. "
            f"Please check the Ollama logs."
        )
        logger.error(error_message)
        status_placeholder.empty()
        raise ModelLoadingError(error_message)

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to connect to Ollama at {settings.ollama_url}. Please ensure the Ollama service is running and accessible."
        logger.error(f"{error_message} - {e}")
        raise ServiceUnavailableError(error_message) from e
    except ModelLoadingError:
        raise  # Re-raise our custom exceptions
    except Exception as e:
        error_message = f"An unexpected error occurred while checking the model: {e}"
        logger.error(error_message, exc_info=True)
        raise ModelLoadingError(error_message) from e


def ensure_tei_is_ready():
    """
    Checks if the Text Embeddings Inference (TEI) service is ready and responsive.
    
    Raises:
        ServiceUnavailableError: If TEI service is not available
    """
    if not settings.is_local_mode:
        return

    # Avoid repeating the wait/info message on each rerun
    state = get_session_state()
    if state.tei_ready:
        return

    model_name = settings.LOCAL_EMBEDDINGS_MODEL
    tei_health_url = f"{settings.tei_url}/health"
    timeout_seconds = settings.TEI_STARTUP_TIMEOUT
    start_time = time.time()

    status_placeholder = st.empty()
    with status_placeholder.container():
        st.info(f"Waiting for embedding service ({model_name})...")
        with st.spinner("Connecting to Text Embeddings Inference service... This may take a moment on first startup as the model is downloaded."):
            while time.time() - start_time < timeout_seconds:
                try:
                    response = requests.get(tei_health_url, timeout=5)
                    if response.status_code == 200:
                        logger.info("TEI service is ready.")
                        status_placeholder.empty()
                        st.toast("Embedding service is ready.", icon="✅")
                        update_session_state(tei_ready=True)
                        st.rerun()
                        return
                except requests.exceptions.RequestException as e:
                    # Service is not yet available, wait and retry
                    logger.debug(f"TEI health check failed: {e}")
                time.sleep(1)
    
    error_message = f"Could not connect to TEI service at {settings.tei_url} after {timeout_seconds} seconds. Please check the service logs."
    logger.error(error_message)
    status_placeholder.empty()
    raise ServiceUnavailableError(error_message)
