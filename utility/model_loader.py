# utility/model_loader.py
"""
Model loading and service health checking for local LLM services.
"""
import requests
import logging
from utility.config import settings
from ca_core.qa import get_llm

logger = logging.getLogger(__name__)


def pull_ollama_model(model_name: str) -> bool:
    """
    Explicitly pull/download a model from Ollama.
    
    Args:
        model_name: The model name to pull (e.g., 'llama3.2:3b')
        
    Returns:
        True if successful, False otherwise
    """
    normalized_name = _normalize_ollama_model_name(model_name)
    ollama_pull_url = settings.ollama_url + "/api/pull"
    
    try:
        logger.info(f"Initiating pull for model '{normalized_name}' from Ollama...")
        response = requests.post(
            ollama_pull_url,
            json={"name": normalized_name, "stream": False},
            timeout=600  # 10 minute timeout for model download
        )
        response.raise_for_status()
        logger.info(f"Successfully pulled model '{normalized_name}'")
        return True
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while pulling model '{normalized_name}'. The model may be too large or network is slow.")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to pull model '{normalized_name}': {e}")
        return False


def load_local_llm_model():
    """
    Ensures the specified local LLM model is available and loaded.
    If the model is not present locally, it triggers a download from Ollama.
    """
    if not settings.is_local_mode:
        return True

    model_name = _normalize_ollama_model_name(settings.LOCAL_LLM_MODEL)
    
    # First, check if the model is already available
    if check_ollama_status():
        logger.info(f"Model '{model_name}' is already available")
        try:
            # Quick test to ensure it's working
            llm = get_llm()
            llm.invoke("Test")
            return True
        except Exception as e:
            logger.error(f"Model exists but failed to invoke: {e}")
            return False
    
    # Model not available, need to download it
    logger.info(f"Model '{model_name}' not found locally, initiating download...")
    
    if not pull_ollama_model(settings.LOCAL_LLM_MODEL):
        return False
    
    # Verify the model is now available
    if not check_ollama_status():
        logger.error(f"Model '{model_name}' was pulled but is not showing as available")
        return False
    
    try:
        llm = get_llm()
        llm.invoke("Respond with a single word.")
        logger.info(f"Successfully loaded model '{model_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}' after download: {e}")
        return False


def _normalize_ollama_model_name(model_name: str) -> str:
    """
    Normalize model name for direct Ollama API calls.
    Accepts values like 'ollama/llama3.2:3b' or 'llama3.2:3b' and returns 'llama3.2:3b'.
    """
    if model_name.startswith("ollama/"):
        return model_name.split("/", 1)[1]
    return model_name


def is_ollama_service_ready() -> bool:
    """Checks if the Ollama service itself is reachable."""
    if not settings.is_local_mode:
        return True
    try:
        response = requests.get(settings.ollama_url, timeout=3)
        response.raise_for_status()
        if "Ollama is running" in response.text:
            return True
        logger.warning("Ollama service is reachable but returned an unexpected response.")
        return False
    except requests.exceptions.RequestException:
        logger.debug(f"Ollama service at {settings.ollama_url} not yet reachable.")
        return False


def check_ollama_status() -> bool:
    """
    Performs a single, non-blocking check to see if the target LLM model
    is available in the local Ollama service.

    Returns:
        True if the target model is available via Ollama, False otherwise.
    """
    if not settings.is_local_mode:
        return True

    model_name = _normalize_ollama_model_name(settings.LOCAL_LLM_MODEL)
    ollama_tags_url = settings.ollama_url + "/api/tags"

    try:
        response = requests.get(ollama_tags_url, timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        ready = any(m.get("name") == model_name for m in models)
        if ready:
            logger.info(f"Ollama model '{model_name}' is ready.")
        else:
            logger.info(
                f"Ollama service reachable, but model '{model_name}' not yet available."
            )
        return ready
    except requests.exceptions.RequestException:
        logger.debug(f"Ollama service at {settings.ollama_url} not reachable yet.")
        return False


def check_tei_status() -> bool:
    """
    Perform a single, non-blocking readiness check for the TEI embeddings service.

    Returns:
        True if TEI is healthy, False otherwise.
    """
    if not settings.is_local_mode:
        return True

    tei_health_url = f"{settings.tei_url}/health"

    try:
        response = requests.get(tei_health_url, timeout=5)
        if response.status_code == 200:
            logger.info("TEI service is ready.")
            return True
        return False
    except requests.exceptions.RequestException:
        logger.debug(f"TEI service at {settings.tei_url} not reachable yet.")
        return False