import pytest
from unittest.mock import patch
from utility.config import Settings

# --- Test Default Fallback Behavior ---

def test_resolve_chunking_params_fallback_defaults():
    """
    Tests that fallback character-based values are used when no token-based
    configuration is provided.
    """
    settings = Settings(
        CHUNK_FALLBACK_CHARS=1000,
        CHUNK_FALLBACK_OVERLAP_CHARS=200,
    )
    chunk_size, chunk_overlap = settings.resolve_chunking_params()
    assert chunk_size == 1000
    assert chunk_overlap == 200

# --- Test Token-based Calculation (from EMBEDDINGS_MAX_INPUT_TOKENS) ---

def test_resolve_chunking_params_from_token_limit():
    """
    Tests that chunk size and overlap are correctly derived from the
    embeddings token limit and default fractions.
    """
    settings = Settings(
        EMBEDDINGS_MAX_INPUT_TOKENS=8192,
        CHUNK_SIZE_FRACTION_OF_EMBED_LIMIT=0.5, # 4096 tokens
        CHUNK_OVERLAP_FRACTION=0.1,           # 409 tokens
        AVG_CHARS_PER_TOKEN=4,
    )
    chunk_size, chunk_overlap = settings.resolve_chunking_params()
    assert chunk_size == 4096 * 4  # 16384
    assert chunk_overlap == 409 * 4   # 1636

# --- Test User Overrides (CHUNK_SIZE_TOKENS and CHUNK_OVERLAP_TOKENS) ---

def test_resolve_chunking_params_with_user_overrides():
    """
    Tests that user-defined token overrides for chunk size and overlap
    take precedence over all other calculations.
    """
    settings = Settings(
        EMBEDDINGS_MAX_INPUT_TOKENS=8192,  # Should be ignored
        CHUNK_SIZE_TOKENS=500,
        CHUNK_OVERLAP_TOKENS=50,
        AVG_CHARS_PER_TOKEN=4,
    )
    chunk_size, chunk_overlap = settings.resolve_chunking_params()
    assert chunk_size == 500 * 4  # 2000
    assert chunk_overlap == 50 * 4   # 200

def test_resolve_chunking_params_with_only_chunk_size_override():
    """
    Tests that if only chunk size is overridden, overlap is derived from it,
    not from the embeddings limit.
    """
    settings = Settings(
        EMBEDDINGS_MAX_INPUT_TOKENS=8192, # Should be ignored for overlap calculation
        CHUNK_SIZE_TOKENS=600,
        CHUNK_OVERLAP_FRACTION=0.25, # 150 tokens
        AVG_CHARS_PER_TOKEN=4,
    )
    chunk_size, chunk_overlap = settings.resolve_chunking_params()
    assert chunk_size == 600 * 4  # 2400
    assert chunk_overlap == 150 * 4 # 600

# --- Test Edge Cases ---

def test_resolve_chunking_params_zero_token_limit():
    """
    Tests that a zero token limit falls back to default char values.
    """
    settings = Settings(
        EMBEDDINGS_MAX_INPUT_TOKENS=0,
        CHUNK_FALLBACK_CHARS=999,
        CHUNK_FALLBACK_OVERLAP_CHARS=111,
    )
    chunk_size, chunk_overlap = settings.resolve_chunking_params()
    assert chunk_size == 999
    assert chunk_overlap == 111

def test_resolve_chunking_params_with_partial_overrides():
    """
    Ensures that if chunk size is provided but overlap is not, overlap is calculated
    based on the provided chunk size.
    """
    settings = Settings(
        CHUNK_SIZE_TOKENS=1000,
        CHUNK_OVERLAP_TOKENS=None,
        CHUNK_OVERLAP_FRACTION=0.15,
        AVG_CHARS_PER_TOKEN=4
    )
    chunk_size, chunk_overlap = settings.resolve_chunking_params()
    assert chunk_size == 1000 * 4  # 4000
    assert chunk_overlap == int(1000 * 0.15) * 4  # 150 * 4 = 600
