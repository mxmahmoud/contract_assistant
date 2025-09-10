# ca_core/config.py
import os
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    """Enum for LLM providers."""
    OPENAI = "openai"
    LOCAL = "local"


class DatabaseType(str, Enum):
    """Database types."""
    CHROMA = "chroma"
    POSTGRES = "postgres"
    SQLITE = "sqlite"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # --- Environment Configuration ---
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment (development, staging, production)"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.LOCAL,
        description="The provider to use for LLM and embedding models"
    )
    
    OPENAI_API_KEY: str = Field(
        default="",
        description="API key for OpenAI services"
    )
    
    # --- Service URLs and Ports ---
    APP_HOST: str = Field(
        default="0.0.0.0",
        description="Host for the main application"
    )
    APP_PORT: int = Field(
        default=8501,
        description="Port for the main application"
    )
    
    LITELLM_HOST: str = Field(
        default="litellm",
        description="LiteLLM service hostname"
    )
    LITELLM_PORT: int = Field(
        default=4000,
        description="LiteLLM service port"
    )
    
    OLLAMA_HOST: str = Field(
        default="ollama",
        description="Ollama service hostname"
    )
    OLLAMA_PORT: int = Field(
        default=11434,
        description="Ollama service port"
    )
    
    TEI_HOST: str = Field(
        default="tei",
        description="TEI service hostname"
    )
    TEI_PORT: int = Field(
        default=8504,
        description="TEI service port"
    )
    
    # --- Model Configuration ---
    OPENAI_LLM_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model for chat completions"
    )
    OPENAI_EMBEDDINGS_MODEL: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for generating embeddings"
    )
    
    LOCAL_LLM_MODEL: str = Field(
        default="ollama/llama3.2:3b",
        description="Local LLM model identifier"
    )
    LOCAL_EMBEDDINGS_MODEL: str = Field(
        default="intfloat/multilingual-e5-large-instruct",
        description="Local embeddings model identifier"
    )
    
    LOCAL_LLM_ALIAS: str = Field(
        default="gpt-local",
        description="Alias for local LLM model in LiteLLM config"
    )
    LOCAL_EMBEDDINGS_ALIAS: str = Field(
        default="local-embeddings",
        description="Alias for local embeddings model in LiteLLM config"
    )
    
    # --- Dynamic Chunking Configuration ---
    # The EMBEDDINGS_MAX_INPUT_TOKENS is provider-dependent.
    # For local TEI `multilingual-e5-large-instruct`, it's 512.
    # For OpenAI `text-embedding-3-small`, it's 8192.
    EMBEDDINGS_MAX_INPUT_TOKENS: Optional[int] = Field(
        default=None,
        description="Maximum input tokens for the embeddings model"
    )
    
    CHUNK_SIZE_TOKENS: Optional[int] = Field(
        default=None,
        description="Override for chunk size in tokens"
    )
    CHUNK_OVERLAP_TOKENS: Optional[int] = Field(
        default=None,
        description="Override for chunk overlap in tokens"
    )

    # Fractions for deriving chunk parameters from token limits, if overrides are not set
    CHUNK_SIZE_FRACTION_OF_EMBED_LIMIT: float = Field(
        default=0.8,
        description="Fraction of embeddings token limit to use for chunk size"
    )
    CHUNK_OVERLAP_FRACTION: float = Field(
        default=0.2,
        description="Fraction of chunk size to use for overlap"
    )
    
    AVG_CHARS_PER_TOKEN: int = Field(
        default=4,
        description="Estimated average characters per token"
    )
    
    # Fallback values (character-based) if token limits are not available
    CHUNK_FALLBACK_CHARS: int = Field(
        default=1000,
        description="Fallback chunk size in characters"
    )
    CHUNK_FALLBACK_OVERLAP_CHARS: int = Field(
        default=200,
        description="Fallback chunk overlap in characters"
    )

    # --- Vector Store Configuration ---
    VECTOR_STORE_TYPE: DatabaseType = Field(
        default=DatabaseType.CHROMA,
        description="Type of vector store to use"
    )
    CHROMA_PERSIST_DIR: str = Field(
        default=str(BASE_DIR / "data" / "chroma_data"),
        description="Directory to persist ChromaDB data"
    )
    
    # --- Docker Configuration ---
    DOCKER_NETWORK_NAME: str = Field(
        default="contract-net",
        description="Docker network name for services"
    )
    
    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode."""
        return self.LLM_PROVIDER == LLMProvider.LOCAL
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def litellm_url(self) -> str:
        """Get LiteLLM service URL."""
        return f"http://{self.LITELLM_HOST}:{self.LITELLM_PORT}"
    
    @property
    def ollama_url(self) -> str:
        """Get Ollama service URL."""
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"
    
    @property
    def tei_url(self) -> str:
        """Get TEI service URL."""
        return f"http://{self.TEI_HOST}:{self.TEI_PORT}"
    
    @property
    def llm_model(self) -> str:
        """Get the appropriate LLM model based on provider."""
        if self.is_local_mode:
            return self.LOCAL_LLM_ALIAS
        return self.OPENAI_LLM_MODEL
    
    @property
    def embeddings_model(self) -> str:
        """Get the appropriate embeddings model based on provider."""
        if self.is_local_mode:
            return self.LOCAL_EMBEDDINGS_ALIAS
        return self.OPENAI_EMBEDDINGS_MODEL
    
    @property
    def openai_base_url(self) -> Optional[str]:
        """Get OpenAI base URL based on provider."""
        if self.is_local_mode:
            return f"{self.litellm_url}/v1"
        return None
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key based on provider."""
        if self.is_local_mode:
            return "not-needed"
        return self.OPENAI_API_KEY
    
    @validator('OPENAI_API_KEY')
    def validate_openai_api_key(cls, v, values):
        """Validate OpenAI API key is provided when using OpenAI provider."""
        if values.get('LLM_PROVIDER') == LLMProvider.OPENAI and not v:
            raise ValueError("OPENAI_API_KEY must be provided when using OpenAI provider")
        return v
    
    def resolve_chunking_params(self) -> tuple[int, int]:
        """
        Resolves chunk size and overlap, prioritizing user overrides, then token-based
        calculations, and finally falling back to character-based defaults.
        
        Returns:
            A tuple of (chunk_size_chars, chunk_overlap_chars).
        """
        chunk_size_tokens: Optional[int] = self.CHUNK_SIZE_TOKENS
        chunk_overlap_tokens: Optional[int] = self.CHUNK_OVERLAP_TOKENS
        
        # If chunk size is not explicitly set, derive it from the embeddings model token limit
        if chunk_size_tokens is None and self.EMBEDDINGS_MAX_INPUT_TOKENS:
            chunk_size_tokens = int(
                self.EMBEDDINGS_MAX_INPUT_TOKENS * self.CHUNK_SIZE_FRACTION_OF_EMBED_LIMIT
            )
        
        # If chunk overlap is not explicitly set, derive it from the chunk size
        if chunk_overlap_tokens is None and chunk_size_tokens:
            chunk_overlap_tokens = int(chunk_size_tokens * self.CHUNK_OVERLAP_FRACTION)
            
        # If token-based values were calculated, convert them to characters
        if chunk_size_tokens and chunk_overlap_tokens:
            chunk_size_chars = chunk_size_tokens * self.AVG_CHARS_PER_TOKEN
            chunk_overlap_chars = chunk_overlap_tokens * self.AVG_CHARS_PER_TOKEN
            return chunk_size_chars, chunk_overlap_chars
            
        # Otherwise, use the character-based fallbacks
        return self.CHUNK_FALLBACK_CHARS, self.CHUNK_FALLBACK_OVERLAP_CHARS

    def get_service_url(self, service: str) -> str:
        """Get URL for a specific service."""
        service_urls = {
            'litellm': self.litellm_url,
            'ollama': self.ollama_url,
            'tei': self.tei_url,
        }
        return service_urls.get(service, '')
    
    def get_model_config(self) -> dict:
        """Get model configuration for the current provider."""
        if self.is_local_mode:
            return {
                'llm_model': self.LOCAL_LLM_MODEL,
                'embeddings_model': self.LOCAL_EMBEDDINGS_MODEL,
                'llm_alias': self.LOCAL_LLM_ALIAS,
                'embeddings_alias': self.LOCAL_EMBEDDINGS_ALIAS,
            }
        return {
            'llm_model': self.OPENAI_LLM_MODEL,
            'embeddings_model': self.OPENAI_EMBEDDINGS_MODEL,
        }
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print(f"=== Configuration Summary ===")
        print(f"Environment: {self.ENVIRONMENT}")
        print(f"LLM Provider: {self.LLM_PROVIDER}")
        print(f"Debug Mode: {self.DEBUG}")
        
        if self.is_local_mode:
            print(f"Local Mode: API calls routed to Docker services")
            print(f"  - LiteLLM: {self.litellm_url}")
            print(f"  - Ollama: {self.ollama_url}")
            print(f"  - TEI: {self.tei_url}")
        else:
            print(f"OpenAI Mode: API calls sent to official OpenAI API")
            print(f"  - LLM Model: {self.llm_model}")
            print(f"  - Embeddings Model: {self.embeddings_model}")
        
        print(f"Vector Store: {self.VECTOR_STORE_TYPE}")
        if self.VECTOR_STORE_TYPE == DatabaseType.CHROMA:
            print(f"  - Persist Directory: {self.CHROMA_PERSIST_DIR}")
    
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


# Instantiate settings to be imported by other modules
settings = Settings()

# Print configuration summary on import (only in development)
if settings.is_development:
    settings.print_config_summary()