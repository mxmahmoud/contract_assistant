"""Custom exceptions for the Contract Assistant application."""


class ContractAssistantError(Exception):
    """Base exception for all Contract Assistant errors."""
    pass


class ConfigurationError(ContractAssistantError):
    """Raised when there's a configuration-related error."""
    pass


class ExtractionError(ContractAssistantError):
    """Raised when document extraction fails."""
    pass


class VectorStoreError(ContractAssistantError):
    """Raised when vector store operations fail."""
    pass


class RegistryError(ContractAssistantError):
    """Raised when contract registry operations fail."""
    pass


class ModelLoadingError(ContractAssistantError):
    """Raised when model loading fails."""
    pass


class ValidationError(ContractAssistantError):
    """Raised when validation fails."""
    pass


class ServiceUnavailableError(ContractAssistantError):
    """Raised when a required service is not available."""
    pass
