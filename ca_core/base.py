"""Base classes and interfaces for Contract Assistant services."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ExtractionStrategy(ABC):
    """Interface for extraction strategies."""
    
    @abstractmethod
    def extract(self, pdf_path: str, max_pages: Optional[int] = None, lang: str = "en") -> List[Dict[str, Any]]:
        """Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract
            lang: Language code for OCR
            
        Returns:
            List of dictionaries containing text and metadata for each page
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass


class Singleton(type):
    """Metaclass for implementing the Singleton pattern."""
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
