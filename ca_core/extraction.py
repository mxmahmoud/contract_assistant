"""Unified extraction module with strategy-based implementation and optional OCR.

Exposes `extract_text_from_pdf` for backward compatibility while using
pluggable strategies under the hood and caching heavy OCR models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ca_core.base import ExtractionStrategy
from ca_core.extract_pypdf import PyPDF2Strategy
from ca_core.exceptions import ExtractionError

logger = logging.getLogger(__name__)

# Lazy import PaddleOCR - only load when actually needed
def _get_paddleocr_strategy():
    """Lazy load PaddleOCR strategy to avoid loading heavy models unless OCR is used."""
    from ca_core.extract_ppocr import PaddleOCRStrategy
    return PaddleOCRStrategy()


@dataclass
class ExtractionConfig:
    strategy: str = "auto"  # 'auto' | 'paddleocr' | 'pypdf2'
    ocr_lang: Any = None
    render_dpi: int = 200
    max_pages: Optional[int] = None


class ExtractionService:
    """High-level extraction orchestrator with auto-detection and caching hooks."""

    def __init__(self, config: Optional[ExtractionConfig] = None) -> None:
        extraction_config = config or ExtractionConfig()
        self.extraction_config = extraction_config
        self._strategies = {}
        self._initialized = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self) -> None:
        """Initialize the extraction service."""
        if self._initialized:
            self.logger.warning(f"{self.__class__.__name__} already initialized")
            return
        self.logger.info(f"Initializing ExtractionService with strategy: {self.extraction_config.strategy}")
        self._initialized = True

    def _make_strategy(self, name: str) -> ExtractionStrategy:
        """Create or retrieve a cached strategy instance.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy instance
            
        Raises:
            ExtractionError: If strategy name is unknown
        """
        if name not in self._strategies:
            if name == "pypdf2":
                self._strategies[name] = PyPDF2Strategy()
            elif name == "paddleocr":
                # Lazy load - only import PaddleOCR when actually needed
                self._strategies[name] = _get_paddleocr_strategy()
            else:
                raise ExtractionError(f"Unknown extraction strategy: {name}")
                
        return self._strategies[name]

    def _is_paddleocr_available(self) -> bool:
        """Check if PaddleOCR is available for import."""
        try:
            import paddleocr
            return True
        except ImportError:
            return False
    
    def _is_text_extraction_reasonable(self, pages: List[Dict[str, Any]]) -> bool:
        """Check if text extraction from pypdf2 produced reasonable results."""
        if not pages:
            return False
        lengths = [len(p.get("text", "").strip()) for p in pages]
        avg_len = sum(lengths) / max(len(lengths), 1)
        return avg_len >= 80

    def extract(self, pdf_path: str, lang: str = "en") -> List[Dict[str, Any]]:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            lang: Language code for OCR
            
        Returns:
            List of page content dictionaries
            
        Raises:
            ExtractionError: If extraction fails
        """
        if not self._initialized:
            self.initialize()
            
        cfg = self.extraction_config

        try:
            # Strategy: paddleocr - use OCR directly
            if cfg.strategy == "paddleocr":
                self.logger.info(f"Using PaddleOCR extraction for {pdf_path}")
                strategy = self._make_strategy("paddleocr")
                return strategy.extract(pdf_path, max_pages=cfg.max_pages, lang=lang)

            # Strategy: pypdf2 - use text extraction directly
            if cfg.strategy == "pypdf2":
                self.logger.info(f"Using PyPDF2 extraction for {pdf_path}")
                strategy = self._make_strategy("pypdf2")
                return strategy.extract(pdf_path, max_pages=cfg.max_pages, lang=lang)

            # Strategy: auto - try pypdf2, fall back to paddleocr if available and needed
            if cfg.strategy == "auto":
                self.logger.info(f"Using auto-detection for {pdf_path}")
                
                # First, try pypdf2
                try:
                    text_strategy = self._make_strategy("pypdf2")
                    pages = text_strategy.extract(pdf_path, max_pages=cfg.max_pages, lang=lang)
                    
                    if self._is_text_extraction_reasonable(pages):
                        self.logger.info("PyPDF2 extraction successful")
                        return pages
                    
                    # Text extraction was poor, check if we can use OCR
                    self.logger.info("PyPDF2 extraction produced poor results")
                    
                except Exception as e:
                    self.logger.warning(f"PyPDF2 extraction failed: {e}")
                
                # Fall back to PaddleOCR if available
                if self._is_paddleocr_available():
                    self.logger.info("Falling back to PaddleOCR")
                    ocr_strategy = self._make_strategy("paddleocr")
                    return ocr_strategy.extract(pdf_path, max_pages=cfg.max_pages, lang=lang)
                else:
                    self.logger.warning("PaddleOCR is not available, returning pypdf2 results")
                    # Return whatever we got from pypdf2, even if poor quality
                    return pages if 'pages' in locals() else []
            
            # Unknown strategy
            raise ExtractionError(f"Unknown extraction strategy: {cfg.strategy}")
            
        except ExtractionError:
            raise  # Re-raise extraction errors
        except Exception as e:
            self.logger.error(f"Unexpected error during extraction: {e}", exc_info=True)
            raise ExtractionError(f"Unexpected error during extraction: {e}") from e


def extract_text_from_pdf(
    pdf_path: str,
    *,
    strategy: str = "auto",
    ocr_lang: Any = "en",
    render_dpi: int = 200,
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract text and metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        strategy: Extraction strategy ('auto', 'pypdf2', or 'paddleocr')
                  - 'auto': Try pypdf2, fall back to paddleocr if available and needed
                  - 'pypdf2': Use PyPDF2 text extraction only
                  - 'paddleocr': Use PaddleOCR (OCR) only
        ocr_lang: Language code for OCR
        render_dpi: DPI for PDF rendering (if needed)
        max_pages: Maximum number of pages to extract
        
    Returns:
        List of page dictionaries with text and metadata
        
    Raises:
        ExtractionError: If extraction fails
    """
    logger.info(f"Starting extraction for '{pdf_path}' with strategy='{strategy}'")
    
    config = ExtractionConfig(
        strategy=strategy,
        ocr_lang=ocr_lang,
        render_dpi=render_dpi,
        max_pages=max_pages,
    )
    service = ExtractionService(config)
    pages = service.extract(pdf_path, lang=ocr_lang)
    
    logger.info(f"Successfully extracted {len(pages)} pages from '{pdf_path}'")
    return pages


