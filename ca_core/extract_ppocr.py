

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ca_core.base import Singleton, ExtractionStrategy
from ca_core.exceptions import ExtractionError
from utility.utility import normalize_whitespace_preserve_newlines

logger = logging.getLogger(__name__)


class PaddleOCRProcessor(metaclass=Singleton):
    """Singleton for managing PaddleOCR instances."""
    
    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_instance(self, lang: str = "en"):
        """Get or create a PaddleOCR instance for the specified language.
        
        Args:
            lang: Language code for OCR
            
        Returns:
            PaddleOCR instance
            
        Raises:
            ExtractionError: If PaddleOCR cannot be initialized
        """
        if lang not in self._instances:
            try:
                from paddleocr import PaddleOCR
            except ImportError as e:
                raise ExtractionError("PaddleOCR is not installed. Please install with: pip install paddleocr") from e
                
            self.logger.info(f"Initializing new PaddleOCR instance for lang='{lang}'")
            # NOTE: Any additional languages need to be downloaded, e.g., via
            # !python -m paddleocr.download --lang=fr --lang=es
            try:
                self._instances[lang] = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="gpu",
                    lang=lang,
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR for lang='{lang}': {e}")
                raise ExtractionError(f"Failed to initialize PaddleOCR: {e}") from e
                
        return self._instances[lang]


class PaddleOCRStrategy(ExtractionStrategy):
    """Run OCR via PaddleOCR directly on a PDF."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._processor = PaddleOCRProcessor()

    def extract(self, pdf_path: str, max_pages: Optional[int] = None, lang: str = "en") -> List[Dict[str, Any]]:
        """Extract text from PDF using PaddleOCR.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to extract
            lang: Language code for OCR
            
        Returns:
            List of page content dictionaries
            
        Raises:
            ExtractionError: If OCR fails
        """
        documents: List[Dict[str, Any]] = []
        
        try:
            ocr_instance = self._processor.get_instance(lang)
            results = ocr_instance.predict(str(pdf_path))
        except Exception as e:
            self.logger.error(f"OCR prediction failed for {pdf_path}: {e}")
            raise ExtractionError(f"OCR prediction failed: {e}") from e
        
        for i, page_result in enumerate(results):
            if max_pages is not None and i >= max_pages:
                break
            
            try:
                # page_result is a dict per page: {'page_index': 0, 'rec_texts': ['text1', ...]}
                page_text = "\n".join(page_result.get("rec_texts", []))
                cleaned = normalize_whitespace_preserve_newlines(page_text)
                
                documents.append({
                    "text": cleaned,
                    "metadata": {
                        "page_number": page_result.get("page_index", i) + 1,
                    },
                })
            except Exception as e:
                self.logger.warning(f"Failed to process OCR results for page {i + 1}: {e}")
                documents.append({
                    "text": "",
                    "metadata": {
                        "page_number": i + 1,
                        "error": str(e),
                    },
                })
        
        return documents
