

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ca_core.base import Singleton, ExtractionStrategy
from ca_core.exceptions import ExtractionError
from utility.utility import normalize_whitespace_preserve_newlines
from utility.config import settings

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
            desired_device = settings.PADDLE_DEVICE
            try:
                self._instances[lang] = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device=desired_device,
                    lang=lang,
                )
                self.logger.info(
                    "Initialized PaddleOCR for lang='%s' on device='%s'", lang, desired_device
                )
            except Exception as e:
                if desired_device != "cpu":
                    self.logger.warning(
                        "Failed to initialize PaddleOCR on device='%s' (%s). Falling back to CPU.",
                        desired_device,
                        e,
                    )
                    try:
                        self._instances[lang] = PaddleOCR(
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_textline_orientation=False,
                            device="cpu",
                            lang=lang,
                        )
                        self.logger.info(
                            "Initialized PaddleOCR for lang='%s' on device='cpu' after fallback", lang
                        )
                    except Exception as cpu_error:
                        self.logger.error(
                            "Failed to initialize PaddleOCR for lang='%s' even on CPU: %s",
                            lang,
                            cpu_error,
                        )
                        raise ExtractionError(
                            f"Failed to initialize PaddleOCR even on CPU: {cpu_error}"
                        ) from cpu_error
                else:
                    self.logger.error(
                        "Failed to initialize PaddleOCR for lang='%s' on CPU: %s",
                        lang,
                        e,
                    )
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
        self.logger.info(f"Starting PaddleOCR extraction for '{pdf_path}'")
        documents: List[Dict[str, Any]] = []
        
        try:
            ocr_instance = self._processor.get_instance(lang)
            results = ocr_instance.predict(str(pdf_path))
        except Exception as e:
            self.logger.error(f"OCR prediction failed for {pdf_path}: {e}")
            raise ExtractionError(f"OCR prediction failed: {e}") from e
        
        self.logger.info(f"Finished PaddleOCR extraction for '{pdf_path}'")
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
