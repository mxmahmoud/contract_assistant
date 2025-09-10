from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional
from pypdf import PdfReader

from ca_core.base import ExtractionStrategy
from ca_core.exceptions import ExtractionError
from utility.utility import normalize_whitespace_preserve_newlines

logger = logging.getLogger(__name__)


class PyPDF2Strategy(ExtractionStrategy):
    """Extract text using PyPDF"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, pdf_path: str, max_pages: Optional[int] = None, lang: str = "en") -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        
        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                
                for i, page in enumerate(reader.pages):
                    if max_pages is not None and i >= max_pages:
                        break
                    page_number = i + 1
                    
                    try:
                        text = page.extract_text() or ""
                        text = re.sub(r"-\n", "", text)
                        cleaned = normalize_whitespace_preserve_newlines(text)
                        
                        documents.append({
                            "text": cleaned,
                            "metadata": {
                                "page_number": page_number,
                            },
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_number}: {e}")
                        documents.append({
                            "text": "",
                            "metadata": {
                                "page_number": page_number,
                                "error": str(e),
                            },
                        })
                        
        except Exception as e:
            self.logger.error(f"Failed to process PDF file {pdf_path}: {e}")
            raise ExtractionError(f"Failed to process PDF file: {e}") from e
            
        return documents
