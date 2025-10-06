

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
            device = settings.PADDLE_DEVICE
            try:
                self._instances[lang] = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device=device,
                    lang=lang,
                )
                self.logger.info(
                    "Initialized PaddleOCR for lang='%s' on device='%s'", lang, device
                )
            except Exception as e:
                if device != "cpu":
                    self.logger.warning(
                        "Failed to initialize PaddleOCR on device='%s' (%s). Falling back to CPU.",
                        device,
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

    def _sort_text_boxes_by_position(self, boxes: List[List[int]], texts: List[str]) -> List[str]:
        """Sort text boxes by their spatial position (top-to-bottom, left-to-right).
        
        This method implements a reading order algorithm that:
        1. Groups text boxes into horizontal "lines" based on Y-coordinate proximity
        2. Sorts boxes within each line by X-coordinate (left to right)
        3. Returns texts in proper reading order
        
        Args:
            boxes: List of bounding boxes in format [x1, y1, x2, y2]
            texts: List of text strings corresponding to each box
            
        Returns:
            List of texts sorted in reading order
        """
        if boxes is None or texts is None:
            return texts
        
        # Check for empty inputs using len() to avoid ambiguous truth value
        if len(boxes) == 0 or len(texts) == 0:
            return texts
        
        # Ensure both lists have the same length
        if len(boxes) != len(texts):
            self.logger.warning(f"Mismatch between boxes ({len(boxes)}) and texts ({len(texts)}), using unsorted texts")
            return texts
        
        # Combine boxes and texts for sorting
        box_text_pairs = list(zip(boxes, texts))
        
        # Calculate vertical threshold for grouping into lines (50% of average box height)
        avg_height = sum(box[3] - box[1] for box in boxes) / len(boxes)
        vertical_threshold = avg_height * 0.5
        
        # Sort by Y-coordinate first (top to bottom)
        box_text_pairs.sort(key=lambda x: x[0][1])
        
        # Group boxes into lines based on Y-coordinate proximity
        lines = []
        current_line = [box_text_pairs[0]]
        
        for box, text in box_text_pairs[1:]:
            # Check if this box is on the same line as the current line
            current_line_y = sum(b[0][1] for b in current_line) / len(current_line)
            if abs(box[1] - current_line_y) <= vertical_threshold:
                current_line.append((box, text))
            else:
                # Start a new line
                lines.append(current_line)
                current_line = [(box, text)]

        if len(current_line) > 0:
            lines.append(current_line)
        
        # Sort boxes within each line by X-coordinate (left to right)
        sorted_texts = []
        for line in lines:
            line.sort(key=lambda x: x[0][0])  # Sort by x1 coordinate
            sorted_texts.extend([text for _, text in line])
        
        return sorted_texts

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
            page_result.save_to_img("output")
            page_result.save_to_json("output")
            if max_pages is not None and i >= max_pages:
                break
            
            try:
                # page_result is a dict with keys: 'rec_boxes', 'rec_texts', 'rec_scores', etc.
                rec_boxes = page_result.get("rec_boxes", [])
                rec_texts = page_result.get("rec_texts", [])
                
                # Sort texts by their spatial position for proper reading order
                if rec_boxes is not None and rec_texts is not None:
                    sorted_texts = self._sort_text_boxes_by_position(rec_boxes, rec_texts)
                    print(f"Sorted texts: {sorted_texts}")
                    page_text = "\n".join(sorted_texts)
                else:
                    # Fallback to original method if boxes are not available
                    page_text = "\n".join(rec_texts)
                print(f"Page text: {page_text}")
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
