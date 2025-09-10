# tests/test_ingestion.py
import pytest
from pathlib import Path

from ca_core.extraction import extract_text_from_pdf
from utility.config import BASE_DIR

# Discover all PDF files in the examples directory
EXAMPLES_DIR = BASE_DIR / "data" / "contracts"
# Create a list of all files ending with .pdf
pdf_files = list(EXAMPLES_DIR.glob("*.pdf"))

# Create user-friendly IDs for the test output
pdf_file_ids = [file.name for file in pdf_files]

# Skip the tests if no PDF files are found to avoid errors
if not pdf_files:
    pytest.skip("No PDF files found in examples directory, skipping extraciton tests.", allow_module_level=True)


# Use @pytest.mark.parametrize to run the test for each discovered file
@pytest.mark.parametrize("pdf_path", pdf_files, ids=pdf_file_ids)
def test_extract_text_from_pdf_for_all_examples(pdf_path: Path):
    """
    Tests that text extraction works for a given PDF, returns the correct data structure,
    and extracts a reasonable amount of text. This test is run for every PDF
    found in the /examples directory.
    """
    # Act: Run the extraction function on the parameterized pdf_path
    documents = extract_text_from_pdf(str(pdf_path))

    # Assert
    assert isinstance(documents, list), f"Expected a list of documents for {pdf_path.name}"
    assert len(documents) > 0, f"No pages were processed for {pdf_path.name}"

    # Check the structure of the first page/document
    first_doc = documents[0]
    assert isinstance(first_doc, dict)
    assert "text" in first_doc
    assert "metadata" in first_doc

    # Check metadata structure
    metadata = first_doc["metadata"]
    assert isinstance(metadata, dict)
    assert "page_number" in metadata
    assert "section" in metadata
    assert metadata["page_number"] == 1, f"Page number should be 1 for the first page in {pdf_path.name}"

    # Use a general assertion instead of a hardcoded string
    # This ensures the test is robust and works for any contract, not just the sample one.
    assert len(first_doc["text"]) > 100, f"Expected more than 100 characters of text to be extracted from {pdf_path.name}"