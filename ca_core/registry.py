from __future__ import annotations

"""
Contract registry module with security validations.

This module handles persistent storage of contract metadata and files
with input sanitization and path validation.
"""

import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utility.config import BASE_DIR, settings
from ca_core.exceptions import RegistryError, ValidationError

logger = logging.getLogger(__name__)

CONTRACTS_DIR = Path(BASE_DIR) / "data" / "contracts"

# Security: Allowed filename characters
SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-. ]+$')


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def _validate_contract_id(contract_id: str) -> None:
    """Validate contract ID format (should be a hex hash)."""
    if not contract_id or not re.match(r'^[a-f0-9]{64}$', contract_id):
        raise ValidationError(f"Invalid contract ID format: {contract_id}")


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
        
    Raises:
        ValidationError: If filename is invalid or unsafe
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")
    
    # Remove any path components
    filename = Path(filename).name
    
    # Replace unsafe characters
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    filename = filename.replace('/', '_').replace('\\', '_')
    
    # Ensure it ends with .pdf
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    
    # Check length
    if len(filename) > 255:
        # Truncate but keep extension
        filename = filename[:251] + '.pdf'
    
    logger.debug(f"Sanitized filename: {filename}")
    return filename


def _contract_dir(contract_id: str) -> Path:
    """Get contract directory, validating the contract ID."""
    _validate_contract_id(contract_id)
    return CONTRACTS_DIR / contract_id


def _meta_path(contract_id: str) -> Path:
    """Get path to contract metadata file."""
    return _contract_dir(contract_id) / "meta.json"


def _entities_path(contract_id: str) -> Path:
    """Get path to contract entities file."""
    return _contract_dir(contract_id) / "entities.json"


def _pdf_path(contract_id: str, original_name: str | None = None) -> Path:
    """Get path to PDF file with sanitized filename."""
    safe_name = _sanitize_filename(original_name or "document.pdf")
    return _contract_dir(contract_id) / safe_name


@dataclass
class ContractMeta:
    contract_id: str
    original_filename: str
    stored_pdf_path: str
    uploaded_at: str
    num_pages: int


def save_contract(
    *,
    contract_id: str,
    original_filename: str,
    source_pdf_path: Optional[Path] = None,
    uploaded_bytes: Optional[bytes] = None,
    num_pages: int = 0,
    entities: Optional[List[Dict[str, Any]]] = None,
) -> ContractMeta:
    """
    Persist the uploaded/ingested contract and its metadata.

    Args:
        contract_id: Unique contract identifier
        original_filename: Original filename (will be sanitized)
        source_pdf_path: Path to source PDF file
        uploaded_bytes: PDF content as bytes
        num_pages: Number of pages in the document
        entities: Extracted entities
        
    Returns:
        ContractMeta object with saved contract information
        
    Raises:
        ValueError: If neither source_pdf_path nor uploaded_bytes provided
        ValidationError: If validation fails
        RegistryError: If saving fails
    """
    logger.info(f"Saving contract: id={contract_id}, filename={original_filename}, pages={num_pages}")
    
    if not source_pdf_path and uploaded_bytes is None:
        raise ValueError("Either source_pdf_path or uploaded_bytes must be provided")
    
    # Validate contract_id
    _validate_contract_id(contract_id)
    
    # Validate number of pages
    if num_pages > settings.MAX_PDF_PAGES:
        raise ValidationError(f"PDF has {num_pages} pages, exceeds maximum of {settings.MAX_PDF_PAGES}")
    
    # Validate file size
    if uploaded_bytes is not None:
        size_mb = len(uploaded_bytes) / (1024 * 1024)
        if size_mb > settings.MAX_PDF_SIZE_MB:
            raise ValidationError(f"PDF size {size_mb:.2f}MB exceeds maximum of {settings.MAX_PDF_SIZE_MB}MB")
        logger.debug(f"PDF size: {size_mb:.2f}MB")

    try:
        dest_dir = _contract_dir(contract_id)
        _ensure_dir(dest_dir)

        pdf_dest = _pdf_path(contract_id, original_filename)

        if source_pdf_path:
            # Validate source path exists
            if not source_pdf_path.exists():
                raise ValidationError(f"Source PDF not found: {source_pdf_path}")
            shutil.copy2(source_pdf_path, pdf_dest)
            logger.debug(f"Copied PDF from {source_pdf_path} to {pdf_dest}")
        else:
            pdf_dest.write_bytes(uploaded_bytes or b"")
            logger.debug(f"Wrote {len(uploaded_bytes or b'')} bytes to {pdf_dest}")

        meta = ContractMeta(
            contract_id=contract_id,
            original_filename=original_filename,
            stored_pdf_path=str(pdf_dest.resolve()),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            num_pages=num_pages,
        )
        _meta_path(contract_id).write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        logger.debug(f"Saved metadata to {_meta_path(contract_id)}")

        if entities is not None:
            _entities_path(contract_id).write_text(json.dumps(entities, indent=2), encoding="utf-8")
            logger.info(f"Saved {len(entities)} entities for contract {contract_id}")

        logger.info(f"Successfully saved contract {contract_id}")
        return meta
        
    except (ValidationError, ValueError):
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Failed to save contract {contract_id}: {e}", exc_info=True)
        raise RegistryError(f"Failed to save contract: {e}") from e


def contract_exists(contract_id: str) -> bool:
    """
    Check if a contract with the given ID already exists.
    
    Args:
        contract_id: The contract ID to check
        
    Returns:
        True if contract exists, False otherwise
    """
    try:
        _validate_contract_id(contract_id)
        meta_path = _meta_path(contract_id)
        exists = meta_path.exists()
        logger.debug(f"Contract {contract_id} exists: {exists}")
        return exists
    except ValidationError:
        logger.warning(f"Invalid contract ID format: {contract_id}")
        return False
    except Exception as e:
        logger.error(f"Error checking if contract exists: {e}")
        return False


def list_contracts() -> List[Dict[str, Any]]:
    """
    Return a list of saved contracts with minimal metadata.
    
    Returns:
        List of contract metadata dictionaries, sorted by upload date
    """
    logger.debug("Listing all contracts")
    _ensure_dir(CONTRACTS_DIR)
    contracts: List[Dict[str, Any]] = []
    
    for meta_file in CONTRACTS_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            contracts.append(meta)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_file}: {e}")
            continue
    
    # Sort by uploaded_at descending
    contracts.sort(key=lambda m: m.get("uploaded_at", ""), reverse=True)
    logger.info(f"Found {len(contracts)} contracts")
    return contracts


def load_contract_meta(contract_id: str) -> Optional[Dict[str, Any]]:
    """
    Load contract metadata by ID.
    
    Args:
        contract_id: Contract identifier
        
    Returns:
        Contract metadata dictionary or None if not found
    """
    logger.debug(f"Loading metadata for contract {contract_id}")
    
    try:
        meta_file = _meta_path(contract_id)
        if not meta_file.exists():
            logger.warning(f"Metadata file not found for contract {contract_id}")
            return None
        
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        logger.debug(f"Successfully loaded metadata for contract {contract_id}")
        return meta
    except Exception as e:
        logger.error(f"Failed to load metadata for contract {contract_id}: {e}", exc_info=True)
        return None


def load_contract_entities(contract_id: str) -> List[Dict[str, Any]]:
    """
    Load contract entities by ID.
    
    Args:
        contract_id: Contract identifier
        
    Returns:
        List of entity dictionaries, or empty list if not found
    """
    logger.debug(f"Loading entities for contract {contract_id}")
    
    try:
        ent_file = _entities_path(contract_id)
        if not ent_file.exists():
            logger.debug(f"Entities file not found for contract {contract_id}")
            return []
        
        entities = json.loads(ent_file.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(entities)} entities for contract {contract_id}")
        return entities
    except Exception as e:
        logger.warning(f"Failed to load entities for contract {contract_id}: {e}")
        return []


def delete_contract(contract_id: str, delete_from_vectorstore: bool = True) -> bool:
    """
    Delete a contract and all its associated data.
    
    Args:
        contract_id: Contract identifier
        delete_from_vectorstore: Whether to also delete from vector store (default: True)
        
    Returns:
        True if deletion was successful, False otherwise
        
    Raises:
        ValidationError: If contract ID is invalid
        RegistryError: If deletion fails
    """
    logger.info(f"Deleting contract {contract_id}")
    
    try:
        _validate_contract_id(contract_id)
        
        contract_dir = _contract_dir(contract_id)
        
        if not contract_dir.exists():
            logger.warning(f"Contract directory does not exist: {contract_dir}")
            return False
        
        # Delete from vector store first (if requested)
        if delete_from_vectorstore:
            try:
                from ca_core.vectorstore import delete_contract_from_vector_store
                deleted_chunks = delete_contract_from_vector_store(contract_id)
                logger.info(f"Deleted {deleted_chunks} chunks from vector store for contract {contract_id}")
            except Exception as e:
                logger.error(f"Failed to delete from vector store: {e}", exc_info=True)
                # Continue with file deletion even if vector store deletion fails
        
        # Delete contract directory and all files
        shutil.rmtree(contract_dir)
        logger.info(f"Successfully deleted contract {contract_id} from filesystem")
        
        return True
        
    except ValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Failed to delete contract {contract_id}: {e}", exc_info=True)
        raise RegistryError(f"Failed to delete contract: {e}") from e


