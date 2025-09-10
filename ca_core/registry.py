from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utility.config import BASE_DIR

CONTRACTS_DIR = Path(BASE_DIR) / "data" / "contracts"

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _contract_dir(contract_id: str) -> Path:
    return CONTRACTS_DIR / contract_id


def _meta_path(contract_id: str) -> Path:
    return _contract_dir(contract_id) / "meta.json"


def _entities_path(contract_id: str) -> Path:
    return _contract_dir(contract_id) / "entities.json"


def _pdf_path(contract_id: str, original_name: str | None = None) -> Path:
    safe_name = (original_name or "document.pdf").replace("/", "_").replace("\\", "_")
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
    """Persist the uploaded/ingested contract and its metadata.

    One of source_pdf_path or uploaded_bytes must be provided.
    """
    if not source_pdf_path and uploaded_bytes is None:
        raise ValueError("Either source_pdf_path or uploaded_bytes must be provided")

    dest_dir = _contract_dir(contract_id)
    _ensure_dir(dest_dir)

    pdf_dest = _pdf_path(contract_id, original_filename)

    if source_pdf_path:
        shutil.copy2(source_pdf_path, pdf_dest)
    else:
        pdf_dest.write_bytes(uploaded_bytes or b"")

    meta = ContractMeta(
        contract_id=contract_id,
        original_filename=original_filename,
        stored_pdf_path=str(pdf_dest.resolve()),
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        num_pages=num_pages,
    )
    _meta_path(contract_id).write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    if entities is not None:
        _entities_path(contract_id).write_text(json.dumps(entities, indent=2), encoding="utf-8")

    return meta


def list_contracts() -> List[Dict[str, Any]]:
    """Return a list of saved contracts with minimal metadata."""
    _ensure_dir(CONTRACTS_DIR)
    contracts: List[Dict[str, Any]] = []
    for meta_file in CONTRACTS_DIR.glob("*/meta.json"):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            contracts.append(meta)
        except Exception:
            continue
    # Sort by uploaded_at descending
    contracts.sort(key=lambda m: m.get("uploaded_at", ""), reverse=True)
    return contracts


def load_contract_meta(contract_id: str) -> Optional[Dict[str, Any]]:
    meta_file = _meta_path(contract_id)
    if not meta_file.exists():
        return None
    return json.loads(meta_file.read_text(encoding="utf-8"))


def load_contract_entities(contract_id: str) -> List[Dict[str, Any]]:
    ent_file = _entities_path(contract_id)
    if not ent_file.exists():
        return []
    try:
        return json.loads(ent_file.read_text(encoding="utf-8"))
    except Exception:
        return []


