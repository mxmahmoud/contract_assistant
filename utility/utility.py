import io, re, hashlib
from pathlib import Path
from pypdf import PdfReader, PdfWriter

def get_hash_of_file(file_bytes: bytes) -> str:
    """Creates a unique ID for the contract based on its content."""
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def normalize_whitespace_preserve_newlines(text: str) -> str:
    return "\n".join(re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines())

def save_flattened_pdf(pdf: bytes | Path, target_path: Path):
    src = pdf if isinstance(pdf, Path) else io.BytesIO(pdf)
    reader = PdfReader(src)
    writer = PdfWriter()
    writer.append(reader)

    # bake pdfs containing annotations
    fields = reader.get_fields() or {}
    values = {}
    for name, fobj in fields.items():
        v = fobj.get("/V")
        if v is not None:
            try:
                v = v.get_object()
            except Exception:
                pass
            values[name] = v

    if values:
        writer.update_page_form_field_values(
            page=None, fields=values, auto_regenerate=False, flatten=True
        )
        writer.remove_annotations(subtypes="/Widget")

    writer.write(target_path)
