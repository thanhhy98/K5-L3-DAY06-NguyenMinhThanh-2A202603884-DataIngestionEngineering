"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while keeping useful structure."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_text_into_pages(raw_text: str) -> list[str]:
    """Split a text file into page-like blocks using a generous page marker syntax."""
    marker_pattern = re.compile(r"(?im)^\s*(?:\[)?PAGE\s+(\d+)\s*(?:\])?\s*$")
    lines = raw_text.splitlines()
    pages: list[str] = []
    current: list[str] = []

    for line in lines:
        if marker_pattern.match(line):
            if current or pages:
                pages.append("\n".join(current).strip())
            current = []
            continue
        current.append(line)

    if current or not pages:
        pages.append("\n".join(current).strip())

    cleaned = [normalize_text(page) for page in pages if normalize_text(page)]
    return cleaned or [normalize_text(raw_text)]


def _parse_page_metadata(page_text: str, fallback_page: int) -> dict:
    metadata = {"page_type": "text_block"}
    title_match = re.search(r"(?im)^Title:\s*(.+)$", page_text)
    section_match = re.search(r"(?im)^Section:\s*(.+)$", page_text)
    audience_match = re.search(r"(?im)^Audience:\s*(.+)$", page_text)
    year_match = re.search(r"(?im)^Year:\s*(.+)$", page_text)
    if title_match:
        metadata["title"] = title_match.group(1).strip()
    if section_match:
        metadata["section"] = section_match.group(1).strip()
    if audience_match:
        metadata["audience"] = audience_match.group(1).strip()
    if year_match:
        metadata["year"] = year_match.group(1).strip()
    metadata["page"] = fallback_page
    return metadata


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page and keep its page number and source."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    text_pages: list[DocumentPage] = []
    if path.suffix.lower() == ".pdf":
        pdf = PdfReader(str(path))
        for page_number, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_text() or ""
            text_pages.append(
                DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=page_number,
                    text=normalize_text(extracted),
                    metadata={"source_file": str(path), "page_type": "pdf_page"},
                )
            )
        return text_pages

    text = path.read_text(encoding="utf-8", errors="replace")
    text = normalize_text(text)
    candidate_pages = _split_text_into_pages(text)

    if len(candidate_pages) == 1 and "[PAGE" not in text and "PAGE " not in text:
        candidate_pages = [text]

    for page_index, page_text in enumerate(candidate_pages, start=1):
        normalized_page = normalize_text(page_text)
        metadata = _parse_page_metadata(normalized_page, page_index)
        metadata["source_file"] = str(path)
        text_pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_index,
                text=normalized_page,
                metadata=metadata,
            )
        )

    if not text_pages:
        text_pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=1,
                text="",
                metadata={"source_file": str(path), "page_type": "empty"},
            )
        )
    return text_pages
