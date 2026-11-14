"""Implement fixed-size and structure-aware chunking."""
from __future__ import annotations

from .models import Chunk, DocumentPage


def _base_metadata(page: DocumentPage, strategy: str) -> dict:
    metadata = dict(page.metadata)
    metadata.update({
        "document_id": page.document_id,
        "source": page.source,
        "page": page.page,
        "strategy": strategy,
    })
    return metadata


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be between 0 and size")

    text = (page.text or "").strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    step = size - overlap

    while start < len(text):
        end = min(start + size, len(text))
        segment = text[start:end].strip()
        if not segment:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-fixed-{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=segment,
                metadata={**_base_metadata(page, "fixed"), "chunk_index": index, "start_char": start, "end_char": end},
            )
        )
        index += 1
        if end == len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs, split oversized paragraphs and retain metadata."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    text = (page.text or "").strip()
    if not text:
        return []

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    index = 0

    def flush_buffer() -> None:
        nonlocal index
        if not buffer:
            return
        text_buffer = "\n\n".join(buffer).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-para-{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text_buffer,
                metadata={**_base_metadata(page, "paragraph"), "chunk_index": index},
            )
        )
        index += 1
        buffer.clear()

    for paragraph in paragraphs:
        if len(paragraph) <= max_size:
            if buffer_len + len(paragraph) + 2 <= max_size:
                buffer.append(paragraph)
                buffer_len += len(paragraph) + 2
            else:
                flush_buffer()
                buffer.append(paragraph)
                buffer_len = len(paragraph)
        else:
            if buffer:
                flush_buffer()
            sentences = paragraph.split(". ")
            sentence_buffer: list[str] = []
            sentence_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if sentence_len + len(sentence) + 2 <= max_size:
                    sentence_buffer.append(sentence)
                    sentence_len += len(sentence) + 2
                else:
                    if sentence_buffer:
                        chunk_text = ". ".join(sentence_buffer).strip()
                        chunks.append(
                            Chunk(
                                chunk_id=f"{page.document_id}-p{page.page}-para-{index}",
                                document_id=page.document_id,
                                source=page.source,
                                page=page.page,
                                text=chunk_text,
                                metadata={**_base_metadata(page, "paragraph"), "chunk_index": index},
                            )
                        )
                        index += 1
                    sentence_buffer = [sentence]
                    sentence_len = len(sentence)
            if sentence_buffer:
                chunk_text = ". ".join(sentence_buffer).strip()
                chunks.append(
                    Chunk(
                        chunk_id=f"{page.document_id}-p{page.page}-para-{index}",
                        document_id=page.document_id,
                        source=page.source,
                        page=page.page,
                        text=chunk_text,
                        metadata={**_base_metadata(page, "paragraph"), "chunk_index": index},
                    )
                )
                index += 1

    flush_buffer()
    return chunks
