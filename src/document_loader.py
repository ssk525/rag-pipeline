"""
Document Loader Module
Handles loading and preprocessing of PDF, TXT, and Markdown documents.
Supports both batch loading from a directory and single file uploads.
"""

import os
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


@dataclass
class DocumentMetadata:
    """Structured metadata for each loaded document."""
    source: str
    filename: str
    file_type: str
    page_number: Optional[int] = None
    doc_hash: str = ""
    loaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_pages: Optional[int] = None
    char_count: int = 0


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,  # TextLoader handles markdown raw text fine
    ".markdown": TextLoader,
}


def _compute_hash(content: str) -> str:
    """Compute SHA-256 hash of document content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _clean_text(text: str) -> str:
    """Clean extracted text: remove excessive whitespace, fix encoding artifacts."""
    # Replace multiple newlines with double newline
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)
    # Remove null bytes and other control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def load_single_file(filepath: str) -> List[Document]:
    """
    Load a single file and return a list of LangChain Documents
    with enriched metadata.

    Args:
        filepath: Absolute or relative path to the file.

    Returns:
        List of Document objects with cleaned text and metadata.

    Raises:
        ValueError: If file type is not supported.
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    loader_cls = LOADER_MAP[ext]
    loader = loader_cls(filepath)
    raw_docs = loader.load()

    processed_docs = []
    for i, doc in enumerate(raw_docs):
        cleaned_content = _clean_text(doc.page_content)
        if len(cleaned_content) < 10:
            continue  # Skip near-empty pages

        metadata = DocumentMetadata(
            source=filepath,
            filename=os.path.basename(filepath),
            file_type=ext.lstrip("."),
            page_number=doc.metadata.get("page", i + 1),
            doc_hash=_compute_hash(cleaned_content),
            total_pages=len(raw_docs),
            char_count=len(cleaned_content),
        )

        processed_docs.append(
            Document(
                page_content=cleaned_content,
                metadata=metadata.__dict__,
            )
        )

    return processed_docs


def load_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[Document]:
    """
    Load all supported documents from a directory.

    Args:
        directory: Path to the directory.
        extensions: List of extensions to filter (e.g., [".pdf"]).
                    Defaults to all supported types.
        recursive: Whether to search subdirectories.

    Returns:
        List of Document objects from all files.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    allowed_exts = set(extensions) if extensions else SUPPORTED_EXTENSIONS
    all_docs = []
    loaded_files = 0
    skipped_files = 0

    for root, _, files in os.walk(directory):
        if not recursive and root != directory:
            continue

        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in allowed_exts:
                continue

            filepath = os.path.join(root, fname)
            try:
                docs = load_single_file(filepath)
                all_docs.extend(docs)
                loaded_files += 1
                print(f"  ✅ Loaded: {fname} ({len(docs)} pages/sections)")
            except Exception as e:
                skipped_files += 1
                print(f"  ⚠️  Skipped: {fname} — {e}")

    print(
        f"\n📄 Loaded {loaded_files} files ({len(all_docs)} document chunks). "
        f"Skipped {skipped_files} files."
    )
    return all_docs


def get_corpus_stats(documents: List[Document]) -> Dict:
    """Get summary statistics about the loaded corpus."""
    if not documents:
        return {"total_documents": 0}

    total_chars = sum(len(d.page_content) for d in documents)
    unique_sources = set(d.metadata.get("filename", "") for d in documents)
    file_types = {}
    for d in documents:
        ft = d.metadata.get("file_type", "unknown")
        file_types[ft] = file_types.get(ft, 0) + 1

    return {
        "total_documents": len(documents),
        "total_characters": total_chars,
        "avg_chars_per_doc": total_chars // len(documents),
        "unique_files": len(unique_sources),
        "file_types": file_types,
        "source_files": sorted(unique_sources),
    }
