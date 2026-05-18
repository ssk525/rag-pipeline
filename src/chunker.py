"""
Chunking Engine
Implements both semantic chunking and fixed-size chunking with overlap.
Provides comparison utilities to benchmark chunking strategies.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    # Fixed-size params
    chunk_size: int = 500  # tokens (approx 4 chars per token)
    chunk_overlap: int = 50
    # Semantic params
    breakpoint_threshold: float = 0.5
    min_chunk_size: int = 100  # minimum chars per chunk
    max_chunk_size: int = 3000  # maximum chars per chunk


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (1 token ≈ 4 characters for English)."""
    return len(text) // 4


def _semantic_split(text: str, min_size: int = 100, max_size: int = 3000) -> List[str]:
    """
    Split text at semantic boundaries (paragraphs, sections, sentence groups).
    Falls back to sentence splitting if paragraphs are too large.

    This is a lightweight semantic chunker that doesn't require an embedding
    model call per split — making it fast and free.
    """
    # Step 1: Split by double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If this single paragraph exceeds max_size, split by sentences
        if len(para) > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sentence_chunk = ""
            for sent in sentences:
                if len(sentence_chunk) + len(sent) > max_size and sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = sent
                else:
                    sentence_chunk = (
                        sentence_chunk + " " + sent if sentence_chunk else sent
                    )
            if sentence_chunk:
                current_chunk = sentence_chunk
            continue

        # If adding this paragraph would exceed max_size, start a new chunk
        if len(current_chunk) + len(para) > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = (
                current_chunk + "\n\n" + para if current_chunk else para
            )

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Merge chunks that are too small
    merged = []
    buffer = ""
    for chunk in chunks:
        if len(buffer) + len(chunk) < min_size:
            buffer = buffer + "\n\n" + chunk if buffer else chunk
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = chunk
    if buffer:
        merged.append(buffer.strip())

    return merged


def chunk_documents(
    documents: List[Document],
    config: Optional[ChunkingConfig] = None,
) -> List[Document]:
    """
    Chunk documents using the specified strategy.

    Args:
        documents: List of LangChain Documents to chunk.
        config: Chunking configuration. Defaults to semantic chunking.

    Returns:
        List of chunked Documents with preserved and enriched metadata.
    """
    if config is None:
        config = ChunkingConfig()

    if config.strategy == ChunkingStrategy.FIXED_SIZE:
        return _fixed_size_chunk(documents, config)
    else:
        return _semantic_chunk(documents, config)


def _fixed_size_chunk(
    documents: List[Document], config: ChunkingConfig
) -> List[Document]:
    """Split documents using fixed-size chunks with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size * 4,  # convert tokens to chars
        chunk_overlap=config.chunk_overlap * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for i, split_text in enumerate(splits):
            chunk_doc = Document(
                page_content=split_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_total": len(splits),
                    "chunk_strategy": "fixed_size",
                    "chunk_size_chars": len(split_text),
                    "chunk_size_tokens_est": _estimate_tokens(split_text),
                },
            )
            all_chunks.append(chunk_doc)

    return all_chunks


def _semantic_chunk(
    documents: List[Document], config: ChunkingConfig
) -> List[Document]:
    """Split documents using semantic boundaries."""
    all_chunks = []
    for doc in documents:
        splits = _semantic_split(
            doc.page_content,
            min_size=config.min_chunk_size,
            max_size=config.max_chunk_size,
        )
        for i, split_text in enumerate(splits):
            chunk_doc = Document(
                page_content=split_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_total": len(splits),
                    "chunk_strategy": "semantic",
                    "chunk_size_chars": len(split_text),
                    "chunk_size_tokens_est": _estimate_tokens(split_text),
                },
            )
            all_chunks.append(chunk_doc)

    return all_chunks


def compare_strategies(
    documents: List[Document],
) -> Dict[str, Dict]:
    """
    Run both chunking strategies and return comparative stats.
    This comparison alone signals depth to recruiters.

    Returns:
        Dictionary with stats for each strategy.
    """
    fixed_config = ChunkingConfig(
        strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=500,
        chunk_overlap=50,
    )
    semantic_config = ChunkingConfig(
        strategy=ChunkingStrategy.SEMANTIC,
        min_chunk_size=100,
        max_chunk_size=3000,
    )

    fixed_chunks = chunk_documents(documents, fixed_config)
    semantic_chunks = chunk_documents(documents, semantic_config)

    def _get_stats(chunks: List[Document]) -> Dict:
        sizes = [len(c.page_content) for c in chunks]
        token_sizes = [_estimate_tokens(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size_chars": sum(sizes) // len(sizes) if sizes else 0,
            "min_chunk_size_chars": min(sizes) if sizes else 0,
            "max_chunk_size_chars": max(sizes) if sizes else 0,
            "avg_tokens_per_chunk": sum(token_sizes) // len(token_sizes) if token_sizes else 0,
            "total_tokens": sum(token_sizes),
        }

    return {
        "fixed_size_500": _get_stats(fixed_chunks),
        "semantic": _get_stats(semantic_chunks),
        "recommendation": (
            "Semantic chunking preserves paragraph-level context, "
            "reducing mid-sentence splits. Fixed-size is more predictable "
            "for embedding models. Test both on your eval set."
        ),
    }
