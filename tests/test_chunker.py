from langchain_core.documents import Document

from src.chunker import ChunkingConfig, ChunkingStrategy, chunk_documents, compare_strategies


def _doc(text: str, name: str = "sample.md") -> Document:
    return Document(page_content=text, metadata={"filename": name, "source": name})


def test_semantic_keeps_paragraphs_together():
    text = (
        "First paragraph about attention and queries.\n\n"
        "Second paragraph about keys and values.\n\n"
        "Third paragraph about softmax scaling."
    )
    chunks = chunk_documents([_doc(text)], ChunkingConfig(strategy=ChunkingStrategy.SEMANTIC))
    assert chunks
    assert all(c.metadata["chunk_strategy"] == "semantic" for c in chunks)
    assert all(c.metadata["filename"] == "sample.md" for c in chunks)


def test_fixed_size_splits_long_text():
    text = "word " * 800
    chunks = chunk_documents(
        [_doc(text)],
        ChunkingConfig(strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=50, chunk_overlap=5),
    )
    assert len(chunks) > 1
    assert all(c.metadata["chunk_strategy"] == "fixed_size" for c in chunks)


def test_compare_strategies_returns_both_counts():
    docs = [_doc("Section one.\n\nSection two about transformers.\n\nSection three.")]
    stats = compare_strategies(docs)
    assert stats["fixed_size_500"]["total_chunks"] >= 1
    assert stats["semantic"]["total_chunks"] >= 1
