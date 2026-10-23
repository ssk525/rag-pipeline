from langchain_core.documents import Document

from src.retriever import HybridRetriever


def _doc(text: str, name: str, digest: str) -> Document:
    return Document(
        page_content=text,
        metadata={"filename": name, "source": name, "doc_hash": digest, "chunk_index": 0},
    )


def test_bm25_prefers_the_document_with_the_rare_token():
    retriever = HybridRetriever(vector_store=None, use_reranker=False)
    retriever.build_bm25_index(
        [
            _doc("transformers use self attention in every layer", "a.md", "h1"),
            _doc("bm25 ranks rare keywords such as zxqwvtoken", "b.md", "h2"),
            _doc("another document about retrieval only", "c.md", "h3"),
        ]
    )
    hits = retriever._bm25_search("zxqwvtoken", k=2)
    assert hits
    assert hits[0][0].metadata["filename"] == "b.md"


def test_rrf_boosts_a_document_that_ranks_in_both_lists():
    retriever = HybridRetriever(vector_store=None, use_reranker=False)
    shared = _doc("shared chunk about rag", "shared.md", "s1")
    only_vec = _doc("vector only", "vec.md", "v1")
    only_bm = _doc("bm25 only", "bm.md", "b1")
    merged = retriever._reciprocal_rank_fusion(
        [(only_vec, 0.9), (shared, 0.8)],
        [(only_bm, 0.9), (shared, 0.7)],
        k=60,
    )
    assert merged[0][0].metadata["filename"] == "shared.md"
    # both lists contribute 1/(60+rank+1)
    assert merged[0][1] == (1.0 / 62) + (1.0 / 62)
