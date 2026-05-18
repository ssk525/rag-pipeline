"""
Hybrid Retriever Module
Combines BM25 (sparse) + Vector (dense) retrieval with Cohere Reranking.
This hybrid approach is the key differentiator for production RAG.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Structured retrieval result with source tracking."""
    content: str
    source: str
    filename: str
    page_number: Optional[int]
    chunk_index: int
    relevance_score: float
    retrieval_method: str  # "vector", "bm25", or "hybrid"


class HybridRetriever:
    """
    Production-grade retriever combining:
    1. Dense retrieval (vector similarity via ChromaDB)
    2. Sparse retrieval (BM25 keyword matching)
    3. Reciprocal Rank Fusion (RRF) to merge results
    4. Cohere Reranker for final ranking

    Why hybrid? Vector search captures semantic meaning but misses exact
    keywords. BM25 catches exact terms but misses paraphrases. Together
    they cover both failure modes.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        use_reranker: bool = True,
        cohere_api_key: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.bm25_index = None
        self.bm25_docs = []

        # Initialize Cohere reranker if available
        self.reranker = None
        if use_reranker:
            api_key = cohere_api_key or os.getenv("COHERE_API_KEY")
            if api_key:
                try:
                    import cohere
                    self.reranker = cohere.ClientV2(api_key=api_key)
                    print("  ✅ Cohere Reranker initialized")
                except ImportError:
                    print("  ⚠️  cohere package not installed. Skipping reranker.")
            else:
                print(
                    "  ⚠️  No COHERE_API_KEY found. Reranking disabled. "
                    "Set it in .env for better retrieval quality."
                )

    def build_bm25_index(self, documents: List[Document]):
        """
        Build BM25 index from document chunks.
        Call this after loading documents into the vector store.

        Args:
            documents: The same chunked documents stored in ChromaDB.
        """
        self.bm25_docs = documents
        tokenized_corpus = [
            doc.page_content.lower().split() for doc in documents
        ]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        print(f"  📊 BM25 index built over {len(documents)} chunks")

    def _vector_search(
        self, query: str, k: int = 20, metadata_filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """Dense retrieval via ChromaDB."""
        return self.vector_store.similarity_search(
            query, k=k, metadata_filter=metadata_filter
        )

    def _bm25_search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """Sparse retrieval via BM25."""
        if self.bm25_index is None or not self.bm25_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.bm25_docs[idx], float(scores[idx])))

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """
        Merge results from multiple retrieval methods using RRF.
        RRF score = Σ 1 / (k + rank_i) for each retrieval method.
        """
        doc_scores: Dict[str, Tuple[Document, float]] = {}

        for rank, (doc, _score) in enumerate(vector_results):
            doc_key = doc.metadata.get("doc_hash", doc.page_content[:100])
            rrf_score = 1.0 / (k + rank + 1)
            if doc_key in doc_scores:
                doc_scores[doc_key] = (
                    doc,
                    doc_scores[doc_key][1] + rrf_score,
                )
            else:
                doc_scores[doc_key] = (doc, rrf_score)

        for rank, (doc, _score) in enumerate(bm25_results):
            doc_key = doc.metadata.get("doc_hash", doc.page_content[:100])
            rrf_score = 1.0 / (k + rank + 1)
            if doc_key in doc_scores:
                doc_scores[doc_key] = (
                    doc,
                    doc_scores[doc_key][1] + rrf_score,
                )
            else:
                doc_scores[doc_key] = (doc, rrf_score)

        # Sort by combined RRF score
        sorted_results = sorted(
            doc_scores.values(), key=lambda x: x[1], reverse=True
        )
        return sorted_results

    def _rerank(
        self, query: str, documents: List[Tuple[Document, float]], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank using Cohere Rerank API."""
        if not self.reranker or not documents:
            return documents[:top_k]

        texts = [doc.page_content for doc, _ in documents]

        try:
            response = self.reranker.rerank(
                model="rerank-v3.5",
                query=query,
                documents=texts,
                top_n=top_k,
            )

            reranked = []
            for result in response.results:
                original_doc = documents[result.index][0]
                reranked.append((original_doc, result.relevance_score))

            return reranked

        except Exception as e:
            print(f"  ⚠️  Reranking failed: {e}. Using RRF scores instead.")
            return documents[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        metadata_filter: Optional[Dict] = None,
        use_hybrid: bool = True,
    ) -> List[RetrievalResult]:
        """
        Full retrieval pipeline: Vector + BM25 → RRF → Rerank.

        Args:
            query: User's question.
            top_k: Number of final results to return.
            initial_k: Number of candidates from each retriever.
            metadata_filter: Optional ChromaDB metadata filter.
            use_hybrid: If False, use only vector search.

        Returns:
            List of RetrievalResult objects, sorted by relevance.
        """
        # Step 1: Vector search
        vector_results = self._vector_search(
            query, k=initial_k, metadata_filter=metadata_filter
        )

        # Step 2: BM25 search (if hybrid enabled and index exists)
        bm25_results = []
        if use_hybrid and self.bm25_index is not None:
            bm25_results = self._bm25_search(query, k=initial_k)

        # Step 3: Merge with RRF (or just use vector results)
        if bm25_results:
            merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
            method = "hybrid"
        else:
            merged = vector_results
            method = "vector"

        # Step 4: Rerank
        if self.reranker and len(merged) > top_k:
            final_results = self._rerank(query, merged, top_k=top_k)
            method += "+reranked"
        else:
            final_results = merged[:top_k]

        # Step 5: Convert to structured results
        structured_results = []
        for doc, score in final_results:
            structured_results.append(
                RetrievalResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    filename=doc.metadata.get("filename", "unknown"),
                    page_number=doc.metadata.get("page_number"),
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    relevance_score=float(score),
                    retrieval_method=method,
                )
            )

        return structured_results
