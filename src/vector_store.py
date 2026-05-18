"""
Vector Store Module
ChromaDB-backed vector store with metadata filtering support.
Handles ingestion, persistence, and similarity search.
"""

import os
from typing import List, Dict, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.embeddings import EmbeddingEngine


# Default persistence directory
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "rag_knowledge_base"


class VectorStore:
    """
    ChromaDB vector store with metadata filtering.
    Supports persistent storage and incremental updates.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_engine = embedding_engine or EmbeddingEngine()

        os.makedirs(persist_directory, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        self.langchain_store = Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=self.embedding_engine.get_langchain_embeddings(),
        )

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
    ) -> int:
        """
        Add documents to the vector store.
        Deduplicates by doc_hash metadata if present.

        Args:
            documents: List of chunked Documents to store.
            batch_size: Number of documents per batch.

        Returns:
            Number of documents added.
        """
        # Get existing hashes to avoid duplicates
        existing_hashes = set()
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            existing = collection.get(include=["metadatas"])
            if existing and existing["metadatas"]:
                existing_hashes = {
                    m.get("doc_hash", "")
                    for m in existing["metadatas"]
                    if m.get("doc_hash")
                }
        except Exception:
            pass

        # Filter out duplicates
        new_docs = [
            doc
            for doc in documents
            if doc.metadata.get("doc_hash", "") not in existing_hashes
        ]

        if not new_docs:
            print("  ℹ️  No new documents to add (all duplicates).")
            return 0

        # Add in batches
        added = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            self.langchain_store.add_documents(batch)
            added += len(batch)
            print(
                f"  💾 Stored batch {(i // batch_size) + 1}/"
                f"{(len(new_docs) + batch_size - 1) // batch_size} "
                f"({len(batch)} chunks)"
            )

        print(f"\n✅ Added {added} new chunks to vector store.")
        return added

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with optional metadata filtering.

        Args:
            query: Search query string.
            k: Number of results to return.
            metadata_filter: ChromaDB-style metadata filter dict.
                Example: {"file_type": "pdf"} or {"source": {"$contains": "report"}}

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        if metadata_filter:
            results = self.langchain_store.similarity_search_with_relevance_scores(
                query, k=k, filter=metadata_filter
            )
        else:
            results = self.langchain_store.similarity_search_with_relevance_scores(
                query, k=k
            )

        return results

    def get_collection_stats(self) -> Dict:
        """Get statistics about the stored collection."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()

            # Get metadata distribution
            all_data = collection.get(include=["metadatas"])
            sources = set()
            strategies = {}
            if all_data and all_data["metadatas"]:
                for m in all_data["metadatas"]:
                    sources.add(m.get("filename", "unknown"))
                    strat = m.get("chunk_strategy", "unknown")
                    strategies[strat] = strategies.get(strat, 0) + 1

            return {
                "total_chunks": count,
                "unique_sources": len(sources),
                "source_files": sorted(sources),
                "chunk_strategies": strategies,
            }
        except Exception:
            return {"total_chunks": 0, "error": "Collection not found"}

    def clear(self):
        """Delete all documents from the collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            # Recreate empty collection
            self.langchain_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_engine.get_langchain_embeddings(),
            )
            print("🗑️  Vector store cleared.")
        except Exception as e:
            print(f"⚠️  Could not clear store: {e}")
