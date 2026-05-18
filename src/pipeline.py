"""
RAG Pipeline Orchestrator
Ties all components together: load → chunk → embed → store → retrieve → generate.
Single entry point for both the CLI and the Streamlit app.
"""

import os
import json
from typing import List, Dict, Optional

from src.document_loader import load_directory, load_single_file, get_corpus_stats
from src.chunker import (
    chunk_documents, ChunkingConfig, ChunkingStrategy, compare_strategies,
)
from src.embeddings import EmbeddingEngine
from src.vector_store import VectorStore
from src.retriever import HybridRetriever, RetrievalResult
from src.generator import RAGGenerator, GeneratedAnswer


class RAGPipeline:
    """
    End-to-end RAG pipeline orchestrator.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_directory("data/sample_docs")
        answer = pipeline.query("What is attention mechanism?")
        print(answer.answer)
    """

    def __init__(
        self,
        persist_dir: str = "chroma_db",
        chunking_strategy: str = "semantic",
        use_reranker: bool = True,
        model_name: str = "gemini-2.0-flash",
    ):
        print("🔧 Initializing RAG Pipeline...")

        # Chunking config
        strategy = (
            ChunkingStrategy.SEMANTIC
            if chunking_strategy == "semantic"
            else ChunkingStrategy.FIXED_SIZE
        )
        self.chunk_config = ChunkingConfig(strategy=strategy)

        # Components
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore(
            persist_directory=persist_dir,
            embedding_engine=self.embedding_engine,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            use_reranker=use_reranker,
        )
        self.generator = RAGGenerator(model_name=model_name)

        # State
        self._all_chunks: List = []
        self._corpus_stats: Dict = {}
        self._chat_history: List[Dict[str, str]] = []

        print("✅ Pipeline ready.\n")

    def ingest_directory(self, directory: str, extensions: Optional[List[str]] = None):
        """Load, chunk, embed, and store all documents from a directory."""
        print(f"📂 Ingesting documents from: {directory}")

        # Load
        documents = load_directory(directory, extensions=extensions)
        if not documents:
            print("⚠️  No documents found.")
            return

        self._corpus_stats = get_corpus_stats(documents)
        print(f"  📊 Corpus: {self._corpus_stats['total_documents']} pages from "
              f"{self._corpus_stats['unique_files']} files\n")

        # Chunk
        print(f"✂️  Chunking with strategy: {self.chunk_config.strategy.value}")
        chunks = chunk_documents(documents, self.chunk_config)
        self._all_chunks = chunks
        print(f"  Generated {len(chunks)} chunks\n")

        # Store in vector DB
        print("💾 Storing in ChromaDB...")
        self.vector_store.add_documents(chunks)

        # Build BM25 index for hybrid search
        print("\n📊 Building BM25 index for hybrid search...")
        self.retriever.build_bm25_index(chunks)

        print("\n🎉 Ingestion complete!")

    def ingest_files(self, file_paths: List[str]):
        """Ingest specific files (for Streamlit upload)."""
        all_docs = []
        for fp in file_paths:
            try:
                docs = load_single_file(fp)
                all_docs.extend(docs)
                print(f"  ✅ Loaded: {os.path.basename(fp)}")
            except Exception as e:
                print(f"  ⚠️  Failed: {os.path.basename(fp)} — {e}")

        if not all_docs:
            return

        chunks = chunk_documents(all_docs, self.chunk_config)
        self._all_chunks.extend(chunks)
        self.vector_store.add_documents(chunks)
        self.retriever.build_bm25_index(self._all_chunks)

    def query(
        self, question: str, top_k: int = 5, use_hybrid: bool = True,
    ) -> GeneratedAnswer:
        """
        Full RAG query: retrieve → generate with citations.

        Args:
            question: User's question.
            top_k: Number of context chunks to retrieve.
            use_hybrid: Use BM25 + vector search.

        Returns:
            GeneratedAnswer with answer, citations, and metadata.
        """
        # Retrieve
        results = self.retriever.retrieve(
            query=question, top_k=top_k, use_hybrid=use_hybrid,
        )

        # Generate
        answer = self.generator.generate(
            query=question,
            retrieval_results=results,
            chat_history=self._chat_history,
        )

        # Update chat history
        self._chat_history.append({"role": "user", "content": question})
        self._chat_history.append({"role": "model", "content": answer.answer})

        # Keep history manageable
        if len(self._chat_history) > 10:
            self._chat_history = self._chat_history[-10:]

        return answer

    def get_retrieval_only(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve without generating — useful for debugging."""
        return self.retriever.retrieve(query=question, top_k=top_k)

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        store_stats = self.vector_store.get_collection_stats()
        return {
            "corpus": self._corpus_stats,
            "vector_store": store_stats,
            "chunking_strategy": self.chunk_config.strategy.value,
            "model": self.generator.model_name,
            "chat_history_length": len(self._chat_history),
        }

    def clear_history(self):
        """Clear chat history."""
        self._chat_history = []

    def compare_chunking(self) -> Dict:
        """Run chunking strategy comparison (for README)."""
        if not self._all_chunks:
            return {"error": "No documents ingested yet."}
        # Reload raw documents from the chunks' sources
        sources = set()
        for c in self._all_chunks:
            sources.add(c.metadata.get("source", ""))
        all_docs = []
        for s in sources:
            if os.path.exists(s):
                try:
                    all_docs.extend(load_single_file(s))
                except Exception:
                    pass
        if all_docs:
            return compare_strategies(all_docs)
        return {"error": "Could not reload source documents for comparison."}
