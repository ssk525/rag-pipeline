"""
Embedding Layer
Generates embeddings using Google Gemini's embedding model.
Supports batch processing with rate limiting for the free tier.
"""

import time
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Gemini embedding model — free tier: 1,500 requests/min
DEFAULT_MODEL = "models/gemini-embedding-001"


class EmbeddingEngine:
    """
    Wrapper around Google Gemini embeddings with batch support
    and rate limiting for free tier compliance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 100,
        requests_per_minute: int = 1400,  # stay under 1500 limit
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self._delay = 60.0 / requests_per_minute

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type="retrieval_document",
        )

        # For query embeddings, we use retrieval_query task type
        self.query_embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            task_type="retrieval_query",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts with batching and rate limiting.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            try:
                embeddings = self.embeddings.embed_documents(batch)
                all_embeddings.extend(embeddings)
                print(
                    f"  📐 Embedded batch {batch_num}/{total_batches} "
                    f"({len(batch)} texts)"
                )
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"  ⏳ Rate limited. Waiting 60s...")
                    time.sleep(60)
                    embeddings = self.embeddings.embed_documents(batch)
                    all_embeddings.extend(embeddings)
                else:
                    raise

            # Rate limiting delay between batches
            if batch_num < total_batches:
                time.sleep(self._delay * len(batch))

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            query: The search query string.

        Returns:
            Embedding vector for the query.
        """
        return self.query_embeddings.embed_query(query)

    def get_langchain_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Return the LangChain-compatible embedding object for ChromaDB."""
        return self.embeddings

    @property
    def dimension(self) -> int:
        """Return embedding dimension (gemini-embedding-001 = 3072)."""
        return 3072
