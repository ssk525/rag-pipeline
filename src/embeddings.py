"""
Embedding Layer
Generates embeddings using Google Gemini's embedding model.
Supports batch processing with rate limiting for the free tier.
"""

import re
import time
from typing import Callable, List, Optional, TypeVar
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Gemini embedding model. The free tier allows 100 embed requests per minute,
# and every text in a batch counts as one request.
DEFAULT_MODEL = "models/gemini-embedding-001"

T = TypeVar("T")


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _retry_delay(exc: Exception, default: float) -> float:
    match = re.search(r"retry in ([0-9.]+)s", str(exc))
    return float(match.group(1)) + 1.0 if match else default


def call_with_backoff(fn: Callable[[], T], attempts: int = 6, base_delay: float = 10.0) -> T:
    """Retry fn on rate-limit errors, honouring the server's retry hint when given."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == attempts - 1:
                raise
            wait = _retry_delay(exc, base_delay * (attempt + 1))
            print(f"  ⏳ Rate limited. Waiting {wait:.0f}s (attempt {attempt + 1}/{attempts})...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


class EmbeddingEngine:
    """
    Wrapper around Google Gemini embeddings with batch support
    and rate limiting for free tier compliance.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 50,
        requests_per_minute: int = 90,  # stay under the free tier's 100/min
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

            embeddings = call_with_backoff(lambda: self.embeddings.embed_documents(batch))
            all_embeddings.extend(embeddings)
            print(
                f"  📐 Embedded batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )

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
        time.sleep(self._delay)
        return call_with_backoff(lambda: self.query_embeddings.embed_query(query))

    def get_langchain_embeddings(self) -> Embeddings:
        """
        LangChain embedding object for ChromaDB. Documents go through the batched,
        throttled retrieval_document path; queries use the retrieval_query task type.
        """
        return _EngineEmbeddings(self)

    @property
    def dimension(self) -> int:
        """Return embedding dimension (gemini-embedding-001 = 3072)."""
        return 3072


class _EngineEmbeddings(Embeddings):
    """Adapter so ChromaDB uses EmbeddingEngine's routing, throttling, and retries."""

    def __init__(self, engine: "EmbeddingEngine"):
        self.engine = engine

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.engine.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.engine.embed_query(text)
