"""
Generator Module
LLM-powered answer generation with inline citations.
Forces the model to ONLY answer from retrieved context.
"""

import os
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.retriever import RetrievalResult

load_dotenv()


@dataclass
class GeneratedAnswer:
    """Structured answer with citations and metadata."""
    answer: str
    citations: List[Dict[str, str]]
    context_used: List[str]
    query: str
    model: str
    latency_ms: float
    refused: bool = False


SYSTEM_PROMPT = """You are a precise research assistant. Answer ONLY from the provided context.

RULES:
1. Use ONLY information from the provided context passages.
2. Cite every claim inline: [Source: filename, Page X]
3. If context is insufficient, say: "I cannot answer this question based on the available documents."
4. Do NOT hallucinate information beyond the context.
5. If partial info is available, answer what you can and state what's missing.
6. Keep answers concise and factual. Use bullet points for multi-part answers.
"""


def _format_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results into context block for LLM."""
    parts = []
    for i, r in enumerate(results, 1):
        page = f", Page {r.page_number}" if r.page_number else ""
        parts.append(
            f"--- Passage {i} ---\n"
            f"Source: {r.filename}{page}\n"
            f"Content:\n{r.content}\n"
        )
    return "\n".join(parts)


def _extract_citations(answer: str) -> List[Dict[str, str]]:
    """Extract [Source: filename, Page X] citations from answer text."""
    pattern = r"\[Source:\s*([^,\]]+)(?:,\s*Page\s*(\d+))?\]"
    matches = re.findall(pattern, answer)
    seen, citations = set(), []
    for filename, page in matches:
        key = f"{filename.strip()}_{page}"
        if key not in seen:
            c = {"filename": filename.strip()}
            if page:
                c["page"] = page
            citations.append(c)
            seen.add(key)
    return citations


class RAGGenerator:
    """LLM answer generator with citation enforcement using Google Gemini."""

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.1, max_tokens=1024):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Set it in .env file.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, query: str, retrieval_results: List[RetrievalResult],
                 chat_history: Optional[List[Dict[str, str]]] = None) -> GeneratedAnswer:
        """Generate an answer from retrieved context with inline citations."""
        if not retrieval_results:
            return GeneratedAnswer(
                answer="I cannot answer — no relevant documents found in the knowledge base. "
                       "Please upload documents first.",
                citations=[], context_used=[], query=query,
                model=self.model_name, latency_ms=0, refused=True,
            )

        context_block = _format_context(retrieval_results)
        user_prompt = (
            f"CONTEXT PASSAGES:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer using ONLY the context. Cite with [Source: filename, Page X]."
        )

        # Build conversation contents
        contents = []
        if chat_history:
            for turn in chat_history[-5:]:
                role = "user" if turn["role"] == "user" else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])])
                )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        )

        start = time.time()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            answer_text = response.text
        except Exception as e:
            answer_text = f"Error generating answer: {e}"
        latency_ms = (time.time() - start) * 1000

        citations = _extract_citations(answer_text)
        refused = any(p in answer_text.lower() for p in [
            "cannot answer", "don't have enough", "no relevant information"
        ])

        return GeneratedAnswer(
            answer=answer_text, citations=citations,
            context_used=[r.content[:200] + "..." for r in retrieval_results],
            query=query, model=self.model_name,
            latency_ms=round(latency_ms, 1), refused=refused,
        )
