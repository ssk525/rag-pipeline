"""
Generator Module
LLM-powered answer generation with inline citations.
Forces the model to ONLY answer from retrieved context.
Supports Google Gemini (cloud) or Ollama (local).
"""

import json
import os
import time
import re
import urllib.request
from typing import List, Dict, Optional
from dataclasses import dataclass

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


def _use_ollama_preferred() -> bool:
    return os.getenv("USE_OLLAMA", "").strip().lower() in {"1", "true", "yes"}


def _ollama_available(model: str = "gemma4:e4b") -> bool:
    """Return True if Ollama is running and has the requested model."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as resp:
            data = json.loads(resp.read().decode())
        models = [m.get("name", "") for m in data.get("models", [])]
        return any(model in name or name.startswith(model) for name in models)
    except Exception:
        return False


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


def _is_refusal(answer_text: str) -> bool:
    return any(p in answer_text.lower() for p in [
        "cannot answer", "don't have enough", "no relevant information"
    ])


class RAGGenerator:
    """LLM answer generator with citation enforcement (Gemini or Ollama)."""

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.1, max_tokens=1024):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_ollama = False
        self.llm = None
        self.client = None

        ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
        if _use_ollama_preferred() and _ollama_available(ollama_model):
            from langchain_ollama import ChatOllama
            self.use_ollama = True
            self.model_name = ollama_model
            self.llm = ChatOllama(
                model=ollama_model,
                temperature=temperature,
                num_predict=max_tokens,
            )
            return

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "No LLM configured. Set USE_OLLAMA=1 with Ollama running, "
                "or add GOOGLE_API_KEY to your .env file."
            )

        from google import genai
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

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

        start = time.time()
        if self.use_ollama:
            answer_text = self._generate_ollama(user_prompt, chat_history)
        else:
            answer_text = self._generate_gemini(user_prompt, chat_history)
        latency_ms = (time.time() - start) * 1000

        citations = _extract_citations(answer_text)
        return GeneratedAnswer(
            answer=answer_text, citations=citations,
            context_used=[r.content[:200] + "..." for r in retrieval_results],
            query=query, model=self.model_name,
            latency_ms=round(latency_ms, 1), refused=_is_refusal(answer_text),
        )

    def _generate_ollama(
        self, user_prompt: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if chat_history:
            for turn in chat_history[-5:]:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                else:
                    messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=user_prompt))

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating answer with Ollama: {e}"

    def _generate_gemini(
        self, user_prompt: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        from google.genai import types

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
            return response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                return (
                    "The Gemini API quota is exhausted. Set USE_OLLAMA=1 in .env "
                    "to use your local Ollama model instead."
                )
            if "401" in err or "API_KEY" in err or "invalid" in err.lower():
                return "Invalid or missing GOOGLE_API_KEY. Add a valid key to your .env file."
            return f"Error generating answer: {e}"
