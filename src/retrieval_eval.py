"""
Retrieval-quality metrics against the labelled eval set.

Each eval question names the file that answers it (`source_context`) and a ground-truth
answer. These metrics need no LLM judge, so the numbers are reproducible:

- hit@1 / hit@k: the top chunk / any of the top-k chunks comes from the labelled file
- MRR: 1 / rank of the first chunk from the labelled file (0 if absent)
- answer coverage@k: share of ground-truth content words that appear in the top-k text
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "into", "each", "than",
    "its", "their", "they", "them", "which", "where", "when", "what", "how", "does",
    "can", "not", "but", "also", "such", "used", "uses", "using", "use", "both", "all",
    "one", "two", "more", "most", "only", "other", "these", "those", "has", "have",
    "was", "were", "been", "being", "will", "would", "about", "over", "between", "like",
    "while", "then", "via", "per", "any", "out", "who", "why", "you", "your",
}


def content_words(text: str) -> set:
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


@dataclass
class QuestionScore:
    question: str
    expected_source: str
    retrieved_sources: List[str]
    hit_at_1: float
    hit_at_k: float
    reciprocal_rank: float
    answer_coverage: float


def score_question(
    question: str,
    expected_source: str,
    retrieved_sources: Sequence[str],
    retrieved_texts: Sequence[str],
    ground_truth: str,
) -> QuestionScore:
    rank = next(
        (i for i, src in enumerate(retrieved_sources, 1) if src == expected_source), None
    )
    truth = content_words(ground_truth)
    found = content_words(" ".join(retrieved_texts))
    coverage = len(truth & found) / len(truth) if truth else 0.0
    return QuestionScore(
        question=question,
        expected_source=expected_source,
        retrieved_sources=list(retrieved_sources),
        hit_at_1=1.0 if rank == 1 else 0.0,
        hit_at_k=1.0 if rank else 0.0,
        reciprocal_rank=1.0 / rank if rank else 0.0,
        answer_coverage=coverage,
    )


def aggregate(scores: Sequence[QuestionScore]) -> Dict[str, float]:
    n = len(scores) or 1
    return {
        "hit@1": round(sum(s.hit_at_1 for s in scores) / n, 4),
        "hit@k": round(sum(s.hit_at_k for s in scores) / n, 4),
        "mrr": round(sum(s.reciprocal_rank for s in scores) / n, 4),
        "answer_coverage@k": round(sum(s.answer_coverage for s in scores) / n, 4),
    }
