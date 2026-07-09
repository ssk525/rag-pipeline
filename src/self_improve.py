"""
Self-improvement loop for the RAG pipeline.

Runs evaluation, tries retrieval configurations, and persists the best settings.
This closes the loop between evaluation metrics and runtime retrieval behavior.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from src.config_store import PipelineConfig, append_improvement_record, load_config, save_config
from src.evaluator import load_eval_set, run_ragas_evaluation
from src.logging_config import get_logger
from src.pipeline import RAGPipeline

logger = get_logger("rag_pipeline.self_improve")

EVAL_PATH = "eval/test_questions.json"


def _score_from_results(results: Dict[str, Any]) -> float:
    agg = results.get("aggregate", {})
    values = []
    for key in ("faithfulness", "answer_relevancy", "context_precision"):
        val = agg.get(key)
        if isinstance(val, (int, float)):
            values.append(float(val))
    return sum(values) / len(values) if values else 0.0


def _evaluate_config(
    pipeline: RAGPipeline,
    questions: List[str],
    ground_truths: List[str],
    top_k: int,
    use_hybrid: bool,
) -> Tuple[float, Dict[str, Any]]:
    answers: List[str] = []
    contexts: List[List[str]] = []

    for question in questions:
        retrieval = pipeline.retriever.retrieve(
            query=question, top_k=top_k, use_hybrid=use_hybrid
        )
        answer = pipeline.generator.generate(
            query=question,
            retrieval_results=retrieval,
            chat_history=None,
        )
        answers.append(answer.answer)
        contexts.append([r.content for r in retrieval])
        pipeline.clear_history()
        time.sleep(0.5)

    results = run_ragas_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )
    return _score_from_results(results), results


def run_self_improvement(
    eval_limit: int = 5,
    sample_dir: str = "data/sample_docs",
) -> Dict[str, Any]:
    """
    Evaluate candidate retrieval settings and persist the best configuration.
    """
    logger.info("Starting self-improvement cycle")
    current = load_config()
    eval_questions = load_eval_set(EVAL_PATH)[:eval_limit]
    if not eval_questions:
        raise ValueError("No evaluation questions found.")

    questions = [q.question for q in eval_questions]
    ground_truths = [q.ground_truth for q in eval_questions]

    pipeline = RAGPipeline(
        chunking_strategy=current.chunking_strategy,
        use_reranker=current.use_reranker,
    )

    if pipeline.get_stats().get("vector_store", {}).get("total_chunks", 0) == 0:
        pipeline.ingest_directory(sample_dir)

    candidates = [
        {"top_k": 3, "use_hybrid": False},
        {"top_k": 5, "use_hybrid": True},
        {"top_k": 5, "use_hybrid": False},
        {"top_k": 7, "use_hybrid": True},
    ]

    best_score = -1.0
    best_candidate = current.to_dict()
    best_results: Dict[str, Any] = {}

    for candidate in candidates:
        logger.info("Evaluating candidate config: %s", candidate)
        score, results = _evaluate_config(
            pipeline,
            questions,
            ground_truths,
            top_k=candidate["top_k"],
            use_hybrid=candidate["use_hybrid"],
        )
        logger.info("Candidate score: %.4f", score)
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_results = results

    improved = PipelineConfig(
        top_k=best_candidate["top_k"],
        use_hybrid=best_candidate["use_hybrid"],
        use_reranker=current.use_reranker,
        chunking_strategy=current.chunking_strategy,
        last_eval_score=round(best_score, 4),
        notes="Updated by self-improvement loop from eval metrics",
    )
    save_config(improved)

    record = {
        "score": round(best_score, 4),
        "selected_config": improved.to_dict(),
        "aggregate_metrics": best_results.get("aggregate", {}),
        "candidates_tested": len(candidates),
        "eval_questions": len(questions),
    }
    append_improvement_record(record)
    logger.info("Self-improvement complete. Best score=%.4f", best_score)
    return record
