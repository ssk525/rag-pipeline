"""
RAG Evaluation Module
Runs Ragas evaluation suite: faithfulness, answer relevancy, context precision.
Publishes results as a table for the README.
"""

import json
import os
import time
from typing import List, Dict, Optional
from dataclasses import dataclass

import pandas as pd


@dataclass
class EvalQuestion:
    """A single evaluation question with ground truth."""
    question: str
    ground_truth: str
    source_context: Optional[str] = None


def load_eval_set(filepath: str) -> List[EvalQuestion]:
    """Load evaluation questions from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [EvalQuestion(**q) for q in data]


def save_eval_set(questions: List[EvalQuestion], filepath: str):
    """Save evaluation questions to a JSON file."""
    data = [
        {"question": q.question, "ground_truth": q.ground_truth,
         "source_context": q.source_context}
        for q in questions
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 Saved {len(questions)} eval questions to {filepath}")


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict:
    """
    Run Ragas evaluation suite.

    Args:
        questions: List of questions asked.
        answers: List of generated answers.
        contexts: List of context chunks used for each answer.
        ground_truths: List of expected correct answers.

    Returns:
        Dictionary with per-question and aggregate scores.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return {
            "aggregate": {
                "faithfulness": round(result["faithfulness"], 4),
                "answer_relevancy": round(result["answer_relevancy"], 4),
                "context_precision": round(result["context_precision"], 4),
                "context_recall": round(result["context_recall"], 4),
            },
            "per_question": result.to_pandas().to_dict("records")
            if hasattr(result, "to_pandas") else [],
            "num_questions": len(questions),
        }

    except ImportError:
        print("  ⚠️  Ragas not installed. Running manual evaluation instead.")
        return _manual_evaluation(questions, answers, contexts, ground_truths)


def _manual_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
) -> Dict:
    """
    Fallback manual evaluation when Ragas is unavailable.
    Computes simple overlap-based metrics.
    """
    results = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        # Simple word overlap as proxy for faithfulness
        answer_words = set(a.lower().split())
        context_words = set()
        for c in ctx:
            context_words.update(c.lower().split())
        gt_words = set(gt.lower().split())

        # Faithfulness: % of answer words found in context
        if answer_words:
            faith = len(answer_words & context_words) / len(answer_words)
        else:
            faith = 0.0

        # Answer relevancy: % of ground truth words found in answer
        if gt_words:
            relevancy = len(gt_words & answer_words) / len(gt_words)
        else:
            relevancy = 0.0

        # Context precision: % of context words relevant to ground truth
        if context_words:
            precision = len(context_words & gt_words) / len(context_words)
        else:
            precision = 0.0

        results.append({
            "question": q, "faithfulness": round(faith, 4),
            "answer_relevancy": round(relevancy, 4),
            "context_precision": round(precision, 4),
        })

    # Aggregate
    n = len(results) if results else 1
    agg = {
        "faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 4),
        "context_precision": round(sum(r["context_precision"] for r in results) / n, 4),
        "context_recall": "N/A (manual eval)",
    }

    return {"aggregate": agg, "per_question": results, "num_questions": len(results)}


def save_eval_results(results: Dict, output_dir: str = "eval/results"):
    """Save evaluation results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Save full results as JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save aggregate as CSV for README table
    csv_path = os.path.join(output_dir, "eval_summary.csv")
    agg = results.get("aggregate", {})
    df = pd.DataFrame([agg])
    df.to_csv(csv_path, index=False)

    # Save per-question details
    if results.get("per_question"):
        detail_path = os.path.join(output_dir, "eval_per_question.csv")
        pd.DataFrame(results["per_question"]).to_csv(detail_path, index=False)

    print(f"\n📊 Evaluation Results:")
    print(f"  Faithfulness:       {agg.get('faithfulness', 'N/A')}")
    print(f"  Answer Relevancy:   {agg.get('answer_relevancy', 'N/A')}")
    print(f"  Context Precision:  {agg.get('context_precision', 'N/A')}")
    print(f"  Context Recall:     {agg.get('context_recall', 'N/A')}")
    print(f"\n  Full results saved to: {output_dir}/")

    return results


def generate_readme_table(results: Dict) -> str:
    """Generate a Markdown table from evaluation results for the README."""
    agg = results.get("aggregate", {})
    table = (
        "| Metric | Score |\n"
        "|--------|-------|\n"
        f"| Faithfulness | {agg.get('faithfulness', 'N/A')} |\n"
        f"| Answer Relevancy | {agg.get('answer_relevancy', 'N/A')} |\n"
        f"| Context Precision | {agg.get('context_precision', 'N/A')} |\n"
        f"| Context Recall | {agg.get('context_recall', 'N/A')} |\n"
        f"\n*Evaluated on {results.get('num_questions', 0)} test questions.*\n"
    )
    return table
