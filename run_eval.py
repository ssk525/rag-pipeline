"""
Evaluation Runner Script
Runs the full RAG pipeline against the eval set and saves results.

Usage:
    python run_eval.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import RAGPipeline
from src.evaluator import (
    load_eval_set, run_ragas_evaluation, save_eval_results, generate_readme_table,
)


def main():
    print("=" * 60)
    print("🧪 RAG Pipeline Evaluation")
    print("=" * 60)

    # Initialize pipeline
    pipeline = RAGPipeline(
        persist_dir="chroma_db",
        chunking_strategy="semantic",
        use_reranker=bool(os.getenv("COHERE_API_KEY")),
    )

    # Ingest sample docs
    sample_dir = os.path.join("data", "sample_docs")
    if os.path.isdir(sample_dir):
        pipeline.ingest_directory(sample_dir)
    else:
        print(f"❌ Sample directory not found: {sample_dir}")
        sys.exit(1)

    # Load eval questions
    eval_path = os.path.join("eval", "test_questions.json")
    eval_questions = load_eval_set(eval_path)
    print(f"\n📝 Loaded {len(eval_questions)} evaluation questions\n")

    # Run each question through the pipeline
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, eq in enumerate(eval_questions, 1):
        print(f"  [{i}/{len(eval_questions)}] {eq.question[:80]}...")

        # Get answer from pipeline
        result = pipeline.query(eq.question, top_k=5, use_hybrid=True)

        questions.append(eq.question)
        answers.append(result.answer)
        contexts.append(result.context_used)
        ground_truths.append(eq.ground_truth)

        # Clear chat history between eval questions
        pipeline.clear_history()

        # Rate limiting
        time.sleep(1)

    print(f"\n{'=' * 60}")
    print("📊 Running Evaluation...")
    print("=" * 60)

    # Run evaluation
    results = run_ragas_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )

    # Save results
    save_eval_results(results, output_dir="eval/results")

    # Generate README table
    table = generate_readme_table(results)
    print(f"\n📋 README Table:\n{table}")

    # Save table to file
    with open("eval/results/readme_table.md", "w") as f:
        f.write(table)

    print("\n✅ Evaluation complete! Results saved to eval/results/")


if __name__ == "__main__":
    main()
