"""
Compare chunking and retrieval settings on the 25-question eval set.

Default (--bm25-only): no API keys. Each chunking strategy is scored with BM25.
That is the comparison committed in eval/results/.

With Gemini keys, omit --bm25-only to also score hybrid BM25+vector vs vector only.

Usage:
    python run_retrieval_eval.py --bm25-only
    python run_retrieval_eval.py
    python run_retrieval_eval.py --rerank
"""

import argparse
import json
import os
import tempfile
from datetime import date

from src.evaluator import load_eval_set
from src.retrieval_eval import aggregate, score_question

TOP_K = 5
RESULTS_DIR = os.path.join("eval", "results")


def run_config(chunking: str, use_hybrid: bool, rerank: bool, questions) -> dict:
    from dotenv import load_dotenv

    from src.pipeline import RAGPipeline

    load_dotenv()
    with tempfile.TemporaryDirectory() as persist_dir:
        pipeline = RAGPipeline(
            persist_dir=persist_dir, chunking_strategy=chunking, use_reranker=rerank
        )
        pipeline.ingest_directory(os.path.join("data", "sample_docs"))
        n_chunks = len(pipeline._all_chunks)
        scores = []
        for q in questions:
            results = pipeline.retriever.retrieve(q.question, top_k=TOP_K, use_hybrid=use_hybrid)
            scores.append(
                score_question(
                    question=q.question,
                    expected_source=q.source_context,
                    retrieved_sources=[r.filename for r in results],
                    retrieved_texts=[r.content for r in results],
                    ground_truth=q.ground_truth,
                )
            )
    return {
        "chunking": chunking,
        "retrieval": ("hybrid" if use_hybrid else "vector") + ("+rerank" if rerank else ""),
        "chunks": n_chunks,
        "metrics": aggregate(scores),
        "per_question": [s.__dict__ for s in scores],
    }


def run_bm25_config(chunking: str, questions) -> dict:
    from src.chunker import ChunkingConfig, ChunkingStrategy, chunk_documents
    from src.document_loader import load_directory
    from src.retriever import HybridRetriever

    docs = load_directory(os.path.join("data", "sample_docs"), extensions=[".md", ".txt"])
    strategy = ChunkingStrategy.SEMANTIC if chunking == "semantic" else ChunkingStrategy.FIXED_SIZE
    chunks = chunk_documents(docs, ChunkingConfig(strategy=strategy))
    retriever = HybridRetriever(vector_store=None, use_reranker=False)
    retriever.build_bm25_index(chunks)

    scores = []
    for q in questions:
        hits = retriever._bm25_search(q.question, k=TOP_K)
        results = [
            type("R", (), {"filename": doc.metadata.get("filename", ""), "content": doc.page_content})()
            for doc, _ in hits
        ]
        scores.append(
            score_question(
                question=q.question,
                expected_source=q.source_context,
                retrieved_sources=[r.filename for r in results],
                retrieved_texts=[r.content for r in results],
                ground_truth=q.ground_truth,
            )
        )
    return {
        "chunking": chunking,
        "retrieval": "bm25",
        "chunks": len(chunks),
        "metrics": aggregate(scores),
        "per_question": [s.__dict__ for s in scores],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerank", action="store_true", help="Apply Cohere rerank-v3.5")
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        default=True,
        help="Score semantic vs fixed chunking with BM25 (no API keys). Default.",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also run hybrid vs vector using Gemini embeddings.",
    )
    args = parser.parse_args()

    questions = load_eval_set(os.path.join("eval", "test_questions.json"))
    runs = [run_bm25_config(chunking, questions) for chunking in ("semantic", "fixed")]
    if args.with_embeddings:
        runs.extend(
            run_config(chunking, hybrid, args.rerank, questions)
            for chunking in ("semantic", "fixed")
            for hybrid in (True, False)
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "retrieval_comparison.json"), "w") as f:
        json.dump({"date": str(date.today()), "top_k": TOP_K, "runs": runs}, f, indent=2)

    lines = [
        f"Retrieval comparison, {len(questions)} questions, top_k={TOP_K}, "
        f"retriever={'embeddings' if args.with_embeddings else 'bm25'}, {date.today()}.",
        "",
        "| chunking | retrieval | chunks | hit@1 | hit@5 | MRR | answer coverage@5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        m = r["metrics"]
        lines.append(
            f"| {r['chunking']} | {r['retrieval']} | {r['chunks']} | {m['hit@1']:.3f} | "
            f"{m['hit@k']:.3f} | {m['mrr']:.3f} | {m['answer_coverage@k']:.3f} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "retrieval_comparison.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
