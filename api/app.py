"""
Flask REST API for the Self-Improving RAG Pipeline.

Endpoints:
  GET  /health
  GET  /stats
  GET  /config
  POST /documents/ingest
  POST /documents/upload
  POST /retrieve
  POST /query
  POST /eval/run
  POST /improve/run
  POST /feedback
"""

import os
import tempfile
from functools import wraps
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

load_dotenv()

from src.config_store import (  # noqa: E402
    PipelineConfig,
    append_feedback,
    load_config,
    save_config,
)
from src.logging_config import setup_logging  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.self_improve import run_self_improvement  # noqa: E402

logger = setup_logging("rag_pipeline.api")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        cfg = load_config()
        logger.info("Initializing pipeline with config: %s", cfg.to_dict())
        _pipeline = RAGPipeline(
            chunking_strategy=cfg.chunking_strategy,
            use_reranker=cfg.use_reranker,
        )
    return _pipeline


def reset_pipeline() -> None:
    global _pipeline
    _pipeline = None


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

    def log_request(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            logger.info("%s %s", request.method, request.path)
            return fn(*args, **kwargs)

        return wrapper

    @app.get("/health")
    @log_request
    def health():
        return jsonify({"status": "ok", "service": "self-improving-rag-pipeline"})

    @app.get("/stats")
    @log_request
    def stats():
        pipeline = get_pipeline()
        return jsonify(pipeline.get_stats())

    @app.get("/config")
    @log_request
    def get_config():
        return jsonify(load_config().to_dict())

    @app.put("/config")
    @log_request
    def update_config():
        data = request.get_json(silent=True) or {}
        cfg = load_config()
        for field in ("top_k", "use_hybrid", "use_reranker", "chunking_strategy"):
            if field in data:
                setattr(cfg, field, data[field])
        saved = save_config(cfg)
        reset_pipeline()
        logger.info("Pipeline config updated: %s", saved.to_dict())
        return jsonify(saved.to_dict())

    @app.post("/documents/ingest")
    @log_request
    def ingest_directory():
        data = request.get_json(silent=True) or {}
        directory = data.get("directory")
        if not directory or not os.path.isdir(directory):
            return jsonify({"error": "Provide a valid 'directory' path"}), 400

        pipeline = get_pipeline()
        pipeline.ingest_directory(directory)
        logger.info("Ingested directory: %s", directory)
        return jsonify({"status": "ingested", "directory": directory, "stats": pipeline.get_stats()})

    @app.post("/documents/upload")
    @log_request
    def upload_documents():
        if not request.files:
            return jsonify({"error": "Upload one or more files as multipart form field 'files'"}), 400

        pipeline = get_pipeline()
        temp_paths = []
        saved_names = []

        try:
            for file_storage in request.files.getlist("files"):
                if not file_storage.filename:
                    continue
                filename = secure_filename(file_storage.filename)
                temp_path = os.path.join(tempfile.gettempdir(), filename)
                file_storage.save(temp_path)
                temp_paths.append(temp_path)
                saved_names.append(filename)

            if not temp_paths:
                return jsonify({"error": "No valid files uploaded"}), 400

            pipeline.ingest_files(temp_paths)
            logger.info("Uploaded and ingested files: %s", saved_names)
            return jsonify({
                "status": "ingested",
                "files": saved_names,
                "stats": pipeline.get_stats(),
            })
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)

    @app.post("/retrieve")
    @log_request
    def retrieve():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400

        cfg = load_config()
        top_k = int(data.get("top_k", cfg.top_k))
        use_hybrid = bool(data.get("use_hybrid", cfg.use_hybrid))

        pipeline = get_pipeline()
        results = pipeline.retriever.retrieve(
            query=question, top_k=top_k, use_hybrid=use_hybrid
        )

        payload = [
            {
                "content": r.content,
                "source": r.source,
                "filename": r.filename,
                "page_number": r.page_number,
                "relevance_score": r.relevance_score,
                "retrieval_method": r.retrieval_method,
            }
            for r in results
        ]
        logger.info("Retrieved %s chunks for question", len(payload))
        return jsonify({"question": question, "results": payload})

    @app.post("/query")
    @log_request
    def query():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400

        cfg = load_config()
        top_k = int(data.get("top_k", cfg.top_k))
        use_hybrid = bool(data.get("use_hybrid", cfg.use_hybrid))

        pipeline = get_pipeline()
        answer = pipeline.query(question=question, top_k=top_k, use_hybrid=use_hybrid)
        logger.info("Generated answer for question (latency=%sms)", answer.latency_ms)
        return jsonify({
            "question": question,
            "answer": answer.answer,
            "citations": answer.citations,
            "context_used": answer.context_used,
            "model": answer.model,
            "latency_ms": answer.latency_ms,
            "refused": answer.refused,
        })

    @app.post("/eval/run")
    @log_request
    def run_eval():
        from run_eval import main as run_eval_main

        logger.info("Evaluation run requested via API")
        try:
            run_eval_main()
            return jsonify({"status": "completed", "results_dir": "eval/results"})
        except Exception as exc:
            logger.exception("Evaluation failed")
            return jsonify({"error": str(exc)}), 500

    @app.post("/improve/run")
    @log_request
    def improve():
        data = request.get_json(silent=True) or {}
        eval_limit = int(data.get("eval_limit", 5))
        logger.info("Self-improvement run requested (eval_limit=%s)", eval_limit)
        try:
            record = run_self_improvement(eval_limit=eval_limit)
            reset_pipeline()
            return jsonify({"status": "completed", "improvement": record})
        except Exception as exc:
            logger.exception("Self-improvement failed")
            return jsonify({"error": str(exc)}), 500

    @app.post("/feedback")
    @log_request
    def feedback():
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400

        record: Dict[str, Any] = {
            "question": question,
            "answer": data.get("answer", ""),
            "rating": data.get("rating", "unknown"),
            "comment": data.get("comment", ""),
        }
        append_feedback(record)
        logger.info("Stored feedback for question")
        return jsonify({"status": "saved"})

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("API_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
