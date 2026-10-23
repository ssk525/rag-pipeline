from run_retrieval_eval import run_bm25_config
from src.evaluator import load_eval_set


def test_bm25_comparison_finds_labelled_files():
    questions = load_eval_set("eval/test_questions.json")
    assert len(questions) == 25
    semantic = run_bm25_config("semantic", questions)
    fixed = run_bm25_config("fixed", questions)
    assert semantic["chunks"] >= 1
    assert fixed["chunks"] >= 1
    assert semantic["metrics"]["hit@k"] > 0
    assert fixed["metrics"]["hit@k"] > 0
