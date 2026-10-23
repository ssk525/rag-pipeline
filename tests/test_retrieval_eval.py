from src.retrieval_eval import aggregate, content_words, score_question


def test_content_words_drops_stopwords():
    words = content_words("The attention mechanism uses queries and keys")
    assert "attention" in words
    assert "the" not in words
    assert "and" not in words


def test_hit_at_1_when_first_source_matches():
    score = score_question(
        question="q",
        expected_source="attention_mechanisms.md",
        retrieved_sources=["attention_mechanisms.md", "rag_overview.md"],
        retrieved_texts=["scaled dot-product attention softmax"],
        ground_truth="scaled dot-product attention",
    )
    assert score.hit_at_1 == 1.0
    assert score.hit_at_k == 1.0
    assert score.reciprocal_rank == 1.0
    assert score.answer_coverage > 0


def test_miss_when_source_absent():
    score = score_question(
        question="q",
        expected_source="attention_mechanisms.md",
        retrieved_sources=["rag_overview.md"],
        retrieved_texts=["retrieval augmented generation"],
        ground_truth="scaled attention",
    )
    assert score.hit_at_1 == 0.0
    assert score.hit_at_k == 0.0
    assert score.reciprocal_rank == 0.0


def test_aggregate_averages_four_metrics():
    a = score_question("q1", "a.md", ["a.md"], ["hello world"], "hello world")
    b = score_question("q2", "b.md", ["c.md"], ["other"], "hello world")
    metrics = aggregate([a, b])
    assert metrics["hit@1"] == 0.5
    assert metrics["hit@k"] == 0.5
    assert metrics["mrr"] == 0.5
