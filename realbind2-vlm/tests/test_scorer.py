from src.common.scorer import normalize_output, score_prediction


def test_known_semantic_normalization():
    assert score_prediction("Final answer: Grey.", "gray", "color").correct
    assert score_prediction("LEFT", "LEFT", "position").correct
    assert score_prediction("The answer is B.", "B", "identity").correct


def test_conflicting_mentions_are_invalid():
    normalized, error = normalize_output("blue, not white", "color")
    assert normalized is None
    assert error == "zero_or_multiple_candidate_mentions"

