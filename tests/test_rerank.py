from unittest.mock import Mock
import numpy as np
import pytest
from src.rerank import rerank


def test_reranking_preserves_candidates_and_breaks_ties():
    candidates = [{"text": "a" * 5000, "id": key, "score": 1 / 61} for key in "ba"]
    model = Mock()
    model.batches.return_value = [(np.ones((2, 1)), None)]
    assert rerank(model, "query", [], 3) == []
    model.batches.assert_not_called()
    hits = rerank(model, "query", candidates, 1)
    assert hits == [candidates[1] | {"score": 1., "fusion_score": 1 / 61, "initial_rank": 2}]


@pytest.mark.parametrize("scores", [np.array([[np.nan]]), np.ones((2, 1)), np.ones(1)])
def test_invalid_logits(scores):
    model = Mock()
    model.batches.return_value = [(scores, None)]
    with pytest.raises(ValueError):
        rerank(model, "q", [{"id": "a", "text": "b", "score": 1}], 1)
