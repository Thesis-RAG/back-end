import pytest

from app.core.config import settings
from app.services.retrieval_service import RetrievalService


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls.append((pairs, batch_size, show_progress_bar))
        return self.scores


def _service_with_reranker(scores):
    service = object.__new__(RetrievalService)
    service._reranker = FakeReranker(scores)
    service._reranker_load_failed = False
    return service


def test_rwss_uses_sigmoid_and_clearance_penalty(monkeypatch):
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_alpha", 0.8)
    monkeypatch.setattr(settings, "reranker_beta", 0.2)
    monkeypatch.setattr(settings, "reranker_max_sensitivity", 5)
    monkeypatch.setattr(settings, "reranker_batch_size", 16)

    service = _service_with_reranker([0.0])
    result = service._rerank_after_rrf(
        "query",
        [{
            "chunk_id": "c1",
            "document_text": "candidate",
            "metadata": {"chunk_sensitivity": 3},
            "rrf_score": 0.01,
        }],
        user_clearance=1,
    )[0]

    # sigmoid(0)=0.5; penalty=(3-1)/(5-1)=0.5
    assert result["cross_encoder_probability"] == 0.5
    assert result["clearance_penalty"] == 0.5
    assert result["score"] == pytest.approx(0.8 * 0.5 - 0.2 * 0.5)
    assert result["rerank_score"] == result["score"]


def test_rwss_reranks_candidates_and_preserves_rrf_score(monkeypatch):
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_alpha", 0.8)
    monkeypatch.setattr(settings, "reranker_beta", 0.2)
    monkeypatch.setattr(settings, "reranker_max_sensitivity", 5)
    monkeypatch.setattr(settings, "reranker_batch_size", 16)

    service = _service_with_reranker([0.0, 2.0])
    results = service._rerank_after_rrf(
        "query",
        [
            {"chunk_id": "unsafe", "document_text": "first", "metadata": {"sensitivity": 5}, "rrf_score": 0.02},
            {"chunk_id": "safe", "document_text": "second", "metadata": {"sensitivity": 1}, "rrf_score": 0.01},
        ],
        user_clearance=1,
    )

    assert [item["chunk_id"] for item in results] == ["safe", "unsafe"]
    assert results[0]["rrf_score"] == 0.01
    assert results[1]["rrf_score"] == 0.02


def test_rwss_is_disabled_without_calling_model(monkeypatch):
    monkeypatch.setattr(settings, "reranker_enabled", False)
    service = _service_with_reranker([1.0])
    candidate = {"chunk_id": "c1", "document_text": "candidate", "metadata": {}}

    assert service._rerank_after_rrf("query", [candidate]) == [candidate]
