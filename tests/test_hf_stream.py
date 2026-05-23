from __future__ import annotations

from src.data import hf_stream


def test_iter_dataset_texts_filters_fineweb_edu_quality_fields(monkeypatch) -> None:
    samples = [
        {
            "text": "kept",
            "language": "en",
            "language_score": 0.98,
            "score": 4.0,
            "token_count": 300,
        },
        {
            "text": "wrong language",
            "language": "pt",
            "language_score": 0.99,
            "score": 5.0,
            "token_count": 300,
        },
        {
            "text": "low language score",
            "language": "en",
            "language_score": 0.94,
            "score": 5.0,
            "token_count": 300,
        },
        {
            "text": "low edu score",
            "language": "en",
            "language_score": 0.99,
            "score": 3.4,
            "token_count": 300,
        },
        {
            "text": "too short",
            "language": "en",
            "language_score": 0.99,
            "score": 5.0,
            "token_count": 199,
        },
        {
            "text": "too long",
            "language": "en",
            "language_score": 0.99,
            "score": 5.0,
            "token_count": 4097,
        },
    ]
    monkeypatch.setattr(hf_stream, "load_dataset", lambda *args, **kwargs: samples)

    texts = list(
        hf_stream.iter_dataset_texts(
            {
                "id": "HuggingFaceFW/fineweb-edu",
                "text_field": "text",
                "filters": {
                    "language": "en",
                    "language_score_min": 0.95,
                    "score_min": 3.5,
                    "token_count_min": 200,
                    "token_count_max": 4096,
                },
            }
        )
    )

    assert texts == ["kept"]


def test_iter_dataset_texts_prioritizes_score_then_language_score(monkeypatch) -> None:
    samples = [
        {
            "text": "lower language score",
            "language": "en",
            "language_score": 0.96,
            "score": 4.0,
            "token_count": 300,
        },
        {
            "text": "lower score",
            "language": "en",
            "language_score": 0.99,
            "score": 3.9,
            "token_count": 300,
        },
        {
            "text": "best",
            "language": "en",
            "language_score": 0.99,
            "score": 4.0,
            "token_count": 300,
        },
    ]
    monkeypatch.setattr(hf_stream, "load_dataset", lambda *args, **kwargs: samples)

    texts = list(
        hf_stream.iter_dataset_texts(
            {
                "id": "HuggingFaceFW/fineweb-edu",
                "text_field": "text",
                "filters": {
                    "language": "en",
                    "language_score_min": 0.95,
                    "score_min": 3.5,
                    "token_count_min": 200,
                    "token_count_max": 4096,
                },
                "selection": {
                    "prioritize_quality": True,
                    "quality_buffer_size": 10,
                    "shuffle": True,
                },
            }
        )
    )

    assert texts == ["best", "lower language score", "lower score"]
