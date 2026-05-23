from __future__ import annotations

from pathlib import Path

from src.data import hf_stream
from src.data.fineweb_edu import write_training_corpus


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


def test_iter_dataset_texts_can_read_one_shard(monkeypatch) -> None:
    samples = [{"text": f"sample {index}"} for index in range(6)]
    monkeypatch.setattr(hf_stream, "load_dataset", lambda *args, **kwargs: samples)

    texts = list(
        hf_stream.iter_dataset_texts(
            {
                "id": "local",
                "text_field": "text",
                "selection": {"prioritize_quality": False},
            },
            shard_index=1,
            num_shards=2,
        )
    )

    assert texts == ["sample 1", "sample 3", "sample 5"]


def test_write_training_corpus_parallel_shards(monkeypatch, tmp_path) -> None:
    samples = [{"text": f"sample {index}"} for index in range(6)]
    monkeypatch.setattr(hf_stream, "load_dataset", lambda *args, **kwargs: samples)

    stats = write_training_corpus(
        {
            "id": "local",
            "text_field": "text",
            "selection": {"prioritize_quality": False},
        },
        tmp_path,
        max_samples=5,
        chunk_samples=2,
        num_workers=2,
    )

    written = []
    for path in stats.files:
        written.extend(Path(path).read_text(encoding="utf-8").splitlines())

    assert stats.samples_written == 5
    assert written == ["sample 0", "sample 2", "sample 4", "sample 1", "sample 3"]
