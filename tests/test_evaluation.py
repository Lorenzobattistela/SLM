from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch
import pytest

from src.model import ModelConfig, TransformerLM
from src.evaluation.benchmarks import get_log_likelihood, evaluate_perplexity, run_all_benchmarks

class FakeTokenizer:
    def __init__(self):
        self.special_token_ids = {"bos_token": 1, "eos_token": 2}
    
    def encode(self, text: str, **kwargs) -> list[int]:
        # Return a deterministic token list based on the string length
        return [3, 4, len(text)]
    
    def decode(self, token_ids: list[int]) -> str:
        return "fake decoded"

@pytest.fixture
def fake_model_and_tokenizer():
    # Construct a minimal model config
    config = ModelConfig(
        vocab_size=100,
        context_length=64,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_dim=64,
        tie_embeddings=True,
    )
    model = TransformerLM(config)
    tokenizer = FakeTokenizer()
    return model, tokenizer

def test_get_log_likelihood(fake_model_and_tokenizer) -> None:
    model, tokenizer = fake_model_and_tokenizer
    model.eval()
    
    context = "Hello world"
    candidate = "Yes indeed"
    device = torch.device("cpu")
    bos_id = tokenizer.special_token_ids["bos_token"]
    
    log_likelihood = get_log_likelihood(
        model=model,
        tokenizer=tokenizer,
        context=context,
        candidate=candidate,
        device=device,
        bos_id=bos_id
    )
    
    assert isinstance(log_likelihood, float)
    assert not torch.isnan(torch.tensor(log_likelihood))

def test_evaluate_perplexity_mocked(fake_model_and_tokenizer, monkeypatch) -> None:
    model, tokenizer = fake_model_and_tokenizer
    
    # Mock load_dataset to return a simple iterable dataset
    mock_samples = [{"text": "Hello"}, {"text": "World"}]
    monkeypatch.setattr("src.evaluation.benchmarks.load_dataset", lambda *args, **kwargs: mock_samples)
    
    ppl = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name="dummy_dataset",
        device=torch.device("cpu"),
        limit=2
    )
    
    assert isinstance(ppl, float)
    assert ppl > 0.0

def test_run_all_benchmarks(fake_model_and_tokenizer, monkeypatch) -> None:
    model, tokenizer = fake_model_and_tokenizer
    
    # Mock all benchmark evaluations to keep it fast and avoid network calls
    monkeypatch.setattr("src.evaluation.benchmarks.evaluate_perplexity", lambda *args, **kwargs: 10.5)
    monkeypatch.setattr("src.evaluation.benchmarks.evaluate_hellaswag", lambda *args, **kwargs: 0.75)
    monkeypatch.setattr("src.evaluation.benchmarks.evaluate_arc_easy", lambda *args, **kwargs: 0.80)
    monkeypatch.setattr("src.evaluation.benchmarks.evaluate_piqa", lambda *args, **kwargs: 0.85)
    monkeypatch.setattr("src.evaluation.benchmarks.evaluate_winogrande", lambda *args, **kwargs: 0.60)
    monkeypatch.setattr("src.evaluation.benchmarks.evaluate_gsm8k", lambda *args, **kwargs: 0.05)
    
    results = run_all_benchmarks(
        model=model,
        tokenizer=tokenizer,
        benchmarks_list=["perplexity", "hellaswag", "arc_easy", "piqa", "winogrande", "gsm8k"],
        ppl_dataset="dummy_dataset",
        device=torch.device("cpu"),
        limit=2
    )
    
    assert results["perplexity"] == 10.5
    assert results["hellaswag_accuracy"] == 0.75
    assert results["arc_easy_accuracy"] == 0.80
    assert results["piqa_accuracy"] == 0.85
    assert results["winogrande_accuracy"] == 0.60
    assert results["gsm8k_accuracy"] == 0.05
