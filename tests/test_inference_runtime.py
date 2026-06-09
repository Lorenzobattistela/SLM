from __future__ import annotations

from pathlib import Path

import torch

from src.inference.runtime import (
    DecoderRuntime,
    build_chat_prompt,
    encode_prompt_for_context,
    generate_text,
    trim_stop_sequences,
)


class FakeTokenizer:
    eos_token_id = None

    def encode(self, text: str) -> list[int]:
        return [int(part) for part in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"context_length": 4})()

    def forward(self, input_ids: torch.Tensor):
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            10,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, -1, 7] = 1.0
        return logits, None


def test_build_chat_prompt_appends_assistant_turn() -> None:
    prompt = build_chat_prompt(
        [
            {"role": "user", "content": "Oi"},
            {"role": "assistant", "content": "Ola"},
            {"role": "user", "content": "Explique decoder-only"},
        ],
        system_prompt="Responda curto.",
    )

    assert prompt == (
        "System: Responda curto.\n"
        "User: Oi\n"
        "Assistant: Ola\n"
        "User: Explique decoder-only\n"
        "Assistant:"
    )


def test_trim_stop_sequences_uses_first_marker() -> None:
    assert trim_stop_sequences("resposta\nUser: nova pergunta", ("\nUser:",)) == "resposta"
    assert trim_stop_sequences("abc\nAssistant: x\nUser: y", ("\nUser:", "\nAssistant:")) == "abc"


def test_encode_prompt_truncates_left_to_context_length() -> None:
    encoded = encode_prompt_for_context(
        FakeTokenizer(),
        "1 2 3 4 5 6",
        context_length=4,
        device=torch.device("cpu"),
    )

    assert encoded.prompt_tokens == 6
    assert encoded.context_tokens == 4
    assert encoded.truncated
    assert encoded.input_ids.tolist() == [[3, 4, 5, 6]]


def test_generate_text_returns_only_new_tokens() -> None:
    runtime = DecoderRuntime(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        config={},
        model_config=type("ModelConfig", (), {"context_length": 4})(),
        device=torch.device("cpu"),
        checkpoint_path=Path("checkpoint.pt"),
    )

    result = generate_text(
        runtime,
        "1 2 3",
        max_new_tokens=2,
        temperature=0.0,
        top_k=0,
        stop_sequences=(),
    )

    assert result.text == "7 7"
    assert result.full_text == "7 7"
    assert result.output_text == "1 2 3 7 7"
    assert result.prompt_tokens == 3
    assert result.generated_tokens == 2
