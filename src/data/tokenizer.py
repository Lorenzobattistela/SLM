from __future__ import annotations

from dataclasses import dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover - dependency is checked at runtime.
    tiktoken = None


@dataclass
class TokenizerWrapper:
    name: str
    vocab_size: int
    eot_token_id: int
    _encoding: object

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode_ordinary(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)


def get_tokenizer(name: str) -> TokenizerWrapper:
    if tiktoken is None:
        raise RuntimeError("tiktoken is not installed. Run `pip install -e .` first.")
    if name != "gpt2":
        raise ValueError(f"Unsupported tokenizer: {name}")
    encoding = tiktoken.get_encoding("gpt2")
    return TokenizerWrapper(
        name=name,
        vocab_size=encoding.n_vocab,
        eot_token_id=encoding.eot_token,
        _encoding=encoding,
    )
