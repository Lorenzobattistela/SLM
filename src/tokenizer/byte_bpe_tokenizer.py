from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


class ByteBPEError(RuntimeError):
    pass


@dataclass
class ByteBPETokenizer:
    name: str
    encoding: Any
    vocab_size: int
    eos_token_id: int

    @property
    def eot_token_id(self) -> int:
        return self.eos_token_id

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {"eos_token": self.eos_token_id, "eot_token": self.eos_token_id}

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids = list(self.encoding.encode_ordinary(text))
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.encoding.decode(list(token_ids))


def ensure_byte_bpe_type(tokenizer_cfg: dict[str, Any]) -> None:
    tokenizer_type = tokenizer_cfg.get("type")
    if tokenizer_type != "byte_bpe":
        raise ByteBPEError(f"Unsupported tokenizer.type={tokenizer_type!r}; expected 'byte_bpe'.")


def load_byte_bpe_tokenizer(tokenizer_cfg: dict[str, Any]) -> ByteBPETokenizer:
    ensure_byte_bpe_type(tokenizer_cfg)
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - dependency is checked at runtime.
        raise ByteBPEError("Byte-level BPE tokenizer support requires `tiktoken`.") from exc

    name = str(tokenizer_cfg.get("name", "gpt2"))
    try:
        encoding = tiktoken.get_encoding(name)
    except ValueError as exc:
        raise ByteBPEError(f"Unsupported tiktoken byte-level BPE encoding: {name!r}") from exc

    vocab_size = int(encoding.n_vocab)
    expected_vocab_size = int(tokenizer_cfg.get("vocab_size", vocab_size))
    if expected_vocab_size != vocab_size:
        raise ByteBPEError(
            f"Loaded byte-level BPE tokenizer {name!r} has vocab size {vocab_size}, "
            f"but tokenizer.vocab_size is {expected_vocab_size}. For the ready GPT-2 "
            "byte-level BPE tokenizer, set tokenizer.vocab_size and model.vocab_size to 50257."
        )

    eos_token_id = int(tokenizer_cfg.get("eos_token_id", encoding.eot_token))
    if eos_token_id < 0 or eos_token_id >= vocab_size:
        raise ByteBPEError(
            f"Configured byte-level BPE eos_token_id {eos_token_id} is outside vocab size "
            f"{vocab_size}."
        )

    return ByteBPETokenizer(
        name=name,
        encoding=encoding,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
    )
