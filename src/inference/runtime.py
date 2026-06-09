from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from src.config import load_run_config, resolve_project_path
from src.data.tokenizer import get_tokenizer
from src.inference.generate import generate
from src.model.config import ModelConfig
from src.model.transformer import TransformerLM
from src.tokenizer import load_tokenizer
from src.train.checkpoint import load_checkpoint as load_legacy_checkpoint
from src.training.checkpointing import (
    find_checkpoint,
    load_checkpoint as load_training_checkpoint,
)
from src.utils import select_device

DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_STOP_SEQUENCES = ("\nUser:", "\nAssistant:", "\nSystem:")


@dataclass(frozen=True)
class DecoderRuntime:
    model: torch.nn.Module
    tokenizer: Any
    config: dict[str, Any]
    model_config: ModelConfig
    device: torch.device
    checkpoint_path: Path


@dataclass(frozen=True)
class EncodedPrompt:
    input_ids: torch.Tensor
    prompt_tokens: int
    context_tokens: int
    truncated: bool


@dataclass(frozen=True)
class GenerationResult:
    text: str
    full_text: str
    output_text: str
    prompt_tokens: int
    context_tokens: int
    generated_tokens: int
    truncated: bool


def is_modern_config(config: dict[str, Any]) -> bool:
    return "training" in config and "tokenizer" in config and "model" in config


def load_inference_config(
    run_config_path: str | Path | None,
    checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    if run_config_path:
        return load_run_config(run_config_path, create_dirs=False)
    if checkpoint_path is None:
        raise ValueError("Either run_config_path or checkpoint_path is required.")
    checkpoint = torch.load(
        resolve_project_path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    return checkpoint["config"]


def default_prompt(config: dict[str, Any]) -> str:
    if is_modern_config(config):
        return "Scientific progress depends on"
    return str(config.get("train", {}).get("sample_prompt", "Scientific progress depends on"))


def load_tokenizer_for_config(config: dict[str, Any]):
    if is_modern_config(config):
        return load_tokenizer(config["tokenizer"])
    return get_tokenizer(config["data"]["tokenizer"]["name"])


def eos_token_id(tokenizer: Any) -> int | None:
    return getattr(tokenizer, "eos_token_id", getattr(tokenizer, "eot_token_id", None))


def resolve_inference_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
) -> Path:
    if checkpoint_path:
        path = resolve_project_path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    if not is_modern_config(config):
        raise ValueError("checkpoint_path is required for legacy configs.")
    try:
        return find_checkpoint(config)
    except FileNotFoundError as exc:
        checkpointing_cfg = config["training"]["checkpointing"]
        configured_save_dir = Path(checkpointing_cfg["save_dir"])
        if configured_save_dir.is_absolute():
            raise

        nested_save_dir = resolve_project_path(Path("checkpoints") / configured_save_dir)
        for name in ("latest.pt", "final.pt"):
            candidate = nested_save_dir / name
            if candidate.exists():
                return candidate
        raise exc


def load_model_checkpoint(
    checkpoint_path: Path,
    model: TransformerLM,
    config: dict[str, Any],
    device: torch.device,
) -> None:
    if is_modern_config(config):
        load_training_checkpoint(
            checkpoint_path,
            model=model,
            map_location=device,
            restore_rng=False,
        )
    else:
        load_legacy_checkpoint(checkpoint_path, model=model, map_location=device)


def load_decoder_runtime(
    *,
    run_config_path: str | Path | None,
    checkpoint_path: str | Path | None,
    device_name: str = "auto",
) -> DecoderRuntime:
    config = load_inference_config(run_config_path, checkpoint_path)
    checkpoint = resolve_inference_checkpoint(config, checkpoint_path)
    model_config = ModelConfig.from_dict(config["model"])
    tokenizer = load_tokenizer_for_config(config)
    device = select_device(device_name)

    model = TransformerLM(model_config).to(device)
    load_model_checkpoint(checkpoint, model, config, device)
    model.eval()

    return DecoderRuntime(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_config=model_config,
        device=device,
        checkpoint_path=checkpoint,
    )


def build_chat_prompt(
    messages: Sequence[dict[str, str]],
    *,
    system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
) -> str:
    parts: list[str] = []
    stripped_system = system_prompt.strip()
    if stripped_system:
        parts.append(f"System: {stripped_system}")

    for message in messages:
        role = message.get("role", "").strip().lower()
        content = message.get("content", "").strip()
        if not content:
            continue
        if role == "assistant":
            label = "Assistant"
        elif role == "system":
            label = "System"
        else:
            label = "User"
        parts.append(f"{label}: {content}")

    parts.append("Assistant:")
    return "\n".join(parts)


def trim_stop_sequences(text: str, stop_sequences: Iterable[str]) -> str:
    stop_positions = [
        index
        for sequence in stop_sequences
        if sequence and (index := text.find(sequence)) >= 0
    ]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def encode_prompt_for_context(
    tokenizer: Any,
    prompt: str,
    *,
    context_length: int,
    device: torch.device,
) -> EncodedPrompt:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("The prompt produced no tokens.")

    truncated = len(prompt_ids) > context_length
    context_ids = prompt_ids[-context_length:] if truncated else prompt_ids
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
    return EncodedPrompt(
        input_ids=input_ids,
        prompt_tokens=len(prompt_ids),
        context_tokens=len(context_ids),
        truncated=truncated,
    )


def generate_text(
    runtime: DecoderRuntime,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    stop_sequences: Iterable[str] = DEFAULT_STOP_SEQUENCES,
) -> GenerationResult:
    encoded = encode_prompt_for_context(
        runtime.tokenizer,
        prompt,
        context_length=runtime.model_config.context_length,
        device=runtime.device,
    )
    top_k_value = top_k if top_k is not None and top_k > 0 else None
    output_ids = generate(
        model=runtime.model,
        input_ids=encoded.input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k_value,
        eos_token_id=eos_token_id(runtime.tokenizer),
    )
    generated_ids = output_ids[0, encoded.input_ids.shape[1] :].tolist()
    generated_text = runtime.tokenizer.decode(generated_ids)
    output_text = runtime.tokenizer.decode(output_ids[0].tolist())
    text = trim_stop_sequences(generated_text, stop_sequences)

    return GenerationResult(
        text=text,
        full_text=generated_text,
        output_text=output_text,
        prompt_tokens=encoded.prompt_tokens,
        context_tokens=encoded.context_tokens,
        generated_tokens=len(generated_ids),
        truncated=encoded.truncated,
    )
