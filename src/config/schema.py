from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


REQUIRED_SECTIONS = (
    "project",
    "dataset",
    "tokenizer",
    "model",
    "training",
    "evaluation",
    "logging",
    "plots",
)


def _is_multi_source_dataset(dataset: dict[str, Any]) -> bool:
    return isinstance(dataset.get("sources"), list) and len(dataset["sources"]) > 0


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {name}")
    return value


def _required(mapping: dict[str, Any], path: str, label: str | None = None) -> Any:
    display_path = label or path
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required config value: {display_path}")
        current = current[part]
    if current is None or current == "":
        raise ConfigError(f"Missing required config value: {display_path}")
    return current


def _required_any(mapping: dict[str, Any], paths: tuple[str, ...], label: str) -> Any:
    for path in paths:
        current: Any = mapping
        found = True
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None and current != "":
            return current
    raise ConfigError(f"Missing required config value: {label}")


def _require_mapping(
    mapping: dict[str, Any],
    path: str,
    label: str | None = None,
) -> dict[str, Any]:
    display_path = label or path
    value = _required(mapping, path, display_path)
    if not isinstance(value, dict):
        raise ConfigError(f"Config value must be a mapping: {display_path}")
    return value


def _require_positive_int(
    mapping: dict[str, Any],
    path: str,
    label: str | None = None,
) -> int:
    display_path = label or path
    value = _required(mapping, path, display_path)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Config value must be a positive integer: {display_path}")
    return value


def _require_positive_int_any(
    mapping: dict[str, Any],
    paths: tuple[str, ...],
    label: str,
) -> int:
    value = _required_any(mapping, paths, label)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Config value must be a positive integer: {label}")
    return value


def _require_optional_positive_int(
    mapping: dict[str, Any],
    path: str,
    label: str | None = None,
) -> int | None:
    display_path = label or path
    if path not in mapping:
        return None
    value = mapping[path]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Config value must be a positive integer: {display_path}")
    return value


def _require_positive_number(
    mapping: dict[str, Any],
    path: str,
    label: str | None = None,
) -> float:
    display_path = label or path
    value = _required(mapping, path, display_path)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"Config value must be a positive number: {display_path}")
    return float(value)


def _require_bool(mapping: dict[str, Any], path: str, label: str | None = None) -> bool:
    display_path = label or path
    value = _required(mapping, path, display_path)
    if not isinstance(value, bool):
        raise ConfigError(f"Config value must be a boolean: {display_path}")
    return value


def _require_bool_any(mapping: dict[str, Any], paths: tuple[str, ...], label: str) -> bool:
    value = _required_any(mapping, paths, label)
    if not isinstance(value, bool):
        raise ConfigError(f"Config value must be a boolean: {label}")
    return value


def _require_directory_value(
    mapping: dict[str, Any],
    path: str,
    label: str | None = None,
) -> str:
    display_path = label or path
    value = _required(mapping, path, display_path)
    if not isinstance(value, str):
        raise ConfigError(f"Directory config value must be a string: {display_path}")
    return value


def _validate_dataset_section(dataset: dict[str, Any]) -> None:
    multi_source = _is_multi_source_dataset(dataset)
    if multi_source:
        _require_positive_int(dataset, "target_train_tokens", "dataset.target_train_tokens")
        _require_directory_value(dataset, "cache_dir", "dataset.cache_dir")
        _require_directory_value(dataset, "processed_dir", "dataset.processed_dir")
        _require_optional_positive_int(
            dataset,
            "tokenize_num_workers",
            "dataset.tokenize_num_workers",
        )
        return

    _required(dataset, "name", "dataset.name")
    _required(dataset, "split", "dataset.split")
    _required(dataset, "text_column", "dataset.text_column")
    _required(dataset, "streaming", "dataset.streaming")
    _require_positive_int(dataset, "target_train_tokens", "dataset.target_train_tokens")
    _require_positive_int(dataset, "validation_tokens", "dataset.validation_tokens")
    _require_directory_value(dataset, "cache_dir", "dataset.cache_dir")
    _require_directory_value(dataset, "raw_dir", "dataset.raw_dir")
    _require_directory_value(dataset, "processed_dir", "dataset.processed_dir")
    _require_optional_positive_int(
        dataset,
        "tokenize_num_workers",
        "dataset.tokenize_num_workers",
    )


def validate_run_config(config: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        _section(config, section)

    dataset = _section(config, "dataset")
    _validate_dataset_section(dataset)

    tokenizer = _section(config, "tokenizer")
    tokenizer_type = _required(tokenizer, "type", "tokenizer.type")
    if tokenizer_type not in {"superbpe", "byte_bpe"}:
        raise ConfigError("tokenizer.type must be either 'superbpe' or 'byte_bpe'")
    _require_positive_int(tokenizer, "vocab_size", "tokenizer.vocab_size")
    if tokenizer_type == "superbpe":
        pretrained = tokenizer.get("pretrained")
        if pretrained is not None:
            pretrained_cfg = _require_mapping(tokenizer, "pretrained", "tokenizer.pretrained")
            _required(pretrained_cfg, "base_url", "tokenizer.pretrained.base_url")
        else:
            _require_positive_int(tokenizer, "train_samples", "tokenizer.train_samples")
            _require_optional_positive_int(
                tokenizer,
                "corpus_num_workers",
                "tokenizer.corpus_num_workers",
            )
        special_tokens = _require_mapping(tokenizer, "special_tokens", "tokenizer.special_tokens")
        for token_name in ("pad_token", "bos_token", "eos_token", "unk_token"):
            _required(special_tokens, token_name, f"tokenizer.special_tokens.{token_name}")
    else:
        _required(tokenizer, "name", "tokenizer.name")

    model = _section(config, "model")
    has_parameter_budget = any(
        key in model
        for key in (
            "target_parameters",
            "acceptable_min_parameters",
            "acceptable_max_parameters",
        )
    )
    if has_parameter_budget:
        target_parameters = _require_positive_int(
            model,
            "target_parameters",
            "model.target_parameters",
        )
        min_parameters = _require_positive_int(
            model,
            "acceptable_min_parameters",
            "model.acceptable_min_parameters",
        )
        max_parameters = _require_positive_int(
            model,
            "acceptable_max_parameters",
            "model.acceptable_max_parameters",
        )
        if not min_parameters <= target_parameters <= max_parameters:
            raise ConfigError(
                "model.target_parameters must be inside the acceptable parameter range"
            )
    expected_model_values = {
        "architecture": "decoder_only_transformer",
        "positional_encoding": "rope",
        "attention": "gqa",
        "activation": "swiglu",
        "normalization": "rmsnorm",
    }
    for key, expected in expected_model_values.items():
        value = _required(model, key, f"model.{key}")
        if value != expected:
            raise ConfigError(f"model.{key} must be '{expected}'")

    _require_bool_any(
        model,
        ("flash_attention", "use_flash_attention"),
        "model.flash_attention",
    )
    _require_bool(model, "flash_attention_fallback", "model.flash_attention_fallback")
    model_vocab_size = _require_positive_int(model, "vocab_size", "model.vocab_size")
    if model_vocab_size != int(tokenizer["vocab_size"]):
        raise ConfigError("model.vocab_size must match tokenizer.vocab_size")
    _require_positive_int(model, "max_seq_len", "model.max_seq_len")
    _require_positive_int(model, "n_layers", "model.n_layers")
    d_model = _require_positive_int(model, "d_model", "model.d_model")
    n_heads = _require_positive_int_any(
        model,
        ("num_attention_heads", "n_heads"),
        "model.num_attention_heads",
    )
    num_kv_heads = _require_positive_int_any(
        model,
        ("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
        "model.num_key_value_heads",
    )
    _require_positive_number(model, "ffn_multiplier", "model.ffn_multiplier")
    _require_positive_int(model, "multiple_of", "model.multiple_of")
    _require_positive_number(model, "norm_eps", "model.norm_eps")
    _require_positive_number(model, "rope_theta", "model.rope_theta")
    _required(model, "dropout", "model.dropout")
    _require_bool(model, "tie_embeddings", "model.tie_embeddings")
    if d_model % n_heads != 0:
        raise ConfigError("model.d_model must be divisible by model.num_attention_heads")
    if n_heads % num_kv_heads != 0:
        raise ConfigError(
            "model.num_attention_heads must be divisible by model.num_key_value_heads"
        )
    if _required_any(model, ("flash_attention", "use_flash_attention"), "model.flash_attention"):
        head_dim = d_model // n_heads
        if head_dim % 8 != 0:
            raise ConfigError(
                "model.d_model / model.num_attention_heads must be divisible by 8 "
                "when model.flash_attention=true"
            )

    training = _section(config, "training")
    distributed = _require_mapping(
        training,
        "distributed",
        "training.distributed",
    )
    _required(distributed, "enabled", "training.distributed.enabled")
    strategy = _required(distributed, "strategy", "training.distributed.strategy")
    if strategy != "ddp":
        raise ConfigError("training.distributed.strategy must be 'ddp'")
    _require_positive_int(distributed, "num_gpus", "training.distributed.num_gpus")

    optimizer_name = _required(training, "optimizer.name", "training.optimizer.name")
    if optimizer_name != "adamw":
        raise ConfigError("training.optimizer.name must be 'adamw'")

    scheduler_name = _required(training, "scheduler.name", "training.scheduler.name")
    if scheduler_name != "cosine":
        raise ConfigError("training.scheduler.name must be 'cosine'")

    _require_directory_value(_section(config, "project"), "output_dir", "project.output_dir")
    _require_directory_value(tokenizer, "save_dir", "tokenizer.save_dir")
    _require_directory_value(
        training,
        "checkpointing.save_dir",
        "training.checkpointing.save_dir",
    )
    _require_directory_value(_section(config, "plots"), "output_dir", "plots.output_dir")

    logging = _section(config, "logging")
    if logging.get("use_tensorboard", False):
        _require_directory_value(logging, "tensorboard_dir", "logging.tensorboard_dir")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[2] / candidate


def _mkdir(path: str | Path) -> None:
    _resolve_path(path).mkdir(parents=True, exist_ok=True)


def ensure_output_directories(config: dict[str, Any]) -> None:
    project = config["project"]
    dataset = config["dataset"]
    tokenizer = config["tokenizer"]
    training = config["training"]
    logging = config["logging"]
    plots = config["plots"]

    _mkdir(project["output_dir"])
    _mkdir(_resolve_path(project["output_dir"]) / "logs")
    if project.get("docs_dir"):
        _mkdir(project["docs_dir"])

    _mkdir(dataset["cache_dir"])
    if dataset.get("raw_dir"):
        _mkdir(dataset["raw_dir"])
    _mkdir(dataset["processed_dir"])
    _mkdir(tokenizer["save_dir"])
    _mkdir(training["checkpointing"]["save_dir"])
    if logging.get("use_tensorboard", False):
        _mkdir(logging["tensorboard_dir"])
    _mkdir(plots["output_dir"])
