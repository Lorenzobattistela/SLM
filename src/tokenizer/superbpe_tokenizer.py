from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.config.loader import resolve_project_path

LOGGER = logging.getLogger(__name__)
DEFAULT_PRETRAINED_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "meta.json")

SUPERBPE_STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?"
    r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SUPERBPE_STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


class SuperBPEError(RuntimeError):
    pass


class SuperBPEBackendError(SuperBPEError):
    pass


@dataclass
class SuperBPETokenizer:
    tokenizer: Any
    save_dir: Path
    special_token_ids: dict[str, int]
    vocab_size: int

    @property
    def eos_token_id(self) -> int | None:
        return self.special_token_ids.get("eos_token")

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids = list(self.tokenizer.encode(text).ids)
        if add_eos and self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(token_ids))


def ensure_superbpe_type(tokenizer_cfg: dict[str, Any]) -> None:
    tokenizer_type = tokenizer_cfg.get("type")
    if tokenizer_type != "superbpe":
        raise SuperBPEError(
            f"Unsupported tokenizer.type={tokenizer_type!r}. "
            "Task 02 requires tokenizer.type: 'superbpe'."
        )


def special_token_values(tokenizer_cfg: dict[str, Any]) -> list[str]:
    special_tokens = tokenizer_cfg.get("special_tokens")
    if not isinstance(special_tokens, dict) or not special_tokens:
        raise SuperBPEError("tokenizer.special_tokens must define pad/bos/eos/unk tokens.")
    return [value for value in special_tokens.values() if isinstance(value, str)]


def tokenizer_artifact_path(tokenizer_cfg: dict[str, Any]) -> Path:
    return resolve_project_path(tokenizer_cfg["save_dir"]) / "tokenizer.json"


def _pretrained_cfg(tokenizer_cfg: dict[str, Any]) -> dict[str, Any]:
    pretrained = tokenizer_cfg.get("pretrained")
    return pretrained if isinstance(pretrained, dict) else {}


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp_target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        if tmp_target.exists():
            tmp_target.unlink()
        raise SuperBPEError(f"Could not download pretrained SuperBPE artifact: {url}") from exc
    tmp_target.replace(target)


def ensure_pretrained_superbpe_artifacts(tokenizer_cfg: dict[str, Any]) -> None:
    pretrained = _pretrained_cfg(tokenizer_cfg)
    if not pretrained:
        return

    output_dir = resolve_project_path(tokenizer_cfg["save_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(pretrained.get("base_url", "")).rstrip("/")
    files = pretrained.get("files", DEFAULT_PRETRAINED_FILES)
    if not base_url:
        raise SuperBPEError("tokenizer.pretrained.base_url is required for pretrained SuperBPE.")
    if not isinstance(files, (list, tuple)) or not files:
        raise SuperBPEError("tokenizer.pretrained.files must list at least tokenizer.json.")

    downloaded: list[str] = []
    for filename in files:
        relative_name = str(filename)
        target = output_dir / relative_name
        if target.exists() and target.stat().st_size > 0:
            continue
        LOGGER.info("Downloading pretrained SuperBPE artifact: %s", relative_name)
        _download_file(f"{base_url}/{relative_name}", target)
        downloaded.append(relative_name)

    if downloaded:
        metadata = {
            "tokenizer_type": "superbpe",
            "source": "pretrained",
            "name": pretrained.get("name", "unknown"),
            "base_url": base_url,
            "files": list(files),
            "downloaded_files": downloaded,
            "tokenizer_json": str(tokenizer_artifact_path(tokenizer_cfg)),
        }
        write_tokenizer_metadata(output_dir, metadata)


def _read_direct_url(dist_name: str) -> str:
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return ""
    return dist.read_text("direct_url.json") or ""


def _backend_looks_like_official_superbpe(tokenizers_module: Any) -> bool:
    paths_to_check = [str(getattr(tokenizers_module, "__file__", ""))]
    paths_to_check.append(_read_direct_url("tokenizers"))
    return any("superbpe" in value.lower() for value in paths_to_check)


def _load_tokenizers_backend(tokenizer_cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        from tokenizers import Regex, Tokenizer, pre_tokenizers
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel, Split
        from tokenizers.trainers import BpeTrainer
        import tokenizers as tokenizers_module
    except ImportError as exc:
        raise SuperBPEBackendError(
            "SuperBPE tokenizer support requires the official SuperBPE tokenizer backend. "
            "Install the SuperBPE project in a dedicated environment, including its custom "
            "`tokenizers_superbpe/bindings/python` package from "
            "https://github.com/PythonNut/superbpe. The pipeline did not fall back to "
            "standard BPE."
        ) from exc

    allow_unverified = bool(tokenizer_cfg.get("allow_unverified_superbpe_backend", False))
    if not _backend_looks_like_official_superbpe(tokenizers_module):
        if not allow_unverified:
            raise SuperBPEBackendError(
                "Found a `tokenizers` package, but could not verify that it is the official "
                "SuperBPE fork. Install PythonNut/superbpe with its custom tokenizers backend, "
                "or set tokenizer.allow_unverified_superbpe_backend: true only for an explicit "
                "development experiment. The default path refuses to silently use standard BPE."
            )
        LOGGER.warning(
            "Using an unverified tokenizers backend because "
            "tokenizer.allow_unverified_superbpe_backend=true. This is a development fallback; "
            "verify artifacts before using them for the final run."
        )

    return {
        "Regex": Regex,
        "Tokenizer": Tokenizer,
        "pre_tokenizers": pre_tokenizers,
        "ByteLevelDecoder": ByteLevelDecoder,
        "BPE": BPE,
        "ByteLevel": ByteLevel,
        "Split": Split,
        "BpeTrainer": BpeTrainer,
    }


def validate_superbpe_backend(tokenizer_cfg: dict[str, Any]) -> None:
    ensure_superbpe_type(tokenizer_cfg)
    _load_tokenizers_backend(tokenizer_cfg)


def _set_superbpe_pretokenizer(tokenizer: Any, backend: dict[str, Any], regex_string: str) -> None:
    tokenizer.pre_tokenizer = backend["pre_tokenizers"].Sequence(
        [
            backend["Split"](
                pattern=backend["Regex"](regex_string),
                behavior="isolated",
                invert=False,
            ),
            backend["ByteLevel"](
                add_prefix_space=False,
                trim_offsets=True,
                use_regex=False,
            ),
        ]
    )
    tokenizer.decoder = backend["ByteLevelDecoder"](
        add_prefix_space=True,
        trim_offsets=True,
        use_regex=True,
    )


def _train_stage(
    *,
    backend: dict[str, Any],
    corpus_files: list[Path],
    output_dir: Path,
    vocab_size: int,
    min_frequency: int,
    regex_string: str,
    special_tokens: list[str],
    unk_token: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        tokenizer = backend["Tokenizer"](backend["BPE"](unk_token=unk_token))
        _set_superbpe_pretokenizer(tokenizer, backend, regex_string)
        trainer = backend["BpeTrainer"](
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )
        tokenizer.train([str(path) for path in corpus_files], trainer)
        tokenizer.model.save(".")
        tokenizer.save("tokenizer.json")
    finally:
        os.chdir(previous_cwd)


def _copy_initial_merges(stage1_dir: Path, final_dir: Path, num_inherit_merges: int) -> None:
    source = stage1_dir / "merges.txt"
    target = final_dir / "merges.txt"
    if not source.exists():
        raise SuperBPEError(f"Stage 1 merges file was not created: {source}")

    final_dir.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for index, line in enumerate(src):
            if index >= num_inherit_merges:
                break
            dst.write(line)


def train_superbpe_tokenizer(
    *,
    tokenizer_cfg: dict[str, Any],
    corpus_files: list[Path],
    save_dir: str | Path,
) -> dict[str, Any]:
    ensure_superbpe_type(tokenizer_cfg)
    if not corpus_files:
        raise SuperBPEError("Cannot train SuperBPE tokenizer without corpus files.")

    backend = _load_tokenizers_backend(tokenizer_cfg)
    output_dir = resolve_project_path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = int(tokenizer_cfg["vocab_size"])
    min_frequency = int(tokenizer_cfg.get("min_frequency", 2))
    stage1_vocab_size = int(tokenizer_cfg.get("superbpe_stage1_vocab_size", vocab_size * 4 // 5))
    if stage1_vocab_size <= 0 or stage1_vocab_size > vocab_size:
        raise SuperBPEError(
            "tokenizer.superbpe_stage1_vocab_size must be positive and no larger than "
            "tokenizer.vocab_size."
        )
    num_inherit_merges = int(tokenizer_cfg.get("superbpe_num_inherit_merges", stage1_vocab_size))
    stage1_regex = str(tokenizer_cfg.get("superbpe_stage1_regex", SUPERBPE_STAGE1_REGEX))
    stage2_regex = str(tokenizer_cfg.get("superbpe_stage2_regex", SUPERBPE_STAGE2_REGEX))
    special_tokens = special_token_values(tokenizer_cfg)
    unk_token = tokenizer_cfg.get("special_tokens", {}).get("unk_token")

    stage1_dir = output_dir / "stage1_subword"
    final_dir = output_dir
    LOGGER.info(
        "Training SuperBPE stage 1: vocab_size=%s min_frequency=%s output=%s",
        stage1_vocab_size,
        min_frequency,
        stage1_dir,
    )
    _train_stage(
        backend=backend,
        corpus_files=corpus_files,
        output_dir=stage1_dir,
        vocab_size=stage1_vocab_size,
        min_frequency=min_frequency,
        regex_string=stage1_regex,
        special_tokens=special_tokens,
        unk_token=unk_token,
    )

    _copy_initial_merges(stage1_dir, final_dir, num_inherit_merges)
    LOGGER.info(
        "Training SuperBPE stage 2: vocab_size=%s inherited_merges=%s output=%s",
        vocab_size,
        num_inherit_merges,
        final_dir,
    )
    _train_stage(
        backend=backend,
        corpus_files=corpus_files,
        output_dir=final_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        regex_string=stage2_regex,
        special_tokens=special_tokens,
        unk_token=unk_token,
    )

    tokenizer = load_superbpe_tokenizer(tokenizer_cfg)
    metadata = {
        "tokenizer_type": "superbpe",
        "vocab_size": tokenizer.vocab_size,
        "configured_vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "stage1_vocab_size": stage1_vocab_size,
        "num_inherit_merges": num_inherit_merges,
        "special_tokens": tokenizer_cfg.get("special_tokens", {}),
        "corpus_files": [str(path) for path in corpus_files],
        "tokenizer_json": str(tokenizer_artifact_path(tokenizer_cfg)),
    }
    write_tokenizer_metadata(output_dir, metadata)
    return metadata


def load_superbpe_tokenizer(tokenizer_cfg: dict[str, Any]) -> SuperBPETokenizer:
    ensure_superbpe_type(tokenizer_cfg)
    backend = _load_tokenizers_backend(tokenizer_cfg)
    ensure_pretrained_superbpe_artifacts(tokenizer_cfg)
    tokenizer_path = tokenizer_artifact_path(tokenizer_cfg)
    if not tokenizer_path.exists():
        hint = (
            "Configure tokenizer.pretrained for automatic download or run "
            "scripts/train_tokenizer.py first."
        )
        raise FileNotFoundError(
            f"SuperBPE tokenizer artifact not found at {tokenizer_path}. {hint}"
        )

    tokenizer = backend["Tokenizer"].from_file(str(tokenizer_path))
    configured_specials = tokenizer_cfg.get("special_tokens", {})
    special_token_ids: dict[str, int] = {}
    missing_specials: list[str] = []
    for name, token in configured_specials.items():
        if not isinstance(token, str):
            continue
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            missing_specials.append(name)
        else:
            special_token_ids[name] = token_id
    if missing_specials:
        raise SuperBPEError(
            "Loaded tokenizer is missing configured special tokens: "
            + ", ".join(sorted(missing_specials))
        )
    vocab_size = tokenizer.get_vocab_size()
    expected_vocab_size = int(tokenizer_cfg.get("vocab_size", vocab_size))
    if vocab_size != expected_vocab_size:
        raise SuperBPEError(
            f"Loaded tokenizer vocab size {vocab_size} does not match configured "
            f"tokenizer.vocab_size {expected_vocab_size}."
        )

    return SuperBPETokenizer(
        tokenizer=tokenizer,
        save_dir=tokenizer_path.parent,
        special_token_ids=special_token_ids,
        vocab_size=vocab_size,
    )


def remove_existing_tokenizer(save_dir: str | Path) -> None:
    target = resolve_project_path(save_dir)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def write_tokenizer_metadata(save_dir: str | Path, metadata: dict[str, Any]) -> Path:
    target = resolve_project_path(save_dir) / "tokenizer_metadata.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return target
