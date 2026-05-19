from __future__ import annotations

from collections.abc import Iterator

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - dependency is checked at runtime.
    load_dataset = None


def _infer_text_field(sample: dict, preferred: str | None = None) -> str:
    if preferred and preferred in sample:
        return preferred
    for field_name in ("text", "content", "raw_content", "document"):
        value = sample.get(field_name)
        if isinstance(value, str):
            return field_name
    raise KeyError(
        "Could not infer a text field from the dataset sample. "
        "Set `dataset.text_field` in the data config."
    )


def iter_dataset_texts(dataset_cfg: dict) -> Iterator[str]:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed. Run `pip install -e .` first.")

    dataset = load_dataset(
        dataset_cfg["id"],
        name=dataset_cfg.get("name"),
        split=dataset_cfg.get("split", "train"),
        revision=dataset_cfg.get("revision"),
        streaming=dataset_cfg.get("streaming", True),
    )
    text_field = dataset_cfg.get("text_field")

    for sample in dataset:
        if text_field is None:
            text_field = _infer_text_field(sample)
        text = sample.get(text_field)
        if isinstance(text, str):
            yield text
