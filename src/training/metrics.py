from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LossStats:
    loss: float
    perplexity: float


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(min(100.0, float(loss)))
    except OverflowError:
        return float("inf")


def append_metrics(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
