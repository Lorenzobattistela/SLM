from __future__ import annotations

import hashlib


def is_validation_text(text: str, validation_ratio: float, salt: str) -> bool:
    digest = hashlib.sha1((salt + text).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return bucket / 2**64 < validation_ratio
