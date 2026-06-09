from __future__ import annotations

import importlib.util
import sys
from array import array
from pathlib import Path

from src.data.token_dataset import TokenBinWriter, token_file_token_count

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "retokenize_superbpe_to_byte_bpe.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "retokenize_superbpe_to_byte_bpe",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None
retokenize = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
sys.modules[SCRIPT_SPEC.name] = retokenize
SCRIPT_SPEC.loader.exec_module(retokenize)


class FakeSuperBPETokenizer:
    vocab_size = 64
    eos_token_id = 63

    def decode(self, token_ids):
        return "".join(chr(ord("a") + int(token_id)) for token_id in token_ids)


class FakeByteBPETokenizer:
    name = "fake-byte-bpe"
    vocab_size = 128
    eos_token_id = 9

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        token_ids = [10 + ord(char) - ord("a") for char in text]
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids


def _read_uint16(path: Path) -> list[int]:
    values = array("H")
    with path.open("rb") as handle:
        values.fromfile(handle, token_file_token_count(path, "uint16"))
    return list(values)


def test_retokenize_file_splits_superbpe_documents_and_appends_byte_bpe_eot(tmp_path) -> None:
    source_path = tmp_path / "source.bin"
    target_path = tmp_path / "target.bin"
    with TokenBinWriter(source_path, vocab_size=FakeSuperBPETokenizer.vocab_size, target_tokens=8) as writer:
        writer.write([0, 1, FakeSuperBPETokenizer.eos_token_id])
        writer.write([2, 3, FakeSuperBPETokenizer.eos_token_id])
        writer.write([4, 5])

    stats = retokenize.retokenize_file(
        source_path=source_path,
        target_path=target_path,
        source_dtype="uint16",
        source_tokenizer=FakeSuperBPETokenizer(),
        target_tokenizer=FakeByteBPETokenizer(),
        source_eos_token_id=FakeSuperBPETokenizer.eos_token_id,
        append_eot=True,
        chunk_tokens=2,
    )

    assert _read_uint16(target_path) == [10, 11, 9, 12, 13, 9, 14, 15, 9]
    assert stats.source_tokens == 8
    assert stats.target_tokens == 9
    assert stats.documents == 3
    assert stats.partial_source_documents == 1


def test_prepare_output_dir_allows_existing_empty_directory(tmp_path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "empty_output"
    source_dir.mkdir()
    output_dir.mkdir()

    retokenize._prepare_output_dir(output_dir, source_dir, overwrite=False)

    assert output_dir.exists()
