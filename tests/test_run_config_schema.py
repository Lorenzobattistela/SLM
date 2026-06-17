from __future__ import annotations

from src.config import load_run_config


def test_load_run_config_accepts_multi_source_sft_yaml() -> None:
    config = load_run_config(
        "supervised-fine-tuning/configs/sft_200m_byte_bpe_gpt2.yml",
        create_dirs=False,
    )

    assert config["dataset"]["sources"]
    assert config["dataset"]["processed_dir"].endswith("processed")
    assert config["model"]["vocab_size"] == config["tokenizer"]["vocab_size"]


def test_load_run_config_accepts_multi_source_mid_train_yaml() -> None:
    config = load_run_config(
        "mid-training/configs/mid_train_200m_byte_bpe_gpt2.yml",
        create_dirs=False,
    )

    assert config["dataset"]["sources"]
    assert config["training"]["checkpointing"]["save_dir"].endswith("checkpoints")
