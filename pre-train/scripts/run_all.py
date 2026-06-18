from __future__ import annotations
# ruff: noqa: E402

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch


def _find_project_root(start: Path) -> Path:
    for parent in (start.parent, *start.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import add_run_config_argument, load_config_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or guide the full FineWeb-Edu pipeline.")
    add_run_config_argument(parser)
    parser.add_argument(
        "--skip-ddp",
        action="store_true",
        help="Do not launch torchrun internally; print the required training command instead.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def should_launch_ddp(
    ddp_cfg: dict,
    *,
    skip_ddp: bool = False,
    cuda_available: bool | None = None,
    cuda_device_count: int | None = None,
) -> bool:
    if skip_ddp:
        return False

    uses_nccl = str(ddp_cfg.get("backend", "")).lower() == "nccl"
    if not uses_nccl:
        return True

    requested_gpus = int(ddp_cfg["num_gpus"])
    has_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    device_count = torch.cuda.device_count() if cuda_device_count is None else cuda_device_count
    return has_cuda and device_count >= requested_gpus


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config_from_args(args)
    run_config = args.run_config
    python = sys.executable

    _run([python, "pre-train/scripts/tokenize_dataset.py", "--run-config", run_config])
    _run([python, "scripts/count_parameters.py", "--run-config", run_config])

    ddp_cfg = config["training"]["distributed"]
    ddp_command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={int(ddp_cfg['num_gpus'])}",
        "pre-train/scripts/train.py",
        "--run-config",
        run_config,
    ]

    if not should_launch_ddp(ddp_cfg, skip_ddp=args.skip_ddp):
        print("Run training with:")
        print(" ".join(ddp_command))
        return

    try:
        _run(ddp_command)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print("Could not complete DDP training from run_all.py.")
        print("Run training manually with:")
        print(" ".join(ddp_command))
        raise SystemExit(1) from exc

    _run([python, "benchmarks/scripts/evaluate.py", "--run-config", run_config])

    metrics_path = Path(config["project"]["output_dir"]) / "logs" / "metrics.jsonl"
    if metrics_path.exists():
        _run(
            [
                python,
                "scripts/plot_training.py",
                "--run-config",
                run_config,
            ]
        )
    else:
        logging.warning("Metrics file not found yet, skipping plot generation: %s", metrics_path)

    checkpoint_path = Path(config["training"]["checkpointing"]["save_dir"]) / "latest.pt"
    _run(
        [
            python,
            "scripts/sample_checkpoint.py",
            "--run-config",
            run_config,
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            "Scientific progress depends on",
        ]
    )


if __name__ == "__main__":
    main()
