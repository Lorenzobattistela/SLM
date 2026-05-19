PYTHON ?= python3

.PHONY: install prepare-tiny prepare-debug smoke train-tiny train-smoke sample plot

install:
	$(PYTHON) -m pip install -e ".[dev]"

prepare-tiny:
	$(PYTHON) scripts/prepare_pretrain_data.py --data-config configs/data/fineweb_edu_tiny.yaml

prepare-debug:
	$(PYTHON) scripts/prepare_pretrain_data.py --data-config configs/data/fineweb_edu_debug.yaml

smoke:
	$(PYTHON) scripts/smoke_overfit_batch.py --run-config configs/run/pretrain_local_tiny.yaml

train-tiny:
	$(PYTHON) -m src.train.pretrain --run-config configs/run/pretrain_local_tiny.yaml

train-smoke:
	$(PYTHON) -m src.train.pretrain --run-config configs/run/pretrain_remote_smoke.yaml

sample:
	$(PYTHON) scripts/sample_checkpoint.py --checkpoint runs/pretrain_local_tiny/checkpoints/latest.pt

plot:
	$(PYTHON) scripts/plot_train_loss.py --metrics runs/pretrain_local_tiny/metrics.jsonl
