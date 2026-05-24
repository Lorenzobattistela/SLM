PYTHON ?= python3
RUN_CONFIG ?= configs/train_200m_fineweb_edu.yml
DEBUG_CONFIG ?= configs/train_200m_fineweb_edu_debug.yml
NPROC_PER_NODE ?= 2
CHECKPOINT ?= checkpoints/llm_200m_fineweb_edu/latest.pt
PROMPT ?= Scientific progress depends on

.PHONY: install prepare-tokenizer tokenize count train-ddp evaluate sample plot run-all debug-tokenizer debug-tokenize debug-count test

install:
	$(PYTHON) -m pip install -e ".[dev]"

prepare-tokenizer:
	$(PYTHON) -c "from src.config.loader import load_run_config; from src.tokenizer import load_superbpe_tokenizer; load_superbpe_tokenizer(load_run_config('$(RUN_CONFIG)')['tokenizer'])"

tokenize:
	$(PYTHON) scripts/tokenize_dataset.py --run-config $(RUN_CONFIG)

count:
	$(PYTHON) scripts/count_parameters.py --run-config $(RUN_CONFIG)

train-ddp:
	torchrun --standalone --nproc_per_node=$(NPROC_PER_NODE) scripts/train.py --run-config $(RUN_CONFIG)

evaluate:
	$(PYTHON) scripts/evaluate.py --run-config $(RUN_CONFIG)

sample:
	$(PYTHON) scripts/sample_checkpoint.py --run-config $(RUN_CONFIG) --checkpoint $(CHECKPOINT) --prompt "$(PROMPT)"

plot:
	$(PYTHON) scripts/plot_training.py --run-config $(RUN_CONFIG)

run-all:
	$(PYTHON) scripts/run_all.py --run-config $(RUN_CONFIG)

debug-tokenizer:
	$(PYTHON) -c "from src.config.loader import load_run_config; from src.tokenizer import load_superbpe_tokenizer; load_superbpe_tokenizer(load_run_config('$(DEBUG_CONFIG)')['tokenizer'])"

debug-tokenize:
	$(PYTHON) scripts/tokenize_dataset.py --run-config $(DEBUG_CONFIG)

debug-count:
	$(PYTHON) scripts/count_parameters.py --run-config $(DEBUG_CONFIG)

test:
	$(PYTHON) -m pytest
