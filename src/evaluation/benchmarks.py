from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def evaluate_perplexity(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_name: str,
    device: torch.device,
    limit: int = 10,
) -> float:
    """Compute perplexity on FineWeb-Edu using exp(mean_cross_entropy_loss)."""
    model.eval()
    LOGGER.info("Evaluating perplexity on %s (limit=%s)...", dataset_name, limit)

    try:
        # Load sample-10BT split or default to train split with streaming
        ds = load_dataset(dataset_name, name="sample-10BT", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)
        except Exception as e:
            LOGGER.error("Failed to load perplexity dataset %s: %s", dataset_name, e)
            return 0.0

    bos_id = tokenizer.special_token_ids.get("bos_token")
    eos_id = tokenizer.special_token_ids.get("eos_token")

    total_loss_tokens = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(ds):
            if i >= limit:
                break
            text = sample.get("text", "")
            if not text:
                continue
            token_ids = [bos_id] + tokenizer.encode(text) + [eos_id]

            block_size = model.config.context_length
            if len(token_ids) > block_size + 1:
                token_ids = token_ids[: block_size + 1]
            if len(token_ids) < 2:
                continue

            inputs = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

            _, loss = model(inputs, targets)
            num_tokens = inputs.numel()
            total_loss_tokens += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens == 0:
        return 0.0

    mean_loss = total_loss_tokens / total_tokens
    perplexity = math.exp(mean_loss)
    return perplexity


@torch.no_grad()
def get_log_likelihood(
    model: torch.nn.Module,
    tokenizer: Any,
    context: str,
    candidate: str,
    device: torch.device,
    bos_id: int,
) -> float:
    """Compute conditional log-likelihood of a candidate answer given the context."""
    context_tokens = [bos_id] + tokenizer.encode(context)
    candidate_tokens = tokenizer.encode(candidate)

    full_tokens = context_tokens + candidate_tokens
    block_size = model.config.context_length
    if len(full_tokens) > block_size:
        # Truncate context from the left if it exceeds context length
        excess = len(full_tokens) - block_size
        context_tokens = context_tokens[excess:]
        full_tokens = context_tokens + candidate_tokens

    inputs = torch.tensor([full_tokens], dtype=torch.long, device=device)

    logits, _ = model(inputs)

    # Shift logits to match predicting next token
    # Logit for predicting candidate_tokens[0] is at index len(context_tokens) - 1 in full logits
    # Logit for predicting candidate_tokens[-1] is at index len(full_tokens) - 2 in full logits
    start_idx = len(context_tokens) - 1
    end_idx = len(full_tokens) - 1

    target_logits = logits[0, start_idx:end_idx, :]
    target_tokens = torch.tensor(candidate_tokens, dtype=torch.long, device=device)

    log_probs = F.log_softmax(target_logits, dim=-1)
    gathered_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

    return gathered_log_probs.sum().item()


def evaluate_hellaswag(
    model: torch.nn.Module, tokenizer: Any, device: torch.device, limit: int = 10
) -> float:
    """Evaluate HellaSwag accuracy using conditional log-likelihood."""
    LOGGER.info("Evaluating HellaSwag (limit=%s)...", limit)
    try:
        ds = load_dataset("allenai/hellaswag", split="validation", streaming=True)
    except Exception as e:
        LOGGER.error("Failed to load HellaSwag: %s", e)
        return 0.0

    bos_id = tokenizer.special_token_ids.get("bos_token")
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        if i >= limit:
            break
        context = sample.get("ctx", "")
        endings = sample.get("endings", [])
        label = sample.get("label", "")
        if not endings or label == "":
            continue

        try:
            label_idx = int(label)
        except ValueError:
            continue

        log_likes = []
        for ending in endings:
            log_likes.append(
                get_log_likelihood(model, tokenizer, context, ending, device, bos_id)
            )

        pred = int(np.argmax(log_likes))
        if pred == label_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_arc_easy(
    model: torch.nn.Module, tokenizer: Any, device: torch.device, limit: int = 10
) -> float:
    """Evaluate ARC-Easy accuracy using conditional log-likelihood."""
    LOGGER.info("Evaluating ARC-Easy (limit=%s)...", limit)
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", streaming=True)
    except Exception as e:
        LOGGER.error("Failed to load ARC-Easy: %s", e)
        return 0.0

    bos_id = tokenizer.special_token_ids.get("bos_token")
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        if i >= limit:
            break
        question = sample.get("question", "")
        choices = sample.get("choices", {})
        answer_key = sample.get("answerKey", "")
        if not choices or not answer_key:
            continue

        texts = choices.get("text", [])
        labels = choices.get("label", [])
        if not texts or not labels or answer_key not in labels:
            continue

        label_idx = labels.index(answer_key)

        log_likes = []
        for text in texts:
            log_likes.append(
                get_log_likelihood(model, tokenizer, question, text, device, bos_id)
            )

        pred = int(np.argmax(log_likes))
        if pred == label_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_piqa(
    model: torch.nn.Module, tokenizer: Any, device: torch.device, limit: int = 10
) -> float:
    """Evaluate PIQA accuracy using conditional log-likelihood."""
    LOGGER.info("Evaluating PIQA (limit=%s)...", limit)
    try:
        ds = load_dataset("lighteval/piqa", split="validation", streaming=True)
    except Exception as e:
        LOGGER.error("Failed to load PIQA: %s", e)
        return 0.0

    bos_id = tokenizer.special_token_ids.get("bos_token")
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        if i >= limit:
            break
        goal = sample.get("goal", "")
        sol1 = sample.get("sol1", "")
        sol2 = sample.get("sol2", "")
        label = sample.get("label", -1)
        if label not in (0, 1):
            continue

        log_likes = []
        for sol in (sol1, sol2):
            log_likes.append(
                get_log_likelihood(model, tokenizer, goal, sol, device, bos_id)
            )

        pred = int(np.argmax(log_likes))
        if pred == label:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_winogrande(
    model: torch.nn.Module, tokenizer: Any, device: torch.device, limit: int = 10
) -> float:
    """Evaluate WinoGrande accuracy using conditional log-likelihood."""
    LOGGER.info("Evaluating WinoGrande (limit=%s)...", limit)
    try:
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", streaming=True)
    except Exception as e:
        LOGGER.error("Failed to load WinoGrande: %s", e)
        return 0.0

    bos_id = tokenizer.special_token_ids.get("bos_token")
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        if i >= limit:
            break
        sentence = sample.get("sentence", "")
        option1 = sample.get("option1", "")
        option2 = sample.get("option2", "")
        answer = sample.get("answer", "")
        if not sentence or not option1 or not option2 or answer not in ("1", "2"):
            continue

        label_idx = int(answer) - 1

        # Evaluate log-likelihood of option completions
        log_likes = []
        for opt in (option1, option2):
            # Replace '_' with candidate option
            context, ending = sentence.split("_", 1)
            ending = opt + ending
            log_likes.append(
                get_log_likelihood(model, tokenizer, context, ending, device, bos_id)
            )

        pred = int(np.argmax(log_likes))
        if pred == label_idx:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def evaluate_gsm8k(
    model: torch.nn.Module, tokenizer: Any, device: torch.device, limit: int = 10
) -> float:
    """Evaluate GSM8K accuracy using exact match parsing on numeric answers."""
    LOGGER.info("Evaluating GSM8K (limit=%s)...", limit)
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    except Exception as e:
        LOGGER.error("Failed to load GSM8K: %s", e)
        return 0.0

    bos_id = tokenizer.special_token_ids.get("bos_token")
    eos_id = tokenizer.special_token_ids.get("eos_token")
    correct = 0
    total = 0

    for i, sample in enumerate(ds):
        if i >= limit:
            break
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        if not question or not answer:
            continue

        # Extract target numeric answer
        target_val = ""
        if "####" in answer:
            target_val = answer.split("####")[-1].strip()

        # Greedy generation of output response
        model.eval()
        prompt_text = f"User: {question}\nAssistant:"
        prompt_tokens = [bos_id] + tokenizer.encode(prompt_text)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        generated_ids = []
        with torch.no_grad():
            for _ in range(128):
                logits, _ = model(input_ids)
                next_token_logits = logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()

                if next_token_id == eos_id:
                    break
                generated_ids.append(next_token_id)
                input_ids = torch.cat(
                    [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
                )
                if input_ids.shape[1] > model.config.context_length:
                    input_ids = input_ids[:, -model.config.context_length :]

        gen_text = tokenizer.decode(generated_ids)

        # Parse generated numeric value (e.g. looking for ####, or last digits sequence)
        gen_val = ""
        if "####" in gen_text:
            gen_val = gen_text.split("####")[-1].strip()
        else:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", gen_text)
            if numbers:
                gen_val = numbers[-1]

        if target_val and gen_val == target_val:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def run_all_benchmarks(
    model: torch.nn.Module,
    tokenizer: Any,
    benchmarks_list: list[str],
    ppl_dataset: str,
    device: torch.device,
    limit: int = 10,
) -> dict[str, float]:
    """Run perplexity and all configured benchmarks."""
    results = {}

    for benchmark in benchmarks_list:
        bench_lower = benchmark.lower()
        if bench_lower == "perplexity":
            results["perplexity"] = evaluate_perplexity(
                model, tokenizer, ppl_dataset, device, limit=limit
            )
        elif bench_lower == "hellaswag":
            results["hellaswag_accuracy"] = evaluate_hellaswag(
                model, tokenizer, device, limit=limit
            )
        elif bench_lower in ("arc_easy", "arc-easy"):
            results["arc_easy_accuracy"] = evaluate_arc_easy(
                model, tokenizer, device, limit=limit
            )
        elif bench_lower == "piqa":
            results["piqa_accuracy"] = evaluate_piqa(model, tokenizer, device, limit=limit)
        elif bench_lower == "winogrande":
            results["winogrande_accuracy"] = evaluate_winogrande(
                model, tokenizer, device, limit=limit
            )
        elif bench_lower == "gsm8k":
            # For gsm8k, limit to half for speed if limit is large
            results["gsm8k_accuracy"] = evaluate_gsm8k(
                model, tokenizer, device, limit=max(1, limit // 2)
            )
        else:
            LOGGER.warning("Unknown benchmark: %s. Skipping.", benchmark)

    return results
