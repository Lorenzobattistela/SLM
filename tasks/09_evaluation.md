# Task 09: Evaluation Suite (Perplexity and Downstream Benchmarks)

## Objective

Implement the evaluation suite to measure the model's perplexity (PPL) and downstream performance on multiple-choice benchmarks and math reasoning tasks. To simplify development, we will utilize an evaluation library (such as Hugging Face's `evaluate` or EleutherAI's `lm-evaluation-harness`, referred to as `lm-eval`).

---

## Required Features

The evaluation suite must support:

1. **Perplexity (PPL) Evaluation**:
   - Compute PPL on a validation split of the pre-training dataset (FineWeb-Edu).
   - Use the formula: `PPL = exp(mean_cross_entropy_loss)`.
   - Track validation perplexity over time across model checkpoints.

2. **Multiple-Choice Benchmarks (log-likelihood method)**:
   - Evaluate model accuracy on:
     - **HellaSwag** (common-sense sentence completion)
     - **ARC-Easy / ARC-Challenge** (science questions)
     - **PIQA** (physical common-sense reasoning)
     - **WinoGrande** (pronoun resolution)
   - *Log-Likelihood Evaluation*: For each question/prompt, calculate the conditional log-likelihood of each candidate answer. The candidate with the highest log-likelihood is chosen as the prediction.

3. **Reasoning Benchmark (GSM8K Accuracy)**:
   - Evaluate model accuracy on the GSM8K dataset using exact match parsing on the final numeric answer (e.g., extracting numbers after `####`).

4. **Evaluation Library Integration**:
   - Wrap the custom model architecture and tokenizer to make it compatible with EleutherAI's `lm-eval` or Hugging Face's `evaluate` library, allowing zero-shot execution of standard benchmarks.

---

## Required Command

Execute the evaluation script via the CLI:

```bash
python scripts/evaluate_benchmarks.py --run-config configs/eval.yml --checkpoint path/to/model.pt
```

---

## Required Source Structure

Create or adapt the following files:

```text
configs/eval.yml
scripts/evaluate_benchmarks.py
src/evaluation/benchmarks.py
```

### Config File (`configs/eval.yml`)
Configure evaluation datasets and steps:
```yaml
project:
  name: "llm_200m_evaluation"

evaluation:
  ppl_dataset: "HuggingFaceFW/fineweb-edu"
  benchmarks:
    - hellaswag
    - arc_easy
    - piqa
    - winogrande
    - gsm8k
  batch_size: 16
```

### Evaluation script (`scripts/evaluate_benchmarks.py`)
This script must load a checkpoint and run evaluations:
1. Initialize the model and tokenizer from configuration and the specified checkpoint.
2. Initialize standard evaluation library adapters (e.g., `lm-eval` API wrapper).
3. Compute validation perplexity.
4. Execute multiple-choice benchmarks and parse predictions.
5. Print and save results into a JSON file (e.g., `outputs/evaluation_results.json`).

---

## Testing

Run the evaluation script on any available checkpoint:

```bash
python scripts/evaluate_benchmarks.py --run-config configs/eval.yml --checkpoint outputs/llm_200m_sft/checkpoints/final.pt
```

Expected outcomes:
- Perplexity is printed.
- Zero-shot benchmarks are run and results are saved in JSON format.
- Output file contains accuracy scores for all selected benchmarks.

---

## Acceptance Criteria

This task is complete when:
- `scripts/evaluate_benchmarks.py` is functional.
- Perplexity is correctly computed.
- Multiple-choice benchmarks (e.g. HellaSwag/ARC) are evaluated via conditional log-likelihood.
- Results are saved to a JSON file.
