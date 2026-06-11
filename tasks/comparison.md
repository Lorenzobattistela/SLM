# Comparison Tasks: Evaluating Tokenizer Impact on SLM Performance

This document outlines the evaluation tasks to compare two Small Language Models (SLMs) that have identical architectures but use different tokenizers:
1. **Model A**: Uses the **SuperBPE** tokenizer.
2. **Model B**: Uses a standard **Byte-level BPE** (GPT-2 style) tokenizer.

Because the tokenizers are different, raw perplexity and training steps cannot be directly compared without adjustment. The following comparative tasks must be conducted to evaluate the impact of the tokenizer on compression, representation, training efficiency, and downstream task capabilities.

---

## Task 1: Tokenizer Fertility & Compression Efficiency Comparison

### Objective
Measure how efficiently each tokenizer partitions text and how many tokens are required to represent a standard corpus.

### Method
1. Select a held-out evaluation corpus (e.g., a test split of FineWeb-Edu or a raw text document).
2. For each tokenizer, calculate:
   - **Fertility Rate**: Average number of tokens per word (`total_tokens / total_words`) and average number of tokens per character (`total_tokens / total_characters`).
   - **Vocabulary Coverage**: Rate of out-of-vocabulary (OOV) or unknown tokens.
   - **Sequence Length Savings**: Average context length consumption.
3. **Analysis**: A lower fertility rate is generally better, as it allows the model to process more text in the same context window (e.g., 2048 tokens can contain more words/information).

---

## Task 2: Character-Normalized Perplexity (Fair PPL Comparison)

### Objective
Obtain a mathematically valid perplexity comparison. Standard perplexity (PPL) is calculated per token and depends directly on the tokenization vocabulary, making direct cross-tokenizer PPL comparisons invalid.

### Method
1. Evaluate both models on the same validation text dataset.
2. Calculate the cross-entropy loss.
3. Compute the **Character-Normalized Cross-Entropy** or **Bits-per-Character (BPC)**:
   $$\text{BPC} = \frac{\text{Total Loss (in base 2)}}{\text{Total Characters}}$$
   $$\text{Perplexity per character} = \exp\left( \frac{\text{Total Loss (in base } e\text{)}}{\text{Total Characters}} \right)$$
4. **Analysis**: Since the denominator (number of characters) is constant across both models, this provides a fair, normalized baseline of how well each model predicts the next character.

---

## Task 3: Zero-Shot Downstream Task Accuracy

### Objective
Determine if vocabulary choices and boundary partitions affect the model's ability to learn semantic relationships and factual knowledge.

### Method
1. Evaluate both models using the evaluation suite (`scripts/evaluate_benchmarks.py`) on:
   - HellaSwag (Common sense reasoning)
   - ARC-Easy / ARC-Challenge (Science questions)
   - PIQA (Physical reasoning)
   - WinoGrande (Pronoun resolution)
2. Compare the accuracies and standard errors.
3. **Analysis**: Analyze if one model outperforms the other on tasks involving specialized terms (e.g., ARC or GSM8K) where SuperBPE's vocabulary partition may capture word roots or math tokens differently.

---

## Task 4: Training Throughput and Efficiency

### Objective
Assess how tokenizer choices affect training speed and compute requirements.

### Method
1. Record:
   - Tokens per second processed during pretraining/fine-tuning.
   - **Characters per second** processed during pretraining/fine-tuning:
     $$\text{Chars/sec} = \text{Tokens/sec} \times \frac{1}{\text{Fertility}}$$
2. **Analysis**: Since model compute is proportional to the number of processed tokens, a model with a lower fertility tokenizer can process more raw text characters per second for the same amount of FLOPS. Compare the true training speed in terms of training time required to see a fixed number of raw characters.

---

## Task 5: Qualitative Chat Generation and Latency

### Objective
Assess real-world generation behavior and end-to-end latency in chat mode.

### Method
1. Run both models on a standardized set of 20 conversational prompts.
2. Record:
   - Time to first token (latency).
   - Generation speed in tokens/second and words/second.
   - Formatting coherence (e.g., correct stop/EOS token generation, avoidance of repetition).
3. **Analysis**: A tokenizer with lower fertility should generate more words per second for the same token generation speed. Compare chat outputs for quality, vocabulary diversity, and completion speed.
