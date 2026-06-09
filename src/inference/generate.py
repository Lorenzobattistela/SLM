from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    output = input_ids
    context_length = model.config.context_length

    for _ in range(max_new_tokens):
        idx_cond = output[:, -context_length:]
        logits, _ = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        if temperature <= 0: #greedy
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None: #top k sampling
                values, _ = torch.topk(next_token_logits, k=top_k)
                threshold = values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        output = torch.cat([output, next_token], dim=1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return output
