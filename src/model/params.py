from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class ParameterReport:
    total: int
    trainable: int
    embedding: int
    attention: int
    ffn: int
    norm: int
    lm_head: int
    other: int
    tied_embeddings: bool


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def describe_parameters(model: nn.Module) -> ParameterReport:
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)

    embedding = 0
    attention = 0
    ffn = 0
    norm = 0
    lm_head = 0
    other = 0

    for name, parameter in model.named_parameters():
        n_params = parameter.numel()
        if name.startswith("tok_embeddings."):
            embedding += n_params
        elif ".attn." in name:
            attention += n_params
        elif ".mlp." in name:
            ffn += n_params
        elif name.startswith("lm_head."):
            lm_head += n_params
        elif "norm" in name:
            norm += n_params
        else:
            other += n_params

    tied_embeddings = getattr(model, "lm_head", None) is not None and (
        getattr(model.lm_head, "weight", None) is getattr(model, "tok_embeddings").weight
    )

    return ParameterReport(
        total=total,
        trainable=trainable,
        embedding=embedding,
        attention=attention,
        ffn=ffn,
        norm=norm,
        lm_head=lm_head,
        other=other,
        tied_embeddings=tied_embeddings,
    )
