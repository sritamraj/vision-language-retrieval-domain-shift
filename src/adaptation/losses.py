"""Loss functions shared by Method A and Method B.

- `clip_contrastive_loss`: standard symmetric InfoNCE over a batch.
- `hard_negative_weights`: reweights the InfoNCE denominator toward the
  hardest negatives in the batch, which matters more under domain shift
  because random negatives are "too easy" once the embedding space has
  drifted (see README section 2, point 3).
- `mmd_loss`: Maximum Mean Discrepancy between a batch of target embeddings
  and a cached bank of source embeddings, used only by Method B to keep the
  adapted target distribution anchored near the source distribution instead
  of drifting freely to fit a handful of target pairs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 0.07,
    hard_negative_top_k: int | None = None,
) -> torch.Tensor:
    logits = image_embeds @ text_embeds.t() / temperature
    targets = torch.arange(logits.size(0), device=logits.device)

    if hard_negative_top_k is not None:
        logits = _mask_to_hardest_negatives(logits, targets, hard_negative_top_k)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) / 2


def _mask_to_hardest_negatives(logits: torch.Tensor, targets: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep the diagonal (positive) plus the top-k hardest off-diagonal
    entries per row; push everything else to -inf so it doesn't contribute
    to the softmax denominator. This concentrates gradient on the negatives
    that are actually confusable post-shift, rather than the whole batch.
    """
    b = logits.size(0)
    masked = logits.clone()
    off_diag = masked.masked_fill(torch.eye(b, dtype=torch.bool, device=logits.device), float("-inf"))
    top_k = min(top_k, b - 1)
    _, hard_idx = off_diag.topk(top_k, dim=1)

    keep = torch.eye(b, dtype=torch.bool, device=logits.device)
    keep.scatter_(1, hard_idx, True)
    return masked.masked_fill(~keep, float("-inf"))


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    xx = (x**2).sum(1, keepdim=True)
    yy = (y**2).sum(1, keepdim=True)
    dist = xx + yy.t() - 2 * x @ y.t()
    dist = dist.clamp(min=0)
    if sigma is None:
        sigma = dist.detach().median().clamp(min=1e-6)
    return torch.exp(-dist / (2 * sigma))


def mmd_loss(target_embeds: torch.Tensor, source_bank: torch.Tensor) -> torch.Tensor:
    """Biased MMD^2 estimator with an RBF kernel between a batch of target
    embeddings and a sampled bank of source embeddings.
    """
    k_tt = rbf_kernel(target_embeds, target_embeds).mean()
    k_ss = rbf_kernel(source_bank, source_bank).mean()
    k_ts = rbf_kernel(target_embeds, source_bank).mean()
    return k_tt + k_ss - 2 * k_ts
