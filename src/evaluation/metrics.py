"""Standard cross-modal retrieval metrics, computed bidirectionally
(image->text and text->image) and averaged, which is the convention used
by Flickr30k/COCO retrieval benchmarks so numbers here are comparable to
published CLIP results.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def compute_similarity_matrix(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
    return image_embeds @ text_embeds.t()


def recall_at_k(sim_matrix: torch.Tensor, k_values=(1, 5, 10)) -> dict:
    """Assumes sim_matrix[i, j] pairs image i with text j, and that the
    ground-truth match for row i is column i (caller is responsible for
    aligning embeddings this way, which PairedRetrievalDataset guarantees
    for one-caption-per-image eval splits).
    """
    n = sim_matrix.size(0)
    ranks_i2t = _ranks_of_correct_match(sim_matrix, torch.arange(n))
    ranks_t2i = _ranks_of_correct_match(sim_matrix.t(), torch.arange(n))

    results = {}
    for k in k_values:
        results[f"i2t_recall@{k}"] = float((ranks_i2t < k).float().mean())
        results[f"t2i_recall@{k}"] = float((ranks_t2i < k).float().mean())
        results[f"recall@{k}"] = (results[f"i2t_recall@{k}"] + results[f"t2i_recall@{k}"]) / 2
    return results


def _ranks_of_correct_match(sim_matrix: torch.Tensor, correct_idx: torch.Tensor) -> torch.Tensor:
    order = sim_matrix.argsort(dim=1, descending=True)
    ranks = torch.zeros(sim_matrix.size(0), dtype=torch.long)
    for i in range(sim_matrix.size(0)):
        ranks[i] = (order[i] == correct_idx[i]).nonzero(as_tuple=True)[0].item()
    return ranks


def mean_average_precision(sim_matrix: torch.Tensor) -> float:
    """mAP for the single-relevant-item-per-query case reduces to the mean
    reciprocal rank of the correct match; kept as a separate named metric
    since that's the convention the results table uses.
    """
    n = sim_matrix.size(0)
    ranks = _ranks_of_correct_match(sim_matrix, torch.arange(n))
    return float((1.0 / (ranks.float() + 1)).mean())


def full_eval(image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> dict:
    sim = compute_similarity_matrix(image_embeds, text_embeds)
    metrics = recall_at_k(sim)
    metrics["mAP"] = mean_average_precision(sim)
    return metrics
