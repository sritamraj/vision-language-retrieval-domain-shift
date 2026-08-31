"""Qualitative side-by-side grids: for a given text query (or image query),
show the top-K retrieved images from the baseline model vs. the adapted
model. This is what the Gradio demo (app/demo.py) reuses, and what
notebooks/03_error_analysis.ipynb uses to build the failure-case gallery.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
from PIL import Image


def topk_indices(sim_row, k: int = 5):
    return sim_row.argsort(descending=True)[:k].tolist()


def plot_retrieval_comparison(
    query_text: str,
    baseline_paths: Sequence[str],
    adapted_paths: Sequence[str],
    image_root: str,
    out_path: str,
):
    k = max(len(baseline_paths), len(adapted_paths))
    fig, axes = plt.subplots(2, k, figsize=(2.2 * k, 5))

    for row, (paths, row_label) in enumerate([(baseline_paths, "Baseline"), (adapted_paths, "Adapted (Method B)")]):
        for col in range(k):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(paths):
                img = Image.open(os.path.join(image_root, paths[col])).convert("RGB")
                ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(row_label, fontsize=10)
            if row == 0:
                ax.set_title(f"#{col + 1}", fontsize=9)

    fig.suptitle(f'Query: "{query_text}"', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
