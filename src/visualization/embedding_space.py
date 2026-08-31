"""t-SNE / PCA views of the shared embedding space, used to *show* the
domain gap rather than just report a recall number for it — this is the
figure that makes "why the gap exists" visually obvious: source and target
image embeddings form separated clusters before adaptation, and move closer
together after Method B's MMD alignment.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_domain_embedding_gap(
    source_embeds: np.ndarray,
    target_embeds_before: np.ndarray,
    target_embeds_after: np.ndarray,
    out_path: str,
    n_sample: int = 500,
):
    rng = np.random.default_rng(42)

    def sample(x):
        idx = rng.choice(len(x), size=min(n_sample, len(x)), replace=False)
        return x[idx]

    source_s = sample(source_embeds)
    target_before_s = sample(target_embeds_before)
    target_after_s = sample(target_embeds_after)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, target_s, title in [
        (axes[0], target_before_s, "Before adaptation"),
        (axes[1], target_after_s, "After Method B (adapter + MMD)"),
    ]:
        combined = np.concatenate([source_s, target_s], axis=0)
        proj = TSNE(n_components=2, init="pca", random_state=42, perplexity=30).fit_transform(combined)
        n_src = len(source_s)
        ax.scatter(proj[:n_src, 0], proj[:n_src, 1], s=8, alpha=0.5, label="source", color="#4C72B0")
        ax.scatter(proj[n_src:, 0], proj[n_src:, 1], s=8, alpha=0.5, label="target", color="#DD8452")
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Image embedding space: source vs. target domain")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
