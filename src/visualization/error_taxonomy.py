"""Buckets failed target-domain retrievals into a small taxonomy and plots
recovery rate per bucket, before vs. after adaptation. This is the figure
that operationalizes README section 2 ("why the gap exists") instead of
leaving it as a narrative claim: each bucket maps to one of the three
causes (visual style shift, semantic granularity mismatch, contrastive
batch geometry).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

TAXONOMY = {
    "style_shift": "Visual style shift (sketch/clipart texture vs. photo)",
    "granularity_mismatch": "Caption specificity mismatch",
    "hard_negative_confusion": "Confused with a near-duplicate class",
    "other": "Unattributed / multiple factors",
}


def build_error_dataframe(records: list[dict]) -> pd.DataFrame:
    """`records`: list of {"bucket": str, "recovered_after_adaptation": bool}
    produced by the manual/heuristic labeling step in notebook 03.
    """
    df = pd.DataFrame(records)
    summary = (
        df.groupby("bucket")["recovered_after_adaptation"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_failures_pre_adaptation", "mean": "recovery_rate"})
        .reset_index()
    )
    summary["bucket_label"] = summary["bucket"].map(TAXONOMY)
    return summary.sort_values("n_failures_pre_adaptation", ascending=False)


def plot_error_taxonomy(summary: pd.DataFrame, out_path: str):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(summary["bucket_label"], summary["n_failures_pre_adaptation"], color="#C44E52", alpha=0.8)
    ax1.set_ylabel("# failures (pre-adaptation)")
    ax1.set_xticklabels(summary["bucket_label"], rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(summary["bucket_label"], summary["recovery_rate"], color="#55A868", marker="o", linewidth=2)
    ax2.set_ylabel("Recovery rate after Method B")
    ax2.set_ylim(0, 1)

    fig.suptitle("Failure taxonomy: volume vs. recovery after adaptation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
