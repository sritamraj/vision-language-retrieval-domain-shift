"""Pulls together every results/tables/*.json file into the README's
results table and the "performance drop -> after adaptation" chart.

Usage:
    python -m src.evaluation.report --results_dir results/tables \
        --out results/figures/performance_drop.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

RUN_ORDER = [
    ("source_eval.json", "Source → Source"),
    ("target_zero_shot_eval.json", "Source → Target (zero-shot)"),
    ("method_a_eval.json", "Method A: full fine-tune"),
    ("method_b_eval.json", "Method B: adapter + MMD"),
]


def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for fname, label in RUN_ORDER:
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        m = data["metrics"]
        rows.append(
            {
                "run": label,
                "recall@1": m.get("recall@1"),
                "recall@5": m.get("recall@5"),
                "recall@10": m.get("recall@10"),
                "mAP": m.get("mAP"),
            }
        )
    return pd.DataFrame(rows)


def plot_drop_and_recovery(df: pd.DataFrame, out_path: str, metric: str = "recall@1"):
    if df.empty:
        print("No results found yet — run the experiments first.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["run"], df[metric], marker="o", linewidth=2)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across the domain-shift pipeline")
    ax.set_xticklabels(df["run"], rotation=20, ha="right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/tables")
    parser.add_argument("--out", default="results/figures/performance_drop.png")
    parser.add_argument("--metric", default="recall@1")
    args = parser.parse_args()

    df = load_results(args.results_dir)
    print(df.to_markdown(index=False) if not df.empty else "No results yet.")
    plot_drop_and_recovery(df, args.out, args.metric)


if __name__ == "__main__":
    main()
