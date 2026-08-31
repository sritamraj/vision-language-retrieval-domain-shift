"""Ablations are just Method B configs with one component toggled off —
this file forwards to the same runner so there's a single source of truth
for the adaptation loop instead of a forked copy that could drift.

    python -m experiments.ablations.run --config configs/ablation_no_mmd.yaml
    python -m experiments.ablations.run --config configs/ablation_no_adapter.yaml
    python -m experiments.ablations.run --config configs/ablation_no_hard_negatives.yaml
"""
from experiments.adaptation.run import main

if __name__ == "__main__":
    main()
