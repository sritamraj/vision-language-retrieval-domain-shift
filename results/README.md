# Results

Populated by running `experiments/`. Nothing here is committed pre-filled —
this directory is the audit trail: every number in the root README's
results table traces back to a JSON file here.

```
results/
├── tables/    # {source_eval, target_zero_shot_eval, method_a_eval, method_b_eval,
│                ablation_*_eval}.json — metrics + path to saved embeddings
├── figures/   # performance_drop.png, embedding_domain_gap.png, error_taxonomy.png,
│                retrieval_comparison_example.png
└── logs/      # source_embedding_bank.npz (cached for Method B / ablations), training logs
```

Regenerate the summary table + drop chart at any time with:
```bash
python -m src.evaluation.report --results_dir results/tables --out results/figures/performance_drop.png
```
