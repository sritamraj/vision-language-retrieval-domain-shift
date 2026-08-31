"""Stage 2 of the pipeline: measure the performance drop by evaluating the
*source-trained, untouched* model directly on the target domain.

    python -m experiments.baseline.eval_target --config configs/target.yaml
"""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import build_target_dataset
from src.evaluation import evaluate
from src.models import VisionLanguageRetriever


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model_cfg = load_cfg(cfg["model"])
    data_cfg = load_cfg(cfg["data"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VisionLanguageRetriever.from_config(model_cfg)
    model.load_state_dict(torch.load(cfg["load_checkpoint"], map_location=device))
    tokenizer = model.tokenizer

    test_ds = build_target_dataset(data_cfg, split="test", tokenizer=tokenizer)
    dl_cfg = data_cfg["dataloader"]
    test_loader = DataLoader(test_ds, batch_size=dl_cfg["batch_size"], shuffle=False, num_workers=dl_cfg["num_workers"])

    result = evaluate(model, test_loader, device=device, out_file=cfg["eval"]["out_file"])
    print("Zero-shot transfer to target domain:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
