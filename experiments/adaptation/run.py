"""Stage 3 of the pipeline: run an adaptation method and evaluate on target.

    python -m experiments.adaptation.run --config configs/adapt_finetune.yaml
    python -m experiments.adaptation.run --config configs/adapt_adapter_mmd.yaml

Also doubles as the ablation runner (experiments/ablations/run.py just
forwards here) since ablations are "adapter_mmd with one config flag off."
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.adaptation import (
    build_source_embedding_bank,
    load_source_embedding_bank,
    run_adapter_mmd,
    run_finetune,
)
from src.datasets import build_source_dataset, build_target_dataset
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

    adapter_override = cfg.get("adapter_override")
    model = VisionLanguageRetriever.from_config(model_cfg, adapter_override=adapter_override)
    model.load_state_dict(torch.load(cfg["load_checkpoint"], map_location=device), strict=False)
    tokenizer = model.tokenizer

    dl_cfg = data_cfg["dataloader"]
    train_ds = build_target_dataset(data_cfg, split="train", tokenizer=tokenizer, few_shot=True)
    train_loader = DataLoader(train_ds, batch_size=dl_cfg["batch_size"], shuffle=True, num_workers=dl_cfg["num_workers"])

    method = cfg["adaptation_method"]
    if method == "full_finetune":
        model, _ = run_finetune(model, train_loader, cfg["train"], device=device)

    elif method == "adapter_mmd":
        mmd_cfg = cfg["train"].get("mmd", {"enabled": False})
        source_bank = None
        if mmd_cfg.get("enabled", False):
            bank_path = mmd_cfg["source_bank_path"]
            if os.path.exists(bank_path):
                source_bank = load_source_embedding_bank(bank_path)
            else:
                source_ds = build_source_dataset(data_cfg, split="train", tokenizer=tokenizer)
                source_loader = DataLoader(source_ds, batch_size=dl_cfg["batch_size"], shuffle=True)
                source_bank = build_source_embedding_bank(model, source_loader, cfg["train"], device=device)
        else:
            source_bank = torch.zeros(1, model.embed_dim)
        model, _ = run_adapter_mmd(model, train_loader, source_bank, cfg["train"], device=device)

    else:
        raise ValueError(f"Unknown adaptation_method: {method}")

    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "adapted.pt"))

    test_ds = build_target_dataset(data_cfg, split="test", tokenizer=tokenizer)
    test_loader = DataLoader(test_ds, batch_size=dl_cfg["batch_size"], shuffle=False)
    result = evaluate(model, test_loader, device=device, out_file=cfg["eval"]["out_file"])

    print(f"[{cfg['experiment_name']}] target evaluation after adaptation:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
