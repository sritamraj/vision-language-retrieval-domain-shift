"""Stage 1 of the pipeline: train the V-L retriever on the source domain.

    python -m experiments.baseline.train --config configs/source.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.adaptation import clip_contrastive_loss
from src.datasets import build_source_dataset
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
    tokenizer = model.tokenizer

    train_ds = build_source_dataset(data_cfg, split="train", tokenizer=tokenizer)
    val_ds = build_source_dataset(data_cfg, split="val", tokenizer=tokenizer)

    dl_cfg = data_cfg["dataloader"]
    train_loader = DataLoader(train_ds, batch_size=dl_cfg["batch_size"], shuffle=True, num_workers=dl_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=dl_cfg["batch_size"], shuffle=False, num_workers=dl_cfg["num_workers"])

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    model.to(device)

    best_recall1 = 0.0
    save_dir = cfg["checkpoint"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            tokens = batch["text"].to(device)
            img_e, txt_e = model(images, tokens)
            loss = clip_contrastive_loss(img_e, txt_e, temperature=cfg["train"]["temperature"])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_result = evaluate(model, val_loader, device=device)
        r1 = val_result["metrics"]["recall@1"]
        print(f"epoch {epoch + 1}: train_loss={avg_loss:.4f} val_recall@1={r1:.4f}")

        if r1 > best_recall1:
            best_recall1 = r1
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))

    # final held-out source evaluation, written to results/tables/source_eval.json
    test_ds = build_source_dataset(data_cfg, split="test", tokenizer=tokenizer)
    test_loader = DataLoader(test_ds, batch_size=dl_cfg["batch_size"], shuffle=False)
    evaluate(model, test_loader, device=device, out_file=cfg["eval"]["out_file"])


if __name__ == "__main__":
    main()
