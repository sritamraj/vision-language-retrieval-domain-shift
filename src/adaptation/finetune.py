"""Method A — the obvious baseline adaptation: unfreeze everything and
fine-tune on the (small) target training split with plain contrastive loss.

This is deliberately simple. Its purpose in the study is to be the thing
Method B has to beat, and to demonstrate the failure mode that motivates
Method B: with few target pairs, full fine-tuning overfits fast (see
ablations + notebooks/03_error_analysis.ipynb for the overfitting curve).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import clip_contrastive_loss


def run_finetune(model, train_loader: DataLoader, cfg: dict, device: str = "cuda"):
    model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    history = []

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Method A] epoch {epoch + 1}/{cfg['epochs']}"):
            images = batch["image"].to(device)
            tokens = batch["text"].to(device)

            img_embeds, txt_embeds = model(images, tokens)
            loss = clip_contrastive_loss(img_embeds, txt_embeds, temperature=cfg["temperature"])

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        history.append({"epoch": epoch + 1, "train_loss": avg_loss})
        print(f"epoch {epoch + 1}: train_loss={avg_loss:.4f}")

    return model, history
