"""Method B — the adaptation strategy this project argues for.

Three ingredients, each independently switchable (see experiments/ablations):
  1. Bottleneck adapters (src/models/adapters.py) instead of full fine-tuning
     -> far fewer trainable params, less overfitting risk on a tiny target set.
  2. Hard-negative-weighted contrastive loss on the target pairs that do exist.
  3. MMD alignment against a cached bank of *source* embeddings, so the
     target embedding distribution is pulled toward where the source
     distribution lives rather than just fitting a handful of target points
     in isolation.

The backbone is frozen throughout; only adapters (and optionally nothing
else) receive gradient.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import clip_contrastive_loss, mmd_loss


def build_source_embedding_bank(model, source_loader: DataLoader, cfg: dict, device: str = "cuda") -> torch.Tensor:
    """Runs the frozen (pre-adaptation) backbone over a sample of source
    images to build the fixed comparison distribution MMD aligns against.
    Cached to disk so repeated adaptation runs / ablations don't recompute it.
    """
    model.to(device)
    model.eval()
    embeds = []
    target_n = cfg["mmd"]["source_bank_size"]

    with torch.no_grad():
        for batch in source_loader:
            images = batch["image"].to(device)
            feats = model.encode_image(images)
            embeds.append(feats.cpu())
            if sum(e.shape[0] for e in embeds) >= target_n:
                break

    bank = torch.cat(embeds, dim=0)[:target_n]
    np.savez(cfg["mmd"]["source_bank_path"], bank=bank.numpy())
    return bank


def load_source_embedding_bank(path: str) -> torch.Tensor:
    data = np.load(path)
    return torch.from_numpy(data["bank"])


def run_adapter_mmd(model, train_loader: DataLoader, source_bank: torch.Tensor, cfg: dict, device: str = "cuda"):
    model.to(device)
    model.train()
    # Keep the backbone in eval-mode behavior (no dropout drift) even though
    # its params are frozen; only adapters actually receive gradients.
    for p in model.clip.parameters():
        p.requires_grad = False

    trainable = model.trainable_parameters()
    optim = None

    if trainable:
        optim = torch.optim.AdamW(
            trainable,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

    source_bank = source_bank.to(device)

    mmd_cfg = cfg.get("mmd", {"enabled": False})
    hard_k = cfg["hard_negative_top_k"] if cfg.get("hard_negative_mining", False) else None

    history = []
    for epoch in range(cfg["epochs"]):
        epoch_contrastive, epoch_mmd = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"[Method B] epoch {epoch + 1}/{cfg['epochs']}"):
            images = batch["image"].to(device)
            tokens = batch["text"].to(device)

            img_embeds, txt_embeds = model(images, tokens)
            contrastive = clip_contrastive_loss(
                img_embeds, txt_embeds, temperature=cfg["temperature"], hard_negative_top_k=hard_k
            )

            loss = contrastive
            mmd_term = torch.tensor(0.0, device=device)
            if mmd_cfg.get("enabled", False):
                bank_sample = source_bank[torch.randperm(source_bank.size(0))[: img_embeds.size(0)]]
                mmd_term = mmd_loss(img_embeds, bank_sample)
                loss = loss + mmd_cfg.get("weight", 1.0) * mmd_term

            if optim is not None:
                optim.zero_grad()
                loss.backward()
                optim.step()

            epoch_contrastive += contrastive.item()
            epoch_mmd += float(mmd_term.detach())

        n = max(len(train_loader), 1)
        history.append({"epoch": epoch + 1, "contrastive_loss": epoch_contrastive / n, "mmd_loss": epoch_mmd / n})
        print(f"epoch {epoch + 1}: contrastive={epoch_contrastive / n:.4f} mmd={epoch_mmd / n:.4f}")

    return model, history
