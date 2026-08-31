from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader

from .metrics import full_eval


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str = "cuda", out_file: str | None = None) -> dict:
    model.to(device)
    model.eval()

    all_image_embeds, all_text_embeds, all_labels, all_paths = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        tokens = batch["text"].to(device)
        img_e, txt_e = model(images, tokens)
        all_image_embeds.append(img_e.cpu())
        all_text_embeds.append(txt_e.cpu())
        all_labels.extend(batch["label"])
        all_paths.extend(batch["image_path"])

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    metrics = full_eval(image_embeds, text_embeds)

    result = {
        "metrics": metrics,
        "n_samples": image_embeds.size(0),
    }

    if out_file:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        # embeddings + labels saved alongside for error analysis (notebook 03)
        emb_path = out_file.replace(".json", "_embeddings.pt")
        torch.save(
            {"image_embeds": image_embeds, "text_embeds": text_embeds, "labels": all_labels, "paths": all_paths},
            emb_path,
        )

    return result
