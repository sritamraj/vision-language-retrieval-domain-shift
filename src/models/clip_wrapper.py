"""Thin wrapper around open_clip so the rest of the codebase depends on a
stable, minimal interface (`encode_image`, `encode_text`) rather than the
open_clip API directly. This is also where adapters get spliced in for
Method B, so trainers never need to know whether they're holding a plain
CLIP model or an adapter-augmented one.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    import open_clip
except ImportError:  # pragma: no cover - allows docs/tests to import without the dep installed
    open_clip = None

from .adapters import Adapter


class VisionLanguageRetriever(nn.Module):
    def __init__(
        self,
        backbone_name: str = "ViT-B-32",
        pretrained: str = "openai",
        freeze_backbone: bool = False,
        adapter_cfg: dict | None = None,
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError("open_clip_torch is required: pip install open_clip_torch")

        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            backbone_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(backbone_name)
        self.embed_dim = self.clip.text_projection.shape[1] if hasattr(self.clip, "text_projection") else 512

        if freeze_backbone:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.image_adapter: Adapter | None = None
        self.text_adapter: Adapter | None = None
        if adapter_cfg and adapter_cfg.get("enabled", False):
            targets = adapter_cfg.get("apply_to", ["image", "text"])
            bottleneck = adapter_cfg.get("bottleneck_dim", 64)
            ratio = adapter_cfg.get("residual_ratio", 0.2)
            if "image" in targets:
                self.image_adapter = Adapter(self.embed_dim, bottleneck, ratio)
            if "text" in targets:
                self.text_adapter = Adapter(self.embed_dim, bottleneck, ratio)

    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        feats = self.clip.encode_image(images)
        if self.image_adapter is not None:
            feats = self.image_adapter(feats)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def encode_text(self, tokens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        feats = self.clip.encode_text(tokens)
        if self.text_adapter is not None:
            feats = self.text_adapter(feats)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        return self.encode_image(images), self.encode_text(tokens)

    def trainable_parameters(self):
        """Used by the adapter+MMD trainer to build an optimizer that only
        touches adapters (backbone stays frozen for Method B)."""
        params = []
        if self.image_adapter is not None:
            params += list(self.image_adapter.parameters())
        if self.text_adapter is not None:
            params += list(self.text_adapter.parameters())
        return params

    @classmethod
    def from_config(cls, model_cfg: dict, adapter_override: dict | None = None):
        backbone = model_cfg["backbone"]
        adapter_cfg = adapter_override if adapter_override is not None else model_cfg.get("adapter")
        return cls(
            backbone_name=backbone["name"],
            pretrained=backbone["pretrained"],
            freeze_backbone=backbone.get("freeze_backbone", False),
            adapter_cfg=adapter_cfg,
        )
