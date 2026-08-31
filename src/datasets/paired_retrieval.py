"""Image-caption paired datasets for retrieval, plus a domain-shift wrapper.

Design note: the source and target datasets share the exact same interface
(`PairedRetrievalDataset`) so every downstream component — model, trainer,
evaluator — is domain-agnostic. The only thing that differs between domains
is *which* directory and split file get passed in via configs/data.yaml.
This is what makes the "source vs target" comparison apples-to-apples.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class RetrievalSample:
    image_path: str
    caption: str
    label: str  # coarse category, used for hard-negative mining and error analysis


class PairedRetrievalDataset(Dataset):
    """Generic (image, caption) retrieval dataset.

    Expects `split_file` to be a JSON list of records:
        {"image": "relative/path.jpg", "caption": "a photo of ...", "label": "dog"}

    This format is intentionally dataset-agnostic — `data/scripts/` contains
    the prep scripts that convert COCO Captions (source) and a DomainNet
    sketch/clipart subset (target) into this shape.
    """

    def __init__(
        self,
        root: str,
        split_file: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):
        self.root = root
        self.transform = transform
        self.tokenizer = tokenizer

        with open(split_file, "r") as f:
            manifest = json.load(f)
        records = manifest[split] if isinstance(manifest, dict) else manifest

        self.samples = [
            RetrievalSample(image_path=r["image"], caption=r["caption"], label=r.get("label", "unknown"))
            for r in records
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_path = os.path.join(self.root, sample.image_path)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        text = sample.caption
        if self.tokenizer is not None:
            text = self.tokenizer(text)
            if hasattr(text, "squeeze"):
                text = text.squeeze(0)  # tokenizer returns [1, seq_len] for a single string; drop that leading dim

        return {
            "image": image,
            "text": text,
            "raw_caption": sample.caption,
            "label": sample.label,
            "image_path": sample.image_path,
        }

    def labels(self):
        return [s.label for s in self.samples]


class FewShotTargetSampler:
    """Caps the number of (image, caption) pairs per class for the target
    domain's *training* split, to keep the "low-supervision adaptation"
    premise honest — Method A/B never see as much target data as the model
    saw at source time.
    """

    def __init__(self, dataset: PairedRetrievalDataset, k_per_class: int, seed: int = 42):
        import random

        rng = random.Random(seed)
        by_label: dict[str, list[int]] = {}
        for i, s in enumerate(dataset.samples):
            by_label.setdefault(s.label, []).append(i)

        kept = []
        for label, idxs in by_label.items():
            rng.shuffle(idxs)
            kept.extend(idxs[:k_per_class])
        self.indices = sorted(kept)

    def apply(self, dataset: PairedRetrievalDataset) -> PairedRetrievalDataset:
        dataset.samples = [dataset.samples[i] for i in self.indices]
        return dataset
