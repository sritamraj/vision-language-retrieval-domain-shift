"""Thin, explicit wrappers so "source" and "target" are never ambiguous
elsewhere in the codebase. Both call into PairedRetrievalDataset —
the point of this file is just to name the two domains clearly and attach
the right image transforms.
"""
from __future__ import annotations

from torchvision import transforms

from .paired_retrieval import FewShotTargetSampler, PairedRetrievalDataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def build_source_dataset(cfg: dict, split: str, tokenizer=None) -> PairedRetrievalDataset:
    src = cfg["source_domain"]
    return PairedRetrievalDataset(
        root=src["root"],
        split_file=src["split_file"],
        split=split,
        transform=build_transform(src["image_size"], train=(split == "train")),
        tokenizer=tokenizer,
    )


def build_target_dataset(cfg: dict, split: str, tokenizer=None, few_shot: bool = False) -> PairedRetrievalDataset:
    tgt = cfg["target_domain"]
    ds = PairedRetrievalDataset(
        root=tgt["root"],
        split_file=tgt["split_file"],
        split=split,
        transform=build_transform(tgt["image_size"], train=(split == "train")),
        tokenizer=tokenizer,
    )
    if few_shot and split == "train":
        sampler = FewShotTargetSampler(ds, k_per_class=tgt["few_shot_pairs_per_class"])
        ds = sampler.apply(ds)
    return ds
