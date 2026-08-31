"""Build the source-domain manifest from COCO Captions.

This script expects the COCO 2017 `train2017`/`val2017` images and
`annotations/captions_*.json` to already be downloaded locally (COCO's
license doesn't permit redistribution, so this repo can't fetch/commit them
for you — see https://cocodataset.org/#download). Point --coco_root at the
extracted directory.

Output: data/source/images/*.jpg (symlinked or copied) + data/source/splits.json
in the PairedRetrievalDataset manifest format.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", required=True, help="Path to extracted COCO 2017 root")
    parser.add_argument("--out_dir", default="data/source")
    parser.add_argument("--n_images", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ann_path = os.path.join(args.coco_root, "annotations", "captions_train2017.json")
    with open(ann_path) as f:
        coco = json.load(f)

    img_by_id = {img["id"]: img for img in coco["images"]}
    caps_by_img = {}
    for ann in coco["annotations"]:
        caps_by_img.setdefault(ann["image_id"], []).append(ann["caption"])

    image_ids = list(caps_by_img.keys())
    random.shuffle(image_ids)
    image_ids = image_ids[: args.n_images]

    images_out = os.path.join(args.out_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    records = []
    for img_id in image_ids:
        img_meta = img_by_id[img_id]
        src_path = os.path.join(args.coco_root, "train2017", img_meta["file_name"])
        dst_rel = os.path.join("images", img_meta["file_name"])
        dst_path = os.path.join(args.out_dir, dst_rel)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

        caption = random.choice(caps_by_img[img_id])
        records.append({"image": dst_rel, "caption": caption, "label": "coco_object"})
        # NOTE: swap "label" for a real category via COCO's instance
        # annotations (categories.json) if you want per-class few-shot
        # sampling on the source side too — target-side labeling is what
        # actually matters for FewShotTargetSampler.

    n = len(records)
    train_end, val_end = int(n * 0.85), int(n * 0.92)
    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    print(f"Wrote {n} source records -> {args.out_dir}/splits.json")
    for split, recs in splits.items():
        print(f"  {split}: {len(recs)}")


if __name__ == "__main__":
    main()
