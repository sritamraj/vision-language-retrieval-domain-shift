# Data

No raw images/captions are committed to this repo. `data/scripts/` downloads
and reformats public datasets into the shared manifest format that
`src/datasets/PairedRetrievalDataset` expects:

```json
[
  {"image": "images/000123.jpg", "caption": "a dog running on the beach", "label": "dog"},
  ...
]
```

split into `data/source/splits.json` and `data/target/splits.json`, each a
dict with `"train"`, `"val"`, `"test"` keys mapping to lists of records.

## Source domain — natural photos

`data/scripts/prepare_source.py` builds a subset from **COCO Captions**
(photographic images, human-written captions, broad object vocabulary).
This is the "in-domain" world the model is trained on.

## Target domain — sketch/clipart

`data/scripts/prepare_target.py` builds a subset from **DomainNet**
(`sketch` and `clipart` subdomains), re-captioned by template
(`"a sketch of a {class}"` / `"a clipart image of a {class}"`) filtered to
the class vocabulary that overlaps with the source label set — so any
performance drop we measure is attributable to *domain*, not to the model
having never seen the concept at all.

Target `train` split is intentionally small (`few_shot_pairs_per_class` in
`configs/data.yaml`, default 8) — the whole point of the study is adapting
under low supervision, not retraining on an equally large target set.

## Why this pairing

COCO → DomainNet(sketch/clipart) is a standard, well-documented domain-shift
setup: same rough object vocabulary, very different visual statistics
(photographic texture vs. line-art/flat-color), which isolates *visual*
domain shift from *vocabulary* shift. Swap in a different pair (e.g.
product photos → user-generated photos) by writing a new
`data/scripts/prepare_*.py` that emits the same manifest format — nothing
else in the repo needs to change.

## Running the prep scripts

```bash
python data/scripts/prepare_source.py --out_dir data/source --n_images 8000
python data/scripts/prepare_target.py --out_dir data/target --n_images 1500
```

Both scripts print dataset sizes per split and per label so you can sanity
check class balance before training.
