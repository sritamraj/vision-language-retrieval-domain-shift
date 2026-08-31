# Adaptive Vision-Language Retrieval Under Domain Shift

**Core question:** How robust is vision-language (image↔text) retrieval when the
target domain differs from the training domain — and can a lightweight
adaptation method close the gap without full retraining?

This repo trains a CLIP-style retrieval model on a **source domain** (natural
photos), measures how much performance degrades on a **target domain**
(sketch/clipart-style images with domain-shifted visual statistics), and then
compares two adaptation strategies against that drop.

---

## 1. Pipeline

```
Source Domain (COCO-style photos, captions)
        │
        ▼
Train V-L Model  (CLIP ViT-B/32 backbone, contrastive fine-tune)
        │
        ▼
Source Evaluation  (Recall@1/5/10, mAP on held-out source pairs)
        │
        ▼
Target Domain  (DomainNet-style sketch/clipart images, re-paired captions)
        │
        ▼
Performance Drop  (zero-shot transfer of the source model to target)
        │
        ▼
Adaptation Method
   ├─ Method A: full fine-tuning on target
   └─ Method B: adapter + domain-alignment (CLIP-Adapter + MMD, few-shot)
        │
        ▼
Target Evaluation  (same metrics, post-adaptation)
        │
        ▼
Error Analysis  (which query types recover, which don't, and why)
```

## 2. Why the gap exists

Domain shift in vision-language retrieval comes from (at least) three
compounding sources, and the error analysis notebook (`03_error_analysis.ipynb`)
attributes failures to each:

1. **Visual style shift** — the image encoder was trained on photographic
   texture statistics; sketches/clipart have different edge density, color
   distribution, and lack photographic shading cues that CLIP's ViT patches
   rely on.
2. **Semantic granularity mismatch** — target captions and source captions
   don't always describe objects at the same level of specificity, so the
   text encoder's embedding neighborhood shifts even when the image content
   is nominally "the same class."
3. **Contrastive batch geometry** — the source model's embedding space was
   shaped by *source* hard negatives. Under shift, what counts as a "hard
   negative" changes, so the decision boundary that used to separate classes
   cleanly no longer does, without touching a single target label.

Method B directly targets (1) and (3): a lightweight adapter absorbs the
visual style shift with far fewer parameters than full fine-tuning, and an
MMD-based alignment term explicitly pulls the target embedding distribution
toward the source distribution rather than just fitting target pairs in
isolation (which is what Method A / plain fine-tuning does, and why it risks
overfitting to a small target set — see ablations).

## 3. Results

The following results come from a **small-scale Colab experiment** using
300 source-domain COCO validation images and 300 target-domain Quick, Draw!
sketches across 12 overlapping classes.

> **Important:** this is a portfolio-scale experiment, not a full COCO/DomainNet
> benchmark. The goal is to demonstrate the complete domain-shift retrieval
> pipeline and make the experiment reproducible.

| Setting | R@1 | R@5 | R@10 | mAP |
|---|---:|---:|---:|---:|
| Source → Source (upper bound) | — | — | — | — |
| Source → Target (zero-shot) | 4.17% | 21.39% | 38.89% | 13.33% |
| Method A: full fine-tune | 4.44% | 24.44% | 49.72% | 15.30% |
| Method B: adapter + MMD | 5.00% | 22.50% | 40.56% | 16.13% |

### Interpretation

On this small target-domain experiment, Method B improves **R@1 and mAP**
over zero-shot transfer, while Method A achieves the strongest R@5/R@10
retrieval in this particular run.

The result does **not** establish that adapter + MMD universally outperforms
full fine-tuning. Instead, it demonstrates that lightweight adaptation can
recover part of the domain-shift gap while using an adapter-based approach.

Because the target evaluation set is small, these differences should be
treated as preliminary rather than statistically conclusive. A larger
COCO/DomainNet-scale experiment would be required for a stronger comparison.

### Qualitative retrieval

The repository also includes a qualitative comparison of baseline and
adapted retrieval results:

![Retrieval comparison](results/figures/retrieval_comparison.png)

### Embedding visualization

The target-domain image/text embeddings after Method B adaptation are
visualized with t-SNE:

![Method B t-SNE](results/figures/method_b_tsne.png)

## 4. Repo structure

```
vision-language-retrieval-domain-shift/
├── README.md
├── requirements.txt
├── Dockerfile                  # builds the HF Spaces demo (app/demo.py)
├── configs/                    # YAML configs: model, data, adaptation, experiment
├── data/                       # dataset download/prep scripts (no raw data committed)
├── notebooks/
│   ├── 01_source_domain.ipynb  # train + evaluate on source
│   ├── 02_target_domain.ipynb  # zero-shot transfer + adaptation
│   └── 03_error_analysis.ipynb # failure taxonomy, qualitative retrieval grids
├── src/
│   ├── datasets/                # PairedRetrievalDataset, domain wrappers
│   ├── models/                  # CLIP wrapper, adapter modules
│   ├── adaptation/               # Method A (fine-tune), Method B (adapter+MMD)
│   ├── evaluation/               # Recall@K, mAP, reporting
│   └── visualization/            # embedding t-SNE, retrieval grids, drop charts
├── experiments/
│   ├── baseline/                 # source-trained model evaluated zero-shot on target
│   ├── adaptation/                # Method A and Method B runs
│   └── ablations/                 # remove one Method B component at a time
├── results/                      # metrics tables, figures, logs (git-tracked, data isn't)
└── app/
    └── demo.py                   # Gradio demo: baseline vs adapted, deployed via Docker
```

## 5. Running it

```bash
pip install -r requirements.txt

# 1. Source training + eval
python -m experiments.baseline.train --config configs/source.yaml

# 2. Zero-shot transfer to target (the "performance drop" measurement)
python -m experiments.baseline.eval_target --config configs/target.yaml

# 3a. Method A — full fine-tune on target
python -m experiments.adaptation.run --config configs/adapt_finetune.yaml

# 3b. Method B — adapter + domain alignment
python -m experiments.adaptation.run --config configs/adapt_adapter_mmd.yaml

# 4. Ablations (drop one component of Method B at a time)
python -m experiments.ablations.run --config configs/ablation_no_mmd.yaml
python -m experiments.ablations.run --config configs/ablation_no_adapter.yaml
python -m experiments.ablations.run --config configs/ablation_no_hard_negatives.yaml

# 5. Error analysis + figures
jupyter notebook notebooks/03_error_analysis.ipynb
```

## 6. Demo

A research demo (not a production inference service) lets you upload an
image or type a text query, retrieve top-K matches, and compare the
**baseline (source-only)** model against the **adapted (Method B)** model
side by side.

Run locally:
```bash
docker build -t vlr-demo .
docker run -p 7860:7860 vlr-demo
# open http://localhost:7860
```

Deployed via **Hugging Face Docker Spaces** (see `Dockerfile`, which follows
the [Spaces Docker SDK guide](https://huggingface.co/docs/hub/spaces-sdks-docker):
listens on port `7860`, runs as a non-root user, and reads `app_port` from the
Space's README frontmatter if overridden).

To push this repo as a Space, add YAML frontmatter to the top of the Space's
`README.md` (Spaces reads this to pick the SDK):

```yaml
---
title: VL Retrieval Under Domain Shift
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

then `git push` to the Space's git remote the same way you'd push to GitHub
— HF builds the image from the root `Dockerfile` and starts the container.

## 7. Run the Demo in Google Colab

The research demo can be run without local GPU setup.

### Quick start

Open the project in Google Colab, clone the repository, install the dependencies, and launch the Gradio application:

```python
!git clone https://github.com/sritamraj/vision-language-retrieval-domain-shift.git
%cd vision-language-retrieval-domain-shift
!pip install -r requirements.txt
!python app/demo.py
```

The application loads the trained baseline and Method B checkpoints and provides text-to-image and image-to-image retrieval over the target-domain gallery.

> The Colab/Gradio public URL is temporary and remains available only while the Colab runtime and demo process are running. No paid hosting is required.

## 8. Citation / notes

This is a research/portfolio project rather than a production system. The
reported results come from the small-scale Colab experiment described above.
The experiment is intentionally limited in scale, so the results should be
interpreted as preliminary rather than as a statistically definitive benchmark.
