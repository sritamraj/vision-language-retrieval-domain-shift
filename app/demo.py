"""Research demo, not a production inference service.

Upload an image or type a text query -> retrieve top-K matching items from a
small target-domain gallery -> compare the baseline (source-only) model
against the adapted (Method B) model side by side.

Deployed on Hugging Face via the **Docker SDK** (see /Dockerfile). Spaces
routes traffic to port 7860 by default; this app binds to
0.0.0.0:7860 explicitly so it matches regardless of how it's invoked
(`python app/demo.py` locally, or `CMD` in the Dockerfile).

Docs: https://huggingface.co/docs/hub/spaces-sdks-docker
"""
from __future__ import annotations

import glob
import os

import os
import glob

import gradio as gr
import torch
from PIL import Image

from src.models import VisionLanguageRetriever

CHECKPOINT_BASELINE = os.environ.get("BASELINE_CHECKPOINT", "experiments/baseline/checkpoints/best.pt")
CHECKPOINT_ADAPTED = os.environ.get("ADAPTED_CHECKPOINT", "experiments/adaptation/checkpoints/method_b/adapted.pt")
GALLERY_DIR = os.environ.get("GALLERY_DIR", "data/target/images")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5


def _load_model(checkpoint_path: str, with_adapter: bool) -> VisionLanguageRetriever | None:
    """Loads a model if its checkpoint exists; returns None otherwise so the
    demo still boots (with a clear message) before any experiments have
    been run — useful for reviewing the UI/UX without training first.
    """
    if not os.path.exists(checkpoint_path):
        return None
    adapter_cfg = {"enabled": True, "bottleneck_dim": 64, "residual_ratio": 0.2, "apply_to": ["image", "text"]} if with_adapter else None
    model = VisionLanguageRetriever(freeze_backbone=with_adapter, adapter_cfg=adapter_cfg)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


print("Loading models (this may take a minute on first boot)...")
baseline_model = _load_model(CHECKPOINT_BASELINE, with_adapter=False)
adapted_model = _load_model(CHECKPOINT_ADAPTED, with_adapter=True)
gallery_paths = sorted(glob.glob(os.path.join(GALLERY_DIR, "**", "*.jpg"), recursive=True))[:500]
print(f"Loaded {len(gallery_paths)} gallery images. baseline={'ok' if baseline_model else 'missing'} adapted={'ok' if adapted_model else 'missing'}")


@torch.no_grad()
def _encode_gallery(model: VisionLanguageRetriever):
    if model is None or not gallery_paths:
        return None
    preprocess = model.preprocess
    embeds = []
    for p in gallery_paths:
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
        embeds.append(model.encode_image(img))
    return torch.cat(embeds, dim=0)


print("Pre-encoding gallery for both models...")
baseline_gallery_embeds = _encode_gallery(baseline_model)
adapted_gallery_embeds = _encode_gallery(adapted_model)


@torch.no_grad()
def retrieve(model, gallery_embeds, query_text: str | None, query_image):
    if model is None or gallery_embeds is None:
        return []
    if query_image is not None:
        img = model.preprocess(query_image.convert("RGB")).unsqueeze(0).to(DEVICE)
        query_embed = model.encode_image(img)
    elif query_text:
        tokens = model.tokenizer([query_text]).to(DEVICE)
        query_embed = model.encode_text(tokens)
    else:
        return []

    sims = (query_embed @ gallery_embeds.t()).squeeze(0)
    top_idx = sims.argsort(descending=True)[:TOP_K].tolist()
    return [gallery_paths[i] for i in top_idx]


def run_comparison(query_text: str, query_image):
    if not query_text and query_image is None:
        return [], [], "Enter a text query or upload an image."

    baseline_results = retrieve(baseline_model, baseline_gallery_embeds, query_text, query_image)
    adapted_results = retrieve(adapted_model, adapted_gallery_embeds, query_text, query_image)

    status_bits = []
    if baseline_model is None:
        status_bits.append("baseline checkpoint not found — run `experiments/baseline/train.py` first.")
    if adapted_model is None:
        status_bits.append("adapted checkpoint not found — run `experiments/adaptation/run.py` (Method B) first.")
    status = " ".join(status_bits) if status_bits else "Showing top-5 target-domain matches from each model."

    return baseline_results, adapted_results, status


with gr.Blocks(title="V-L Retrieval Under Domain Shift") as demo:
    gr.Markdown(
        """
        # Adaptive Vision-Language Retrieval Under Domain Shift — Demo
        Research demo, not a production service. Query the **target domain**
        gallery (sketch/clipart-style images) with text or an image, and
        compare the **source-trained baseline** model against the
        **Method B (adapter + MMD) adapted** model, side by side.
        """
    )
    with gr.Row():
        text_in = gr.Textbox(label="Text query", placeholder='e.g. "a sketch of a dog"')
        image_in = gr.Image(label="...or upload an image query", type="pil")
    run_btn = gr.Button("Retrieve", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline (source-trained)")
            baseline_gallery = gr.Gallery(columns=5, height=180)
        with gr.Column():
            gr.Markdown("### Adapted (Method B)")
            adapted_gallery = gr.Gallery(columns=5, height=180)

    run_btn.click(run_comparison, inputs=[text_in, image_in], outputs=[baseline_gallery, adapted_gallery, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=True)
