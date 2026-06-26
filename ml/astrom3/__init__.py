"""
Achernar ↔ AstroM3 integration.

AstroM3 (Rizhko & Bloom 2024, arXiv:2411.08842) is a self-supervised *trimodal*
CLIP model over photometric light curves + spectra + metadata. The upstream
code is vendored in this repo at ``AstroM3-main/`` and the pretrained weights
live on the HuggingFace Hub under the ``AstroMLCore`` org.

- ``download.py`` — fetch the pretrained weights from HuggingFace.
- ``loader.py``   — load a model (local checkpoint or HF) and extract embeddings.
- ``edge.py``     — ``AstroM3Edge``: the daemon retrofit (see below).

AstroM3 is used in Achernar in two deployments — one model family, one shared
512-d embedding space:

  * **cloud / planning** (``apps.ai_engine``): AstroM3 as published —
    light curve + spectrum + metadata. Use ``loader.py``.
  * **edge / daemon**: AstroM3 *retrofit* to live modalities — image crop +
    capture metadata + on-mount telemetry — reusing AstroM3's Informer +
    MetaModel + CLIPLoss with a new image encoder. Use ``edge.py``.

The pretrained AstroM3 weights cannot run directly on the daemon (no image
encoder, no spectra on an imaging rig, telemetry != survey photometry); the edge
model reuses AstroM3's architecture/recipe and is trained on Achernar data.
"""

HF_ORG = "AstroMLCore"

# Logical name -> HuggingFace repo id.
HF_MODELS = {
    "base": f"{HF_ORG}/AstroM3-CLIP",          # trimodal CLIP — use for embeddings
    "all": f"{HF_ORG}/AstroM3-CLIP-all",       # multimodal classifier
    "photo": f"{HF_ORG}/AstroM3-CLIP-photo",   # photometry-only classifier (Informer)
    "spectra": f"{HF_ORG}/AstroM3-CLIP-spectra",  # spectra-only classifier (GalSpecNet)
    "meta": f"{HF_ORG}/AstroM3-CLIP-meta",     # metadata-only classifier (MetaModel)
}

# Seed variants of the base CLIP model (load the dataset with the matching seed).
HF_SEED_MODELS = {seed: f"{HF_ORG}/AstroM3-CLIP-{seed}" for seed in (42, 0, 66, 12, 123)}
