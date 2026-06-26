"""
tri_modal_embedding — Section 7.6.9.

The Tri-Modal Embedding (TME) in Achernar **is AstroM3** (Rizhko & Bloom 2024,
arXiv:2411.08842) — a self-supervised CLIP model that aligns three modalities in
one shared embedding space. The upstream implementation is vendored at
``AstroM3-main/`` and integrated under ``ml/astrom3/``. There is no separate
from-scratch model.

AstroM3 is deployed in Achernar in two places, same model family + same 512-d
embedding space:

1. **Cloud / planning** (``apps.ai_engine``) — AstroM3 as published:
   light curve + spectrum + metadata. Load it with::

       from ml.models.tri_modal_embedding import load_cloud
       model = load_cloud("base")          # -> ml.astrom3.loader.load_astrom3

2. **Edge / daemon** — AstroM3 *retrofit* to the daemon's live modalities
   (image crop + capture metadata + on-mount telemetry), reusing AstroM3's
   Informer + MetaModel + CLIPLoss with a new image encoder::

       from ml.models.tri_modal_embedding import build_edge
       model = build_edge()                # -> ml.astrom3.edge.AstroM3Edge

Imports are lazy so this module stays importable without torch / the AstroM3
dependencies present.
"""
from __future__ import annotations


def load_cloud(which: str = "base", source=None):
    """Cloud AstroM3 (light curve + spectrum + metadata). See ml/astrom3/loader.py."""
    from ml.astrom3.loader import load_astrom3
    return load_astrom3(which=which, source=source)


def build_edge(**kwargs):
    """Edge AstroM3 retrofit (image + metadata + telemetry). See ml/astrom3/edge.py."""
    from ml.astrom3.edge import AstroM3Edge
    return AstroM3Edge(**kwargs)
