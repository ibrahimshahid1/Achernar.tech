"""
Load AstroM3 models and extract trimodal embeddings.

Uses the vendored upstream code in ``AstroM3-main/src`` plus pretrained weights
(downloaded via ml/astrom3/download.py, or fetched from HuggingFace on first use).

    from ml.astrom3.loader import load_astrom3, embed
    model = load_astrom3("base")                     # AstroM3-CLIP (trimodal)
    p, s, m = model.get_embeddings(photometry, photometry_mask, spectra, metadata)
    joint = embed(model, photometry, photometry_mask, spectra, metadata)  # mean of present modalities

Requires: torch (see AstroM3-main/requirements.txt), huggingface_hub, safetensors.

Input contracts (after AstroM3 preprocessing — see the model card / data.py):
    photometry      : (B, seq_len=200, p_enc_in)  float   light curve + aux features
    photometry_mask : (B, seq_len)                float   1=valid, 0=pad
    spectra         : (B, 1, C)                    float   interpolated to 3850-9000 A grid
    metadata        : (B, m_input_dim)             float   normalized scalar features
"""
from __future__ import annotations

import pathlib
import sys

import ml.astrom3 as A3

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASTROM3_SRC = ROOT / "AstroM3-main" / "src"
CHECKPOINTS = ROOT / "ml" / "checkpoints" / "astrom3"


def _ensure_src_on_path() -> None:
    """Vendored AstroM3 uses bare imports (e.g. `from informer import ...`)."""
    p = str(ASTROM3_SRC)
    if not ASTROM3_SRC.exists():
        raise FileNotFoundError(
            f"AstroM3 source not found at {ASTROM3_SRC}. The upstream repo should be "
            f"vendored at AstroM3-main/ (git clone https://github.com/MeriDK/AstroM3)."
        )
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_source(which: str) -> str:
    """Prefer a locally-downloaded checkpoint dir; otherwise the HF repo id."""
    repo = A3.HF_MODELS[which]
    local = CHECKPOINTS / repo.split("/")[-1]
    if (local / "model.safetensors").exists():
        return str(local)
    return repo  # PyTorchModelHubMixin.from_pretrained downloads on demand


def load_astrom3(which: str = "base", source: str | None = None):
    """
    Load an AstroM3 model.

    which : "base"    -> AstroM3 (trimodal CLIP; use .get_embeddings)
            "all"     -> AstroM3 multimodal classifier
            "photo"   -> Informer    (photometry classifier)
            "spectra" -> GalSpecNet  (spectra classifier)
            "meta"    -> MetaModel   (metadata classifier)
    source: explicit local dir or HF repo id (overrides auto-resolution).
    """
    _ensure_src_on_path()
    from model import AstroM3, GalSpecNet, Informer, MetaModel  # type: ignore

    cls = {"base": AstroM3, "all": AstroM3, "photo": Informer,
           "spectra": GalSpecNet, "meta": MetaModel}[which]
    src = source or _resolve_source(which)
    model = cls.from_pretrained(src)
    return model.eval()


def embed(model, photometry, photometry_mask, spectra, metadata):
    """Mean of the per-modality embeddings (drop a modality by zeroing its input)."""
    p_emb, s_emb, m_emb = model.get_embeddings(photometry, photometry_mask, spectra, metadata)
    return (p_emb + s_emb + m_emb) / 3.0


if __name__ == "__main__":
    # Smoke check (needs torch + downloaded/network weights).
    import torch

    model = load_astrom3("base")
    B, seq_len = 2, 200
    photometry = torch.randn(B, seq_len, 3)
    photometry_mask = torch.ones(B, seq_len)
    spectra = torch.randn(B, 1, 1024)
    metadata = torch.randn(B, model.metadata_encoder.input_dim)
    joint = embed(model, photometry, photometry_mask, spectra, metadata)
    print("AstroM3 joint embedding:", tuple(joint.shape))
