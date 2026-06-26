"""
AstroM3 — daemon (edge) retrofit.

The published AstroM3 model CANNOT run directly on the daemon's live data:
  * it has no image encoder (its modalities are light curve / spectrum / metadata),
  * an imaging rig produces no spectra,
  * its "photometry" branch encodes a star's brightness-vs-time light curve, not
    focus/guider telemetry, and its weights were trained on survey data.

What we CAN do — and what this module is — is retrofit AstroM3's *architecture
and training recipe* onto the daemon's modalities, reusing AstroM3's own encoder
classes so the two deployments stay one model family in one shared space:

    AstroM3 (cloud)            AstroM3Edge (daemon, this file)
    ----------------------     --------------------------------------
    photometry  (Informer)  →  on-mount telemetry   (Informer, reused)
    metadata    (MetaModel) →  capture metadata     (MetaModel, reused)
    spectra     (GalSpecNet) → image crop           (ImageEncoder, NEW)
    shared 512-d CLIP space  =  shared 512-d CLIP space (identical dim)

Because the shared space is the same size (512) and trained with the same
CLIPLoss, edge embeddings can be aligned/compared with cloud AstroM3 target
embeddings (e.g. "does this capture match the intended target?").

Requires torch + the vendored AstroM3 source at ../../AstroM3-main/src.
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ASTROM3_SRC = _ROOT / "AstroM3-main" / "src"
if not _ASTROM3_SRC.exists():
    raise FileNotFoundError(
        f"AstroM3 source not found at {_ASTROM3_SRC}. Vendor it at AstroM3-main/ "
        f"(git clone https://github.com/MeriDK/AstroM3)."
    )
if str(_ASTROM3_SRC) not in sys.path:
    sys.path.insert(0, str(_ASTROM3_SRC))

# Reused, unmodified AstroM3 components.
from model import Informer, MetaModel  # noqa: E402  (AstroM3 encoders)
from loss import CLIPLoss  # noqa: E402  (AstroM3 contrastive loss)


class ImageEncoder(nn.Module):
    """
    NEW modality branch (AstroM3 has none). Compact CNN over a grayscale image
    crop -> feature vector -> projected to the shared embedding dim.
    Input: (B, in_ch, H, W)   Output: (B, out_dim)
    """

    def __init__(self, in_ch: int = 1, out_dim: int = 512, width: int = 32):
        super().__init__()

        def block(i: int, o: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.GELU(),
            )

        self.net = nn.Sequential(
            block(in_ch, width, 2),       # 64 -> 32
            block(width, width * 2, 2),   # 32 -> 16
            block(width * 2, width * 4, 2),  # 16 -> 8
            block(width * 4, width * 4, 2),  # 8  -> 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(width * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


class AstroM3Edge(nn.Module):
    """
    Trimodal CLIP for the daemon: image crop + capture metadata + on-mount
    telemetry, fused in the same 512-d space AstroM3 uses. Mirrors
    ``AstroM3.get_embeddings`` / ``forward`` so training code transfers 1:1.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        # image branch
        img_in_ch: int = 1,
        # metadata branch (AstroM3 MetaModel) — capture metadata feature count
        meta_input_dim: int = 16,
        meta_hidden_dim: int = 512,
        meta_dropout: float = 0.16,
        # telemetry branch (AstroM3 Informer) — HFR / guider / focus channels
        ts_seq_len: int = 200,
        ts_enc_in: int = 4,
        ts_d_model: int = 128,
        ts_dropout: float = 0.25,
        ts_factor: int = 1,
        ts_n_heads: int = 4,
        ts_d_ff: int = 512,
        ts_e_layers: int = 4,
        ts_activation: str = "gelu",
        # classifier head (optional; CLIP pre-training uses classification=False)
        classification: bool = False,
        num_classes: int | None = None,
        fusion: str = "avg",
    ):
        super().__init__()
        self.classification = classification
        self.fusion = fusion

        # Encoders: image is NEW; metadata + telemetry reuse AstroM3 classes.
        self.image_encoder = ImageEncoder(in_ch=img_in_ch, out_dim=embed_dim)
        self.metadata_encoder = MetaModel(
            classification=False, num_classes=None,
            input_dim=meta_input_dim, hidden_dim=meta_hidden_dim, dropout=meta_dropout,
        )
        self.timeseries_encoder = Informer(
            classification=False, num_classes=None,
            seq_len=ts_seq_len, enc_in=ts_enc_in, d_model=ts_d_model, dropout=ts_dropout,
            factor=ts_factor, output_attention=False, n_heads=ts_n_heads, d_ff=ts_d_ff,
            activation=ts_activation, e_layers=ts_e_layers,
        )

        # Projection heads -> shared 512-d space.
        self.image_proj = nn.Linear(embed_dim, embed_dim)
        self.metadata_proj = nn.Linear(meta_hidden_dim, embed_dim)
        self.timeseries_proj = nn.Linear(ts_seq_len * ts_d_model, embed_dim)

        # Learnable contrastive temperatures (mirrors AstroM3's logit scales).
        self.logit_scale_it = nn.Parameter(torch.log(torch.ones([]) * 100))  # image <-> telemetry
        self.logit_scale_tm = nn.Parameter(torch.log(torch.ones([]) * 100))  # telemetry <-> metadata
        self.logit_scale_mi = nn.Parameter(torch.log(torch.ones([]) * 100))  # metadata <-> image

        if classification:
            in_features = embed_dim * 3 if fusion == "concat" else embed_dim
            self.fc = nn.Linear(in_features, num_classes)

    def get_embeddings(self, image, metadata, timeseries, timeseries_mask):
        """Unit-normalized per-modality embeddings, each (B, embed_dim)."""
        i = self.image_proj(self.image_encoder(image))
        m = self.metadata_proj(self.metadata_encoder(metadata))
        t = self.timeseries_proj(self.timeseries_encoder(timeseries, timeseries_mask))
        i = i / i.norm(dim=-1, keepdim=True)
        m = m / m.norm(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True)
        return i, m, t

    def joint_embedding(self, image, metadata, timeseries, timeseries_mask):
        """Mean of present modalities — the vector to store / compare with cloud."""
        i, m, t = self.get_embeddings(image, metadata, timeseries, timeseries_mask)
        return (i + m + t) / 3.0

    def forward(self, image, metadata, timeseries, timeseries_mask):
        i, m, t = self.get_embeddings(image, metadata, timeseries, timeseries_mask)
        if self.classification:
            emb = torch.cat((i, m, t), dim=1) if self.fusion == "concat" else (i + m + t) / 3
            return self.fc(emb)
        s_it = torch.clamp(self.logit_scale_it.exp(), min=1, max=100)
        s_tm = torch.clamp(self.logit_scale_tm.exp(), min=1, max=100)
        s_mi = torch.clamp(self.logit_scale_mi.exp(), min=1, max=100)
        return s_it * (i @ t.T), s_tm * (t @ m.T), s_mi * (m @ i.T)


def clip_loss() -> CLIPLoss:
    """AstroM3's CLIPLoss, reused unchanged. Call on (logits_it, logits_tm, logits_mi)."""
    return CLIPLoss()


def warm_start_from_astrom3(model: "AstroM3Edge", astrom3) -> list[str]:
    """
    Best-effort transfer of the metadata + telemetry encoder weights from a loaded
    cloud AstroM3 (ml.astrom3.loader.load_astrom3). The first input-embedding layer
    differs (capture metadata != 34 catalog features; telemetry channels != 9
    photometry features), so those are skipped; deeper layers may transfer. Returns
    the list of successfully copied parameter names. Expect to train/fine-tune after.
    """
    copied: list[str] = []
    src = dict(astrom3.state_dict())
    dst = model.state_dict()
    for dst_key in list(dst.keys()):
        # map our encoder names onto AstroM3's encoder names
        src_key = (
            dst_key.replace("metadata_encoder", "metadata_encoder")
                   .replace("timeseries_encoder", "photometry_encoder")
        )
        if src_key in src and src[src_key].shape == dst[dst_key].shape:
            dst[dst_key] = src[src_key]
            copied.append(dst_key)
    model.load_state_dict(dst, strict=False)
    return copied


if __name__ == "__main__":  # shape self-check (needs torch + AstroM3 deps)
    m = AstroM3Edge().eval()
    B = 2
    image = torch.randn(B, 1, 64, 64)
    metadata = torch.randn(B, 16)
    telemetry = torch.randn(B, 200, 4)
    mask = torch.ones(B, 200)
    emb = m.joint_embedding(image, metadata, telemetry, mask)
    print("AstroM3Edge joint embedding:", tuple(emb.shape))  # (2, 512)
