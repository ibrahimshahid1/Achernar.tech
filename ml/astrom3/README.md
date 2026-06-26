# AstroM3 integration

**AstroM³** (Rizhko & Bloom 2024, [arXiv:2411.08842](https://arxiv.org/abs/2411.08842))
is a self-supervised **trimodal** CLIP model for astronomy that aligns three
modalities in one embedding space:

- **Photometry** (light curves) → `Informer` (transformer)
- **Spectra** → `GalSpecNet` (1D CNN)
- **Metadata** (astrophysical scalars) → `MetaModel` (MLP)

Each modality is encoded and projected into a shared space; a symmetric
contrastive loss aligns them. The learned embeddings support classification,
similarity search, and anomaly detection — the exact downstream tasks Achernar's
planning / AI-engine layer needs for catalog targets.

Upstream code is vendored at [`AstroM3-main/`](../../AstroM3-main); weights are on
the HuggingFace Hub under [`AstroMLCore`](https://huggingface.co/AstroMLCore).

## Download the weights

> The Cowork sandbox cannot reach huggingface.co (egress proxy returns 403), so
> the weights are **not** bundled. Run this once on your machine:

```bash
./scripts/download_astrom3.sh            # core set into ml/checkpoints/astrom3/
# or, explicitly:
python ml/astrom3/download.py --set core
```

Models fetched (each ~66 MB, 16.5M F32 params):

| logical name | HF repo                          | role                              |
|--------------|----------------------------------|-----------------------------------|
| `base`       | `AstroMLCore/AstroM3-CLIP`       | trimodal CLIP — **embeddings**    |
| `all`        | `AstroMLCore/AstroM3-CLIP-all`   | multimodal classifier             |
| `photo`      | `AstroMLCore/AstroM3-CLIP-photo` | photometry-only classifier        |
| `spectra`    | `AstroMLCore/AstroM3-CLIP-spectra` | spectra-only classifier         |
| `meta`       | `AstroMLCore/AstroM3-CLIP-meta`  | metadata-only classifier          |

Seed variants (`--set seeds`): `AstroM3-CLIP-{42,0,66,12,123}`.
`ml/checkpoints/` is git-ignored — weights are artifacts, not source.

## Use it

```python
import torch
from ml.astrom3.loader import load_astrom3, embed

model = load_astrom3("base")          # local checkpoint if present, else HF
p, s, m = model.get_embeddings(photometry, photometry_mask, spectra, metadata)
joint = embed(model, photometry, photometry_mask, spectra, metadata)  # (B, hidden_dim)
```

Input contracts (after AstroM3 preprocessing):

| input             | shape                    | notes                                   |
|-------------------|--------------------------|-----------------------------------------|
| `photometry`      | `(B, 200, p_enc_in)`     | light curve + aux features, seq_len 200 |
| `photometry_mask` | `(B, 200)`               | 1 = valid, 0 = pad                      |
| `spectra`         | `(B, 1, C)`              | interpolated to the 3850–9000 Å grid    |
| `metadata`        | `(B, m_input_dim)`       | normalized scalar features              |

Preprocessing (interpolation grid, mean/MAD normalization, auxiliary features)
must match the dataset — see `AstroM3-main/src/data.py` and the
[`AstroM3Processed`](https://huggingface.co/datasets/AstroMLCore/AstroM3Processed)
dataset the user linked.

## Two deployments in Achernar

AstroM3 is used **twice** — one model family, one shared 512-d embedding space.

### 1. Cloud / planning (`apps.ai_engine`) — AstroM3 as published

Native modalities (light curve + spectrum + metadata) over catalog targets:
target ranking, similarity search, anomaly detection. Load via `loader.py`.

### 2. Edge / daemon — AstroM3 retrofit (`edge.py`, `AstroM3Edge`)

The **pretrained AstroM3 cannot run directly on the daemon**:

- it has **no image encoder** (its modalities are light curve / spectrum / metadata);
- an imaging rig produces **no spectra**;
- AstroM3's "photometry" is a star's brightness-vs-time light curve, **not**
  focus/guider telemetry, and its weights were trained on survey data.

So `edge.py` **retrofits** AstroM3 to the daemon's live modalities, reusing
AstroM3's own components:

| AstroM3 (cloud)         | AstroM3Edge (daemon)                    |
|-------------------------|-----------------------------------------|
| photometry — `Informer` | on-mount telemetry — `Informer` (reused)|
| metadata — `MetaModel`  | capture metadata — `MetaModel` (reused) |
| spectra — `GalSpecNet`  | image crop — `ImageEncoder` (**new**)   |
| 512-d CLIP space        | 512-d CLIP space (identical)            |

Same dim + same `CLIPLoss` ⇒ edge embeddings are comparable to cloud target
embeddings. `warm_start_from_astrom3()` copies the metadata/telemetry encoder
weights where shapes match; then train on Achernar data and export to ONNX.

```python
from ml.astrom3.edge import AstroM3Edge, clip_loss
model = AstroM3Edge()   # image + metadata + telemetry -> 512-d joint embedding
emb = model.joint_embedding(image, metadata, telemetry, telemetry_mask)
```

## Citation

```bibtex
@article{rizhko2024astrom,
  title={AstroM3: A self-supervised multimodal model for astronomy},
  author={Rizhko, Mariia and Bloom, Joshua S},
  journal={arXiv preprint arXiv:2411.08842},
  year={2024}
}
```
