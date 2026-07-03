# AI Engine App

## Purpose

Cloud-based AI/ML services for embeddings, recommendations, quality scoring, and anomaly detection.

## Structure

- **`embeddings/`** — Tri-modal CLIP-style embeddings (Image ↔ Time-Series ↔ Spectra ↔ Metadata)
- **`recommender/`** — Contextual bandits and Bayesian hierarchical models
- **`quality/`** — Image quality scoring with calibration and explainability (SHAP)
- **`anomaly/`** — Isolation Forest and autoencoders for transient detection

## Models

- `EmbeddingModel` — Tracks embedding model versions and deployment
- `ObjectEmbedding` — Stores 512-D AstroM3 embeddings for astronomical objects

## API Endpoints

- `POST /api/ai/embed/` — Generate embeddings
- `POST /api/ai/recommend/` — Get exposure/gain recommendations
- `POST /api/ai/quality/` — Score frame quality
- `POST /api/ai/anomaly/` — Detect anomalies

## PDF Reference

Section 16.4 - AI/ML Cloud Services
