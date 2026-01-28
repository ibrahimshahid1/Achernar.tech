# Core App

## Purpose

Shared utilities and base abstract models for reuse across all Django apps.

## Base Models

- **TimestampedModel** — Auto-tracking of created_at/updated_at
- **UUIDModel** — UUID primary key
- **SoftDeleteModel** — Soft delete capability

## Utilities

- `utils.py` — Shared helper functions (coordinate conversion, airmass calculation)

## Usage

```python
from core.models import TimestampedModel, UUIDModel

class MyModel(UUIDModel, TimestampedModel):
    name = models.CharField(max_length=255)
```

## PDF Reference

Core utilities not explicitly in PDF, but essential for DRY principles.
