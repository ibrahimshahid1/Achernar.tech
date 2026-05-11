"""Compatibility package for Django app imports.

The Django settings and app configs use short paths such as ``apps.core``.
In this repository the app packages live under ``backend/apps``. Exposing that
directory as this package path keeps those imports stable for local runs,
Docker, and existing code.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "backend" / "apps")]
