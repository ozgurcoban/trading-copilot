"""Stable JSON renderers for TechnicalSnapshot."""

from __future__ import annotations

import json

from .models import TechnicalSnapshot


def canonical_snapshot_json(snapshot: TechnicalSnapshot) -> str:
    """Return compact, byte-stable JSON for a fixed snapshot."""

    return json.dumps(
        snapshot.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def pretty_snapshot_json(snapshot: TechnicalSnapshot) -> str:
    """Return human-readable JSON without changing snapshot content."""

    return json.dumps(
        snapshot.model_dump(mode="json"),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
