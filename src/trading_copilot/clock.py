"""Clock abstractions used to make snapshots reproducible in tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Minimal injectable clock contract."""

    def now_utc(self) -> datetime:
        """Return a timezone-aware UTC timestamp."""


class SystemClock:
    """Production clock."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class FixedClock:
    """Clock with a fixed timestamp for deterministic tests."""

    value: datetime

    def now_utc(self) -> datetime:
        if self.value.tzinfo is None or self.value.utcoffset() is None:
            raise ValueError("FixedClock requires a timezone-aware datetime")
        return self.value.astimezone(timezone.utc)
