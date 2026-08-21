"""Environment-only configuration for the local Telegram bot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os

TELEGRAM_PENDING_TTL_SECONDS = 15 * 60
TELEGRAM_FULL_REPORT_TTL_SECONDS = TELEGRAM_PENDING_TTL_SECONDS


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str = field(repr=False)
    allowed_user_ids: frozenset[int]

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> TelegramConfig:
        values = environ if environ is not None else os.environ
        token = values.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN must be set")

        raw_user_ids = values.get("TELEGRAM_ALLOWED_USER_IDS", "")
        allowed_user_ids: set[int] = set()
        for raw_value in raw_user_ids.split(","):
            value = raw_value.strip()
            if not value:
                continue
            try:
                user_id = int(value)
            except ValueError as exc:
                raise ValueError(
                    "TELEGRAM_ALLOWED_USER_IDS must be comma-separated integers"
                ) from exc
            if user_id <= 0:
                raise ValueError("Telegram user IDs must be positive integers")
            allowed_user_ids.add(user_id)
        if not allowed_user_ids:
            raise ValueError("TELEGRAM_ALLOWED_USER_IDS must contain at least one user ID")
        return cls(
            bot_token=token,
            allowed_user_ids=frozenset(allowed_user_ids),
        )

    def allows(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.allowed_user_ids
