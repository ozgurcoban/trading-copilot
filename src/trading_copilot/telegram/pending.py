"""Small in-memory request store with strict callback verification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
import re
import secrets
from time import monotonic

from ..analysis.models import AnalysisModel
from ..models import TechnicalSnapshot
from .config import TELEGRAM_PENDING_TTL_SECONDS

_CALLBACK_PREFIX = "ta"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,32}$")


class TelegramAnalysisChoice(str, Enum):
    GPT_5_6_SOL = "s"
    CLAUDE_OPUS_5 = "o"
    CLAUDE_FABLE_5 = "f"
    COMPARE_ALL = "a"

    @property
    def model(self) -> AnalysisModel | None:
        return {
            TelegramAnalysisChoice.GPT_5_6_SOL: AnalysisModel.GPT_5_6_SOL,
            TelegramAnalysisChoice.CLAUDE_OPUS_5: AnalysisModel.CLAUDE_OPUS_5,
            TelegramAnalysisChoice.CLAUDE_FABLE_5: AnalysisModel.CLAUDE_FABLE_5,
            TelegramAnalysisChoice.COMPARE_ALL: None,
        }[self]

    @property
    def button_label(self) -> str:
        return {
            TelegramAnalysisChoice.GPT_5_6_SOL: "GPT-5.6 Sol",
            TelegramAnalysisChoice.CLAUDE_OPUS_5: "Claude Opus 5",
            TelegramAnalysisChoice.CLAUDE_FABLE_5: "Claude Fable 5",
            TelegramAnalysisChoice.COMPARE_ALL: "Jämför alla",
        }[self]


@dataclass(frozen=True)
class PendingRequest:
    request_id: str
    user_id: int
    chat_id: int
    message_id: int | None
    snapshot: TechnicalSnapshot
    created_at_monotonic: float
    expires_at_monotonic: float


class PendingRequestError(ValueError):
    """Callback data did not identify a valid pending request."""


class PendingRequestStore:
    def __init__(
        self,
        *,
        ttl_seconds: int = TELEGRAM_PENDING_TTL_SECONDS,
        timer: Callable[[], float] = monotonic,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._timer = timer
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(9))
        self._requests: dict[str, PendingRequest] = {}

    def create(
        self,
        *,
        user_id: int,
        chat_id: int,
        snapshot: TechnicalSnapshot,
    ) -> PendingRequest:
        now = self._timer()
        self._purge_expired(now)
        request_id = self._unique_request_id()
        request = PendingRequest(
            request_id=request_id,
            user_id=user_id,
            chat_id=chat_id,
            message_id=None,
            snapshot=snapshot,
            created_at_monotonic=now,
            expires_at_monotonic=now + self._ttl_seconds,
        )
        self._requests[request_id] = request
        return request

    def bind_message(self, request_id: str, message_id: int) -> PendingRequest:
        request = self._active_request(request_id)
        bound = replace(request, message_id=message_id)
        self._requests[request_id] = bound
        return bound

    def consume_callback(
        self,
        callback_data: str,
        *,
        user_id: int,
        chat_id: int,
        message_id: int,
    ) -> tuple[PendingRequest, TelegramAnalysisChoice]:
        request_id, choice = decode_callback_data(callback_data)
        request = self._active_request(request_id)
        if request.message_id is None:
            raise PendingRequestError("request is not yet bound to a Telegram message")
        if request.user_id != user_id:
            raise PendingRequestError("callback user does not own this request")
        if request.chat_id != chat_id or request.message_id != message_id:
            raise PendingRequestError("callback did not originate from the request message")
        self._requests.pop(request_id, None)
        return request, choice

    def __len__(self) -> int:
        self._purge_expired(self._timer())
        return len(self._requests)

    def _active_request(self, request_id: str) -> PendingRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise PendingRequestError("pending request was not found")
        if self._timer() >= request.expires_at_monotonic:
            self._requests.pop(request_id, None)
            raise PendingRequestError("pending request has expired")
        return request

    def _purge_expired(self, now: float) -> None:
        expired = [
            request_id
            for request_id, request in self._requests.items()
            if now >= request.expires_at_monotonic
        ]
        for request_id in expired:
            self._requests.pop(request_id, None)

    def _unique_request_id(self) -> str:
        for _ in range(5):
            request_id = self._token_factory()
            if not _REQUEST_ID_PATTERN.fullmatch(request_id):
                raise ValueError("token_factory returned an invalid request ID")
            if request_id not in self._requests:
                return request_id
        raise RuntimeError("could not allocate a unique pending request ID")


def encode_callback_data(
    request_id: str,
    choice: TelegramAnalysisChoice,
) -> str:
    if not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise ValueError("invalid request ID")
    callback_data = f"{_CALLBACK_PREFIX}:{request_id}:{choice.value}"
    if len(callback_data.encode("utf-8")) > 64:
        raise ValueError("Telegram callback_data exceeds 64 bytes")
    return callback_data


def decode_callback_data(
    callback_data: str,
) -> tuple[str, TelegramAnalysisChoice]:
    parts = callback_data.split(":") if isinstance(callback_data, str) else []
    if len(parts) != 3 or parts[0] != _CALLBACK_PREFIX:
        raise PendingRequestError("invalid callback data")
    request_id = parts[1]
    if not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise PendingRequestError("invalid callback request ID")
    try:
        choice = TelegramAnalysisChoice(parts[2])
    except ValueError as exc:
        raise PendingRequestError("invalid analysis choice") from exc
    return request_id, choice
