"""In-memory references for showing completed Telegram reports on demand."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import re
import secrets
from time import monotonic

from ..analysis.models import AnalysisReport
from ..models import TechnicalSnapshot
from .config import TELEGRAM_FULL_REPORT_TTL_SECONDS

_CALLBACK_PREFIX = "tf"
_REPORT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,32}$")


@dataclass(frozen=True)
class FullReportRequest:
    report_id: str
    user_id: int
    chat_id: int
    message_id: int | None
    snapshot: TechnicalSnapshot
    report: AnalysisReport
    comparison_position: tuple[int, int] | None
    created_at_monotonic: float
    expires_at_monotonic: float


class FullReportRequestError(ValueError):
    """Callback data did not identify an accessible completed report."""


class FullReportStore:
    def __init__(
        self,
        *,
        ttl_seconds: int = TELEGRAM_FULL_REPORT_TTL_SECONDS,
        timer: Callable[[], float] = monotonic,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._timer = timer
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(9))
        self._reports: dict[str, FullReportRequest] = {}

    def create(
        self,
        *,
        user_id: int,
        chat_id: int,
        snapshot: TechnicalSnapshot,
        report: AnalysisReport,
        comparison_position: tuple[int, int] | None,
    ) -> FullReportRequest:
        now = self._timer()
        self._purge_expired(now)
        report_id = self._unique_report_id()
        request = FullReportRequest(
            report_id=report_id,
            user_id=user_id,
            chat_id=chat_id,
            message_id=None,
            snapshot=snapshot,
            report=report,
            comparison_position=comparison_position,
            created_at_monotonic=now,
            expires_at_monotonic=now + self._ttl_seconds,
        )
        self._reports[report_id] = request
        return request

    def bind_message(self, report_id: str, message_id: int) -> FullReportRequest:
        request = self._active_request(report_id)
        bound = replace(request, message_id=message_id)
        self._reports[report_id] = bound
        return bound

    def resolve_callback(
        self,
        callback_data: str,
        *,
        user_id: int,
        chat_id: int,
        message_id: int,
    ) -> FullReportRequest:
        report_id = decode_full_report_callback_data(callback_data)
        request = self._active_request(report_id)
        if request.message_id is None:
            raise FullReportRequestError("report is not yet bound to a Telegram message")
        if request.user_id != user_id:
            raise FullReportRequestError("callback user does not own this report")
        if request.chat_id != chat_id or request.message_id != message_id:
            raise FullReportRequestError(
                "callback did not originate from the compact report message"
            )
        return request

    def discard(self, report_id: str) -> None:
        self._reports.pop(report_id, None)

    def __len__(self) -> int:
        self._purge_expired(self._timer())
        return len(self._reports)

    def _active_request(self, report_id: str) -> FullReportRequest:
        request = self._reports.get(report_id)
        if request is None:
            raise FullReportRequestError("full report was not found")
        if self._timer() >= request.expires_at_monotonic:
            self._reports.pop(report_id, None)
            raise FullReportRequestError("full report has expired")
        return request

    def _purge_expired(self, now: float) -> None:
        expired = [
            report_id
            for report_id, request in self._reports.items()
            if now >= request.expires_at_monotonic
        ]
        for report_id in expired:
            self._reports.pop(report_id, None)

    def _unique_report_id(self) -> str:
        for _ in range(5):
            report_id = self._token_factory()
            if not _REPORT_ID_PATTERN.fullmatch(report_id):
                raise ValueError("token_factory returned an invalid report ID")
            if report_id not in self._reports:
                return report_id
        raise RuntimeError("could not allocate a unique report ID")


def encode_full_report_callback_data(report_id: str) -> str:
    if not _REPORT_ID_PATTERN.fullmatch(report_id):
        raise ValueError("invalid report ID")
    callback_data = f"{_CALLBACK_PREFIX}:{report_id}"
    if len(callback_data.encode("utf-8")) > 64:
        raise ValueError("Telegram callback_data exceeds 64 bytes")
    return callback_data


def decode_full_report_callback_data(callback_data: str) -> str:
    parts = callback_data.split(":") if isinstance(callback_data, str) else []
    if len(parts) != 2 or parts[0] != _CALLBACK_PREFIX:
        raise FullReportRequestError("invalid full-report callback data")
    report_id = parts[1]
    if not _REPORT_ID_PATTERN.fullmatch(report_id):
        raise FullReportRequestError("invalid full-report ID")
    return report_id


def is_full_report_callback(callback_data: str | None) -> bool:
    return isinstance(callback_data, str) and callback_data.startswith(
        f"{_CALLBACK_PREFIX}:"
    )
