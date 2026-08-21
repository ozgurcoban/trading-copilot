"""Thin Telegram interface for Trading Copilot Milestone 3."""

from .config import (
    TELEGRAM_FULL_REPORT_TTL_SECONDS,
    TELEGRAM_PENDING_TTL_SECONDS,
    TelegramConfig,
)
from .pending import (
    PendingRequest,
    PendingRequestError,
    PendingRequestStore,
    TelegramAnalysisChoice,
    encode_callback_data,
)
from .rendering import (
    TELEGRAM_COMPACT_MAX_LENGTH,
    TELEGRAM_SAFE_MESSAGE_LENGTH,
    render_analysis_report,
    render_compact_analysis_report,
    render_full_analysis_report,
)
from .reports import (
    FullReportRequest,
    FullReportRequestError,
    FullReportStore,
    encode_full_report_callback_data,
)

__all__ = [
    "FullReportRequest",
    "FullReportRequestError",
    "FullReportStore",
    "PendingRequest",
    "PendingRequestError",
    "PendingRequestStore",
    "TELEGRAM_FULL_REPORT_TTL_SECONDS",
    "TELEGRAM_COMPACT_MAX_LENGTH",
    "TELEGRAM_PENDING_TTL_SECONDS",
    "TELEGRAM_SAFE_MESSAGE_LENGTH",
    "TelegramAnalysisChoice",
    "TelegramConfig",
    "encode_callback_data",
    "encode_full_report_callback_data",
    "render_analysis_report",
    "render_compact_analysis_report",
    "render_full_analysis_report",
]
