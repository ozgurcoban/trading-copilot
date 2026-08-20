"""Fetch, normalize, and validate adjusted daily OHLCV data from yfinance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from .errors import (
    InvalidAsOfError,
    InvalidTickerError,
    MarketDataFetchError,
    MarketDataValidationError,
)
from .models import SnapshotWarning

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
MINIMUM_OBSERVATIONS = 252
MARKET_DATA_STALE_AFTER_CALENDAR_DAYS = 7
HISTORY_LOOKBACK_CALENDAR_DAYS = 800

_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=\-]{0,31}$")


@dataclass(frozen=True)
class MarketDataBundle:
    ticker: str
    frame: pd.DataFrame
    requested_as_of: date
    as_of_session: date
    exchange_today: date
    exchange: str | None
    market_currency: str | None
    exchange_timezone: str
    market_data_age_days: int
    is_stale: bool
    warnings: tuple[SnapshotWarning, ...] = ()


def normalize_ticker(ticker: str) -> str:
    """Return the canonical Yahoo ticker or reject malformed user input."""

    if not isinstance(ticker, str):
        raise InvalidTickerError("Ticker must be a string")
    normalized = ticker.strip().upper()
    if not normalized or not _TICKER_PATTERN.fullmatch(normalized):
        raise InvalidTickerError(
            "Ticker must contain only Yahoo-compatible letters, numbers, '.', '-', '^', or '='"
        )
    return normalized


def fetch_market_data(
    ticker_object: Any,
    ticker: str,
    *,
    as_of: date | None,
    now_utc: datetime,
) -> MarketDataBundle:
    """Fetch roughly two years of adjusted daily data and enforce snapshot rules."""

    normalized_ticker = normalize_ticker(ticker)
    _validate_clock_value(now_utc)
    if as_of is not None and type(as_of) is not date:
        raise InvalidAsOfError("as_of must be a date or None")

    # A small UTC-side margin covers exchanges whose local date is ahead of UTC.
    today_utc = now_utc.astimezone(timezone.utc).date()
    provisional_as_of = as_of or (today_utc + timedelta(days=1))
    provisional_as_of = min(provisional_as_of, today_utc + timedelta(days=1))
    start = provisional_as_of - timedelta(days=HISTORY_LOOKBACK_CALENDAR_DAYS)
    end = provisional_as_of + timedelta(days=2)  # yfinance's end is exclusive.

    try:
        raw_frame = ticker_object.history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=True,
            actions=False,
            repair=False,
            timeout=20,
        )
    except Exception as exc:  # yfinance exposes several transport-specific errors.
        raise MarketDataFetchError(
            f"Could not fetch daily market data for {normalized_ticker}: {exc}"
        ) from exc

    if raw_frame is None or raw_frame.empty:
        raise InvalidTickerError(
            f"No daily market data was returned for ticker {normalized_ticker}"
        )

    metadata, metadata_warning = _read_history_metadata(ticker_object)
    exchange_timezone, timezone_warning = _resolve_exchange_timezone(
        metadata, raw_frame.index
    )
    local_zone = ZoneInfo(exchange_timezone)
    exchange_today = now_utc.astimezone(local_zone).date()
    effective_as_of = as_of or exchange_today
    if effective_as_of > exchange_today:
        raise InvalidAsOfError(
            f"as_of {effective_as_of.isoformat()} is in the future for "
            f"{exchange_timezone} (today is {exchange_today.isoformat()})"
        )

    frame = _normalize_session_index(raw_frame, exchange_timezone)
    frame = frame.loc[frame.index.date <= effective_as_of]

    warnings: list[SnapshotWarning] = []
    if metadata_warning is not None:
        warnings.append(metadata_warning)
    if timezone_warning is not None:
        warnings.append(timezone_warning)

    if (
        not frame.empty
        and effective_as_of == exchange_today
        and frame.index[-1].date() == exchange_today
        and not _current_regular_session_is_complete(metadata, now_utc)
    ):
        frame = frame.iloc[:-1]
        warnings.append(
            SnapshotWarning(
                code="current_session_excluded_unconfirmed_close",
                message=(
                    "The current exchange-local daily candle was excluded because "
                    "the regular session close could not yet be confirmed."
                ),
            )
        )

    frame = validate_daily_ohlcv(frame)
    as_of_session = frame.index[-1].date()
    age_days = (effective_as_of - as_of_session).days
    is_stale = age_days > MARKET_DATA_STALE_AFTER_CALENDAR_DAYS
    if is_stale:
        warnings.append(
            SnapshotWarning(
                code="market_data_stale",
                message=(
                    f"Latest completed market session is {age_days} calendar days "
                    f"before the requested as_of date; the stale threshold is more "
                    f"than {MARKET_DATA_STALE_AFTER_CALENDAR_DAYS} days."
                ),
            )
        )

    return MarketDataBundle(
        ticker=normalized_ticker,
        frame=frame,
        requested_as_of=effective_as_of,
        as_of_session=as_of_session,
        exchange_today=exchange_today,
        exchange=_string_or_none(
            metadata.get("fullExchangeName") or metadata.get("exchangeName")
        ),
        market_currency=_string_or_none(metadata.get("currency")),
        exchange_timezone=exchange_timezone,
        market_data_age_days=age_days,
        is_stale=is_stale,
        warnings=tuple(warnings),
    )


def validate_daily_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a defensive daily OHLCV copy."""

    if frame is None or frame.empty:
        raise MarketDataValidationError("No completed daily OHLCV rows remain")
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame]
    if missing:
        raise MarketDataValidationError(
            f"Daily market data is missing required columns: {', '.join(missing)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise MarketDataValidationError("Daily OHLCV index must be a DatetimeIndex")
    if frame.index.has_duplicates:
        raise MarketDataValidationError("Daily OHLCV contains duplicate session dates")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataValidationError("Daily OHLCV must be ordered oldest to newest")

    validated = frame.loc[:, REQUIRED_OHLCV_COLUMNS].copy().astype(float)
    numeric = validated.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise MarketDataValidationError("Daily OHLCV contains missing or non-finite values")
    if (validated.loc[:, ("Open", "High", "Low", "Close")] <= 0.0).any().any():
        raise MarketDataValidationError("Daily OHLC prices must all be positive")
    if (validated["Volume"] < 0.0).any():
        raise MarketDataValidationError("Daily volume cannot be negative")

    price_max = validated.loc[:, ("Open", "Low", "Close")].max(axis=1)
    price_min = validated.loc[:, ("Open", "High", "Close")].min(axis=1)
    if (validated["High"] < price_max).any():
        raise MarketDataValidationError(
            "Daily high must be at least open, low, and close"
        )
    if (validated["Low"] > price_min).any():
        raise MarketDataValidationError(
            "Daily low must be at most open, high, and close"
        )
    if len(validated) < MINIMUM_OBSERVATIONS:
        raise MarketDataValidationError(
            f"At least {MINIMUM_OBSERVATIONS} completed sessions are required, "
            f"received {len(validated)}"
        )
    return validated


def stale_market_data(as_of: date, latest_session: date) -> tuple[int, bool]:
    """Apply the explicit calendar-day stale rule (7 fresh, 8 stale)."""

    age_days = (as_of - latest_session).days
    if age_days < 0:
        raise MarketDataValidationError("Latest market session cannot be after as_of")
    return age_days, age_days > MARKET_DATA_STALE_AFTER_CALENDAR_DAYS


def _normalize_session_index(
    frame: pd.DataFrame, exchange_timezone: str
) -> pd.DataFrame:
    normalized = frame.copy()
    index = pd.DatetimeIndex(normalized.index)
    if index.tz is None:
        index = index.tz_localize(exchange_timezone)
    else:
        index = index.tz_convert(exchange_timezone)
    normalized.index = pd.DatetimeIndex(index.date)
    normalized = normalized.sort_index()
    return normalized


def _read_history_metadata(
    ticker_object: Any,
) -> tuple[dict[str, Any], SnapshotWarning | None]:
    try:
        metadata = ticker_object.get_history_metadata()
        if isinstance(metadata, dict):
            return metadata, None
    except Exception:
        pass
    return {}, SnapshotWarning(
        code="history_metadata_unavailable",
        message="yfinance history metadata was unavailable; available OHLCV fields were still validated.",
    )


def _resolve_exchange_timezone(
    metadata: dict[str, Any], index: pd.Index
) -> tuple[str, SnapshotWarning | None]:
    candidate = metadata.get("exchangeTimezoneName")
    if isinstance(candidate, str):
        try:
            ZoneInfo(candidate)
            return candidate, None
        except ZoneInfoNotFoundError:
            pass

    index_timezone = getattr(index, "tz", None)
    if index_timezone is not None:
        name = str(index_timezone)
        try:
            ZoneInfo(name)
            return name, SnapshotWarning(
                code="exchange_timezone_inferred_from_prices",
                message="Exchange timezone was inferred from the yfinance price index.",
            )
        except ZoneInfoNotFoundError:
            pass

    return "UTC", SnapshotWarning(
        code="exchange_timezone_unavailable_defaulted_utc",
        message="Exchange timezone was unavailable; UTC was used as a conservative fallback.",
    )


def _current_regular_session_is_complete(
    metadata: dict[str, Any], now_utc: datetime
) -> bool:
    current_period = metadata.get("currentTradingPeriod")
    if not isinstance(current_period, dict):
        return False
    regular = current_period.get("regular")
    if not isinstance(regular, dict):
        return False
    close_epoch = regular.get("end")
    if not isinstance(close_epoch, (int, float)) or not np.isfinite(close_epoch):
        return False
    return now_utc.timestamp() >= float(close_epoch)


def _validate_clock_value(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Clock must return a timezone-aware datetime")


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
