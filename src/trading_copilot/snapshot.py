"""Public orchestration for TechnicalSnapshot Milestone 1."""

from __future__ import annotations

from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import yfinance as yf

from .clock import Clock, SystemClock
from .fundamentals import (
    fetch_current_fundamental_context,
    historical_fundamental_context,
)
from .indicators import calculate_technical_sections
from .market_data import MarketDataBundle, fetch_market_data, normalize_ticker
from .models import (
    FundamentalRiskContext,
    Instrument,
    SnapshotMetadata,
    TechnicalSnapshot,
)


def build_snapshot(
    ticker: str,
    as_of: date | None = None,
    clock: Clock | None = None,
) -> TechnicalSnapshot:
    """Build one validated, provider-neutral TechnicalSnapshot."""

    active_clock = clock or SystemClock()
    generated_at_utc = active_clock.now_utc()
    if generated_at_utc.tzinfo is None or generated_at_utc.utcoffset() is None:
        raise ValueError("Clock must return a timezone-aware datetime")
    generated_at_utc = generated_at_utc.astimezone(timezone.utc)

    normalized_ticker = normalize_ticker(ticker)
    ticker_object = yf.Ticker(normalized_ticker)
    market = fetch_market_data(
        ticker_object,
        normalized_ticker,
        as_of=as_of,
        now_utc=generated_at_utc,
    )

    if market.requested_as_of < market.exchange_today:
        fundamentals = historical_fundamental_context()
        company_name = None
    else:
        fundamentals, company_name = fetch_current_fundamental_context(
            ticker_object,
            as_of=market.requested_as_of,
        )

    return assemble_snapshot(
        market=market,
        fundamentals=fundamentals,
        generated_at_utc=generated_at_utc,
        company_name=company_name,
        source_version=_yfinance_version(),
    )


def assemble_snapshot(
    *,
    market: MarketDataBundle,
    fundamentals: FundamentalRiskContext,
    generated_at_utc: datetime,
    company_name: str | None = None,
    source_version: str | None = None,
) -> TechnicalSnapshot:
    """Assemble a snapshot from already-fetched inputs (also used by tests)."""

    technical = calculate_technical_sections(market.frame)
    return TechnicalSnapshot(
        instrument=Instrument(
            ticker=market.ticker,
            name=company_name,
            exchange=market.exchange,
            market_currency=market.market_currency,
            exchange_timezone=market.exchange_timezone,
        ),
        metadata=SnapshotMetadata(
            requested_as_of=market.requested_as_of,
            as_of_session=market.as_of_session,
            generated_at_utc=generated_at_utc,
            source_version=source_version or _yfinance_version(),
            observation_count=len(market.frame),
            market_data_age_days=market.market_data_age_days,
            is_stale=market.is_stale,
            data_quality_warnings=market.warnings + technical.warnings,
        ),
        current_candle=technical.current_candle,
        trend=technical.trend,
        momentum=technical.momentum,
        volume=technical.volume,
        volatility=technical.volatility,
        price_structure=technical.price_structure,
        fundamental_risk_context=fundamentals,
    )


def _yfinance_version() -> str:
    try:
        return version("yfinance")
    except PackageNotFoundError:
        return getattr(yf, "__version__", "unknown")
