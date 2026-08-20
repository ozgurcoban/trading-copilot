from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trading_copilot.errors import (
    InvalidAsOfError,
    InvalidTickerError,
    MarketDataValidationError,
)
from trading_copilot.market_data import (
    fetch_market_data,
    normalize_ticker,
    stale_market_data,
    validate_daily_ohlcv,
)

from .helpers import FakeTicker, make_ohlcv, utc


def test_ticker_is_normalized_and_malformed_input_is_rejected() -> None:
    assert normalize_ticker("  iren ") == "IREN"
    assert normalize_ticker("brk-b") == "BRK-B"
    with pytest.raises(InvalidTickerError):
        normalize_ticker("IREN please")


def test_stale_rule_is_fresh_at_seven_days_and_stale_at_eight() -> None:
    latest = date(2026, 8, 10)
    assert stale_market_data(date(2026, 8, 17), latest) == (7, False)
    assert stale_market_data(date(2026, 8, 18), latest) == (8, True)


def test_current_unfinished_daily_candle_is_excluded() -> None:
    frame = make_ohlcv(periods=320, end="2026-08-20", timezone="America/New_York")
    bundle = fetch_market_data(
        FakeTicker(frame),
        "IREN",
        as_of=None,
        now_utc=utc("2026-08-20T15:00:00Z"),
    )

    assert bundle.requested_as_of == date(2026, 8, 20)
    assert bundle.as_of_session == date(2026, 8, 19)
    assert bundle.frame.index[-1].date() == date(2026, 8, 19)
    assert "current_session_excluded_unconfirmed_close" in {
        warning.code for warning in bundle.warnings
    }


def test_completed_current_daily_candle_is_retained_after_regular_close() -> None:
    frame = make_ohlcv(periods=320, end="2026-08-20", timezone="America/New_York")
    bundle = fetch_market_data(
        FakeTicker(frame),
        "IREN",
        as_of=None,
        now_utc=utc("2026-08-20T21:00:00Z"),
    )

    assert bundle.as_of_session == date(2026, 8, 20)


def test_historical_as_of_is_inclusive_and_future_is_rejected() -> None:
    frame = make_ohlcv(periods=340, end="2026-08-20", timezone="America/New_York")
    ticker = FakeTicker(frame)
    bundle = fetch_market_data(
        ticker,
        "IREN",
        as_of=date(2026, 8, 10),
        now_utc=utc("2026-08-20T21:00:00Z"),
    )
    assert bundle.frame.index[-1].date() <= date(2026, 8, 10)
    assert bundle.requested_as_of == date(2026, 8, 10)

    with pytest.raises(InvalidAsOfError):
        fetch_market_data(
            ticker,
            "IREN",
            as_of=date(2026, 8, 21),
            now_utc=utc("2026-08-20T21:00:00Z"),
        )


@pytest.mark.parametrize("column", ["Open", "High", "Low", "Close", "Volume"])
def test_non_finite_ohlcv_is_rejected(column: str) -> None:
    frame = make_ohlcv(periods=260)
    frame.iloc[-1, frame.columns.get_loc(column)] = np.nan
    with pytest.raises(MarketDataValidationError, match="non-finite"):
        validate_daily_ohlcv(frame)


def test_impossible_price_bar_and_insufficient_history_are_rejected() -> None:
    frame = make_ohlcv(periods=260)
    frame.iloc[-1, frame.columns.get_loc("High")] = frame["Low"].iloc[-1] - 1.0
    with pytest.raises(MarketDataValidationError, match="high"):
        validate_daily_ohlcv(frame)

    with pytest.raises(MarketDataValidationError, match="252"):
        validate_daily_ohlcv(make_ohlcv(periods=251))
