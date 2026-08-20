from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from trading_copilot import FixedClock, build_snapshot, canonical_snapshot_json
from trading_copilot.fundamentals import (
    extract_fundamental_context,
    historical_fundamental_context,
)
from trading_copilot.market_data import MarketDataBundle
from trading_copilot.snapshot import assemble_snapshot

from .helpers import FakeTicker, make_ohlcv, utc


def make_market_bundle() -> MarketDataBundle:
    frame = make_ohlcv(periods=320, end="2026-08-20")
    return MarketDataBundle(
        ticker="IREN",
        frame=frame,
        requested_as_of=date(2026, 8, 20),
        as_of_session=date(2026, 8, 20),
        exchange_today=date(2026, 8, 20),
        exchange="NasdaqGS",
        market_currency="USD",
        exchange_timezone="America/New_York",
        market_data_age_days=0,
        is_stale=False,
    )


def test_canonical_snapshot_is_byte_deterministic_for_fixed_inputs_and_clock() -> None:
    timestamp = utc("2026-08-20T21:00:00Z")
    kwargs = {
        "market": make_market_bundle(),
        "fundamentals": historical_fundamental_context(),
        "generated_at_utc": timestamp,
        "source_version": "test",
    }
    first = assemble_snapshot(**kwargs)
    second = assemble_snapshot(**kwargs)

    assert canonical_snapshot_json(first).encode() == canonical_snapshot_json(second).encode()
    assert "NaN" not in canonical_snapshot_json(first)
    assert "Infinity" not in canonical_snapshot_json(first)


def test_only_generation_time_changes_when_only_clock_value_changes() -> None:
    common = {
        "market": make_market_bundle(),
        "fundamentals": historical_fundamental_context(),
        "source_version": "test",
    }
    first = assemble_snapshot(
        **common, generated_at_utc=utc("2026-08-20T21:00:00Z")
    ).model_dump(mode="json")
    second = assemble_snapshot(
        **common, generated_at_utc=utc("2026-08-20T21:01:00Z")
    ).model_dump(mode="json")

    first["metadata"].pop("generated_at_utc")
    second["metadata"].pop("generated_at_utc")
    assert first == second


def test_snapshot_models_are_immutable() -> None:
    snapshot = assemble_snapshot(
        market=make_market_bundle(),
        fundamentals=historical_fundamental_context(),
        generated_at_utc=utc("2026-08-20T21:00:00Z"),
        source_version="test",
    )
    with pytest.raises(ValidationError):
        snapshot.instrument.ticker = "OTHER"  # type: ignore[misc]


def test_serialized_snapshot_carries_report_date_age_and_freshness_warning() -> None:
    report_date = date(2026, 3, 31)
    balance_sheet = pd.DataFrame(
        {report_date: [2_000_000_000.0, 3_000_000_000.0]},
        index=["CashAndCashEquivalents", "TotalDebt"],
    )
    fundamentals = extract_fundamental_context(
        balance_sheet,
        as_of=date(2026, 8, 20),
        reporting_currency="USD",
    )
    snapshot = assemble_snapshot(
        market=make_market_bundle(),
        fundamentals=fundamentals,
        generated_at_utc=utc("2026-08-20T21:00:00Z"),
        source_version="test",
    )
    serialized = json.loads(canonical_snapshot_json(snapshot))[
        "fundamental_risk_context"
    ]

    assert serialized["report_date"] == "2026-03-31"
    assert serialized["report_age_days"] == 142
    assert "fundamental_report_period_stale" in {
        warning["code"] for warning in serialized["warnings"]
    }


def test_public_builder_uses_fixed_clock_and_never_fetches_current_fundamentals_for_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_ohlcv(
        periods=340, end="2026-08-20", timezone="America/New_York"
    )
    fake = FakeTicker(frame)
    monkeypatch.setattr("trading_copilot.snapshot.yf.Ticker", lambda _: fake)
    clock = FixedClock(utc("2026-08-20T21:00:00Z"))

    first = build_snapshot("iren", as_of=date(2026, 8, 18), clock=clock)
    second = build_snapshot("IREN", as_of=date(2026, 8, 18), clock=clock)

    assert canonical_snapshot_json(first) == canonical_snapshot_json(second)
    assert fake.balance_sheet_calls == 0
    assert fake.info_calls == 0
    assert first.metadata.requested_as_of == date(2026, 8, 18)
    assert first.current_candle.date <= date(2026, 8, 18)
    assert json.loads(canonical_snapshot_json(first))["metadata"][
        "generated_at_utc"
    ] == "2026-08-20T21:00:00Z"
    assert "historical_fundamentals_withheld_no_point_in_time_source" in {
        warning.code for warning in first.fundamental_risk_context.warnings
    }
