from __future__ import annotations

import os

import pytest

from trading_copilot import (
    build_snapshot,
    canonical_snapshot_json,
    pretty_snapshot_json,
)
from trading_copilot.fundamentals import (
    FUNDAMENTAL_REPORT_STALE_AFTER_CALENDAR_DAYS,
)


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to call yfinance",
)
def test_live_iren_snapshot() -> None:
    snapshot = build_snapshot("IREN")
    payload = canonical_snapshot_json(snapshot)
    print(pretty_snapshot_json(snapshot))

    assert snapshot.instrument.ticker == "IREN"
    assert snapshot.metadata.observation_count >= 252
    assert snapshot.current_candle.date <= snapshot.metadata.requested_as_of
    assert "NaN" not in payload
    assert "Infinity" not in payload

    fundamentals = snapshot.fundamental_risk_context
    if fundamentals.report_date is not None:
        expected_age = (
            snapshot.metadata.requested_as_of - fundamentals.report_date
        ).days
        assert fundamentals.report_age_days == expected_age
        freshness_codes = {warning.code for warning in fundamentals.warnings}
        assert (
            "fundamental_report_period_stale" in freshness_codes
        ) is (expected_age > FUNDAMENTAL_REPORT_STALE_AFTER_CALENDAR_DAYS)
