from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from trading_copilot.fundamentals import (
    extract_fundamental_context,
    historical_fundamental_context,
)
from trading_copilot.models import (
    FundamentalStatus,
    NetPositionKind,
)


def test_historical_fundamentals_are_explicitly_withheld() -> None:
    context = historical_fundamental_context()

    assert context.status is FundamentalStatus.UNAVAILABLE
    assert context.report_date is None
    assert context.report_age_days is None
    assert context.cash is None
    assert context.total_debt is None
    assert [warning.code for warning in context.warnings] == [
        "historical_fundamentals_withheld_no_point_in_time_source"
    ]


def test_latest_eligible_report_is_used_without_lookahead() -> None:
    sheet = pd.DataFrame(
        {
            pd.Timestamp("2026-09-30"): [900.0, 100.0],
            pd.Timestamp("2026-06-30"): [400.0, 650.0],
            pd.Timestamp("2026-03-31"): [300.0, 700.0],
        },
        index=["CashAndCashEquivalents", "TotalDebt"],
    )
    context = extract_fundamental_context(
        sheet,
        as_of=date(2026, 8, 20),
        reporting_currency="USD",
    )

    assert context.report_date == date(2026, 6, 30)
    assert context.report_age_days == 51
    assert context.cash is not None and context.cash.value == 400.0
    assert context.total_debt is not None and context.total_debt.value == 650.0
    assert context.net_position is not None
    assert context.net_position.kind is NetPositionKind.NET_DEBT
    assert context.net_position.amount == 250.0


def test_cash_fallback_and_net_cash_are_explicitly_defined() -> None:
    sheet = pd.DataFrame(
        {pd.Timestamp("2026-06-30"): [800.0, 250.0]},
        index=["Cash Cash Equivalents And Short Term Investments", "Total Debt"],
    )
    context = extract_fundamental_context(
        sheet,
        as_of=date(2026, 8, 20),
        reporting_currency="USD",
    )

    assert context.status is FundamentalStatus.AVAILABLE
    assert context.cash is not None
    assert context.cash.source_row == "Cash Cash Equivalents And Short Term Investments"
    assert context.net_position is not None
    assert context.net_position.kind is NetPositionKind.NET_CASH
    assert context.net_position.amount == 550.0


def test_report_period_freshness_is_explicit_at_120_and_121_day_boundary() -> None:
    as_of = date(2026, 8, 20)

    def context_with_age(age_days: int):
        report_date = as_of - timedelta(days=age_days)
        sheet = pd.DataFrame(
            {pd.Timestamp(report_date): [800.0, 250.0]},
            index=["CashAndCashEquivalents", "TotalDebt"],
        )
        return extract_fundamental_context(
            sheet,
            as_of=as_of,
            reporting_currency="USD",
        )

    fresh = context_with_age(120)
    stale = context_with_age(121)

    assert fresh.report_age_days == 120
    assert "fundamental_report_period_stale" not in {
        warning.code for warning in fresh.warnings
    }
    assert stale.report_age_days == 121
    stale_warning = next(
        warning
        for warning in stale.warnings
        if warning.code == "fundamental_report_period_stale"
    )
    assert "must not be treated as current balances" in stale_warning.message


def test_partial_and_unavailable_fundamentals_are_not_fabricated() -> None:
    partial_sheet = pd.DataFrame(
        {pd.Timestamp("2026-06-30"): [123.0]}, index=["CashAndCashEquivalents"]
    )
    partial = extract_fundamental_context(
        partial_sheet,
        as_of=date(2026, 8, 20),
        reporting_currency="USD",
    )
    unavailable = extract_fundamental_context(
        pd.DataFrame(),
        as_of=date(2026, 8, 20),
    )

    assert partial.status is FundamentalStatus.PARTIAL
    assert partial.total_debt is None
    assert partial.net_position is None
    assert unavailable.status is FundamentalStatus.UNAVAILABLE
