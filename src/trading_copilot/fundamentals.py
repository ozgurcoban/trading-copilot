"""Minimal, explicitly time-bounded balance-sheet context."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .models import (
    FundamentalMetric,
    FundamentalRiskContext,
    FundamentalStatus,
    NetPosition,
    NetPositionKind,
    SnapshotWarning,
)

_CASH_DIRECT = "CashAndCashEquivalents"
_CASH_FALLBACK = "CashCashEquivalentsAndShortTermInvestments"
_TOTAL_DEBT = "TotalDebt"
FUNDAMENTAL_REPORT_STALE_AFTER_CALENDAR_DAYS = 120


def historical_fundamental_context() -> FundamentalRiskContext:
    """Withhold current-only Yahoo fundamentals from historical snapshots."""

    return FundamentalRiskContext(
        status=FundamentalStatus.UNAVAILABLE,
        warnings=(
            SnapshotWarning(
                code="historical_fundamentals_withheld_no_point_in_time_source",
                message=(
                    "Fundamentals were withheld because yfinance does not provide "
                    "a reliable publication timestamp for point-in-time historical use."
                ),
            ),
        ),
    )


def fetch_current_fundamental_context(
    ticker_object: Any,
    *,
    as_of: date,
) -> tuple[FundamentalRiskContext, str | None]:
    """Fetch the latest eligible quarterly balance sheet and optional company name."""

    warnings: list[SnapshotWarning] = []
    try:
        balance_sheet = ticker_object.get_balance_sheet(
            freq="quarterly", pretty=False
        )
    except Exception as exc:
        balance_sheet = pd.DataFrame()
        warnings.append(
            SnapshotWarning(
                code="fundamentals_fetch_failed",
                message=f"The latest available balance-sheet data could not be fetched: {exc}",
            )
        )

    company_name: str | None = None
    reporting_currency: str | None = None
    try:
        info = ticker_object.get_info()
        if isinstance(info, dict):
            company_name = _string_or_none(
                info.get("longName") or info.get("shortName")
            )
            reporting_currency = _string_or_none(info.get("financialCurrency"))
    except Exception:
        warnings.append(
            SnapshotWarning(
                code="company_info_unavailable",
                message="Company name and reporting currency were unavailable from yfinance.",
            )
        )

    context = extract_fundamental_context(
        balance_sheet,
        as_of=as_of,
        reporting_currency=reporting_currency,
        initial_warnings=warnings,
    )
    return context, company_name


def extract_fundamental_context(
    balance_sheet: pd.DataFrame,
    *,
    as_of: date,
    reporting_currency: str | None = None,
    initial_warnings: list[SnapshotWarning] | None = None,
) -> FundamentalRiskContext:
    """Extract cash and direct total debt from the latest column not after as_of."""

    warnings = list(initial_warnings or [])
    if balance_sheet is None or balance_sheet.empty:
        warnings.append(
            SnapshotWarning(
                code="fundamentals_unavailable",
                message="No quarterly balance sheet was available from yfinance.",
            )
        )
        return FundamentalRiskContext(
            status=FundamentalStatus.UNAVAILABLE,
            reporting_currency=reporting_currency,
            warnings=tuple(warnings),
        )

    eligible_columns: list[tuple[pd.Timestamp, object]] = []
    for column in balance_sheet.columns:
        try:
            timestamp = pd.Timestamp(column)
        except (TypeError, ValueError):
            continue
        if timestamp.date() <= as_of:
            eligible_columns.append((timestamp, column))
    if not eligible_columns:
        warnings.append(
            SnapshotWarning(
                code="fundamentals_unavailable_as_of",
                message="No balance-sheet period ended on or before the requested as_of date.",
            )
        )
        return FundamentalRiskContext(
            status=FundamentalStatus.UNAVAILABLE,
            reporting_currency=reporting_currency,
            warnings=tuple(warnings),
        )

    report_timestamp, selected_column = max(eligible_columns, key=lambda item: item[0])
    report_age_days = (as_of - report_timestamp.date()).days
    if report_age_days > FUNDAMENTAL_REPORT_STALE_AFTER_CALENDAR_DAYS:
        warnings.append(
            SnapshotWarning(
                code="fundamental_report_period_stale",
                message=(
                    f"The latest yfinance balance-sheet period ended {report_age_days} "
                    f"calendar days before snapshot as_of. This exceeds the explicit "
                    f"{FUNDAMENTAL_REPORT_STALE_AFTER_CALENDAR_DAYS}-day freshness "
                    "threshold; cash and debt are period-end figures and must not be "
                    "treated as current balances."
                ),
            )
        )
    selected = balance_sheet[selected_column]
    cash = _metric_from_rows(
        selected,
        (_CASH_DIRECT, _CASH_FALLBACK),
        definitions={
            _CASH_DIRECT: "Cash and cash equivalents reported by yfinance.",
            _CASH_FALLBACK: (
                "Cash, cash equivalents, and short-term investments reported by yfinance."
            ),
        },
    )
    debt = _metric_from_rows(
        selected,
        (_TOTAL_DEBT,),
        definitions={
            _TOTAL_DEBT: "Total debt reported directly by yfinance.",
        },
    )

    if cash is None:
        warnings.append(
            SnapshotWarning(
                code="cash_unavailable",
                message="Neither supported cash balance-sheet row was available.",
            )
        )
    if debt is None:
        warnings.append(
            SnapshotWarning(
                code="total_debt_unavailable",
                message="A directly reported TotalDebt balance-sheet row was unavailable.",
            )
        )

    net_position: NetPosition | None = None
    if cash is not None and debt is not None:
        signed_net_debt = debt.value - cash.value
        if signed_net_debt >= 0.0:
            net_position = NetPosition(
                kind=NetPositionKind.NET_DEBT,
                amount=signed_net_debt,
                derivation="total_debt - cash",
            )
        else:
            net_position = NetPosition(
                kind=NetPositionKind.NET_CASH,
                amount=abs(signed_net_debt),
                derivation="cash - total_debt",
            )

    populated = sum(metric is not None for metric in (cash, debt))
    if populated == 2:
        status = FundamentalStatus.AVAILABLE
    elif populated == 1:
        status = FundamentalStatus.PARTIAL
    else:
        status = FundamentalStatus.UNAVAILABLE

    if reporting_currency is None:
        warnings.append(
            SnapshotWarning(
                code="fundamental_reporting_currency_unavailable",
                message="The currency of reported fundamental values was unavailable.",
            )
        )

    return FundamentalRiskContext(
        status=status,
        report_date=report_timestamp.date(),
        report_age_days=report_age_days,
        reporting_currency=reporting_currency,
        cash=cash,
        total_debt=debt,
        net_position=net_position,
        warnings=tuple(warnings),
    )


def _metric_from_rows(
    selected: pd.Series,
    candidates: tuple[str, ...],
    *,
    definitions: dict[str, str],
) -> FundamentalMetric | None:
    normalized_rows = {_normalize_row_name(row): row for row in selected.index}
    for candidate in candidates:
        actual_row = normalized_rows.get(_normalize_row_name(candidate))
        if actual_row is None:
            continue
        value = selected.loc[actual_row]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric) or numeric < 0.0:
            continue
        return FundamentalMetric(
            value=numeric,
            source_row=str(actual_row),
            definition=definitions[candidate],
        )
    return None


def _normalize_row_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
