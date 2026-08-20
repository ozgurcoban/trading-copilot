from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_copilot.indicators import (
    calculate_technical_sections,
    classify_ma_direction,
    find_recent_confirmed_pivot,
    wilder_atr,
    wilder_rsi,
)
from trading_copilot.models import (
    MovingAverageDirection,
    MovingAverageMetric,
    MomentumSnapshot,
    PriceStructureSnapshot,
    VolumeSnapshot,
)

from .helpers import make_ohlcv


@pytest.mark.parametrize(
    ("slope", "expected"),
    [
        (0.1000001, MovingAverageDirection.RISING),
        (0.10, MovingAverageDirection.FLAT),
        (0.0, MovingAverageDirection.FLAT),
        (-0.10, MovingAverageDirection.FLAT),
        (-0.1000001, MovingAverageDirection.FALLING),
    ],
)
def test_ma_direction_uses_inclusive_explicit_flat_band(
    slope: float, expected: MovingAverageDirection
) -> None:
    assert classify_ma_direction(slope) is expected


def test_wilder_rsi_handles_monotonic_and_constant_prices() -> None:
    increasing = pd.Series(np.arange(1.0, 31.0))
    decreasing = increasing.iloc[::-1].reset_index(drop=True)
    constant = pd.Series(np.full(30, 7.0))

    assert wilder_rsi(increasing).iloc[-1] == pytest.approx(100.0)
    assert wilder_rsi(decreasing).iloc[-1] == pytest.approx(0.0)
    assert wilder_rsi(constant).iloc[-1] == pytest.approx(50.0)


def test_wilder_atr_is_seeded_then_smoothed() -> None:
    index = pd.bdate_range("2026-01-01", periods=30)
    close = pd.Series(np.arange(20.0, 50.0), index=index)
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 100.0,
        }
    )
    atr = wilder_atr(frame)

    assert atr.iloc[12] != atr.iloc[12]  # NaN before the 14-period seed.
    assert atr.iloc[13] == pytest.approx(2.0)
    assert atr.iloc[-1] == pytest.approx(2.0)


def test_relative_volume_excludes_current_session() -> None:
    frame = make_ohlcv(periods=260)
    frame.loc[:, "Volume"] = 10.0
    frame.iloc[-21:-1, frame.columns.get_loc("Volume")] = 100.0
    frame.iloc[-1, frame.columns.get_loc("Volume")] = 400.0

    sections = calculate_technical_sections(frame)

    assert sections.volume.average_20d_prior == pytest.approx(100.0)
    assert sections.volume.relative_volume == pytest.approx(4.0)


def test_recent_pivot_is_strict_confirmed_and_not_last_two_candles() -> None:
    index = pd.bdate_range("2026-01-01", periods=10)
    confirmed = pd.Series(
        [1.0, 2.0, 3.0, 9.0, 3.0, 2.0, 4.0, 5.0, 20.0, 21.0],
        index=index,
    )
    pivot = find_recent_confirmed_pivot(confirmed, kind="high")

    assert pivot is not None
    assert pivot.date == index[3].date()
    assert pivot.price == 9.0

    plateau = pd.Series([1.0, 2.0, 5.0, 5.0, 2.0, 1.0], index=index[:6])
    assert find_recent_confirmed_pivot(plateau, kind="high") is None


def test_all_sections_are_populated_from_valid_daily_data() -> None:
    frame = make_ohlcv()
    sections = calculate_technical_sections(frame)

    assert sections.current_candle.date == frame.index[-1].date()
    assert sections.momentum.return_1d_pct == pytest.approx(
        ((frame["Close"].iloc[-1] / frame["Close"].iloc[-2]) - 1.0) * 100.0
    )
    assert sections.price_structure.high_52w.price == pytest.approx(
        frame["High"].iloc[-252:].max()
    )
    assert sections.price_structure.low_52w.price == pytest.approx(
        frame["Low"].iloc[-252:].min()
    )


def test_distance_percentages_keep_latest_close_as_the_subject() -> None:
    frame = make_ohlcv()
    sections = calculate_technical_sections(frame)
    latest_close = float(frame["Close"].iloc[-1])

    for metric in (
        sections.trend.sma_20,
        sections.trend.sma_50,
        sections.trend.sma_200,
    ):
        assert metric.distance_pct == pytest.approx(
            ((latest_close / metric.value) - 1.0) * 100.0
        )
    assert sections.price_structure.distance_from_52w_high_pct == pytest.approx(
        (
            (latest_close / sections.price_structure.high_52w.price)
            - 1.0
        )
        * 100.0
    )

    ma_description = MovingAverageMetric.model_fields["distance_pct"].description
    high_description = PriceStructureSnapshot.model_fields[
        "distance_from_52w_high_pct"
    ].description
    assert ma_description is not None and "Latest close relative" in ma_description
    assert high_description is not None and "Latest close relative" in high_description


def test_snapshot_descriptions_limit_volume_and_single_pivot_semantics() -> None:
    volume_description = VolumeSnapshot.model_fields["relative_volume"].description
    pivot_high_description = PriceStructureSnapshot.model_fields[
        "recent_confirmed_pivot_high"
    ].description
    pivot_low_description = PriceStructureSnapshot.model_fields[
        "recent_confirmed_pivot_low"
    ].description

    assert volume_description is not None and "relative activity only" in volume_description
    assert pivot_high_description is not None and "single most recent" in pivot_high_description
    assert pivot_low_description is not None and "single most recent" in pivot_low_description


def test_rsi_schema_describes_momentum_not_buying_or_selling_pressure() -> None:
    description = MomentumSnapshot.model_fields["rsi_14"].description

    assert description is not None
    assert "momentum oscillator" in description
    assert "not direct buying pressure" in description
