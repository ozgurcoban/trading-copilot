"""Deterministic technical-indicator calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .errors import InsufficientHistoryError, MarketDataValidationError
from .models import (
    CurrentCandle,
    DatedPriceLevel,
    MomentumSnapshot,
    MovingAverageDirection,
    MovingAverageMetric,
    PriceStructureSnapshot,
    SnapshotWarning,
    TrendSnapshot,
    VolatilitySnapshot,
    VolumeSnapshot,
)

MA_FLAT_TOLERANCE_PCT = 0.10


@dataclass(frozen=True)
class TechnicalSections:
    current_candle: CurrentCandle
    trend: TrendSnapshot
    momentum: MomentumSnapshot
    volume: VolumeSnapshot
    volatility: VolatilitySnapshot
    price_structure: PriceStructureSnapshot
    warnings: tuple[SnapshotWarning, ...] = ()


def classify_ma_direction(
    slope_5d_pct: float,
    *,
    tolerance_pct: float = MA_FLAT_TOLERANCE_PCT,
) -> MovingAverageDirection:
    """Classify a five-session MA change using an inclusive flat band."""

    if slope_5d_pct > tolerance_pct:
        return MovingAverageDirection.RISING
    if slope_5d_pct < -tolerance_pct:
        return MovingAverageDirection.FALLING
    return MovingAverageDirection.FLAT


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's seeded recursive smoothing."""

    close = close.astype(float)
    result = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= period:
        return result

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    avg_gain = float(gains.iloc[1 : period + 1].mean())
    avg_loss = float(losses.iloc[1 : period + 1].mean())

    def rsi_value(gain: float, loss: float) -> float:
        if loss == 0.0 and gain == 0.0:
            return 50.0
        if loss == 0.0:
            return 100.0
        if gain == 0.0:
            return 0.0
        relative_strength = gain / loss
        return 100.0 - (100.0 / (1.0 + relative_strength))

    result.iloc[period] = rsi_value(avg_gain, avg_loss)
    for position in range(period + 1, len(close)):
        avg_gain = ((avg_gain * (period - 1)) + float(gains.iloc[position])) / period
        avg_loss = ((avg_loss * (period - 1)) + float(losses.iloc[position])) / period
        result.iloc[position] = rsi_value(avg_gain, avg_loss)

    return result


def true_range(frame: pd.DataFrame) -> pd.Series:
    """Calculate daily true range."""

    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    previous_close = frame["Close"].astype(float).shift(1)
    components = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return components.max(axis=1, skipna=True)


def wilder_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR using Wilder's seeded recursive smoothing."""

    ranges = true_range(frame)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(ranges) < period:
        return result

    average = float(ranges.iloc[:period].mean())
    result.iloc[period - 1] = average
    for position in range(period, len(ranges)):
        average = ((average * (period - 1)) + float(ranges.iloc[position])) / period
        result.iloc[position] = average
    return result


def find_recent_confirmed_pivot(
    series: pd.Series,
    *,
    kind: str,
    lookback_sessions: int = 60,
) -> DatedPriceLevel | None:
    """Return the latest strict five-candle confirmed pivot."""

    if kind not in {"high", "low"}:
        raise ValueError("kind must be 'high' or 'low'")
    if len(series) < 5:
        return None

    values = series.astype(float).to_numpy()
    first_candidate = max(2, len(values) - lookback_sessions)
    for position in range(len(values) - 3, first_candidate - 1, -1):
        center = values[position]
        neighbours = np.array(
            [
                values[position - 2],
                values[position - 1],
                values[position + 1],
                values[position + 2],
            ]
        )
        is_pivot = bool(np.all(center > neighbours)) if kind == "high" else bool(
            np.all(center < neighbours)
        )
        if is_pivot:
            return DatedPriceLevel(
                date=_index_date(series.index[position]),
                price=float(center),
            )
    return None


def calculate_technical_sections(frame: pd.DataFrame) -> TechnicalSections:
    """Calculate all deterministic TechnicalSnapshot sections."""

    if len(frame) < 252:
        raise InsufficientHistoryError(
            f"At least 252 sessions are required, received {len(frame)}"
        )

    close = frame["Close"].astype(float)
    latest_close = float(close.iloc[-1])
    warnings: list[SnapshotWarning] = []

    moving_averages: dict[int, MovingAverageMetric] = {}
    for window in (20, 50, 200):
        series = close.rolling(window=window, min_periods=window).mean()
        current = float(series.iloc[-1])
        five_sessions_ago = float(series.iloc[-6])
        if not np.isfinite(current) or not np.isfinite(five_sessions_ago):
            raise MarketDataValidationError(
                f"Unable to calculate SMA{window} and its five-session slope"
            )
        slope = ((current / five_sessions_ago) - 1.0) * 100.0
        moving_averages[window] = MovingAverageMetric(
            value=current,
            distance_pct=((latest_close / current) - 1.0) * 100.0,
            slope_5d_pct=slope,
            direction=classify_ma_direction(slope),
        )

    rsi = float(wilder_rsi(close, period=14).iloc[-1])
    atr = float(wilder_atr(frame, period=14).iloc[-1])
    if not np.isfinite(rsi) or not np.isfinite(atr):
        raise MarketDataValidationError("Unable to calculate RSI14 or ATR14")

    previous_close = float(close.iloc[-2])
    latest_open = float(frame["Open"].iloc[-1])
    latest_volume = float(frame["Volume"].iloc[-1])
    average_volume = float(frame["Volume"].astype(float).iloc[-21:-1].mean())
    if average_volume == 0.0:
        relative_volume = None
        warnings.append(
            SnapshotWarning(
                code="relative_volume_unavailable_zero_baseline",
                message="Relative volume is unavailable because prior 20-session average volume is zero.",
            )
        )
    else:
        relative_volume = latest_volume / average_volume

    current_candle = CurrentCandle(
        date=_index_date(frame.index[-1]),
        open=latest_open,
        high=float(frame["High"].iloc[-1]),
        low=float(frame["Low"].iloc[-1]),
        close=latest_close,
        daily_change_absolute=latest_close - previous_close,
        daily_change_pct=((latest_close / previous_close) - 1.0) * 100.0,
        gap_absolute=latest_open - previous_close,
        gap_pct=((latest_open / previous_close) - 1.0) * 100.0,
    )

    high_20 = _dated_extreme(frame["High"].iloc[-20:], kind="high")
    low_20 = _dated_extreme(frame["Low"].iloc[-20:], kind="low")
    high_52 = _dated_extreme(frame["High"].iloc[-252:], kind="high")
    low_52 = _dated_extreme(frame["Low"].iloc[-252:], kind="low")

    return TechnicalSections(
        current_candle=current_candle,
        trend=TrendSnapshot(
            sma_20=moving_averages[20],
            sma_50=moving_averages[50],
            sma_200=moving_averages[200],
        ),
        momentum=MomentumSnapshot(
            rsi_14=rsi,
            return_1d_pct=_return_pct(close, 1),
            return_5d_pct=_return_pct(close, 5),
            return_20d_pct=_return_pct(close, 20),
        ),
        volume=VolumeSnapshot(
            latest=latest_volume,
            average_20d_prior=average_volume,
            relative_volume=relative_volume,
        ),
        volatility=VolatilitySnapshot(
            atr_14=atr,
            atr_14_pct=(atr / latest_close) * 100.0,
        ),
        price_structure=PriceStructureSnapshot(
            high_20d=high_20,
            low_20d=low_20,
            high_52w=high_52,
            low_52w=low_52,
            distance_from_52w_high_pct=((latest_close / high_52.price) - 1.0)
            * 100.0,
            recent_confirmed_pivot_high=find_recent_confirmed_pivot(
                frame["High"], kind="high"
            ),
            recent_confirmed_pivot_low=find_recent_confirmed_pivot(
                frame["Low"], kind="low"
            ),
        ),
        warnings=tuple(warnings),
    )


def _return_pct(close: pd.Series, sessions: int) -> float:
    return ((float(close.iloc[-1]) / float(close.iloc[-(sessions + 1)])) - 1.0) * 100.0


def _dated_extreme(series: pd.Series, *, kind: str) -> DatedPriceLevel:
    values = series.astype(float)
    target = float(values.max()) if kind == "high" else float(values.min())
    matching_positions = np.flatnonzero(np.isclose(values.to_numpy(), target, rtol=0, atol=0))
    position = int(matching_positions[-1])
    return DatedPriceLevel(
        date=_index_date(values.index[position]),
        price=target,
    )


def _index_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()
