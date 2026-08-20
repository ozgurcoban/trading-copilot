"""Immutable Pydantic domain models for TechnicalSnapshot v0.1."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class SnapshotWarning(FrozenModel):
    code: str
    message: str


class Instrument(FrozenModel):
    ticker: str
    name: str | None = None
    exchange: str | None = None
    market_currency: str | None = None
    exchange_timezone: str | None = None


class SnapshotMetadata(FrozenModel):
    requested_as_of: date
    as_of_session: date
    generated_at_utc: datetime
    source: Literal["yfinance"] = "yfinance"
    source_version: str
    interval: Literal["1d"] = "1d"
    adjusted: Literal[True] = True
    repaired: Literal[False] = False
    observation_count: int = Field(ge=252)
    market_data_age_days: int = Field(ge=0)
    is_stale: bool
    data_quality_warnings: tuple[SnapshotWarning, ...] = ()

    @field_validator("generated_at_utc")
    @classmethod
    def timestamp_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)


class CurrentCandle(FrozenModel):
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    daily_change_absolute: float
    daily_change_pct: float
    gap_absolute: float
    gap_pct: float


class MovingAverageDirection(str, Enum):
    RISING = "rising"
    FLAT = "flat"
    FALLING = "falling"


class MovingAverageMetric(FrozenModel):
    value: float = Field(gt=0)
    distance_pct: float = Field(
        description=(
            "Latest close relative to this moving-average value, calculated as "
            "((latest_close / value) - 1) * 100. Positive means the close is above "
            "the moving average; negative means the close is below it."
        )
    )
    slope_5d_pct: float
    direction: MovingAverageDirection


class TrendSnapshot(FrozenModel):
    sma_20: MovingAverageMetric
    sma_50: MovingAverageMetric
    sma_200: MovingAverageMetric


class MomentumSnapshot(FrozenModel):
    rsi_14: float = Field(
        ge=0,
        le=100,
        description=(
            "Fourteen-period RSI momentum oscillator. It describes momentum, not "
            "direct buying pressure, selling pressure, or order flow."
        ),
    )
    return_1d_pct: float
    return_5d_pct: float
    return_20d_pct: float


class VolumeSnapshot(FrozenModel):
    latest: float = Field(ge=0)
    average_20d_prior: float = Field(ge=0)
    relative_volume: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Latest volume divided by the prior 20-session average. It measures "
            "relative activity only and does not identify intent, panic, "
            "capitulation, accumulation, or distribution."
        ),
    )


class VolatilitySnapshot(FrozenModel):
    atr_14: float = Field(ge=0)
    atr_14_pct: float = Field(ge=0)


class DatedPriceLevel(FrozenModel):
    date: date
    price: float = Field(gt=0)


class PriceStructureSnapshot(FrozenModel):
    high_20d: DatedPriceLevel
    low_20d: DatedPriceLevel
    high_52w: DatedPriceLevel
    low_52w: DatedPriceLevel
    distance_from_52w_high_pct: float = Field(
        description=(
            "Latest close relative to the 52-week high, calculated as "
            "((latest_close / high_52w.price) - 1) * 100."
        )
    )
    recent_confirmed_pivot_high: DatedPriceLevel | None = Field(
        default=None,
        description=(
            "The single most recent confirmed pivot high. One pivot does not "
            "establish a sequence of lower or higher highs."
        ),
    )
    recent_confirmed_pivot_low: DatedPriceLevel | None = Field(
        default=None,
        description=(
            "The single most recent confirmed pivot low. One pivot does not "
            "establish a sequence of lower or higher lows."
        ),
    )


class FundamentalStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class FundamentalMetric(FrozenModel):
    value: float = Field(ge=0)
    source_row: str
    definition: str


class NetPositionKind(str, Enum):
    NET_DEBT = "net_debt"
    NET_CASH = "net_cash"


class NetPosition(FrozenModel):
    kind: NetPositionKind
    amount: float = Field(ge=0)
    derivation: str


class FundamentalRiskContext(FrozenModel):
    status: FundamentalStatus
    report_date: date | None = None
    report_age_days: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Calendar days from the balance-sheet period end to snapshot as_of; "
            "this is period age, not publication age."
        ),
    )
    reporting_currency: str | None = None
    cash: FundamentalMetric | None = None
    total_debt: FundamentalMetric | None = None
    net_position: NetPosition | None = None
    warnings: tuple[SnapshotWarning, ...] = ()

    @model_validator(mode="after")
    def report_date_and_age_must_coexist(self) -> Self:
        if (self.report_date is None) != (self.report_age_days is None):
            raise ValueError("report_date and report_age_days must either both be set or both be absent")
        return self


class TechnicalSnapshot(FrozenModel):
    schema_version: Literal["0.1"] = "0.1"
    instrument: Instrument
    metadata: SnapshotMetadata
    current_candle: CurrentCandle
    trend: TrendSnapshot
    momentum: MomentumSnapshot
    volume: VolumeSnapshot
    volatility: VolatilitySnapshot
    price_structure: PriceStructureSnapshot
    fundamental_risk_context: FundamentalRiskContext
