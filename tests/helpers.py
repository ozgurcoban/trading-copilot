from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def make_ohlcv(
    *,
    periods: int = 320,
    end: str = "2026-08-20",
    timezone: str | None = None,
) -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=periods, tz=timezone)
    positions = np.arange(periods, dtype=float)
    close = 20.0 + (positions * 0.08) + (np.sin(positions / 6.0) * 1.5)
    open_price = close + (np.cos(positions / 4.0) * 0.25)
    high = np.maximum(open_price, close) + 0.8
    low = np.minimum(open_price, close) - 0.8
    volume = 1_000_000.0 + ((positions % 19) * 25_000.0)
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


class FakeTicker:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        metadata: dict | None = None,
        balance_sheet: pd.DataFrame | None = None,
        info: dict | None = None,
    ) -> None:
        self.frame = frame
        self.metadata = metadata or {
            "exchangeTimezoneName": "America/New_York",
            "exchangeName": "NMS",
            "currency": "USD",
            "currentTradingPeriod": {
                "regular": {
                    "end": int(
                        pd.Timestamp("2026-08-20T20:00:00Z").timestamp()
                    )
                }
            },
        }
        self.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        self.info = info or {}
        self.balance_sheet_calls = 0
        self.info_calls = 0

    def history(self, **_: object) -> pd.DataFrame:
        return self.frame.copy()

    def get_history_metadata(self) -> dict:
        return self.metadata.copy()

    def get_balance_sheet(self, **_: object) -> pd.DataFrame:
        self.balance_sheet_calls += 1
        return self.balance_sheet.copy()

    def get_info(self) -> dict:
        self.info_calls += 1
        return self.info.copy()


def utc(value: str) -> datetime:
    return pd.Timestamp(value).to_pydatetime()
