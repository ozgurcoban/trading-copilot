# Trading Copilot v0.1

Milestone 1 is a deterministic Python engine that builds a validated
`TechnicalSnapshot` from daily yfinance data. It contains no AI or Telegram
integration.

Fundamental cash and debt values retain their balance-sheet `report_date` and
`report_age_days`. A period older than 120 calendar days receives the explicit
`fundamental_report_period_stale` warning. These period-end values must not be
described downstream as current cash, current debt, or the company's current
balance sheet.

```python
from trading_copilot import build_snapshot, pretty_snapshot_json

snapshot = build_snapshot("IREN")
print(pretty_snapshot_json(snapshot))
```

Run deterministic tests with `python -m pytest`. The live IREN test is opt-in:

```text
RUN_LIVE_TESTS=1 python -m pytest -m live
```
