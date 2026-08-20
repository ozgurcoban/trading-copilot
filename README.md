# Trading Copilot v0.1

Milestone 1 is a deterministic Python engine that builds a validated
`TechnicalSnapshot` from daily yfinance data. Milestone 2 adds a thin,
provider-neutral AI analysis layer for GPT-5.6 Sol, Claude Opus 5, and Claude
Fable 5. Telegram is not part of these milestones.

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

## AI analysis (Milestone 2)

Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in the environment. All three
models receive the same canonical snapshot and prompt, use provider-specific
`high` effort, and return the same immutable `AnalysisReport` schema. The
Claude Opus 5 and Claude Fable 5 choices both use the same thin
`ClaudeAnalysisProvider` adapter.

```python
from trading_copilot import (
    AnalysisModel,
    build_snapshot,
    compare_all_models,
    analyze_with_model,
)

snapshot = build_snapshot("IREN")
# Select exactly one model.
report = analyze_with_model(snapshot, AnalysisModel.CLAUDE_FABLE_5)

# Or call each model once with this exact snapshot and prompt.
comparison = compare_all_models(snapshot)
```

Every report includes provider, model, normalized token usage when exposed by
the API, latency, response ID, generation timestamp, and an estimated USD cost.
Pricing lives in the separate, explicit `MODEL_PRICING_USD` configuration; it
is not embedded in either provider adapter.

Snapshot percentage distances keep their defined subject and direction. For
example, a moving-average `distance_pct` always describes the latest close
relative to that moving average. The AI must not invert or recalculate it; when
the opposite viewpoint has no explicit snapshot field, the report states the
two price levels without inventing a percentage.

Relative volume is limited to describing observed activity versus its prior
20-session baseline; it cannot establish panic, capitulation, accumulation, or
trader intent. A single recent confirmed pivot is an isolated level, not proof
of a sequence of higher or lower pivots. The main fundamental risk block is at
most two short sentences for Telegram reading, while detailed period-end
figures and freshness limitations remain in the report's `limitations`.
RSI is described strictly as momentum, never as a direct measure of buying or
selling pressure.

The shared prompt requires every warning to be surfaced. Fundamental amounts
must always be described as period-end values for `report_date`, never as the
company's current cash, debt, or balance sheet. Identically named effort levels
do not imply equivalent internal compute between providers.
