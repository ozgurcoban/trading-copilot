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

## Telegram (Milestone 3)

Milestone 3 is a thin local Telegram interface. It adds no analysis logic and
keeps the approved AI prompt frozen at v0.4. Copy the variable names from
`.env.example` into the ignored local `.env` and set:

```dotenv
TELEGRAM_BOT_TOKEN=your_botfather_token
TELEGRAM_ALLOWED_USER_IDS=your_numeric_telegram_user_id
```

Load the local environment and start long polling:

```text
set -a
source .env
set +a
.venv/bin/python -m trading_copilot.telegram
```

Send one Yahoo-compatible ticker such as `IREN`. The bot builds exactly one
immutable snapshot, stores it in memory for 15 minutes, and offers this inline
keyboard:

```text
[ GPT-5.6 Sol ]
[ Claude Opus 5 ] [ Claude Fable 5 ]
[ Jämför alla ]
```

Callbacks are bound to the random pending-request ID, allowed user, private
chat, and exact Telegram message, then consumed on first valid use. The bot
acknowledges the callback and replaces the keyboard with `Analys pågår…` before
starting a model call. The first result is one deterministic compact decision
overview of at most 1,200 characters with price, bias, a two-sentence setup,
moving-average direction, momentum, non-repeated support and resistance,
one-sentence scenarios, up to three watch items, compact risk, latency, and
estimated cost. It has a
`Visa full analys` button bound to the allowed user, private chat, and exact
compact message. The completed report stays in memory for 15 minutes and opening
it performs no new snapshot or model call.

The full view renders every `AnalysisReport` content field using Telegram HTML
and deterministic chunks of at most 3,900 characters, below Telegram's
4,096-character limit. Both views exclude `prompt_sha256`, `snapshot_sha256`,
and provider response IDs from the normal user interface.

`Jämför alla` runs the same snapshot separately through Sol, Opus, and Fable,
then renders three labeled compact reports, each with its own full-report
button, without a fourth summarizer. The bot has no
webhook, database, scheduler, history, notifications, portfolio, or journal.

## Future roadmap: Trade Decision Layer

A separate Trade Decision Layer may later translate an approved technical
analysis into explicit entry conditions, invalidation, risk/reward, and
position sizing. It is intentionally not implemented in v0.1. The future layer
must remain separate from `TechnicalSnapshot` and `AnalysisReport`, define its
own validated schema and risk assumptions, and be evaluated independently
before it can influence the Telegram output.
