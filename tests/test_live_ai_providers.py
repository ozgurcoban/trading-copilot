from __future__ import annotations

import os
from pathlib import Path

import pytest

from trading_copilot import (
    AnalysisModel,
    ClaudeAnalysisProvider,
    OpenAIAnalysisProvider,
    analyze_snapshot,
)
from trading_copilot.analysis.models import ProviderName
from trading_copilot.models import TechnicalSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = PROJECT_ROOT / "artifacts" / "comparison_snapshots"
TICKERS = ("IREN", "MU", "AG", "AAPL")


def _load_snapshot(ticker: str) -> TechnicalSnapshot:
    return TechnicalSnapshot.model_validate_json(
        (SNAPSHOT_DIR / f"{ticker.lower()}.json").read_text(encoding="utf-8")
    )


@pytest.mark.ai_live
@pytest.mark.parametrize("ticker", TICKERS)
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_AI_TESTS") != "1" or not os.environ.get("OPENAI_API_KEY"),
    reason="set RUN_LIVE_AI_TESTS=1 and OPENAI_API_KEY to call OpenAI",
)
def test_saved_snapshot_with_openai(ticker: str) -> None:
    snapshot = _load_snapshot(ticker)
    report = analyze_snapshot(snapshot, OpenAIAnalysisProvider())

    assert report.provider is ProviderName.OPENAI
    assert report.ticker == ticker


@pytest.mark.ai_live
@pytest.mark.parametrize("ticker", TICKERS)
@pytest.mark.parametrize(
    "model",
    (AnalysisModel.CLAUDE_OPUS_5, AnalysisModel.CLAUDE_FABLE_5),
)
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_AI_TESTS") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="set RUN_LIVE_AI_TESTS=1 and ANTHROPIC_API_KEY to call Claude",
)
def test_saved_snapshot_with_claude(ticker: str, model: AnalysisModel) -> None:
    snapshot = _load_snapshot(ticker)
    report = analyze_snapshot(snapshot, ClaudeAnalysisProvider(model=model))

    assert report.provider is ProviderName.CLAUDE
    assert report.model is model
    assert report.ticker == ticker
