from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trading_copilot import (
    AnalysisModel,
    ClaudeAnalysisProvider,
    OpenAIAnalysisProvider,
    analyze_snapshot,
    analyze_with_model,
    canonical_snapshot_json,
    compare_all_models,
    compare_openai_and_claude,
    provider_for_model,
)
from trading_copilot.analysis.models import (
    AllModelComparison,
    AnalysisComparison,
    AnalysisContent,
    AnalysisReport,
    ProviderName,
    ScenarioAnalysis,
    TechnicalBias,
)
from trading_copilot.analysis.prompting import (
    ANALYSIS_SYSTEM_PROMPT,
    build_analysis_input,
)
from trading_copilot.analysis.providers import (
    ANALYSIS_EFFORT,
    CLAUDE_MODEL,
    CLAUDE_FABLE_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENAI_MODEL,
    AnalysisProviderError,
)
from trading_copilot.analysis.pricing import MODEL_PRICING_USD
from trading_copilot.models import TechnicalSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_iren_snapshot() -> TechnicalSnapshot:
    return TechnicalSnapshot.model_validate_json(
        (PROJECT_ROOT / "artifacts" / "iren_snapshot_2026-08-20.json").read_text()
    )


def make_content(label: str) -> AnalysisContent:
    return AnalysisContent(
        overall_bias=TechnicalBias.MIXED,
        overall_technical_picture=f"{label}: blandad teknisk bild.",
        trend=f"{label}: trendförklaring.",
        momentum=f"{label}: momentumförklaring.",
        volume_and_volatility=f"{label}: volym och volatilitet.",
        key_price_levels=f"{label}: nivåförklaring.",
        scenarios=ScenarioAnalysis(
            bullish=f"{label}: bullish villkor.",
            neutral=f"{label}: neutralt villkor.",
            bearish=f"{label}: bearish villkor.",
        ),
        what_to_watch=(f"{label}: bevaka pivotnivån.",),
        setup_summary=f"{label}: kort sammanfattning.",
        fundamental_risk_context=(
            f"{label}: periodslutsvärden per rapportdatum, inte aktuella balanser."
        ),
        limitations=(f"{label}: rapportperioden är gammal.",),
    )


class FakeOpenAIResponses:
    def __init__(self, content: AnalysisContent | None) -> None:
        self.content = content
        self.calls: list[dict] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_openai_test",
            output_parsed=self.content,
            usage=SimpleNamespace(
                input_tokens=2_000,
                output_tokens=1_000,
                total_tokens=3_000,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=500,
                    cache_write_tokens=0,
                ),
                output_tokens_details=SimpleNamespace(reasoning_tokens=300),
            ),
        )


class FakeOpenAIClient:
    def __init__(self, content: AnalysisContent | None) -> None:
        self.responses = FakeOpenAIResponses(content)


class FakeClaudeMessages:
    def __init__(
        self,
        content: AnalysisContent | None,
        *,
        stop_reason: str = "end_turn",
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="msg_claude_test",
            parsed_output=self.content,
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(
                input_tokens=1_500,
                cache_read_input_tokens=500,
                cache_creation_input_tokens=100,
                output_tokens=1_000,
                output_tokens_details=SimpleNamespace(thinking_tokens=250),
            ),
        )


class FakeClaudeClient:
    def __init__(
        self,
        content: AnalysisContent | None,
        *,
        stop_reason: str = "end_turn",
    ) -> None:
        self.messages = FakeClaudeMessages(content, stop_reason=stop_reason)


class SequenceTimer:
    def __init__(self, *values: float) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


FIXED_ANALYSIS_TIME = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def test_analysis_input_is_deterministic_and_contains_complete_snapshot() -> None:
    snapshot = load_iren_snapshot()
    first = build_analysis_input(snapshot)
    second = build_analysis_input(snapshot)

    assert first == second
    assert canonical_snapshot_json(snapshot) in first.user_prompt
    assert "fundamental_report_period_stale" in first.user_prompt
    assert "report_age_days" in first.user_prompt
    assert "never call them current cash" in first.system_prompt
    assert "Every numeric claim must be directly traceable" in first.system_prompt
    assert first.prompt_version == "0.4"
    assert "describes only where the latest close is relative" in first.system_prompt
    assert "Never invert either viewpoint" in first.system_prompt
    assert "Never negate, reciprocally transform" in first.system_prompt
    assert "omit that percentage" in first.system_prompt
    assert "cannot establish trader intent or market psychology" in first.system_prompt
    assert "Never infer panic selling, capitulation" in first.system_prompt
    assert "each contain only one isolated latest pivot" in first.system_prompt
    assert "Never claim a series of lower highs" in first.system_prompt
    assert "at most two short, Telegram-ready sentences" in first.system_prompt
    assert "Put the full freshness explanation" in first.system_prompt
    assert "RSI is a momentum oscillator" in first.system_prompt
    assert "not a direct measure of buying pressure" in first.system_prompt
    assert "visar inget tydligt momentumövertag åt något håll" in first.system_prompt


def test_analysis_content_rejects_blank_required_sections() -> None:
    payload = make_content("test").model_dump()
    payload["trend"] = "  "
    with pytest.raises(ValidationError, match="trend must not be blank"):
        AnalysisContent.model_validate(payload)


def test_output_schema_repeats_volume_pivot_and_fundamental_limits() -> None:
    schema = AnalysisContent.model_json_schema()
    properties = schema["properties"]
    scenario_properties = schema["$defs"]["ScenarioAnalysis"]["properties"]

    assert "without inferring trader intent" in properties[
        "volume_and_volatility"
    ]["description"]
    assert "At most two short, Telegram-ready sentences" in properties[
        "fundamental_risk_context"
    ]["description"]
    assert "Do not interpret RSI as direct" in properties["momentum"]["description"]
    assert "single confirmed pivot" in scenario_properties["bullish"]["description"]
    assert "single confirmed pivot" in scenario_properties["bearish"]["description"]


def test_openai_adapter_uses_responses_structured_output_and_high_effort() -> None:
    snapshot = load_iren_snapshot()
    client = FakeOpenAIClient(make_content("OpenAI"))
    report = analyze_snapshot(
        snapshot,
        OpenAIAnalysisProvider(
            client=client,
            clock=lambda: FIXED_ANALYSIS_TIME,
            timer=SequenceTimer(10.0, 10.25),
        ),
    )
    call = client.responses.calls[0]

    assert report.provider is ProviderName.OPENAI
    assert report.model == OPENAI_MODEL
    assert report.effort == ANALYSIS_EFFORT == "high"
    assert report.prompt_version == "0.4"
    assert call["model"] == OPENAI_MODEL
    assert call["reasoning"] == {"effort": "high"}
    assert call["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert call["store"] is False
    assert call["text_format"] is AnalysisContent
    assert call["input"][0] == {  # type: ignore[index]
        "role": "system",
        "content": ANALYSIS_SYSTEM_PROMPT,
    }
    assert report.run_metadata.response_id == "resp_openai_test"
    assert report.run_metadata.generated_at_utc == FIXED_ANALYSIS_TIME
    assert report.run_metadata.latency_ms == pytest.approx(250.0)
    assert report.run_metadata.token_usage is not None
    assert report.run_metadata.token_usage.model_dump() == {
        "input_tokens": 2_000,
        "output_tokens": 1_000,
        "total_tokens": 3_000,
        "cached_input_tokens": 500,
        "cache_write_input_tokens": 0,
        "reasoning_tokens": 300,
    }
    assert report.run_metadata.estimated_cost is not None
    assert report.run_metadata.estimated_cost.total_cost_usd == pytest.approx(0.03775)


def test_claude_adapter_uses_messages_structured_output_and_high_effort() -> None:
    snapshot = load_iren_snapshot()
    client = FakeClaudeClient(make_content("Claude"))
    report = analyze_snapshot(
        snapshot,
        ClaudeAnalysisProvider(
            client=client,
            clock=lambda: FIXED_ANALYSIS_TIME,
            timer=SequenceTimer(20.0, 20.4),
        ),
    )
    call = client.messages.calls[0]

    assert report.provider is ProviderName.CLAUDE
    assert report.model == CLAUDE_MODEL
    assert report.effort == ANALYSIS_EFFORT == "high"
    assert call["model"] == CLAUDE_MODEL
    assert call["output_config"] == {"effort": "high"}
    assert call["max_tokens"] == MAX_OUTPUT_TOKENS
    assert call["output_format"] is AnalysisContent
    assert call["system"] == ANALYSIS_SYSTEM_PROMPT
    assert report.run_metadata.response_id == "msg_claude_test"
    assert report.run_metadata.latency_ms == pytest.approx(400.0)
    assert report.run_metadata.token_usage is not None
    assert report.run_metadata.token_usage.model_dump() == {
        "input_tokens": 2_100,
        "output_tokens": 1_000,
        "total_tokens": 3_100,
        "cached_input_tokens": 500,
        "cache_write_input_tokens": 100,
        "reasoning_tokens": 250,
    }
    assert report.run_metadata.estimated_cost is not None
    assert report.run_metadata.estimated_cost.total_cost_usd == pytest.approx(0.033375)


def test_fable_uses_existing_claude_adapter_and_same_schema() -> None:
    snapshot = load_iren_snapshot()
    client = FakeClaudeClient(make_content("Fable"))

    report = analyze_with_model(
        snapshot,
        AnalysisModel.CLAUDE_FABLE_5,
        client=client,
    )
    call = client.messages.calls[0]

    assert report.provider is ProviderName.CLAUDE
    assert report.model is AnalysisModel.CLAUDE_FABLE_5
    assert call["model"] == CLAUDE_FABLE_MODEL
    assert call["output_config"] == {"effort": "high"}
    assert call["output_format"] is AnalysisContent
    assert report.run_metadata.estimated_cost is not None
    assert report.run_metadata.estimated_cost.total_cost_usd == pytest.approx(0.06675)


def test_model_selection_maps_three_models_to_two_adapters() -> None:
    openai_provider = provider_for_model(
        AnalysisModel.GPT_5_6_SOL,
        client=FakeOpenAIClient(make_content("Sol")),
    )
    opus_provider = provider_for_model(
        AnalysisModel.CLAUDE_OPUS_5,
        client=FakeClaudeClient(make_content("Opus")),
    )
    fable_provider = provider_for_model(
        AnalysisModel.CLAUDE_FABLE_5,
        client=FakeClaudeClient(make_content("Fable")),
    )

    assert isinstance(openai_provider, OpenAIAnalysisProvider)
    assert isinstance(opus_provider, ClaudeAnalysisProvider)
    assert isinstance(fable_provider, ClaudeAnalysisProvider)
    assert [model.value for model in AnalysisModel] == [
        OPENAI_MODEL,
        CLAUDE_MODEL,
        CLAUDE_FABLE_MODEL,
    ]
    assert set(MODEL_PRICING_USD) == set(AnalysisModel)


def test_compare_sends_identical_prompt_and_snapshot_to_both_providers() -> None:
    snapshot = load_iren_snapshot()
    canonical_before = canonical_snapshot_json(snapshot)
    openai_client = FakeOpenAIClient(make_content("OpenAI"))
    claude_client = FakeClaudeClient(make_content("Claude"))

    comparison = compare_openai_and_claude(
        snapshot,
        openai_provider=OpenAIAnalysisProvider(client=openai_client),
        claude_provider=ClaudeAnalysisProvider(client=claude_client),
    )

    openai_input = openai_client.responses.calls[0]["input"]  # type: ignore[index]
    claude_messages = claude_client.messages.calls[0]["messages"]  # type: ignore[index]
    assert openai_input[1]["content"] == claude_messages[0]["content"]
    assert comparison.openai.snapshot_sha256 == comparison.claude.snapshot_sha256
    assert comparison.openai.prompt_sha256 == comparison.claude.prompt_sha256
    assert comparison.snapshot_sha256 == comparison.openai.snapshot_sha256
    assert comparison.prompt_sha256 == comparison.openai.prompt_sha256
    assert canonical_snapshot_json(snapshot) == canonical_before
    assert (
        comparison.effort_comparability_note
        == "Provider effort labels do not imply equivalent internal compute."
    )


def test_compare_all_models_uses_identical_snapshot_prompt_and_schema() -> None:
    snapshot = load_iren_snapshot()
    canonical_before = canonical_snapshot_json(snapshot)
    sol_client = FakeOpenAIClient(make_content("Sol"))
    opus_client = FakeClaudeClient(make_content("Opus"))
    fable_client = FakeClaudeClient(make_content("Fable"))

    comparison = compare_all_models(
        snapshot,
        sol_provider=OpenAIAnalysisProvider(client=sol_client),
        opus_provider=ClaudeAnalysisProvider(
            client=opus_client,
            model=AnalysisModel.CLAUDE_OPUS_5,
        ),
        fable_provider=ClaudeAnalysisProvider(
            client=fable_client,
            model=AnalysisModel.CLAUDE_FABLE_5,
        ),
    )

    assert isinstance(comparison, AllModelComparison)
    sol_prompt = sol_client.responses.calls[0]["input"][1]["content"]  # type: ignore[index]
    opus_prompt = opus_client.messages.calls[0]["messages"][0]["content"]  # type: ignore[index]
    fable_prompt = fable_client.messages.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert sol_prompt == opus_prompt == fable_prompt
    reports = (
        comparison.gpt_5_6_sol,
        comparison.claude_opus_5,
        comparison.claude_fable_5,
    )
    assert {report.snapshot_sha256 for report in reports} == {
        comparison.snapshot_sha256
    }
    assert {report.prompt_sha256 for report in reports} == {
        comparison.prompt_sha256
    }
    assert canonical_snapshot_json(snapshot) == canonical_before
    assert comparison.gpt_5_6_sol.model is AnalysisModel.GPT_5_6_SOL
    assert comparison.claude_opus_5.model is AnalysisModel.CLAUDE_OPUS_5
    assert comparison.claude_fable_5.model is AnalysisModel.CLAUDE_FABLE_5
    assert (
        comparison.effort_comparability_note
        == "Provider effort labels do not imply equivalent internal compute."
    )


def test_comparison_rejects_reports_from_different_snapshots() -> None:
    snapshot = load_iren_snapshot()
    openai = OpenAIAnalysisProvider(
        client=FakeOpenAIClient(make_content("OpenAI"))
    ).analyze(snapshot)
    claude = ClaudeAnalysisProvider(
        client=FakeClaudeClient(make_content("Claude"))
    ).analyze(snapshot)
    mismatched_claude = AnalysisReport.model_validate(
        {
            **claude.model_dump(),
            "snapshot_sha256": "0" * 64,
        }
    )

    with pytest.raises(ValidationError, match="comparison snapshot"):
        AnalysisComparison(
            snapshot_sha256=openai.snapshot_sha256,
            prompt_sha256=openai.prompt_sha256,
            ticker=openai.ticker,
            snapshot_as_of=openai.snapshot_as_of,
            openai=openai,
            claude=mismatched_claude,
        )


@pytest.mark.parametrize(
    "provider",
    [
        OpenAIAnalysisProvider(client=FakeOpenAIClient(None)),
        ClaudeAnalysisProvider(client=FakeClaudeClient(None)),
    ],
)
def test_missing_structured_output_is_wrapped_as_provider_error(provider: object) -> None:
    with pytest.raises(AnalysisProviderError, match="no parsed structured output"):
        provider.analyze(load_iren_snapshot())  # type: ignore[attr-defined]


def test_claude_refusal_is_reported_as_provider_error() -> None:
    provider = ClaudeAnalysisProvider(
        client=FakeClaudeClient(make_content("refusal"), stop_reason="refusal"),
        model=AnalysisModel.CLAUDE_FABLE_5,
    )
    with pytest.raises(AnalysisProviderError, match="refused"):
        provider.analyze(load_iren_snapshot())
