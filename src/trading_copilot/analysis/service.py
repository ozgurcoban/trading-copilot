"""Minimal provider-neutral entrypoints used by future interfaces."""

from __future__ import annotations

from typing import Any

from ..models import TechnicalSnapshot
from .models import (
    AllModelComparison,
    AnalysisComparison,
    AnalysisModel,
    AnalysisReport,
)
from .prompting import build_analysis_input
from .providers import (
    AnalysisProvider,
    ClaudeAnalysisProvider,
    OpenAIAnalysisProvider,
)


def provider_for_model(
    model: AnalysisModel | str,
    *,
    client: Any | None = None,
) -> AnalysisProvider:
    """Map one supported model to one of the two explicit provider adapters."""

    selected_model = AnalysisModel(model)
    if selected_model is AnalysisModel.GPT_5_6_SOL:
        return OpenAIAnalysisProvider(client=client, model=selected_model)
    return ClaudeAnalysisProvider(client=client, model=selected_model)


def analyze_snapshot(
    snapshot: TechnicalSnapshot,
    provider: AnalysisProvider,
) -> AnalysisReport:
    """Send an existing snapshot to exactly one explicitly selected provider."""

    return provider.analyze(snapshot)


def analyze_with_model(
    snapshot: TechnicalSnapshot,
    model: AnalysisModel | str,
    *,
    client: Any | None = None,
) -> AnalysisReport:
    """Analyze a snapshot with one explicitly selected supported model."""

    return provider_for_model(model, client=client).analyze(snapshot)


def compare_openai_and_claude(
    snapshot: TechnicalSnapshot,
    *,
    openai_provider: AnalysisProvider | None = None,
    claude_provider: AnalysisProvider | None = None,
) -> AnalysisComparison:
    """Analyze the exact same snapshot and prompt once with each provider."""

    analysis_input = build_analysis_input(snapshot)
    openai_report = (openai_provider or OpenAIAnalysisProvider()).analyze(snapshot)
    claude_report = (claude_provider or ClaudeAnalysisProvider()).analyze(snapshot)
    return AnalysisComparison(
        snapshot_sha256=analysis_input.snapshot_sha256,
        prompt_sha256=analysis_input.prompt_sha256,
        ticker=snapshot.instrument.ticker,
        snapshot_as_of=snapshot.metadata.requested_as_of,
        openai=openai_report,
        claude=claude_report,
    )


def compare_all_models(
    snapshot: TechnicalSnapshot,
    *,
    sol_provider: AnalysisProvider | None = None,
    opus_provider: AnalysisProvider | None = None,
    fable_provider: AnalysisProvider | None = None,
) -> AllModelComparison:
    """Analyze one immutable snapshot with Sol, Opus, and Fable."""

    analysis_input = build_analysis_input(snapshot)
    sol_report = (
        sol_provider
        or OpenAIAnalysisProvider(model=AnalysisModel.GPT_5_6_SOL)
    ).analyze(snapshot)
    opus_report = (
        opus_provider
        or ClaudeAnalysisProvider(model=AnalysisModel.CLAUDE_OPUS_5)
    ).analyze(snapshot)
    fable_report = (
        fable_provider
        or ClaudeAnalysisProvider(model=AnalysisModel.CLAUDE_FABLE_5)
    ).analyze(snapshot)
    return AllModelComparison(
        snapshot_sha256=analysis_input.snapshot_sha256,
        prompt_sha256=analysis_input.prompt_sha256,
        ticker=snapshot.instrument.ticker,
        snapshot_as_of=snapshot.metadata.requested_as_of,
        gpt_5_6_sol=sol_report,
        claude_opus_5=opus_report,
        claude_fable_5=fable_report,
    )
