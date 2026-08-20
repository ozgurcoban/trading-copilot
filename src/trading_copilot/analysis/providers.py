"""Two explicit provider adapters behind one deliberately small interface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Protocol

from ..models import TechnicalSnapshot
from .models import (
    AnalysisContent,
    AnalysisModel,
    AnalysisReport,
    AnalysisRunMetadata,
    ProviderName,
    TokenUsage,
)
from .pricing import estimate_analysis_cost
from .prompting import AnalysisInput, build_analysis_input

OPENAI_MODEL = AnalysisModel.GPT_5_6_SOL.value
CLAUDE_MODEL = AnalysisModel.CLAUDE_OPUS_5.value
CLAUDE_FABLE_MODEL = AnalysisModel.CLAUDE_FABLE_5.value
ANALYSIS_EFFORT = "high"
MAX_OUTPUT_TOKENS = 16_000


class AnalysisProviderError(RuntimeError):
    """A provider request failed or did not return a valid AnalysisContent."""


class AnalysisProvider(Protocol):
    name: ProviderName
    model: AnalysisModel
    effort: str

    def analyze(self, snapshot: TechnicalSnapshot) -> AnalysisReport:
        """Analyze one already-built snapshot without fetching or mutating data."""


class OpenAIAnalysisProvider:
    name = ProviderName.OPENAI
    effort = ANALYSIS_EFFORT

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: AnalysisModel | str = AnalysisModel.GPT_5_6_SOL,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] | None = None,
    ) -> None:
        selected_model = AnalysisModel(model)
        if selected_model is not AnalysisModel.GPT_5_6_SOL:
            raise ValueError("OpenAIAnalysisProvider only supports gpt-5.6-sol")
        if client is None:
            from openai import OpenAI

            client = OpenAI(timeout=180.0)
        self.client = client
        self.model = selected_model
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timer = timer or perf_counter

    def analyze(self, snapshot: TechnicalSnapshot) -> AnalysisReport:
        analysis_input = build_analysis_input(snapshot)
        started_at = self._timer()
        try:
            response = self.client.responses.parse(
                model=self.model.value,
                reasoning={"effort": self.effort},
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
                input=[
                    {"role": "system", "content": analysis_input.system_prompt},
                    {"role": "user", "content": analysis_input.user_prompt},
                ],
                text_format=AnalysisContent,
            )
            content = _validated_content(response.output_parsed)
            run_metadata = _build_run_metadata(
                provider=self.name,
                model=self.model,
                usage=_extract_openai_usage(_value(response, "usage")),
                latency_ms=(self._timer() - started_at) * 1_000,
                response_id=_response_id(response),
                generated_at_utc=self._clock(),
            )
            return _build_report(
                snapshot=snapshot,
                analysis_input=analysis_input,
                provider=self.name,
                model=self.model,
                content=content,
                run_metadata=run_metadata,
            )
        except Exception as exc:
            raise AnalysisProviderError(f"OpenAI analysis failed: {exc}") from exc


class ClaudeAnalysisProvider:
    name = ProviderName.CLAUDE
    effort = ANALYSIS_EFFORT

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: AnalysisModel | str = AnalysisModel.CLAUDE_OPUS_5,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] | None = None,
    ) -> None:
        selected_model = AnalysisModel(model)
        if selected_model not in {
            AnalysisModel.CLAUDE_OPUS_5,
            AnalysisModel.CLAUDE_FABLE_5,
        }:
            raise ValueError("ClaudeAnalysisProvider supports Claude Opus 5 and Fable 5")
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(timeout=180.0)
        self.client = client
        self.model = selected_model
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timer = timer or perf_counter

    def analyze(self, snapshot: TechnicalSnapshot) -> AnalysisReport:
        analysis_input = build_analysis_input(snapshot)
        started_at = self._timer()
        try:
            response = self.client.messages.parse(
                model=self.model.value,
                max_tokens=MAX_OUTPUT_TOKENS,
                output_config={"effort": self.effort},
                system=analysis_input.system_prompt,
                messages=[{"role": "user", "content": analysis_input.user_prompt}],
                output_format=AnalysisContent,
            )
            if str(_value(response, "stop_reason", "")) == "refusal":
                raise ValueError("model refused the structured analysis request")
            content = _validated_content(response.parsed_output)
            run_metadata = _build_run_metadata(
                provider=self.name,
                model=self.model,
                usage=_extract_claude_usage(_value(response, "usage")),
                latency_ms=(self._timer() - started_at) * 1_000,
                response_id=_response_id(response),
                generated_at_utc=self._clock(),
            )
            return _build_report(
                snapshot=snapshot,
                analysis_input=analysis_input,
                provider=self.name,
                model=self.model,
                content=content,
                run_metadata=run_metadata,
            )
        except Exception as exc:
            raise AnalysisProviderError(f"Claude analysis failed: {exc}") from exc


def _validated_content(value: Any) -> AnalysisContent:
    if value is None:
        raise ValueError("provider returned no parsed structured output")
    if isinstance(value, AnalysisContent):
        return value
    return AnalysisContent.model_validate(value)


def _build_report(
    *,
    snapshot: TechnicalSnapshot,
    analysis_input: AnalysisInput,
    provider: ProviderName,
    model: AnalysisModel,
    content: AnalysisContent,
    run_metadata: AnalysisRunMetadata,
) -> AnalysisReport:
    return AnalysisReport(
        provider=provider,
        model=model,
        prompt_version=analysis_input.prompt_version,
        prompt_sha256=analysis_input.prompt_sha256,
        snapshot_sha256=analysis_input.snapshot_sha256,
        ticker=snapshot.instrument.ticker,
        snapshot_as_of=snapshot.metadata.requested_as_of,
        snapshot_generated_at_utc=snapshot.metadata.generated_at_utc,
        run_metadata=run_metadata,
        **content.model_dump(),
    )


def _build_run_metadata(
    *,
    provider: ProviderName,
    model: AnalysisModel,
    usage: TokenUsage | None,
    latency_ms: float,
    response_id: str | None,
    generated_at_utc: datetime,
) -> AnalysisRunMetadata:
    return AnalysisRunMetadata(
        provider=provider,
        model=model,
        token_usage=usage,
        latency_ms=latency_ms,
        response_id=response_id,
        generated_at_utc=generated_at_utc,
        estimated_cost=estimate_analysis_cost(model, usage) if usage else None,
    )


def _extract_openai_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    input_tokens = _optional_int(usage, "input_tokens")
    output_tokens = _optional_int(usage, "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    input_details = _value(usage, "input_tokens_details")
    output_details = _value(usage, "output_tokens_details")
    cached_input_tokens = _optional_int(input_details, "cached_tokens") or 0
    cache_write_input_tokens = (
        _optional_int(input_details, "cache_write_tokens")
        or _optional_int(input_details, "cache_creation_tokens")
        or 0
    )
    total_tokens = _optional_int(usage, "total_tokens")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_tokens=_optional_int(output_details, "reasoning_tokens"),
    )


def _extract_claude_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    uncached_input_tokens = _optional_int(usage, "input_tokens")
    output_tokens = _optional_int(usage, "output_tokens")
    if uncached_input_tokens is None or output_tokens is None:
        return None
    cached_input_tokens = _optional_int(usage, "cache_read_input_tokens") or 0
    cache_write_input_tokens = _optional_int(usage, "cache_creation_input_tokens") or 0
    input_tokens = (
        uncached_input_tokens + cached_input_tokens + cache_write_input_tokens
    )
    output_details = _value(usage, "output_tokens_details")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_tokens=_optional_int(output_details, "thinking_tokens"),
    )


def _response_id(response: Any) -> str | None:
    value = _value(response, "id")
    return str(value) if value else None


def _optional_int(value: Any, field_name: str) -> int | None:
    field_value = _value(value, field_name)
    return int(field_value) if field_value is not None else None


def _value(value: Any, field_name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)
