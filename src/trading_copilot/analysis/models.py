"""Shared structured-output models returned by every AI provider."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnalysisFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ProviderName(str, Enum):
    OPENAI = "openai"
    CLAUDE = "claude"


class AnalysisModel(str, Enum):
    GPT_5_6_SOL = "gpt-5.6-sol"
    CLAUDE_OPUS_5 = "claude-opus-5"
    CLAUDE_FABLE_5 = "claude-fable-5"


class TechnicalBias(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    MIXED = "mixed"


class ScenarioAnalysis(AnalysisFrozenModel):
    bullish: str = Field(
        description=(
            "Conditional bullish scenario using only supplied levels. A single "
            "confirmed pivot must not be described as a pivot sequence."
        )
    )
    neutral: str = Field(description="Conditional neutral scenario using supplied levels.")
    bearish: str = Field(
        description=(
            "Conditional bearish scenario using only supplied levels. A single "
            "confirmed pivot must not be described as a pivot sequence."
        )
    )


class AnalysisContent(AnalysisFrozenModel):
    """Provider-generated content; both providers are constrained to this schema."""

    overall_bias: TechnicalBias
    overall_technical_picture: str
    trend: str
    momentum: str = Field(
        description=(
            "Describe RSI and returns as momentum. Do not interpret RSI as direct "
            "buying pressure, selling pressure, or order flow."
        )
    )
    volume_and_volatility: str = Field(
        description=(
            "Describe observed relative activity and volatility without inferring "
            "trader intent, panic, capitulation, accumulation, or distribution."
        )
    )
    key_price_levels: str
    scenarios: ScenarioAnalysis
    what_to_watch: tuple[str, ...]
    setup_summary: str
    fundamental_risk_context: str = Field(
        description=(
            "At most two short, Telegram-ready sentences: identify report_date and "
            "report_age_days, state any freshness warning, and summarize historical "
            "period-end net cash or net debt without calling it current. Put the "
            "full freshness disclaimer and detailed cash/debt figures in limitations."
        )
    )
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def required_narrative_fields_must_not_be_blank(self) -> Self:
        narrative_fields = (
            "overall_technical_picture",
            "trend",
            "momentum",
            "volume_and_volatility",
            "key_price_levels",
            "setup_summary",
            "fundamental_risk_context",
        )
        for field_name in narrative_fields:
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        for field_name in ("bullish", "neutral", "bearish"):
            if not getattr(self.scenarios, field_name).strip():
                raise ValueError(f"scenarios.{field_name} must not be blank")
        if not self.what_to_watch or any(not item.strip() for item in self.what_to_watch):
            raise ValueError("what_to_watch must contain non-blank items")
        if any(not item.strip() for item in self.limitations):
            raise ValueError("limitations cannot contain blank items")
        return self


class TokenUsage(AnalysisFrozenModel):
    """Normalized token usage while retaining provider cache categories."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    reasoning_tokens: int | None = None

    @model_validator(mode="after")
    def token_counts_must_be_consistent(self) -> Self:
        counts = (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_input_tokens,
            self.cache_write_input_tokens,
        )
        if self.reasoning_tokens is not None:
            counts += (self.reasoning_tokens,)
        if any(count < 0 for count in counts):
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens + self.cache_write_input_tokens > self.input_tokens:
            raise ValueError("cached and cache-write tokens cannot exceed input_tokens")
        if self.total_tokens < self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens cannot be smaller than input plus output tokens")
        return self


class CostEstimate(AnalysisFrozenModel):
    """Auditable USD estimate produced from one versioned pricing configuration."""

    model: AnalysisModel
    currency: Literal["USD"] = "USD"
    pricing_as_of: date
    uncached_input_cost_usd: float
    cached_input_cost_usd: float
    cache_write_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float

    @model_validator(mode="after")
    def costs_cannot_be_negative(self) -> Self:
        costs = (
            self.uncached_input_cost_usd,
            self.cached_input_cost_usd,
            self.cache_write_input_cost_usd,
            self.output_cost_usd,
            self.total_cost_usd,
        )
        if any(cost < 0 for cost in costs):
            raise ValueError("cost estimates cannot be negative")
        return self


class AnalysisRunMetadata(AnalysisFrozenModel):
    provider: ProviderName
    model: AnalysisModel
    effort: Literal["high"] = "high"
    token_usage: TokenUsage | None
    latency_ms: float
    response_id: str | None
    generated_at_utc: datetime
    estimated_cost: CostEstimate | None

    @field_validator("generated_at_utc")
    @classmethod
    def generated_timestamp_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def metadata_must_be_consistent(self) -> Self:
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        expected_provider = (
            ProviderName.OPENAI
            if self.model is AnalysisModel.GPT_5_6_SOL
            else ProviderName.CLAUDE
        )
        if self.provider is not expected_provider:
            raise ValueError("model does not belong to the selected provider")
        if self.estimated_cost is not None:
            if self.token_usage is None:
                raise ValueError("estimated_cost requires token_usage")
            if self.estimated_cost.model is not self.model:
                raise ValueError("estimated_cost must use the run model")
        return self


class AnalysisReport(AnalysisContent):
    """Common application-level report returned by OpenAI and Claude."""

    schema_version: Literal["0.1"] = "0.1"
    provider: ProviderName
    model: AnalysisModel
    effort: Literal["high"] = "high"
    prompt_version: Literal["0.1", "0.2", "0.3", "0.4"] = "0.4"
    prompt_sha256: str
    snapshot_sha256: str
    ticker: str
    snapshot_as_of: date
    snapshot_generated_at_utc: datetime
    run_metadata: AnalysisRunMetadata

    @field_validator("snapshot_generated_at_utc")
    @classmethod
    def snapshot_timestamp_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot_generated_at_utc must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("prompt_sha256", "snapshot_sha256")
    @classmethod
    def digest_must_be_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
            raise ValueError("digest must be a 64-character SHA-256 hex string")
        return normalized

    @model_validator(mode="after")
    def report_and_run_metadata_must_match(self) -> Self:
        if self.run_metadata.provider is not self.provider:
            raise ValueError("run metadata provider must match the report")
        if self.run_metadata.model is not self.model:
            raise ValueError("run metadata model must match the report")
        if self.run_metadata.effort != self.effort:
            raise ValueError("run metadata effort must match the report")
        return self


class AnalysisComparison(AnalysisFrozenModel):
    """Two reports proven to use the same immutable snapshot and prompt."""

    schema_version: Literal["0.1"] = "0.1"
    snapshot_sha256: str
    prompt_sha256: str
    ticker: str
    snapshot_as_of: date
    openai: AnalysisReport
    claude: AnalysisReport
    effort_comparability_note: Literal[
        "Provider effort labels do not imply equivalent internal compute."
    ] = "Provider effort labels do not imply equivalent internal compute."

    @model_validator(mode="after")
    def reports_must_be_a_fair_pair(self) -> Self:
        if self.openai.provider is not ProviderName.OPENAI:
            raise ValueError("openai report must come from the OpenAI provider")
        if self.claude.provider is not ProviderName.CLAUDE:
            raise ValueError("claude report must come from the Claude provider")
        for report in (self.openai, self.claude):
            if report.snapshot_sha256 != self.snapshot_sha256:
                raise ValueError("both reports must use the comparison snapshot")
            if report.prompt_sha256 != self.prompt_sha256:
                raise ValueError("both reports must use the comparison prompt")
            if report.ticker != self.ticker or report.snapshot_as_of != self.snapshot_as_of:
                raise ValueError("both reports must identify the same ticker and as_of")
        return self


class AllModelComparison(AnalysisFrozenModel):
    """Three reports proven to use the same immutable snapshot and prompt."""

    schema_version: Literal["0.1"] = "0.1"
    snapshot_sha256: str
    prompt_sha256: str
    ticker: str
    snapshot_as_of: date
    gpt_5_6_sol: AnalysisReport
    claude_opus_5: AnalysisReport
    claude_fable_5: AnalysisReport
    effort_comparability_note: Literal[
        "Provider effort labels do not imply equivalent internal compute."
    ] = "Provider effort labels do not imply equivalent internal compute."

    @model_validator(mode="after")
    def reports_must_be_a_fair_three_model_comparison(self) -> Self:
        expected = (
            (self.gpt_5_6_sol, ProviderName.OPENAI, AnalysisModel.GPT_5_6_SOL),
            (self.claude_opus_5, ProviderName.CLAUDE, AnalysisModel.CLAUDE_OPUS_5),
            (self.claude_fable_5, ProviderName.CLAUDE, AnalysisModel.CLAUDE_FABLE_5),
        )
        for report, provider, model in expected:
            if report.provider is not provider or report.model is not model:
                raise ValueError(f"comparison report must use {model.value}")
            if report.snapshot_sha256 != self.snapshot_sha256:
                raise ValueError("all reports must use the comparison snapshot")
            if report.prompt_sha256 != self.prompt_sha256:
                raise ValueError("all reports must use the comparison prompt")
            if report.ticker != self.ticker or report.snapshot_as_of != self.snapshot_as_of:
                raise ValueError("all reports must identify the same ticker and as_of")
        return self
