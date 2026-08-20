"""Simple, explicit model pricing used only for per-run cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from types import MappingProxyType

from .models import AnalysisModel, CostEstimate, TokenUsage

PRICING_AS_OF = date(2026, 8, 21)
_ONE_MILLION = Decimal(1_000_000)
_COST_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: Decimal
    cached_input_usd_per_million: Decimal
    cache_write_input_usd_per_million: Decimal
    output_usd_per_million: Decimal


MODEL_PRICING_USD = MappingProxyType(
    {
        AnalysisModel.GPT_5_6_SOL: ModelPricing(
            input_usd_per_million=Decimal("5"),
            cached_input_usd_per_million=Decimal("0.5"),
            cache_write_input_usd_per_million=Decimal("6.25"),
            output_usd_per_million=Decimal("30"),
        ),
        AnalysisModel.CLAUDE_OPUS_5: ModelPricing(
            input_usd_per_million=Decimal("5"),
            cached_input_usd_per_million=Decimal("0.5"),
            cache_write_input_usd_per_million=Decimal("6.25"),
            output_usd_per_million=Decimal("25"),
        ),
        AnalysisModel.CLAUDE_FABLE_5: ModelPricing(
            input_usd_per_million=Decimal("10"),
            cached_input_usd_per_million=Decimal("1"),
            cache_write_input_usd_per_million=Decimal("12.5"),
            output_usd_per_million=Decimal("50"),
        ),
    }
)


def estimate_analysis_cost(model: AnalysisModel, usage: TokenUsage) -> CostEstimate:
    """Estimate one run at standard API rates, preserving cache categories."""

    pricing = MODEL_PRICING_USD[model]
    uncached_input_tokens = (
        usage.input_tokens
        - usage.cached_input_tokens
        - usage.cache_write_input_tokens
    )
    uncached_input_cost = _cost(
        uncached_input_tokens,
        pricing.input_usd_per_million,
    )
    cached_input_cost = _cost(
        usage.cached_input_tokens,
        pricing.cached_input_usd_per_million,
    )
    cache_write_input_cost = _cost(
        usage.cache_write_input_tokens,
        pricing.cache_write_input_usd_per_million,
    )
    output_cost = _cost(usage.output_tokens, pricing.output_usd_per_million)
    total_cost = (
        uncached_input_cost
        + cached_input_cost
        + cache_write_input_cost
        + output_cost
    ).quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)
    return CostEstimate(
        model=model,
        pricing_as_of=PRICING_AS_OF,
        uncached_input_cost_usd=float(uncached_input_cost),
        cached_input_cost_usd=float(cached_input_cost),
        cache_write_input_cost_usd=float(cache_write_input_cost),
        output_cost_usd=float(output_cost),
        total_cost_usd=float(total_cost),
    )


def _cost(tokens: int, usd_per_million: Decimal) -> Decimal:
    return (
        Decimal(tokens) * usd_per_million / _ONE_MILLION
    ).quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP)
