"""Deterministic OpenRouter token pricing; no model is called to calculate cost."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

PRICING_VERSION = "2026-11-23"
USD_PER_MILLION_TOKENS = Decimal("1000000")


@dataclass(frozen=True)
class PricingOverride:
    min_prompt_tokens: int
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    overrides: tuple[PricingOverride, ...] = ()


@dataclass(frozen=True)
class CostCalculation:
    cost_usd: Decimal | None
    input_usd_per_million: Decimal | None
    output_usd_per_million: Decimal | None
    method: str


MODEL_PRICING = {
    "nvidia/nemotron-3.5-lightning:free": ModelPricing(
        input_usd_per_million=Decimal("0"),
        output_usd_per_million=Decimal("0"),
    ),
    "liquid/lfm-3-1.7b:free": ModelPricing(
        input_usd_per_million=Decimal("0"),
        output_usd_per_million=Decimal("0"),
    ),
    "baidu/ernie-5.5-flash-preview:free": ModelPricing(
        input_usd_per_million=Decimal("0"),
        output_usd_per_million=Decimal("0"),
    ),
    "google/gemma-4-31b-it:free": ModelPricing(
        input_usd_per_million=Decimal("0"),
        output_usd_per_million=Decimal("0"),
    ),
    "qwen/qwen3-embedding-8b": ModelPricing(
        input_usd_per_million=Decimal("0.01"),
        output_usd_per_million=Decimal("0"),
    ),
    "qwen/qwen3.7-flash": ModelPricing(
        input_usd_per_million=Decimal("0.03"),
        output_usd_per_million=Decimal("0.13"),
        overrides=(
            PricingOverride(32000, Decimal("0.10"), Decimal("0.40")),
            PricingOverride(256000, Decimal("0.20"), Decimal("0.80")),
        ),
    ),
}


def calculate_cost_usd(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> CostCalculation:
    """Calculate cost from the local rate card and provider-reported token counts."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return CostCalculation(None, None, None, "unpriced_model")

    applicable_override = None
    if prompt_tokens is not None:
        applicable_override = next(
            (
                override
                for override in reversed(pricing.overrides)
                if prompt_tokens >= override.min_prompt_tokens
            ),
            None,
        )
    input_rate = (
        applicable_override.input_usd_per_million
        if applicable_override
        else pricing.input_usd_per_million
    )
    output_rate = (
        applicable_override.output_usd_per_million
        if applicable_override
        else pricing.output_usd_per_million
    )
    input_tokens = prompt_tokens
    output_tokens = completion_tokens
    if input_tokens is None and input_rate == 0:
        input_tokens = 0
    if output_tokens is None and output_rate == 0:
        output_tokens = 0
    if input_tokens is None or output_tokens is None:
        return CostCalculation(
            None,
            input_rate,
            output_rate,
            "missing_token_usage",
        )

    cost = (
        Decimal(input_tokens) * input_rate
        + Decimal(output_tokens) * output_rate
    ) / USD_PER_MILLION_TOKENS
    method = f"local_rate_card:{PRICING_VERSION}"
    if applicable_override:
        method += f":prompt_tokens>={applicable_override.min_prompt_tokens}"
    return CostCalculation(
        cost,
        input_rate,
        output_rate,
        method,
    )
