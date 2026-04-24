from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

_MISSING = object()


@dataclass(slots=True)
class UsageBreakdown:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(slots=True)
class PricingBreakdown:
    total_usd: float
    input_usd: float = 0.0
    output_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    calculator: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


PriceCalculator = Callable[[dict[str, float | None], Any], PricingBreakdown | None]


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def extract_usage_breakdown(response: Any) -> UsageBreakdown:
    usage = _get_field(response, "usage", response)
    if usage is None:
        return UsageBreakdown()

    prompt_tokens = _as_int(_get_field(usage, "prompt_tokens", 0))
    completion_tokens = _as_int(_get_field(usage, "completion_tokens", 0))

    raw_cache_hit = _get_field(usage, "prompt_cache_hit_tokens", _MISSING)
    raw_cache_miss = _get_field(usage, "prompt_cache_miss_tokens", _MISSING)
    prompt_details = _get_field(usage, "prompt_tokens_details")
    completion_details = _get_field(usage, "completion_tokens_details")

    cached_prompt_tokens = _as_int(_get_field(prompt_details, "cached_tokens", 0))
    prompt_cache_hit_tokens = (
        _as_int(raw_cache_hit)
        if raw_cache_hit is not _MISSING
        else cached_prompt_tokens
    )
    prompt_cache_miss_tokens = (
        _as_int(raw_cache_miss) if raw_cache_miss is not _MISSING else 0
    )

    if cached_prompt_tokens > 0 and prompt_cache_hit_tokens == 0:
        prompt_cache_hit_tokens = cached_prompt_tokens

    if prompt_tokens > 0:
        accounted_prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens
        if accounted_prompt_tokens < prompt_tokens:
            prompt_cache_miss_tokens += prompt_tokens - accounted_prompt_tokens
        elif accounted_prompt_tokens > prompt_tokens:
            overflow = accounted_prompt_tokens - prompt_tokens
            if prompt_cache_miss_tokens >= overflow:
                prompt_cache_miss_tokens -= overflow
            else:
                prompt_cache_hit_tokens = max(
                    prompt_cache_hit_tokens - (overflow - prompt_cache_miss_tokens),
                    0,
                )
                prompt_cache_miss_tokens = 0

    return UsageBreakdown(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        reasoning_tokens=_as_int(_get_field(completion_details, "reasoning_tokens", 0)),
    )


def flat_input_output_v1(
    prices: dict[str, float | None], response: Any
) -> PricingBreakdown | None:
    input_price = _as_float(prices.get("input_per_million_usd"))
    output_price = _as_float(prices.get("output_per_million_usd"))
    if input_price is None or output_price is None:
        return None

    usage = extract_usage_breakdown(response)
    input_usd = (usage.prompt_tokens / 1_000_000) * input_price
    output_usd = (usage.completion_tokens / 1_000_000) * output_price
    return PricingBreakdown(
        total_usd=input_usd + output_usd,
        input_usd=input_usd,
        output_usd=output_usd,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        calculator="flat_input_output_v1",
        details={
            "input_per_million_usd": input_price,
            "output_per_million_usd": output_price,
        },
    )


def cached_input_output_v1(
    prices: dict[str, float | None], response: Any
) -> PricingBreakdown | None:
    cache_hit_price = _as_float(prices.get("input_cache_hit_per_million_usd"))
    cache_miss_price = _as_float(prices.get("input_cache_miss_per_million_usd"))
    output_price = _as_float(prices.get("output_per_million_usd"))
    if cache_hit_price is None or cache_miss_price is None or output_price is None:
        return None

    usage = extract_usage_breakdown(response)
    input_usd = (usage.prompt_cache_hit_tokens / 1_000_000) * cache_hit_price + (
        usage.prompt_cache_miss_tokens / 1_000_000
    ) * cache_miss_price
    output_usd = (usage.completion_tokens / 1_000_000) * output_price
    return PricingBreakdown(
        total_usd=input_usd + output_usd,
        input_usd=input_usd,
        output_usd=output_usd,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        prompt_cache_hit_tokens=usage.prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=usage.prompt_cache_miss_tokens,
        calculator="cached_input_output_v1",
        details={
            "input_cache_hit_per_million_usd": cache_hit_price,
            "input_cache_miss_per_million_usd": cache_miss_price,
            "output_per_million_usd": output_price,
        },
    )


PRICING_CALCULATORS: dict[str, PriceCalculator] = {
    "flat_input_output_v1": flat_input_output_v1,
    "cached_input_output_v1": cached_input_output_v1,
}


def resolve_pricing_calculator(name: str) -> PriceCalculator:
    try:
        return PRICING_CALCULATORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRICING_CALCULATORS))
        raise ValueError(
            f"Unknown pricing calculator '{name}'. Available: {available}"
        ) from exc
