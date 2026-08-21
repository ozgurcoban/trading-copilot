"""Deterministic level selection and scenario language for compact reports."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import TechnicalSnapshot


@dataclass(frozen=True)
class CompactLevel:
    low: float
    high: float
    label: str
    priority: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class CompactLevels:
    decision: tuple[CompactLevel, ...] = ()
    support: tuple[CompactLevel, ...] = ()
    resistance: tuple[CompactLevel, ...] = ()


@dataclass(frozen=True)
class _LevelSource:
    price: float
    key: str
    label: str
    priority: int


_SOURCE_ORDER = {
    "SMA20": 0,
    "SMA50": 1,
    "SMA200": 2,
    "pivot_low": 3,
    "pivot_high": 4,
    "low_20d": 5,
    "high_20d": 6,
    "low_52w": 7,
    "high_52w": 8,
}

_SOURCE_LABELS = {
    "SMA20": "SMA20",
    "SMA50": "SMA50",
    "SMA200": "SMA200",
    "pivot_low": "bekräftad pivotbotten",
    "pivot_high": "bekräftad pivottopp",
    "low_20d": "20d-lägsta",
    "high_20d": "20d-högsta",
    "low_52w": "52v-lägsta",
    "high_52w": "52v-högsta",
}

_SOURCE_NOUNS = {
    "SMA20": "SMA20",
    "SMA50": "SMA50",
    "SMA200": "SMA200",
    "pivot_low": "pivotbotten",
    "pivot_high": "pivottoppen",
    "low_20d": "20-dagarslägsta",
    "high_20d": "20-dagarshögsta",
    "low_52w": "52-veckorslägsta",
    "high_52w": "52-veckorshögsta",
}

_SOURCE_SIGNIFICANCE = {
    "SMA20": "kursens kortsiktiga position mot SMA20",
    "SMA50": "kursens medellånga position mot SMA50",
    "SMA200": "kursens långsiktiga position mot SMA200",
    "pivot_low": "den senaste bekräftade pivotbotten",
    "pivot_high": "den senaste bekräftade pivottoppen",
    "low_20d": "20-dagarslägsta",
    "high_20d": "20-dagarshögsta",
    "low_52w": "52-veckorslägsta",
    "high_52w": "52-veckorshögsta",
}


def build_compact_levels(snapshot: TechnicalSnapshot) -> CompactLevels:
    """Select a small, technically prioritized set of price levels."""

    close = snapshot.current_candle.close
    candidates = [
        *_moving_average_levels(snapshot),
        *_structure_levels(snapshot),
    ]
    candidates = list(_merge_confluent_levels(candidates, snapshot))
    relevance_distance = max(snapshot.volatility.atr_14 * 3, close * 0.12)
    relevant = [
        level
        for level in candidates
        if _level_distance(level, close) <= relevance_distance
    ]
    decision = _rank_levels(
        [level for level in relevant if level.low <= close <= level.high],
        close,
    )[:1]
    support = _rank_levels(
        [level for level in relevant if level.high < close],
        close,
    )[:2]
    resistance = _rank_levels(
        [level for level in relevant if level.low > close],
        close,
    )[:2]

    if not support:
        support = tuple(
            sorted(
                (level for level in candidates if level.high < close),
                key=lambda level: _level_distance(level, close),
            )[:1]
        )
    if not resistance:
        resistance = tuple(
            sorted(
                (level for level in candidates if level.low > close),
                key=lambda level: _level_distance(level, close),
            )[:1]
        )
    return CompactLevels(
        decision=decision,
        support=support,
        resistance=resistance,
    )


def format_compact_level(level: CompactLevel, currency: str) -> str:
    if abs(level.high - level.low) < 0.005:
        value = _format_decimal(level.low)
    else:
        value = f"{_format_decimal(level.low)}–{_format_decimal(level.high)}"
    return f"{value} {currency}".strip()


def build_compact_scenarios(
    levels: CompactLevels,
    currency: str,
) -> tuple[str, str, str]:
    """Describe important levels, why they matter, and what a break changes."""

    primary_support = levels.support[0] if levels.support else None
    primary_resistance = levels.resistance[0] if levels.resistance else None

    if primary_resistance is not None:
        bullish = _scenario_for_level(
            primary_resistance,
            currency,
            role="resistance",
        )
    else:
        bullish = (
            "Ett etablerat brott över nästa bekräftade motstånd skulle placera "
            "kursen över den närmaste tekniska referensnivån."
        )

    if primary_support is not None and primary_resistance is not None:
        support_text = format_compact_level(primary_support, currency)
        resistance_text = format_compact_level(primary_resistance, currency)
        neutral = (
            f"Mellan stödet {support_text} och motståndet {resistance_text} "
            "förblir kursen inom intervallet utan ett bekräftat brott."
        )
    else:
        neutral = (
            "Kursen förblir avvaktande så länge den håller sig inom den "
            "närmaste tekniska beslutszonen."
        )

    if primary_support is not None:
        bearish = _scenario_for_level(
            primary_support,
            currency,
            role="support",
        )
    else:
        bearish = (
            "Ett etablerat brott under nästa bekräftade stöd skulle placera "
            "kursen under den närmaste tekniska referensnivån."
        )
    return bullish, neutral, bearish


def _scenario_for_level(
    level: CompactLevel,
    currency: str,
    *,
    role: str,
) -> str:
    price = format_compact_level(level, currency)
    is_zone = level.low != level.high or len(level.sources) > 1
    reference = "zonen" if is_zone else "nivån"
    direction = "över"
    if role == "support":
        direction = "under"
    significance = _level_significance(level)
    consequence = _break_consequence(level, role=role)
    subject = f"Vid {price}" if len(level.sources) > 1 else price
    return (
        f"{subject} {significance}; ett etablerat "
        f"brott {direction} {reference} skulle {consequence}."
    )


def _level_significance(level: CompactLevel) -> str:
    sources = level.sources
    if len(sources) == 1:
        source = sources[0]
        verb = "visar" if source.startswith("SMA") else "är"
        return f"{verb} {_SOURCE_SIGNIFICANCE[source]}"

    factors = _join_swedish(tuple(_SOURCE_NOUNS[source] for source in sources))
    moving_averages = tuple(
        source for source in ("SMA20", "SMA50", "SMA200") if source in sources
    )
    if len(moving_averages) == len(sources):
        horizon = {
            ("SMA20", "SMA50"): "kort och medellång trend",
            ("SMA50", "SMA200"): "medellång och lång trend",
            ("SMA20", "SMA50", "SMA200"): (
                "trendbilden över flera tidshorisonter"
            ),
        }.get(moving_averages, "flera trendhorisonter")
        return f"sammanfaller {factors}, vilket gör zonen viktig för {horizon}"

    return (
        f"sammanfaller {factors}, vilket gör zonen viktig för "
        f"{_join_swedish(_confluence_impacts(sources))}"
    )


def _confluence_impacts(sources: tuple[str, ...]) -> tuple[str, ...]:
    impacts: list[str] = []
    moving_averages = tuple(
        source for source in ("SMA20", "SMA50", "SMA200") if source in sources
    )
    if len(moving_averages) == 1:
        impacts.append(
            {
                "SMA20": "kort trend",
                "SMA50": "medellång trend",
                "SMA200": "lång trend",
            }[moving_averages[0]]
        )
    elif moving_averages:
        impacts.append("trendpositioner över flera tidshorisonter")

    structural = tuple(source for source in sources if not source.startswith("SMA"))
    if len(structural) > 1:
        count = {2: "två", 3: "tre", 4: "fyra"}.get(
            len(structural),
            "flera",
        )
        impacts.append(f"{count} prisreferenser")
    elif structural:
        impacts.append(
            {
                "pivot_low": "den senaste referensbotten",
                "pivot_high": "den senaste referenstoppen",
                "low_20d": "20-dagarsintervallets golv",
                "high_20d": "20-dagarsintervallets tak",
                "low_52w": "52-veckorsintervallets golv",
                "high_52w": "52-veckorsintervallets tak",
            }[structural[0]]
        )
    return tuple(impacts)


def _break_consequence(level: CompactLevel, *, role: str) -> str:
    sources = set(level.sources)
    if role == "resistance":
        if sources == {"pivot_high"}:
            return "placera kursen över denna referens"
        moving_averages = tuple(
            source for source in ("SMA20", "SMA50", "SMA200") if source in sources
        )
        parts: list[str] = []
        if moving_averages and "pivot_high" not in sources:
            parts.append(_moving_average_consequence(moving_averages, "över"))
        elif moving_averages or "pivot_high" in sources:
            targets = [*moving_averages]
            if "pivot_high" in sources:
                targets.append("pivottoppen")
            parts.append(
                f"placera kursen över {_join_swedish(tuple(targets))}"
            )
        if "high_52w" in sources:
            parts.append("sätta en ny 52-veckorshögsta")
        elif "high_20d" in sources:
            parts.append("sätta en ny 20-dagarshögsta")
        return _join_actions(parts) if parts else "placera kursen över motståndet"

    if sources == {"pivot_low"}:
        return "placera kursen under denna referens"
    moving_averages = tuple(
        source for source in ("SMA20", "SMA50", "SMA200") if source in sources
    )
    parts = []
    if moving_averages and "pivot_low" not in sources:
        parts.append(_moving_average_consequence(moving_averages, "under"))
    elif moving_averages or "pivot_low" in sources:
        targets = [*moving_averages]
        if "pivot_low" in sources:
            targets.append("pivotbotten")
        parts.append(f"placera kursen under {_join_swedish(tuple(targets))}")
    if "low_52w" in sources:
        parts.append("sätta en ny 52-veckorslägsta")
    elif "low_20d" in sources:
        parts.append("sätta en ny 20-dagarslägsta")
    return _join_actions(parts) if parts else "placera kursen under stödet"


def _join_actions(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    first, *rest = parts
    return f"{first} samt {_join_swedish(tuple(rest))}"


def _moving_average_consequence(
    moving_averages: tuple[str, ...],
    direction: str,
) -> str:
    if len(moving_averages) == 1:
        target = moving_averages[0]
    elif len(moving_averages) == 2:
        target = "båda medelvärdena"
    else:
        target = "samtliga tre medelvärden"
    return f"placera kursen {direction} {target}"


def _moving_average_levels(snapshot: TechnicalSnapshot) -> tuple[CompactLevel, ...]:
    moving_averages = (
        ("SMA20", snapshot.trend.sma_20.value, 70),
        ("SMA50", snapshot.trend.sma_50.value, 85),
        ("SMA200", snapshot.trend.sma_200.value, 100),
    )
    zone_width = max(
        snapshot.volatility.atr_14 * 0.8,
        snapshot.current_candle.close * 0.01,
    )
    remaining = list(moving_averages)
    groups: list[list[tuple[str, float, int]]] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        for candidate in tuple(remaining):
            prices = [item[1] for item in (*group, candidate)]
            if max(prices) - min(prices) <= zone_width:
                group.append(candidate)
                remaining.remove(candidate)
        groups.append(group)

    levels: list[CompactLevel] = []
    for group in groups:
        labels = tuple(item[0] for item in group)
        prices = tuple(item[1] for item in group)
        if len(group) > 1:
            levels.append(
                CompactLevel(
                    low=min(prices),
                    high=max(prices),
                    label=f"{'/'.join(labels)}-zon",
                    priority=120 + (5 if "SMA200" in labels else 0),
                    sources=labels,
                )
            )
            continue
        label, price, priority = group[0]
        levels.append(
            CompactLevel(
                low=price,
                high=price,
                label=label,
                priority=priority,
                sources=(label,),
            )
        )
    return tuple(levels)


def _structure_levels(snapshot: TechnicalSnapshot) -> tuple[CompactLevel, ...]:
    structure = snapshot.price_structure
    sources = [
        _LevelSource(structure.low_20d.price, "low_20d", "20d-lägsta", 80),
        _LevelSource(structure.high_20d.price, "high_20d", "20d-högsta", 80),
        _LevelSource(structure.low_52w.price, "low_52w", "52v-lägsta", 90),
        _LevelSource(structure.high_52w.price, "high_52w", "52v-högsta", 90),
    ]
    if structure.recent_confirmed_pivot_low is not None:
        sources.append(
            _LevelSource(
                structure.recent_confirmed_pivot_low.price,
                "pivot_low",
                "bekräftad pivotbotten",
                100,
            )
        )
    if structure.recent_confirmed_pivot_high is not None:
        sources.append(
            _LevelSource(
                structure.recent_confirmed_pivot_high.price,
                "pivot_high",
                "bekräftad pivottopp",
                100,
            )
        )

    tolerance = max(snapshot.current_candle.close * 0.0005, 0.01)
    remaining = sorted(sources, key=lambda source: source.price)
    groups: list[list[_LevelSource]] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        for candidate in tuple(remaining):
            if abs(candidate.price - anchor.price) <= tolerance:
                group.append(candidate)
                remaining.remove(candidate)
        groups.append(group)
    return tuple(_structure_level(group) for group in groups)


def _structure_level(group: list[_LevelSource]) -> CompactLevel:
    ordered = sorted(group, key=lambda source: (-source.priority, source.key))
    price = sum(source.price for source in group) / len(group)
    return CompactLevel(
        low=price,
        high=price,
        label=" + ".join(source.label for source in ordered),
        priority=max(source.priority for source in group) + 15 * (len(group) - 1),
        sources=tuple(source.key for source in ordered),
    )


def _merge_confluent_levels(
    levels: list[CompactLevel],
    snapshot: TechnicalSnapshot,
) -> tuple[CompactLevel, ...]:
    confluence_distance = max(
        snapshot.volatility.atr_14 * 0.35,
        snapshot.current_candle.close * 0.005,
    )
    remaining = sorted(levels, key=lambda level: -level.priority)
    merged: list[CompactLevel] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        for candidate in tuple(remaining):
            if not _is_cross_factor_confluence(group, candidate):
                continue
            if _interval_distance(anchor, candidate) > confluence_distance:
                continue
            group.append(candidate)
            remaining.remove(candidate)
        merged.append(_combine_confluent_levels(group))
    return tuple(merged)


def _is_cross_factor_confluence(
    group: list[CompactLevel],
    candidate: CompactLevel,
) -> bool:
    group_sources = {source for level in group for source in level.sources}
    candidate_sources = set(candidate.sources)
    group_has_ma = any(source.startswith("SMA") for source in group_sources)
    group_has_structure = any(
        not source.startswith("SMA") for source in group_sources
    )
    candidate_has_ma = any(
        source.startswith("SMA") for source in candidate_sources
    )
    candidate_has_structure = any(
        not source.startswith("SMA") for source in candidate_sources
    )
    return (
        (group_has_ma and candidate_has_structure)
        or (group_has_structure and candidate_has_ma)
    )


def _interval_distance(first: CompactLevel, second: CompactLevel) -> float:
    if first.low <= second.high and second.low <= first.high:
        return 0.0
    return min(abs(first.low - second.high), abs(second.low - first.high))


def _combine_confluent_levels(group: list[CompactLevel]) -> CompactLevel:
    if len(group) == 1:
        return group[0]
    sources = _ordered_sources(
        tuple(source for level in group for source in level.sources)
    )
    return CompactLevel(
        low=min(level.low for level in group),
        high=max(level.high for level in group),
        label=_confluence_label(sources),
        priority=max(level.priority for level in group) + 25 * (len(group) - 1),
        sources=sources,
    )


def _ordered_sources(sources: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(sources), key=lambda source: _SOURCE_ORDER[source]))


def _confluence_label(sources: tuple[str, ...]) -> str:
    labels = tuple(_SOURCE_LABELS[source] for source in sources)
    if all(source.startswith("SMA") for source in sources):
        return f"{'/'.join(labels)}-zon"
    return f"{' + '.join(labels)}-zon"


def _rank_levels(
    levels: list[CompactLevel],
    close: float,
) -> tuple[CompactLevel, ...]:
    return tuple(
        sorted(
            levels,
            key=lambda level: (-level.priority, _level_distance(level, close)),
        )
    )


def _level_distance(level: CompactLevel, close: float) -> float:
    if level.low <= close <= level.high:
        return 0.0
    return min(abs(close - level.low), abs(close - level.high))


def _join_swedish(values: tuple[str, ...]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} och {values[1]}"
    return f"{', '.join(values[:-1])} och {values[-1]}"


def _format_decimal(value: float) -> str:
    grouped = f"{value:,.2f}"
    return grouped.replace(",", "\u00a0").replace(".", ",")
