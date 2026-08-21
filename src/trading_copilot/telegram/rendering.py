"""Deterministic compact and full Telegram rendering for AnalysisReport."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from ..analysis.models import (
    AnalysisModel,
    AnalysisReport,
    ProviderName,
    TechnicalBias,
)
from ..models import MovingAverageDirection, TechnicalSnapshot
from .compact_levels import (
    CompactLevel,
    CompactLevels,
    build_compact_levels,
    build_compact_scenarios,
    format_compact_level,
)

TELEGRAM_MESSAGE_LIMIT = 4_096
TELEGRAM_SAFE_MESSAGE_LENGTH = 3_900
TELEGRAM_COMPACT_MAX_LENGTH = 1_800

_MODEL_HEADER_LABELS = {
    AnalysisModel.GPT_5_6_SOL: "GPT-5.6 Sol",
    AnalysisModel.CLAUDE_OPUS_5: "Claude Opus 5",
    AnalysisModel.CLAUDE_FABLE_5: "Claude Fable 5",
}
_MODEL_FOOTER_LABELS = {
    AnalysisModel.GPT_5_6_SOL: "GPT-5.6 Sol",
    AnalysisModel.CLAUDE_OPUS_5: "Opus 5",
    AnalysisModel.CLAUDE_FABLE_5: "Fable 5",
}
_PROVIDER_LABELS = {
    ProviderName.OPENAI: "OpenAI",
    ProviderName.CLAUDE: "Anthropic",
}
_FULL_BIAS_LABELS = {
    TechnicalBias.BULLISH: "Bullish",
    TechnicalBias.NEUTRAL: "Neutral",
    TechnicalBias.BEARISH: "Bearish",
    TechnicalBias.MIXED: "Blandad",
}
_DIRECTION_LABELS = {
    MovingAverageDirection.RISING: "↗",
    MovingAverageDirection.FLAT: "→",
    MovingAverageDirection.FALLING: "↘",
}


@dataclass(frozen=True)
class _RenderBlock:
    prefix_html: str
    body: str = ""


def render_compact_analysis_report(
    report: AnalysisReport,
    snapshot: TechnicalSnapshot,
    *,
    comparison_position: tuple[int, int] | None = None,
    max_message_length: int = TELEGRAM_COMPACT_MAX_LENGTH,
) -> str:
    """Render one deterministic decision overview from an existing report."""

    _validate_report_snapshot(report, snapshot)
    _validate_message_length(max_message_length)

    model_label = _MODEL_HEADER_LABELS[report.model]
    provider_label = _PROVIDER_LABELS[report.provider]
    technical_state = _compact_technical_state(snapshot)
    close = snapshot.current_candle.close
    currency = snapshot.instrument.market_currency or ""
    levels = build_compact_levels(snapshot)
    bullish_scenario, neutral_scenario, bearish_scenario = build_compact_scenarios(
        levels,
        currency,
    )

    sections: list[str] = []
    if comparison_position is not None:
        position, total = _validated_comparison_position(comparison_position)
        sections.append(f"🔎 <b>Jämförelse {position}/{total}</b>")
    sections.extend(
        [
            (
                f"<b>{escape(report.ticker)} · {escape(provider_label)}</b>\n"
                f"<code>{escape(_format_price(close, currency))}</code> · "
                f"Teknisk bild: <b>{escape(technical_state)}</b>"
            ),
            (
                "<b>Kort bild</b>\n"
                f"{escape(_compact_technical_picture(snapshot), quote=False)}"
            ),
            (
                "<b>Trend</b>\n"
                f"{_moving_average_html('SMA20', snapshot.trend.sma_20.value, snapshot.trend.sma_20.direction, currency)}\n"
                f"{_moving_average_html('SMA50', snapshot.trend.sma_50.value, snapshot.trend.sma_50.direction, currency)}\n"
                f"{_moving_average_html('SMA200', snapshot.trend.sma_200.value, snapshot.trend.sma_200.direction, currency)}"
            ),
            (
                "<b>Momentum</b>\n"
                f"{escape(_compact_momentum(snapshot), quote=False)}"
            ),
            (
                "<b>Nivåer</b>\n"
                f"{_compact_levels_html(levels, currency)}"
            ),
            (
                "<b>Scenario</b>\n"
                f"🟢 <b>Bull</b>\n{escape(bullish_scenario, quote=False)}\n\n"
                f"⚪ <b>Neutral</b>\n{escape(neutral_scenario, quote=False)}\n\n"
                f"🔴 <b>Bear</b>\n{escape(bearish_scenario, quote=False)}"
            ),
            (
                "<b>Bevaka</b>\n"
                + "\n".join(
                    f"• {escape(item, quote=False)}"
                    for item in _compact_watch_items(snapshot, levels)
                )
            ),
            (
                "<b>Risk</b>\n"
                f"{escape(_compact_risk(snapshot), quote=False)}"
            ),
            _compact_footer_html(report),
        ]
    )
    rendered = "\n\n".join(sections)
    if len(rendered) > max_message_length:
        raise ValueError("compact Telegram rendering exceeds the message limit")
    return rendered


def render_full_analysis_report(
    report: AnalysisReport,
    snapshot: TechnicalSnapshot,
    *,
    comparison_position: tuple[int, int] | None = None,
    max_message_length: int = TELEGRAM_SAFE_MESSAGE_LENGTH,
) -> tuple[str, ...]:
    """Render every report field and split only at deterministic boundaries."""

    _validate_report_snapshot(report, snapshot)
    _validate_message_length(max_message_length)

    model_label = _MODEL_HEADER_LABELS[report.model]
    provider_label = _PROVIDER_LABELS[report.provider]
    footer_label = _MODEL_FOOTER_LABELS[report.model]
    currency = snapshot.instrument.market_currency or ""
    close = _format_decimal(snapshot.current_candle.close, 2)
    close_label = f"{close} {currency}".strip()
    bias = _FULL_BIAS_LABELS[report.overall_bias]

    blocks: list[_RenderBlock] = []
    if comparison_position is not None:
        position, total = _validated_comparison_position(comparison_position)
        blocks.append(
            _RenderBlock(
                f"🔎 <b>Jämförelse {position}/{total}</b>"
            )
        )
    blocks.extend(
        [
            _RenderBlock(
                (
                    f"<b>{escape(report.ticker)} · {escape(provider_label)} · "
                    f"{escape(model_label)}</b>\n"
                    f"Stängning: <code>{escape(close_label)}</code> · "
                    f"Bias: <b>{escape(bias)}</b>"
                )
            ),
            _RenderBlock("<b>Kort bild</b>", report.setup_summary),
            _RenderBlock("<b>Teknisk helhetsbild</b>", report.overall_technical_picture),
            _RenderBlock("<b>Trend</b>", report.trend),
            _RenderBlock("<b>Momentum</b>", report.momentum),
            _RenderBlock("<b>Volym &amp; volatilitet</b>", report.volume_and_volatility),
            _RenderBlock("<b>Prisnivåer</b>", report.key_price_levels),
            _RenderBlock(
                "<b>Scenarier</b>",
                (
                    f"🟢 Bullish: {report.scenarios.bullish}\n\n"
                    f"⚪ Neutral: {report.scenarios.neutral}\n\n"
                    f"🔴 Bearish: {report.scenarios.bearish}"
                ),
            ),
            _RenderBlock(
                "<b>Viktigast att bevaka</b>",
                "\n".join(f"• {item}" for item in report.what_to_watch),
            ),
            _RenderBlock("<b>Fundamental risk</b>", report.fundamental_risk_context),
            _RenderBlock(
                "<b>Begränsningar</b>",
                "\n".join(f"• {item}" for item in report.limitations),
            ),
            _RenderBlock(
                _footer_html(
                    footer_label=footer_label,
                    effort=report.effort,
                    latency_ms=report.run_metadata.latency_ms,
                    total_cost_usd=(
                        report.run_metadata.estimated_cost.total_cost_usd
                        if report.run_metadata.estimated_cost is not None
                        else None
                    ),
                )
            ),
        ]
    )
    return _split_blocks(blocks, max_message_length=max_message_length)


def render_analysis_report(
    report: AnalysisReport,
    snapshot: TechnicalSnapshot,
    *,
    comparison_position: tuple[int, int] | None = None,
    max_message_length: int = TELEGRAM_SAFE_MESSAGE_LENGTH,
) -> tuple[str, ...]:
    """Backward-compatible alias for the full report renderer."""

    return render_full_analysis_report(
        report,
        snapshot,
        comparison_position=comparison_position,
        max_message_length=max_message_length,
    )


def _validate_report_snapshot(
    report: AnalysisReport,
    snapshot: TechnicalSnapshot,
) -> None:
    if report.ticker != snapshot.instrument.ticker:
        raise ValueError("report ticker does not match snapshot")
    if report.snapshot_as_of != snapshot.metadata.requested_as_of:
        raise ValueError("report as_of does not match snapshot")


def _validate_message_length(max_message_length: int) -> None:
    if not 256 <= max_message_length <= TELEGRAM_MESSAGE_LIMIT:
        raise ValueError("max_message_length must be between 256 and 4096")


def _validated_comparison_position(
    comparison_position: tuple[int, int],
) -> tuple[int, int]:
    position, total = comparison_position
    if position < 1 or total < 1 or position > total:
        raise ValueError("invalid comparison_position")
    return position, total


def _moving_average_html(
    label: str,
    value: float,
    direction: MovingAverageDirection,
    currency: str,
) -> str:
    return (
        f"{escape(label)}: <code>{escape(_format_price(value, currency))}</code> · "
        f"{_DIRECTION_LABELS[direction]}"
    )


def _compact_levels_html(levels: CompactLevels, currency: str) -> str:
    lines: list[str] = []
    if levels.decision:
        lines.append(_compact_level_html("Nyckelzon", levels.decision[0], currency))
    for index, level in enumerate(levels.support):
        prefix = "Primärt stöd" if index == 0 else "Sekundärt stöd"
        lines.append(_compact_level_html(prefix, level, currency))
    for index, level in enumerate(levels.resistance):
        prefix = "Primärt motstånd" if index == 0 else "Sekundärt motstånd"
        lines.append(_compact_level_html(prefix, level, currency))
    return "\n".join(lines) if lines else "Tekniskt relevanta nivåer saknas."


def _compact_level_html(
    prefix: str,
    level: CompactLevel,
    currency: str,
) -> str:
    price = escape(format_compact_level(level, currency))
    label = escape(level.label)
    return f"{escape(prefix)}: <code>{price}</code> · {label}"


def _format_price(value: float, currency: str) -> str:
    return f"{_format_compact_decimal(value, 2)} {currency}".strip()


def _format_signed_pct(value: float) -> str:
    return f"{value:+.1f} %".replace(".", ",")


def _format_compact_decimal(value: float, decimals: int) -> str:
    grouped = f"{value:,.{decimals}f}"
    return grouped.replace(",", "\u00a0").replace(".", ",")


def _compact_momentum(snapshot: TechnicalSnapshot) -> str:
    momentum = snapshot.momentum
    values = (
        f"RSI: {_format_decimal(momentum.rsi_14, 1)} · "
        f"5d: {_format_signed_pct(momentum.return_5d_pct)} · "
        f"20d: {_format_signed_pct(momentum.return_20d_pct)}"
    )
    short = momentum.return_5d_pct
    medium = momentum.return_20d_pct
    rsi = momentum.rsi_14
    threshold = 1.0
    if short >= threshold and medium <= -threshold:
        if rsi >= 55:
            interpretation = (
                "Femdagarsuppgången är en kortsiktig återhämtning efter "
                "20-dagarsnedgången; RSI över mitten stödjer förbättringen men "
                "bekräftar ännu inget bredare skifte."
            )
        elif rsi <= 45:
            interpretation = (
                "Femdagarsuppgången är en kortsiktig återhämtning efter "
                "20-dagarsnedgången, men RSI under mitten ger ännu ingen "
                "momentumkräftelse."
            )
        else:
            interpretation = (
                "Femdagarsuppgången är en kortsiktig återhämtning efter "
                "20-dagarsnedgången, medan RSI nära mitten ännu inte bekräftar "
                "ett bredare momentumskifte."
            )
    elif short <= -threshold and medium >= threshold:
        if rsi <= 45:
            interpretation = (
                "Femdagarsnedgången inom en positiv 20-dagarsrörelse visar en "
                "rekyl, och RSI under mitten bekräftar den kortsiktiga "
                "momentumsvagheten."
            )
        elif rsi >= 55:
            interpretation = (
                "Femdagarsnedgången inom en positiv 20-dagarsrörelse visar en "
                "rekyl, men RSI över mitten håller det bredare momentumet "
                "positivt."
            )
        else:
            interpretation = (
                "Femdagarsnedgången är en rekyl inom den positiva "
                "20-dagarsrörelsen; RSI nära mitten saknar tydligt "
                "momentumövertag."
            )
    elif short >= threshold and medium >= threshold:
        if rsi >= 70:
            interpretation = (
                "Positiv avkastning över 5 och 20 dagar samt RSI över 70 visar "
                "starkt uppåtriktat momentum, men också en utsträckt rörelse."
            )
        elif rsi >= 55:
            interpretation = (
                "Positiv avkastning över 5 och 20 dagar tillsammans med RSI "
                "över mitten bekräftar ett brett uppåtriktat momentum."
            )
        elif rsi <= 45:
            interpretation = (
                "Avkastningen är positiv över 5 och 20 dagar, men RSI under "
                "mitten visar att momentumet ännu inte bekräftar uppgången."
            )
        else:
            interpretation = (
                "Avkastningen är positiv över 5 och 20 dagar, men RSI nära "
                "mitten visar att momentumövertaget ännu är begränsat."
            )
    elif short <= -threshold and medium <= -threshold:
        if rsi <= 30:
            interpretation = (
                "Negativ avkastning över 5 och 20 dagar samt RSI under 30 visar "
                "starkt nedåtriktat momentum, men också en utsträckt rörelse."
            )
        elif rsi <= 45:
            interpretation = (
                "Negativ avkastning över 5 och 20 dagar tillsammans med RSI "
                "under mitten bekräftar ett brett nedåtriktat momentum."
            )
        elif rsi >= 55:
            interpretation = (
                "Avkastningen är negativ över 5 och 20 dagar, men RSI över "
                "mitten gör det nedåtriktade momentumet mindre entydigt."
            )
        else:
            interpretation = (
                "Avkastningen är negativ över 5 och 20 dagar, men RSI nära "
                "mitten visar att momentumövertaget ännu är begränsat."
            )
    elif medium >= threshold:
        interpretation = (
            "Den korta rörelsen har planat ut; den positiva "
            f"20-dagarsavkastningen är bredare drivkraft, medan {_rsi_context(rsi)}."
        )
    elif medium <= -threshold:
        interpretation = (
            "Den korta rörelsen har planat ut; den negativa "
            f"20-dagarsavkastningen är bredare drivkraft, medan {_rsi_context(rsi)}."
        )
    elif short >= threshold:
        interpretation = (
            "Femdagarsuppgången visar en kortsiktig förbättring, medan "
            f"20-dagarsrörelsen saknar riktning och {_rsi_context(rsi)}."
        )
    elif short <= -threshold:
        interpretation = (
            "Femdagarsnedgången visar en kortsiktig försvagning, medan "
            f"20-dagarsrörelsen saknar riktning och {_rsi_context(rsi)}."
        )
    else:
        interpretation = (
            "Avkastningen är begränsad över 5 och 20 dagar, och "
            f"{_rsi_context(rsi)}."
        )
    return f"{values}\n{interpretation}"


def _rsi_context(rsi: float) -> str:
    if rsi >= 70:
        return "RSI visar ett högt och utsträckt momentumläge"
    if rsi >= 55:
        return "RSI visar ett måttligt positivt momentumövertag"
    if rsi > 45:
        return "RSI nära mitten saknar tydligt momentumövertag"
    if rsi > 30:
        return "RSI visar ett måttligt negativt momentumövertag"
    return "RSI visar ett lågt och utsträckt momentumläge"


def _compact_technical_picture(snapshot: TechnicalSnapshot) -> str:
    close = snapshot.current_candle.close
    position_20 = _moving_average_position(
        close,
        snapshot.trend.sma_20.value,
        snapshot.volatility.atr_14,
    )
    position_50 = _moving_average_position(
        close,
        snapshot.trend.sma_50.value,
        snapshot.volatility.atr_14,
    )
    position_200 = _moving_average_position(
        close,
        snapshot.trend.sma_200.value,
        snapshot.volatility.atr_14,
    )
    above_sma20 = position_20 > 0
    above_sma50 = position_50 > 0
    above_sma200 = position_200 > 0
    below_sma20 = position_20 < 0
    below_sma50 = position_50 < 0
    below_sma200 = position_200 < 0
    sma20_rising = snapshot.trend.sma_20.direction is MovingAverageDirection.RISING
    sma50_rising = snapshot.trend.sma_50.direction is MovingAverageDirection.RISING
    sma200_rising = snapshot.trend.sma_200.direction is MovingAverageDirection.RISING
    sma20_falling = snapshot.trend.sma_20.direction is MovingAverageDirection.FALLING
    sma50_falling = snapshot.trend.sma_50.direction is MovingAverageDirection.FALLING
    sma200_falling = snapshot.trend.sma_200.direction is MovingAverageDirection.FALLING
    short_return = snapshot.momentum.return_5d_pct
    medium_return = snapshot.momentum.return_20d_pct

    if (
        above_sma20
        and above_sma50
        and above_sma200
        and sma20_rising
        and sma50_rising
        and sma200_rising
    ):
        return (
            "En samstämmig upptrend dominerar över kort, medellång och lång "
            "horisont, eftersom kursen ligger över tre stigande medelvärden."
        )
    if (
        below_sma20
        and below_sma50
        and below_sma200
        and sma20_falling
        and sma50_falling
        and sma200_falling
    ):
        return (
            "Teknisk svaghet dominerar över kort, medellång och lång horisont, "
            "eftersom kursen ligger tydligt under tre fallande medelvärden."
        )

    if above_sma20 and above_sma50:
        movement = (
            "En kortsiktig återhämtning dominerar efter en större "
            "20-dagarsnedgång"
            if short_return > 1 and medium_return < -1
            else "En kortsiktig uppgång dominerar över SMA20 och SMA50"
        )
        if above_sma200 and sma200_rising:
            return (
                f"{movement}. Den stöds även av en stigande SMA200, så flera "
                "tidshorisonter pekar nu åt samma håll."
            )
        if above_sma200:
            return (
                f"{movement}. Kursen ligger även över SMA200, men det "
                "långsiktiga medelvärdet stiger ännu inte."
            )
        if position_200 == 0:
            return (
                f"{movement}. Kursen testar samtidigt SMA200, så ett "
                "långsiktigt trendskifte är ännu inte bekräftat."
            )
        return (
            f"{movement}. Kursen är fortfarande under SMA200, vilket "
            "begränsar den långsiktiga bilden och skapar en konflikt mellan "
            "kort och lång trend."
        )

    if below_sma20 and below_sma50:
        movement = (
            "En kortsiktig rekyl dominerar efter nedgång under SMA20 och SMA50"
            if short_return < 0
            else "Kortsiktig svaghet dominerar under SMA20 och SMA50"
        )
        if above_sma200 and sma200_rising:
            return (
                f"{movement}. Den långsiktiga upptrenden är fortfarande intakt "
                "över en stigande SMA200, vilket skapar en tydlig konflikt "
                "mellan tidshorisonterna."
            )
        if above_sma200:
            return (
                f"{movement}. Kursen håller sig över SMA200, men den "
                "långsiktiga riktningen är ännu inte stigande."
            )
        if position_200 == 0:
            return (
                f"{movement}. SMA200 testas samtidigt, så den långsiktiga "
                "trendriktningen är ännu inte avgjord."
            )
        return (
            f"{movement}. Svagheten under SMA200 gör att den längre negativa "
            "strukturen fortsatt dominerar."
        )

    position = _short_moving_average_position(position_20, position_50)
    if short_return < -1 and medium_return > 1:
        movement = (
            "En kortsiktig rekyl dominerar inom 20-dagarsuppgången"
        )
    elif short_return > 1 and medium_return < -1:
        movement = (
            "En kortsiktig återhämtning dominerar efter en större "
            "20-dagarsnedgång"
        )
    elif short_return >= 0:
        movement = "En kortsiktig återhämtning försöker ta form"
    else:
        movement = "En kortsiktig rekyl dominerar"

    if above_sma200 and sma200_rising:
        return (
            f"{movement}; {position}. Den stigande SMA200 gör att den "
            "långsiktiga upptrendstrukturen fortfarande är intakt."
        )
    if above_sma200:
        return (
            f"{movement}; {position}. Kursen är över SMA200, men utan en "
            "stigande långtrend saknas samstämmighet mellan tidshorisonterna."
        )
    if position_200 == 0:
        return (
            f"{movement}; {position}. Kursen testar även SMA200, så den "
            "långsiktiga trendriktningen är ännu inte bekräftad."
        )
    if sma200_falling:
        return (
            f"{movement}; {position}. Kursen är fortfarande under en fallande "
            "SMA200, vilket begränsar den långsiktiga bilden."
        )
    return (
        f"{movement}; {position}. Läget under SMA200 gör att den långsiktiga "
        "bilden fortsatt är svagare."
    )


def _compact_technical_state(snapshot: TechnicalSnapshot) -> str:
    close = snapshot.current_candle.close
    position_20 = _moving_average_position(
        close,
        snapshot.trend.sma_20.value,
        snapshot.volatility.atr_14,
    )
    position_50 = _moving_average_position(
        close,
        snapshot.trend.sma_50.value,
        snapshot.volatility.atr_14,
    )
    position_200 = _moving_average_position(
        close,
        snapshot.trend.sma_200.value,
        snapshot.volatility.atr_14,
    )
    above_sma20 = position_20 > 0
    above_sma50 = position_50 > 0
    above_sma200 = position_200 > 0
    below_sma20 = position_20 < 0
    below_sma50 = position_50 < 0
    sma200_rising = (
        snapshot.trend.sma_200.direction is MovingAverageDirection.RISING
    )

    if above_sma20 and above_sma50:
        if above_sma200 and sma200_rising:
            return "Styrka över flera tidshorisonter"
        if above_sma200:
            return "Kortsiktig styrka i osäker långtrend"
        if position_200 == 0:
            return "Kortsiktig styrka vid SMA200-test"
        return "Kortsiktig återhämtning i svagare trend"

    if below_sma20 and below_sma50:
        if above_sma200 and sma200_rising:
            return "Kortsiktig rekyl i stigande långtrend"
        if above_sma200:
            return "Kortsiktig svaghet över långsiktigt stöd"
        if position_200 == 0:
            return "Kortsiktig svaghet vid SMA200-test"
        return "Svaghet över flera tidshorisonter"

    if position_20 == 0 and position_50 > 0:
        if above_sma200 and sma200_rising:
            return "Test av SMA20 i stigande långtrend"
        return "Marginellt test av kort trendstöd"
    if position_20 == 0 and position_50 < 0:
        return "Test av SMA20 under medellångt motstånd"
    if position_50 == 0:
        return "Test av SMA50 på medellång horisont"
    if above_sma200 and sma200_rising:
        return "Konsolidering i stigande långtrend"
    if above_sma200:
        return "Kortsiktig konsolidering över långsiktigt stöd"
    if (
        snapshot.momentum.return_5d_pct <= -1
        and snapshot.momentum.return_20d_pct >= 1
    ):
        return "Kortsiktig rekyl i svagare långtrend"
    if (
        snapshot.momentum.return_5d_pct >= 1
        and snapshot.momentum.return_20d_pct <= -1
    ):
        return "Kortsiktig återhämtning i svagare trend"
    return "Konsolidering i svagare trend"


def _moving_average_position(close: float, value: float, atr: float) -> int:
    tolerance = max(atr * 0.15, close * 0.003)
    if close > value + tolerance:
        return 1
    if close < value - tolerance:
        return -1
    return 0


def _short_moving_average_position(position_20: int, position_50: int) -> str:
    positions = {
        (1, -1): "kursen håller SMA20 men ligger under SMA50",
        (-1, 1): "kursen ligger under SMA20 men håller sig över SMA50",
        (0, 1): "kursen testar SMA20 men håller sig över SMA50",
        (0, -1): "kursen testar SMA20 och ligger under SMA50",
        (1, 0): "kursen håller SMA20 och testar SMA50",
        (-1, 0): "kursen ligger under SMA20 och testar SMA50",
        (0, 0): "kursen handlas i en gemensam SMA20/SMA50-zon",
    }
    return positions[(position_20, position_50)]


def _compact_footer_html(report: AnalysisReport) -> str:
    model_label = _MODEL_FOOTER_LABELS[report.model]
    latency_seconds = round(report.run_metadata.latency_ms / 1_000.0)
    cost_value = (
        report.run_metadata.estimated_cost.total_cost_usd
        if report.run_metadata.estimated_cost is not None
        else None
    )
    cost = f"${cost_value:.3f}" if cost_value is not None else "kostnad saknas"
    return f"<i>{escape(model_label)} · {latency_seconds}s · {cost}</i>"


def _compact_risk(snapshot: TechnicalSnapshot) -> str:
    context = snapshot.fundamental_risk_context
    atr = _format_decimal(snapshot.volatility.atr_14_pct, 1)
    if snapshot.volatility.atr_14_pct >= 5:
        risk = (
            f"ATR {atr} % innebär stora dagsrörelser; bekräfta nivåbrott med "
            "stängning och uppföljning."
        )
    elif snapshot.volatility.atr_14_pct >= 2:
        risk = (
            f"ATR {atr} % innebär märkbara dagsrörelser; bekräfta nivåbrott "
            "med stängning."
        )
    else:
        risk = (
            f"ATR {atr} % innebär begränsade dagsrörelser; även mindre "
            "nivåbrott kan därför vara tekniskt relevanta."
        )
    if any(
        warning.code == "fundamental_report_period_stale"
        for warning in context.warnings
    ):
        return f"{risk} Fundamentaldata är äldre än 120 dagar."
    if context.warnings:
        return f"{risk} Fundamentaldata har en datavarning."
    return risk


def _compact_watch_items(
    snapshot: TechnicalSnapshot,
    levels: CompactLevels,
) -> tuple[str, ...]:
    currency = snapshot.instrument.market_currency or ""
    if levels.resistance:
        nearest = format_compact_level(levels.resistance[0], currency)
        level_watch = f"En bekräftad stängning och uppföljning över {nearest}."
    elif levels.support:
        nearest = format_compact_level(levels.support[0], currency)
        level_watch = f"Om kursen håller eller stänger under {nearest}."
    else:
        level_watch = "Om närmaste stöd eller motstånd bryts med en stängning."

    sma20_position = _moving_average_position(
        snapshot.current_candle.close,
        snapshot.trend.sma_20.value,
        snapshot.volatility.atr_14,
    )
    if sma20_position > 0:
        sma_watch = "Om SMA20 fortsätter hålla som stöd."
    elif sma20_position < 0:
        sma_watch = (
            "Om kursen återtar SMA20 eller om snittet fortsätter fungera som "
            "motstånd."
        )
    else:
        sma_watch = "Om testet av SMA20 bekräftas med en stängning över eller under."

    rsi = snapshot.momentum.rsi_14
    if rsi >= 70:
        momentum_watch = "Om RSI faller tillbaka från höga nivåer."
    elif rsi <= 30:
        momentum_watch = "Om RSI återhämtar sig från låga nivåer."
    else:
        momentum_watch = (
            "Om RSI lämnar mittzonen och visar tydligare momentum."
        )
    return level_watch, sma_watch, momentum_watch


def _footer_html(
    *,
    footer_label: str,
    effort: str,
    latency_ms: float,
    total_cost_usd: float | None,
) -> str:
    latency = _format_decimal(latency_ms / 1_000.0, 1)
    cost = (
        f"${_format_decimal(total_cost_usd, 3)}"
        if total_cost_usd is not None
        else "kostnad saknas"
    )
    return f"<i>{escape(footer_label)} · {escape(effort)} · {latency} s · {cost}</i>"


def _split_blocks(
    blocks: list[_RenderBlock],
    *,
    max_message_length: int,
) -> tuple[str, ...]:
    messages: list[str] = []
    current = ""
    for block in blocks:
        for piece in _block_pieces(block, max_message_length=max_message_length):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= max_message_length:
                current = candidate
                continue
            if current:
                messages.append(current)
            current = piece
    if current:
        messages.append(current)
    if not messages or any(len(message) > max_message_length for message in messages):
        raise AssertionError("Telegram rendering produced an invalid message length")
    return tuple(messages)


def _block_pieces(
    block: _RenderBlock,
    *,
    max_message_length: int,
) -> tuple[str, ...]:
    if not block.body:
        if len(block.prefix_html) > max_message_length:
            raise ValueError("render block heading exceeds Telegram message limit")
        return (block.prefix_html,)

    escaped_body = escape(block.body, quote=False)
    complete = f"{block.prefix_html}\n{escaped_body}"
    if len(complete) <= max_message_length:
        return (complete,)

    pieces: list[str] = []
    remaining = block.body
    first = True
    while remaining:
        prefix = f"{block.prefix_html}\n" if first else ""
        budget = max_message_length - len(prefix)
        part, remaining = _take_plain_prefix(remaining, escaped_budget=budget)
        pieces.append(f"{prefix}{escape(part, quote=False)}")
        first = False
    return tuple(pieces)


def _take_plain_prefix(text: str, *, escaped_budget: int) -> tuple[str, str]:
    if escaped_budget <= 0:
        raise ValueError("escaped_budget must be positive")
    escaped_length = 0
    maximum_end = 0
    for index, character in enumerate(text, start=1):
        character_length = len(escape(character, quote=False))
        if escaped_length + character_length > escaped_budget:
            break
        escaped_length += character_length
        maximum_end = index
    if maximum_end == 0:
        raise ValueError("message budget cannot fit the next character")
    if maximum_end == len(text):
        return text, ""

    newline_boundary = text.rfind("\n", 0, maximum_end + 1)
    space_boundary = text.rfind(" ", 0, maximum_end + 1)
    boundary = max(newline_boundary, space_boundary)
    cut = boundary + 1 if boundary >= maximum_end // 2 else maximum_end
    return text[:cut], text[cut:]


def _format_decimal(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")
