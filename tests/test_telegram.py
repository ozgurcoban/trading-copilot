from __future__ import annotations

import asyncio
from html import unescape
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from trading_copilot.analysis.models import (
    AllModelComparison,
    AnalysisModel,
    AnalysisReport,
    TechnicalBias,
)
from trading_copilot.models import MovingAverageDirection, TechnicalSnapshot
from trading_copilot.telegram.app import (
    TelegramBotController,
    build_full_report_keyboard,
    build_model_keyboard,
)
from trading_copilot.telegram.config import (
    TELEGRAM_PENDING_TTL_SECONDS,
    TelegramConfig,
)
from trading_copilot.telegram.pending import (
    PendingRequestError,
    PendingRequestStore,
    TelegramAnalysisChoice,
    decode_callback_data,
    encode_callback_data,
)
from trading_copilot.telegram.rendering import (
    TELEGRAM_COMPACT_MAX_LENGTH,
    TELEGRAM_SAFE_MESSAGE_LENGTH,
    render_compact_analysis_report,
    render_full_analysis_report,
)
from trading_copilot.telegram.reports import (
    FullReportRequestError,
    FullReportStore,
    decode_full_report_callback_data,
    encode_full_report_callback_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_USER_ID = 123_456_789
CHAT_ID = 123_456_789


def load_snapshot() -> TechnicalSnapshot:
    return TechnicalSnapshot.model_validate_json(
        (
            PROJECT_ROOT
            / "artifacts"
            / "comparison_snapshots"
            / "iren.json"
        ).read_text(encoding="utf-8")
    )


def load_report(filename: str) -> AnalysisReport:
    return AnalysisReport.model_validate_json(
        (
            PROJECT_ROOT
            / "artifacts"
            / "analysis_comparisons"
            / filename
        ).read_text(encoding="utf-8")
    )


def load_comparison() -> AllModelComparison:
    return AllModelComparison.model_validate_json(
        (
            PROJECT_ROOT
            / "artifacts"
            / "analysis_comparisons"
            / "iren_sol_opus_fable.json"
        ).read_text(encoding="utf-8")
    )


def test_telegram_config_requires_token_and_nonempty_private_allowlist() -> None:
    config = TelegramConfig.from_env(
        {
            "TELEGRAM_BOT_TOKEN": "123456:secret-token",
            "TELEGRAM_ALLOWED_USER_IDS": "123, 456,123",
        }
    )

    assert config.allowed_user_ids == frozenset({123, 456})
    assert config.allows(123)
    assert not config.allows(999)
    assert "secret-token" not in repr(config)

    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        TelegramConfig.from_env({"TELEGRAM_ALLOWED_USER_IDS": "123"})
    with pytest.raises(ValueError, match="at least one"):
        TelegramConfig.from_env({"TELEGRAM_BOT_TOKEN": "token"})
    with pytest.raises(ValueError, match="comma-separated integers"):
        TelegramConfig.from_env(
            {
                "TELEGRAM_BOT_TOKEN": "token",
                "TELEGRAM_ALLOWED_USER_IDS": "not-an-id",
            }
        )


def test_pending_request_verifies_ttl_owner_chat_message_and_single_use() -> None:
    now = [100.0]
    snapshot = load_snapshot()
    store = PendingRequestStore(
        timer=lambda: now[0],
        token_factory=lambda: "request_AB12",
    )
    request = store.create(
        user_id=ALLOWED_USER_ID,
        chat_id=CHAT_ID,
        snapshot=snapshot,
    )
    bound = store.bind_message(request.request_id, 77)
    callback_data = encode_callback_data(
        bound.request_id,
        TelegramAnalysisChoice.CLAUDE_FABLE_5,
    )

    assert len(callback_data.encode("utf-8")) <= 64
    assert bound.snapshot is snapshot
    with pytest.raises(PendingRequestError, match="does not own"):
        store.consume_callback(
            callback_data,
            user_id=999,
            chat_id=CHAT_ID,
            message_id=77,
        )
    assert len(store) == 1
    with pytest.raises(PendingRequestError, match="did not originate"):
        store.consume_callback(
            callback_data,
            user_id=ALLOWED_USER_ID,
            chat_id=CHAT_ID,
            message_id=78,
        )
    consumed, choice = store.consume_callback(
        callback_data,
        user_id=ALLOWED_USER_ID,
        chat_id=CHAT_ID,
        message_id=77,
    )
    assert consumed.snapshot is snapshot
    assert choice is TelegramAnalysisChoice.CLAUDE_FABLE_5
    assert len(store) == 0
    with pytest.raises(PendingRequestError, match="not found"):
        store.consume_callback(
            callback_data,
            user_id=ALLOWED_USER_ID,
            chat_id=CHAT_ID,
            message_id=77,
        )

    expired = store.create(
        user_id=ALLOWED_USER_ID,
        chat_id=CHAT_ID,
        snapshot=snapshot,
    )
    store.bind_message(expired.request_id, 79)
    now[0] += TELEGRAM_PENDING_TTL_SECONDS
    with pytest.raises(PendingRequestError, match="expired"):
        store.consume_callback(
            encode_callback_data(expired.request_id, TelegramAnalysisChoice.GPT_5_6_SOL),
            user_id=ALLOWED_USER_ID,
            chat_id=CHAT_ID,
            message_id=79,
        )


def test_callback_data_parser_rejects_tampering() -> None:
    data = encode_callback_data(
        "request_AB12",
        TelegramAnalysisChoice.COMPARE_ALL,
    )
    assert decode_callback_data(data) == (
        "request_AB12",
        TelegramAnalysisChoice.COMPARE_ALL,
    )
    for malformed in (
        "request_AB12:a",
        "ta:short:a",
        "ta:request_AB12:unknown",
        "ta:request_AB12:a:extra",
    ):
        with pytest.raises(PendingRequestError):
            decode_callback_data(malformed)


def test_full_report_store_verifies_ttl_owner_chat_and_message() -> None:
    now = [100.0]
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    store = FullReportStore(
        ttl_seconds=60,
        timer=lambda: now[0],
        token_factory=lambda: "report_AB12",
    )
    request = store.create(
        user_id=ALLOWED_USER_ID,
        chat_id=CHAT_ID,
        snapshot=snapshot,
        report=report,
        comparison_position=None,
    )
    bound = store.bind_message(request.report_id, 88)
    callback_data = encode_full_report_callback_data(bound.report_id)

    assert decode_full_report_callback_data(callback_data) == bound.report_id
    assert len(callback_data.encode("utf-8")) <= 64
    with pytest.raises(FullReportRequestError, match="does not own"):
        store.resolve_callback(
            callback_data,
            user_id=999,
            chat_id=CHAT_ID,
            message_id=88,
        )
    with pytest.raises(FullReportRequestError, match="did not originate"):
        store.resolve_callback(
            callback_data,
            user_id=ALLOWED_USER_ID,
            chat_id=CHAT_ID,
            message_id=89,
        )
    resolved = store.resolve_callback(
        callback_data,
        user_id=ALLOWED_USER_ID,
        chat_id=CHAT_ID,
        message_id=88,
    )
    assert resolved.report is report
    assert resolved.snapshot is snapshot

    now[0] += 60
    with pytest.raises(FullReportRequestError, match="expired"):
        store.resolve_callback(
            callback_data,
            user_id=ALLOWED_USER_ID,
            chat_id=CHAT_ID,
            message_id=88,
        )


def test_model_keyboard_has_exact_required_layout_and_short_callbacks() -> None:
    keyboard = build_model_keyboard("request_AB12")
    rows = keyboard.inline_keyboard

    assert [[button.text for button in row] for row in rows] == [
        ["GPT-5.6 Sol"],
        ["Claude Opus 5", "Claude Fable 5"],
        ["Jämför alla"],
    ]
    callback_values = [button.callback_data for row in rows for button in row]
    assert all(value is not None for value in callback_values)
    assert all(len(value.encode("utf-8")) <= 64 for value in callback_values if value)

    full_keyboard = build_full_report_keyboard("report_AB12")
    full_button = full_keyboard.inline_keyboard[0][0]
    assert full_button.text == "Visa full analys"
    assert full_button.callback_data == "tf:report_AB12"


def test_compact_report_is_one_deterministic_decision_overview() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")

    first = render_compact_analysis_report(report, snapshot)
    second = render_compact_analysis_report(report, snapshot)
    rendered = _plain_text(first)

    assert first == second
    assert 800 <= len(rendered) <= 1_600
    assert len(first) <= TELEGRAM_COMPACT_MAX_LENGTH
    assert "IREN · OpenAI" in rendered
    assert (
        "42,60 USD · Teknisk bild: Kortsiktig rekyl i svagare långtrend"
    ) in rendered
    assert "Bias:" not in rendered
    assert "Kort bild" in rendered
    assert (
        "En kortsiktig rekyl dominerar inom 20-dagarsuppgången; kursen håller "
        "SMA20 men ligger under SMA50. Kursen är fortfarande under en fallande "
        "SMA200, vilket begränsar den långsiktiga bilden."
    ) in rendered
    assert "SMA20: 39,68 USD · ↗" in rendered
    assert "SMA50: 43,75 USD · ↘" in rendered
    assert "SMA200: 46,74 USD · ↘" in rendered
    assert "RSI: 51,9 · 5d: -4,8 % · 20d: +5,0 %" in rendered
    assert (
        "Femdagarsnedgången är en rekyl inom den positiva 20-dagarsrörelsen; "
        "RSI nära mitten saknar tydligt momentumövertag."
    ) in rendered
    assert "Primärt stöd: 37,08 USD · bekräftad pivotbotten" in rendered
    assert "Sekundärt stöd: 39,68 USD · SMA20" in rendered
    assert (
        "Primärt motstånd: 43,75–46,74 USD · SMA50/SMA200-zon"
    ) in rendered
    assert (
        "Sekundärt motstånd: 49,19 USD · bekräftad pivottopp + 20d-högsta"
    ) in rendered
    assert "76,87" not in rendered
    assert "17,22" not in rendered
    lines = rendered.splitlines()
    for label in ("🟢 Bull", "⚪ Neutral", "🔴 Bear"):
        label_index = lines.index(label)
        scenario_sentence = lines[label_index + 1]
        assert scenario_sentence.endswith((".", "!", "?"))
        assert "…" not in scenario_sentence
        assert "..." not in scenario_sentence
    assert (
        "Vid 43,75–46,74 USD sammanfaller SMA50 och SMA200, vilket gör zonen "
        "viktig för medellång och lång trend; ett etablerat brott över zonen "
        "skulle placera kursen över båda medelvärdena."
    ) in rendered
    assert (
        "Mellan stödet 37,08 USD och motståndet 43,75–46,74 USD förblir "
        "kursen inom intervallet utan ett bekräftat brott."
    ) in rendered
    assert (
        "37,08 USD är den senaste bekräftade pivotbotten; ett etablerat brott "
        "under nivån skulle placera kursen under denna referens."
    ) in rendered
    assert "Bevaka" in rendered
    assert len([line for line in rendered.splitlines() if line.startswith("• ")]) == 3
    assert (
        "En bekräftad stängning och uppföljning över 43,75–46,74 USD."
    ) in rendered
    assert "Om SMA20 fortsätter hålla som stöd." in rendered
    assert "Om RSI lämnar mittzonen och visar tydligare momentum." in rendered
    assert (
        "ATR 9,0 % innebär stora dagsrörelser; bekräfta nivåbrott med stängning "
        "och uppföljning. "
        "Fundamentaldata är äldre än 120 dagar."
    ) in rendered
    for imprecise_phrase in (
        "väger tyngst",
        "återta trendgränsen",
        "skapa trendstyrka",
        "skapa struktursignal",
    ):
        assert imprecise_phrase not in rendered
    assert "nettoskuld" not in rendered
    assert "GPT-5.6 Sol · 75s · $0.132" in rendered
    assert " · high · " not in rendered
    assert "…" not in rendered
    assert "..." not in rendered
    assert "Teknisk helhetsbild" not in rendered
    assert "Volym & volatilitet" not in rendered
    assert "Begränsningar" not in rendered
    assert report.fundamental_risk_context not in rendered
    assert all(limitation not in rendered for limitation in report.limitations)
    for technical_metadata in (
        "prompt_sha256",
        "snapshot_sha256",
        "response_id",
        report.prompt_sha256,
        report.snapshot_sha256,
        report.run_metadata.response_id,
    ):
        assert technical_metadata not in rendered
    flat_sma20 = snapshot.trend.sma_20.model_copy(
        update={"direction": MovingAverageDirection.FLAT}
    )
    flat_snapshot = snapshot.model_copy(
        update={
            "trend": snapshot.trend.model_copy(update={"sma_20": flat_sma20})
        }
    )
    flat_rendered = _plain_text(render_compact_analysis_report(report, flat_snapshot))
    assert "SMA20: 39,68 USD · →" in flat_rendered

    large_price_snapshot = snapshot.model_copy(
        update={
            "current_candle": snapshot.current_candle.model_copy(
                update={"close": 1_036.13}
            )
        }
    )
    large_price_rendered = _plain_text(
        render_compact_analysis_report(report, large_price_snapshot)
    )
    assert "1\u00a0036,13 USD" in large_price_rendered

    for bias in TechnicalBias:
        labeled_report = report.model_copy(update={"overall_bias": bias})
        labeled_rendered = _plain_text(
            render_compact_analysis_report(labeled_report, snapshot)
        )
        assert (
            "Teknisk bild: Kortsiktig rekyl i svagare långtrend"
        ) in labeled_rendered

    recovery_snapshot = snapshot.model_copy(
        update={
            "current_candle": snapshot.current_candle.model_copy(
                update={"close": 45.0}
            ),
            "momentum": snapshot.momentum.model_copy(
                update={"return_5d_pct": 4.0}
            ),
        }
    )
    recovery_rendered = _plain_text(
        render_compact_analysis_report(report, recovery_snapshot)
    )
    assert "Teknisk bild: Kortsiktig återhämtning i svagare trend" in recovery_rendered


def test_arm_compact_picture_describes_time_horizon_conflict() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    arm_snapshot = snapshot.model_copy(
        update={
            "instrument": snapshot.instrument.model_copy(update={"ticker": "ARM"}),
            "current_candle": snapshot.current_candle.model_copy(
                update={"close": 250.72}
            ),
            "trend": snapshot.trend.model_copy(
                update={
                    "sma_20": snapshot.trend.sma_20.model_copy(
                        update={
                            "value": 261.61,
                            "direction": MovingAverageDirection.FALLING,
                        }
                    ),
                    "sma_50": snapshot.trend.sma_50.model_copy(
                        update={
                            "value": 303.65,
                            "direction": MovingAverageDirection.FALLING,
                        }
                    ),
                    "sma_200": snapshot.trend.sma_200.model_copy(
                        update={
                            "value": 196.77,
                            "direction": MovingAverageDirection.RISING,
                        }
                    ),
                }
            ),
            "momentum": snapshot.momentum.model_copy(
                update={"return_5d_pct": -10.02}
            ),
        }
    )
    arm_report = report.model_copy(
        update={
            "ticker": "ARM",
            "overall_bias": TechnicalBias.NEUTRAL,
        }
    )

    rendered = _plain_text(render_compact_analysis_report(arm_report, arm_snapshot))

    assert (
        "250,72 USD · Teknisk bild: Kortsiktig rekyl i stigande långtrend"
    ) in rendered
    assert (
        "En kortsiktig rekyl dominerar efter nedgång under SMA20 och SMA50. "
        "Den långsiktiga upptrenden är fortfarande intakt över en stigande "
        "SMA200, vilket skapar en tydlig konflikt mellan tidshorisonterna."
    ) in rendered
    assert "Blandad trendbild" not in rendered


def test_compact_picture_treats_marginal_sma_break_as_a_test() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    marginal_snapshot = snapshot.model_copy(
        update={
            "current_candle": snapshot.current_candle.model_copy(
                update={"close": 39.60}
            ),
            "trend": snapshot.trend.model_copy(
                update={
                    "sma_20": snapshot.trend.sma_20.model_copy(
                        update={
                            "value": 39.68,
                            "direction": MovingAverageDirection.FALLING,
                        }
                    ),
                    "sma_50": snapshot.trend.sma_50.model_copy(
                        update={
                            "value": 38.0,
                            "direction": MovingAverageDirection.RISING,
                        }
                    ),
                    "sma_200": snapshot.trend.sma_200.model_copy(
                        update={
                            "value": 35.0,
                            "direction": MovingAverageDirection.RISING,
                        }
                    ),
                }
            ),
        }
    )

    rendered = _plain_text(
        render_compact_analysis_report(report, marginal_snapshot)
    )

    assert "Teknisk bild: Test av SMA20 i stigande långtrend" in rendered
    assert "kursen testar SMA20 men håller sig över SMA50" in rendered
    assert "samstämmig nedtrend" not in rendered.lower()
    assert (
        "Om testet av SMA20 bekräftas med en stängning över eller under."
    ) in rendered


def test_compact_levels_merge_pivot_and_sma200_into_one_zone() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    confluent_snapshot = snapshot.model_copy(
        update={
            "trend": snapshot.trend.model_copy(
                update={
                    "sma_200": snapshot.trend.sma_200.model_copy(
                        update={"value": 49.0}
                    )
                }
            )
        }
    )

    rendered = _plain_text(
        render_compact_analysis_report(report, confluent_snapshot)
    )

    assert (
        "Primärt motstånd: 49,00–49,19 USD · SMA200 + bekräftad pivottopp + "
        "20d-högsta-zon"
    ) in rendered
    assert "Sekundärt motstånd: 43,75 USD · SMA50" in rendered
    assert (
        "Vid 49,00–49,19 USD sammanfaller SMA200, pivottoppen och "
        "20-dagarshögsta, vilket gör zonen viktig för lång trend och två "
        "prisreferenser; ett etablerat brott över zonen skulle placera kursen "
        "över SMA200 och pivottoppen samt sätta en ny 20-dagarshögsta."
    ) in rendered


def test_compact_picture_describes_aligned_structure_and_momentum_regime() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    aligned_snapshot = snapshot.model_copy(
        update={
            "trend": snapshot.trend.model_copy(
                update={
                    "sma_20": snapshot.trend.sma_20.model_copy(
                        update={"direction": MovingAverageDirection.RISING}
                    ),
                    "sma_50": snapshot.trend.sma_50.model_copy(
                        update={
                            "value": 38.0,
                            "direction": MovingAverageDirection.RISING,
                        }
                    ),
                    "sma_200": snapshot.trend.sma_200.model_copy(
                        update={
                            "value": 30.0,
                            "direction": MovingAverageDirection.RISING,
                        }
                    ),
                }
            ),
            "momentum": snapshot.momentum.model_copy(
                update={"return_5d_pct": 3.0, "return_20d_pct": 10.0}
            ),
        }
    )

    rendered = _plain_text(
        render_compact_analysis_report(report, aligned_snapshot)
    )

    assert "Teknisk bild: Styrka över flera tidshorisonter" in rendered
    assert (
        "En samstämmig upptrend dominerar över kort, medellång och lång "
        "horisont, eftersom kursen ligger över tre stigande medelvärden."
    ) in rendered
    assert (
        "Avkastningen är positiv över 5 och 20 dagar, men RSI nära mitten visar "
        "att momentumövertaget ännu är begränsat."
    ) in rendered

    stretched_snapshot = aligned_snapshot.model_copy(
        update={
            "momentum": aligned_snapshot.momentum.model_copy(
                update={"rsi_14": 72.0}
            )
        }
    )
    stretched_rendered = _plain_text(
        render_compact_analysis_report(report, stretched_snapshot)
    )
    assert (
        "Positiv avkastning över 5 och 20 dagar samt RSI över 70 visar starkt "
        "uppåtriktat momentum, men också en utsträckt rörelse."
    ) in stretched_rendered


def test_compact_picture_calls_ma_alignment_dominant_technical_weakness() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    weak_snapshot = snapshot.model_copy(
        update={
            "current_candle": snapshot.current_candle.model_copy(
                update={"close": 20.0}
            ),
            "trend": snapshot.trend.model_copy(
                update={
                    "sma_20": snapshot.trend.sma_20.model_copy(
                        update={
                            "value": 30.0,
                            "direction": MovingAverageDirection.FALLING,
                        }
                    ),
                    "sma_50": snapshot.trend.sma_50.model_copy(
                        update={
                            "value": 35.0,
                            "direction": MovingAverageDirection.FALLING,
                        }
                    ),
                    "sma_200": snapshot.trend.sma_200.model_copy(
                        update={
                            "value": 40.0,
                            "direction": MovingAverageDirection.FALLING,
                        }
                    ),
                }
            ),
            "momentum": snapshot.momentum.model_copy(
                update={"return_5d_pct": -4.0, "return_20d_pct": -12.0}
            ),
        }
    )

    rendered = _plain_text(
        render_compact_analysis_report(report, weak_snapshot)
    )

    assert (
        "Teknisk svaghet dominerar över kort, medellång och lång horisont, "
        "eftersom kursen ligger tydligt under tre fallande medelvärden."
    ) in rendered
    assert "samstämmig nedtrend" not in rendered.lower()


def test_compact_momentum_identifies_bounce_after_broader_decline() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_gpt-5.6-sol.json")
    bounce_snapshot = snapshot.model_copy(
        update={
            "current_candle": snapshot.current_candle.model_copy(
                update={"close": 45.0}
            ),
            "momentum": snapshot.momentum.model_copy(
                update={"return_5d_pct": 4.0, "return_20d_pct": -12.0}
            ),
        }
    )

    rendered = _plain_text(
        render_compact_analysis_report(report, bounce_snapshot)
    )

    assert "Teknisk bild: Kortsiktig återhämtning i svagare trend" in rendered
    assert (
        "Femdagarsuppgången är en kortsiktig återhämtning efter "
        "20-dagarsnedgången, medan RSI nära mitten ännu inte bekräftar ett "
        "bredare momentumskifte."
    ) in rendered


def test_full_report_is_complete_deterministic_and_split_safely() -> None:
    snapshot = load_snapshot()
    report = load_report("iren_claude-fable-5_prompt-v0.3.json")

    first = render_full_analysis_report(report, snapshot, max_message_length=600)
    second = render_full_analysis_report(report, snapshot, max_message_length=600)
    rendered = _plain_text("\n".join(first))

    assert first == second
    assert len(first) > 1
    assert all(len(chunk) <= 600 for chunk in first)
    assert "IREN · Anthropic · Claude Fable 5" in rendered
    assert "Stängning: 42,60 USD · Bias: Blandad" in rendered
    assert "Kort bild" in rendered
    assert "Trend" in rendered
    assert "Momentum" in rendered
    assert "Volym & volatilitet" in rendered
    assert "Prisnivåer" in rendered
    assert "Bullish:" in rendered and "Neutral:" in rendered and "Bearish:" in rendered
    assert "Viktigast att bevaka" in rendered
    assert "Fundamental risk" in rendered
    assert "Begränsningar" in rendered
    assert "Fable 5 · high · 51,4 s · $0,211" in rendered
    for technical_metadata in (
        "prompt_sha256",
        "snapshot_sha256",
        "response_id",
        report.prompt_sha256,
        report.snapshot_sha256,
        report.run_metadata.response_id,
    ):
        assert technical_metadata not in rendered

    report_fields = (
        report.setup_summary,
        report.overall_technical_picture,
        report.trend,
        report.momentum,
        report.volume_and_volatility,
        report.key_price_levels,
        report.scenarios.bullish,
        report.scenarios.neutral,
        report.scenarios.bearish,
        *report.what_to_watch,
        report.fundamental_risk_context,
        *report.limitations,
    )
    normalized_rendered = _normalize(rendered)
    for field in report_fields:
        assert _normalize(field) in normalized_rendered

    standard_chunks = render_full_analysis_report(report, snapshot)
    assert all(len(chunk) <= TELEGRAM_SAFE_MESSAGE_LENGTH for chunk in standard_chunks)


class FakeStatusMessage:
    def __init__(self, event_log: list[tuple], *, message_id: int = 77) -> None:
        self.message_id = message_id
        self.chat = SimpleNamespace(id=CHAT_ID, type="private")
        self.event_log = event_log
        self.edits: list[dict] = []

    async def edit_text(self, text: str, **kwargs: object) -> None:
        self.event_log.append(("status_edit", text))
        self.edits.append({"text": text, **kwargs})


class FakeIncomingMessage:
    def __init__(self, text: str, event_log: list[tuple]) -> None:
        self.text = text
        self.event_log = event_log
        self.replies: list[dict] = []
        self.status_message = FakeStatusMessage(event_log)

    async def reply_text(self, text: str, **kwargs: object) -> FakeStatusMessage:
        self.event_log.append(("reply", text))
        self.replies.append({"text": text, **kwargs})
        return self.status_message


class FakeCallbackQuery:
    def __init__(
        self,
        *,
        data: str,
        message: FakeStatusMessage,
        event_log: list[tuple],
        user_id: int = ALLOWED_USER_ID,
    ) -> None:
        self.data = data
        self.message = message
        self.from_user = SimpleNamespace(id=user_id)
        self.event_log = event_log
        self.answers: list[dict] = []
        self.edits: list[str] = []

    async def answer(self, text: str | None = None, **kwargs: object) -> None:
        self.event_log.append(("callback_answer", text))
        self.answers.append({"text": text, **kwargs})

    async def edit_message_text(self, text: str, **kwargs: object) -> None:
        self.event_log.append(("callback_edit", text))
        self.edits.append(text)


class FakeBot:
    def __init__(self, event_log: list[tuple]) -> None:
        self.event_log = event_log
        self.sent: list[dict] = []
        self._next_message_id = 1_000

    async def send_message(self, **kwargs: object) -> SimpleNamespace:
        self.event_log.append(("send_message", kwargs["text"]))
        sent = dict(kwargs)
        sent["message_id"] = self._next_message_id
        self.sent.append(sent)
        message = SimpleNamespace(message_id=self._next_message_id)
        self._next_message_id += 1
        return message


class FailingBot(FakeBot):
    async def send_message(self, **kwargs: object) -> None:
        del kwargs
        raise RuntimeError("simulated Telegram delivery failure")


def test_ticker_to_single_model_flow_builds_once_and_shows_progress_first() -> None:
    event_log: list[tuple] = []
    snapshot = load_snapshot()
    report = load_report("iren_claude-fable-5_prompt-v0.3.json")
    builder_snapshots: list[TechnicalSnapshot] = []
    analyzed_snapshots: list[TechnicalSnapshot] = []

    def snapshot_builder(ticker: str) -> TechnicalSnapshot:
        event_log.append(("build_snapshot", ticker))
        builder_snapshots.append(snapshot)
        return snapshot

    def single_analyzer(
        received_snapshot: TechnicalSnapshot,
        model: AnalysisModel,
    ) -> AnalysisReport:
        event_log.append(("analyze", model.value))
        analyzed_snapshots.append(received_snapshot)
        assert model is AnalysisModel.CLAUDE_FABLE_5
        return report

    store = PendingRequestStore(token_factory=lambda: "request_AB12")
    full_reports = FullReportStore(token_factory=lambda: "report_AB12")
    controller = TelegramBotController(
        config=_config(),
        pending_requests=store,
        full_reports=full_reports,
        snapshot_builder=snapshot_builder,
        single_analyzer=single_analyzer,
    )
    incoming = FakeIncomingMessage("iren", event_log)
    message_update = SimpleNamespace(
        effective_message=incoming,
        effective_user=SimpleNamespace(id=ALLOWED_USER_ID),
        effective_chat=SimpleNamespace(id=CHAT_ID, type="private"),
    )
    asyncio.run(controller.handle_ticker(message_update, SimpleNamespace()))

    assert len(builder_snapshots) == 1
    selection_edit = incoming.status_message.edits[-1]
    keyboard = selection_edit["reply_markup"]
    fable_callback = keyboard.inline_keyboard[1][1].callback_data
    assert fable_callback is not None
    fake_bot = FakeBot(event_log)
    query = FakeCallbackQuery(
        data=fable_callback,
        message=incoming.status_message,
        event_log=event_log,
    )
    callback_update = SimpleNamespace(callback_query=query)
    asyncio.run(
        controller.handle_callback(
            callback_update,
            SimpleNamespace(bot=fake_bot),
        )
    )

    assert analyzed_snapshots == [snapshot]
    assert analyzed_snapshots[0] is builder_snapshots[0]
    assert len(store) == 0
    answer_index = next(i for i, event in enumerate(event_log) if event[0] == "callback_answer")
    progress_index = next(
        i
        for i, event in enumerate(event_log)
        if event[0] == "callback_edit" and "Analys pågår" in event[1]
    )
    analysis_index = next(i for i, event in enumerate(event_log) if event[0] == "analyze")
    assert answer_index < progress_index < analysis_index
    assert query.edits[0] == "⏳ IREN · Claude Fable 5\nAnalys pågår…"
    assert query.edits[-1] == "✅ IREN · Claude Fable 5 klar"
    assert fake_bot.sent
    assert len(fake_bot.sent) == 1
    assert all(message["parse_mode"] == "HTML" for message in fake_bot.sent)
    assert all("snapshot_sha256" not in message["text"] for message in fake_bot.sent)
    compact = _plain_text(fake_bot.sent[0]["text"])
    assert "Kort bild" in compact
    assert "Teknisk helhetsbild" not in compact
    full_button = fake_bot.sent[0]["reply_markup"].inline_keyboard[0][0]
    assert full_button.text == "Visa full analys"
    assert len(full_reports) == 1

    full_message = FakeStatusMessage(
        event_log,
        message_id=fake_bot.sent[0]["message_id"],
    )
    full_query = FakeCallbackQuery(
        data=full_button.callback_data,
        message=full_message,
        event_log=event_log,
    )
    asyncio.run(
        controller.handle_callback(
            SimpleNamespace(callback_query=full_query),
            SimpleNamespace(bot=fake_bot),
        )
    )
    assert full_query.answers[-1]["text"] == "Visar full analys…"
    assert len(analyzed_snapshots) == 1
    full_rendered = _plain_text(
        "\n".join(message["text"] for message in fake_bot.sent[1:])
    )
    assert "Teknisk helhetsbild" in full_rendered
    assert "Volym & volatilitet" in full_rendered
    assert "Begränsningar" in full_rendered
    assert report.overall_technical_picture in full_rendered

    asyncio.run(
        controller.handle_callback(
            callback_update,
            SimpleNamespace(bot=fake_bot),
        )
    )
    assert len(analyzed_snapshots) == 1
    assert query.answers[-1]["show_alert"] is True


def test_compare_all_uses_one_snapshot_and_sends_three_separated_reports() -> None:
    event_log: list[tuple] = []
    snapshot = load_snapshot()
    comparison = load_comparison()
    build_count = 0
    comparison_snapshots: list[TechnicalSnapshot] = []

    def snapshot_builder(ticker: str) -> TechnicalSnapshot:
        nonlocal build_count
        build_count += 1
        assert ticker == "IREN"
        return snapshot

    def comparison_analyzer(received_snapshot: TechnicalSnapshot) -> AllModelComparison:
        event_log.append(("compare_all", received_snapshot.instrument.ticker))
        comparison_snapshots.append(received_snapshot)
        return comparison

    controller = TelegramBotController(
        config=_config(),
        pending_requests=PendingRequestStore(token_factory=lambda: "request_CD34"),
        snapshot_builder=snapshot_builder,
        single_analyzer=lambda *_: pytest.fail("single analyzer must not run"),
        comparison_analyzer=comparison_analyzer,
    )
    incoming = FakeIncomingMessage("IREN", event_log)
    asyncio.run(
        controller.handle_ticker(
            SimpleNamespace(
                effective_message=incoming,
                effective_user=SimpleNamespace(id=ALLOWED_USER_ID),
                effective_chat=SimpleNamespace(id=CHAT_ID, type="private"),
            ),
            SimpleNamespace(),
        )
    )
    keyboard = incoming.status_message.edits[-1]["reply_markup"]
    compare_callback = keyboard.inline_keyboard[2][0].callback_data
    fake_bot = FakeBot(event_log)
    query = FakeCallbackQuery(
        data=compare_callback,
        message=incoming.status_message,
        event_log=event_log,
    )
    asyncio.run(
        controller.handle_callback(
            SimpleNamespace(callback_query=query),
            SimpleNamespace(bot=fake_bot),
        )
    )

    assert build_count == 1
    assert comparison_snapshots == [snapshot]
    assert comparison_snapshots[0] is snapshot
    rendered = _plain_text("\n".join(message["text"] for message in fake_bot.sent))
    assert "Jämförelse 1/3" in rendered
    assert "Jämförelse 2/3" in rendered
    assert "Jämförelse 3/3" in rendered
    assert "IREN · OpenAI" in rendered
    assert rendered.count("IREN · Anthropic") == 2
    assert len(fake_bot.sent) == 3
    assert all(
        800 <= len(message["text"]) <= TELEGRAM_COMPACT_MAX_LENGTH
        for message in fake_bot.sent
    )
    assert all(
        message["reply_markup"].inline_keyboard[0][0].text == "Visa full analys"
        for message in fake_bot.sent
    )
    assert "Teknisk helhetsbild" not in rendered
    assert query.edits[0] == "⏳ IREN · Jämför alla\nAnalys pågår…"
    assert query.edits[-1] == "✅ IREN · 3 analyser klara"


def test_completed_analysis_shows_visible_error_if_report_delivery_fails() -> None:
    event_log: list[tuple] = []
    snapshot = load_snapshot()
    report = load_report("iren_claude-fable-5_prompt-v0.3.json")
    store = PendingRequestStore(token_factory=lambda: "request_GH78")
    controller = TelegramBotController(
        config=_config(),
        pending_requests=store,
        snapshot_builder=lambda _: snapshot,
        single_analyzer=lambda *_: report,
    )
    incoming = FakeIncomingMessage("IREN", event_log)
    asyncio.run(
        controller.handle_ticker(
            SimpleNamespace(
                effective_message=incoming,
                effective_user=SimpleNamespace(id=ALLOWED_USER_ID),
                effective_chat=SimpleNamespace(id=CHAT_ID, type="private"),
            ),
            SimpleNamespace(),
        )
    )
    callback_data = (
        incoming.status_message.edits[-1]["reply_markup"]
        .inline_keyboard[1][1]
        .callback_data
    )
    query = FakeCallbackQuery(
        data=callback_data,
        message=incoming.status_message,
        event_log=event_log,
    )

    asyncio.run(
        controller.handle_callback(
            SimpleNamespace(callback_query=query),
            SimpleNamespace(bot=FailingBot(event_log)),
        )
    )

    assert query.edits[-1].startswith(
        "❌ Analysen blev klar men rapporten kunde inte levereras."
    )


def test_unauthorized_user_never_builds_snapshot_or_consumes_callback() -> None:
    event_log: list[tuple] = []
    snapshot = load_snapshot()
    build_count = 0

    def snapshot_builder(_: str) -> TechnicalSnapshot:
        nonlocal build_count
        build_count += 1
        return snapshot

    store = PendingRequestStore(token_factory=lambda: "request_EF56")
    controller = TelegramBotController(
        config=_config(),
        pending_requests=store,
        snapshot_builder=snapshot_builder,
    )
    incoming = FakeIncomingMessage("IREN", event_log)
    asyncio.run(
        controller.handle_ticker(
            SimpleNamespace(
                effective_message=incoming,
                effective_user=SimpleNamespace(id=999),
                effective_chat=SimpleNamespace(id=999, type="private"),
            ),
            SimpleNamespace(),
        )
    )
    assert build_count == 0
    assert incoming.replies[0]["text"] == "Åtkomst nekad."

    pending = store.create(user_id=ALLOWED_USER_ID, chat_id=CHAT_ID, snapshot=snapshot)
    store.bind_message(pending.request_id, 77)
    query = FakeCallbackQuery(
        data=encode_callback_data(
            pending.request_id,
            TelegramAnalysisChoice.GPT_5_6_SOL,
        ),
        message=FakeStatusMessage(event_log),
        event_log=event_log,
        user_id=999,
    )
    asyncio.run(
        controller.handle_callback(
            SimpleNamespace(callback_query=query),
            SimpleNamespace(bot=FakeBot(event_log)),
        )
    )
    assert query.answers[-1]["show_alert"] is True
    assert len(store) == 1


def _config() -> TelegramConfig:
    return TelegramConfig(
        bot_token="123456:secret-token",
        allowed_user_ids=frozenset({ALLOWED_USER_ID}),
    )


def _plain_text(value: str) -> str:
    return unescape(re.sub(r"<[^>]+>", "", value))


def _normalize(value: str) -> str:
    return " ".join(value.split())
