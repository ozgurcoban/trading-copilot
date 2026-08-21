"""Local long-polling Telegram entrypoint for Milestone 3."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..analysis.models import AllModelComparison, AnalysisModel, AnalysisReport
from ..analysis.service import analyze_with_model, compare_all_models
from ..market_data import normalize_ticker
from ..models import TechnicalSnapshot
from ..snapshot import build_snapshot
from .config import TelegramConfig
from .pending import (
    PendingRequest,
    PendingRequestError,
    PendingRequestStore,
    TelegramAnalysisChoice,
    encode_callback_data,
)
from .rendering import (
    render_compact_analysis_report,
    render_full_analysis_report,
)
from .reports import (
    FullReportRequestError,
    FullReportStore,
    encode_full_report_callback_data,
    is_full_report_callback,
)

logger = logging.getLogger(__name__)


class TelegramBotController:
    """Translate Telegram events into existing snapshot and analysis calls."""

    def __init__(
        self,
        *,
        config: TelegramConfig,
        pending_requests: PendingRequestStore | None = None,
        full_reports: FullReportStore | None = None,
        snapshot_builder: Callable[[str], TechnicalSnapshot] = build_snapshot,
        single_analyzer: Callable[
            [TechnicalSnapshot, AnalysisModel], AnalysisReport
        ] = analyze_with_model,
        comparison_analyzer: Callable[
            [TechnicalSnapshot], AllModelComparison
        ] = compare_all_models,
    ) -> None:
        self.config = config
        self.pending_requests = (
            pending_requests
            if pending_requests is not None
            else PendingRequestStore()
        )
        self.full_reports = (
            full_reports if full_reports is not None else FullReportStore()
        )
        self.snapshot_builder = snapshot_builder
        self.single_analyzer = single_analyzer
        self.comparison_analyzer = comparison_analyzer

    async def handle_ticker(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        del context
        message = update.effective_message
        user = update.effective_user
        chat = update.effective_chat
        if message is None or user is None or chat is None:
            return
        if not self.config.allows(user.id):
            await message.reply_text("Åtkomst nekad.")
            return
        if chat.type != ChatType.PRIVATE:
            await message.reply_text("Botten kan endast användas i en privat chatt.")
            return
        try:
            ticker = normalize_ticker(message.text or "")
        except Exception:
            await message.reply_text("Skicka endast en giltig ticker, till exempel IREN.")
            return

        status_message = await message.reply_text(f"⏳ Bygger snapshot för {ticker}…")
        try:
            snapshot = await asyncio.to_thread(self.snapshot_builder, ticker)
        except Exception:
            logger.exception("Could not build Telegram snapshot for %s", ticker)
            await status_message.edit_text(
                f"❌ Kunde inte bygga snapshot för {ticker}. Kontrollera tickern och försök igen."
            )
            return

        pending = self.pending_requests.create(
            user_id=user.id,
            chat_id=chat.id,
            snapshot=snapshot,
        )
        pending = self.pending_requests.bind_message(
            pending.request_id,
            status_message.message_id,
        )
        await status_message.edit_text(
            _selection_text(snapshot),
            reply_markup=build_model_keyboard(pending.request_id),
        )
        logger.info(
            "Telegram snapshot ready for %s (user_id=%s, request_id=%s)",
            snapshot.instrument.ticker,
            user.id,
            pending.request_id,
        )

    async def handle_callback(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        query = update.callback_query
        if query is None or query.from_user is None:
            return
        if not self.config.allows(query.from_user.id):
            await query.answer("Åtkomst nekad.", show_alert=True)
            return
        message = query.message
        if message is None or message.chat.type != ChatType.PRIVATE:
            await query.answer("Ogiltig callback.", show_alert=True)
            return
        if is_full_report_callback(query.data):
            await self._handle_full_report_callback(query, context)
            return
        try:
            pending, choice = self.pending_requests.consume_callback(
                query.data or "",
                user_id=query.from_user.id,
                chat_id=message.chat.id,
                message_id=message.message_id,
            )
        except PendingRequestError:
            await query.answer(
                "Förfrågan är ogiltig eller har löpt ut. Skicka tickern igen.",
                show_alert=True,
            )
            return

        await query.answer("Analysen startar…")
        await query.edit_message_text(_progress_text(pending, choice))
        logger.info(
            "Telegram analysis started for %s (%s, request_id=%s)",
            pending.snapshot.instrument.ticker,
            choice.value,
            pending.request_id,
        )
        try:
            if choice is TelegramAnalysisChoice.COMPARE_ALL:
                comparison = await asyncio.to_thread(
                    self.comparison_analyzer,
                    pending.snapshot,
                )
                reports = (
                    comparison.gpt_5_6_sol,
                    comparison.claude_opus_5,
                    comparison.claude_fable_5,
                )
            else:
                model = choice.model
                if model is None:
                    raise AssertionError("single-model choice has no model")
                report = await asyncio.to_thread(
                    self.single_analyzer,
                    pending.snapshot,
                    model,
                )
                reports = (report,)
        except Exception:
            logger.exception(
                "Telegram analysis failed for %s (%s)",
                pending.snapshot.instrument.ticker,
                choice.value,
            )
            await query.edit_message_text(
                "❌ Analysen misslyckades. Skicka tickern igen för att försöka på nytt."
            )
            return

        sent_reports = 0
        try:
            await query.edit_message_text(_complete_text(pending, choice))
            for index, report in enumerate(reports, start=1):
                comparison_position = (
                    (index, len(reports)) if len(reports) > 1 else None
                )
                compact = render_compact_analysis_report(
                    report,
                    pending.snapshot,
                    comparison_position=comparison_position,
                )
                full_report = self.full_reports.create(
                    user_id=pending.user_id,
                    chat_id=pending.chat_id,
                    snapshot=pending.snapshot,
                    report=report,
                    comparison_position=comparison_position,
                )
                try:
                    sent_message = await context.bot.send_message(
                        chat_id=pending.chat_id,
                        text=compact,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                        reply_markup=build_full_report_keyboard(full_report.report_id),
                    )
                    message_id = getattr(sent_message, "message_id", None)
                    if not isinstance(message_id, int):
                        raise RuntimeError(
                            "Telegram did not return a message ID for the compact report"
                        )
                    self.full_reports.bind_message(
                        full_report.report_id,
                        message_id,
                    )
                except Exception:
                    self.full_reports.discard(full_report.report_id)
                    raise
                sent_reports += 1
        except Exception:
            logger.exception(
                "Telegram compact report delivery failed for %s (%s) after %d reports",
                pending.snapshot.instrument.ticker,
                choice.value,
                sent_reports,
            )
            try:
                await query.edit_message_text(
                    "❌ Analysen blev klar men rapporten kunde inte levereras. "
                    "Skicka tickern igen för att försöka på nytt."
                )
            except Exception:
                logger.exception("Could not show Telegram delivery failure")
            return
        logger.info(
            "Telegram compact report sent for %s (%s, reports=%d)",
            pending.snapshot.instrument.ticker,
            choice.value,
            len(reports),
        )

    async def _handle_full_report_callback(
        self,
        query: Any,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        message = query.message
        try:
            full_report = self.full_reports.resolve_callback(
                query.data or "",
                user_id=query.from_user.id,
                chat_id=message.chat.id,
                message_id=message.message_id,
            )
        except FullReportRequestError:
            await query.answer(
                "Den fullständiga analysen är ogiltig eller har löpt ut. Kör analysen igen.",
                show_alert=True,
            )
            return

        await query.answer("Visar full analys…")
        try:
            chunks = render_full_analysis_report(
                full_report.report,
                full_report.snapshot,
                comparison_position=full_report.comparison_position,
            )
            for chunk in chunks:
                await context.bot.send_message(
                    chat_id=full_report.chat_id,
                    text=chunk,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
        except Exception:
            logger.exception(
                "Telegram full report delivery failed for %s (%s)",
                full_report.report.ticker,
                full_report.report.model.value,
            )
            try:
                await context.bot.send_message(
                    chat_id=full_report.chat_id,
                    text="❌ Den fullständiga analysen kunde inte visas. Försök igen.",
                )
            except Exception:
                logger.exception("Could not show Telegram full-report delivery failure")


def build_model_keyboard(request_id: str) -> InlineKeyboardMarkup:
    callback = lambda choice: encode_callback_data(request_id, choice)
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    TelegramAnalysisChoice.GPT_5_6_SOL.button_label,
                    callback_data=callback(TelegramAnalysisChoice.GPT_5_6_SOL),
                )
            ],
            [
                InlineKeyboardButton(
                    TelegramAnalysisChoice.CLAUDE_OPUS_5.button_label,
                    callback_data=callback(TelegramAnalysisChoice.CLAUDE_OPUS_5),
                ),
                InlineKeyboardButton(
                    TelegramAnalysisChoice.CLAUDE_FABLE_5.button_label,
                    callback_data=callback(TelegramAnalysisChoice.CLAUDE_FABLE_5),
                ),
            ],
            [
                InlineKeyboardButton(
                    TelegramAnalysisChoice.COMPARE_ALL.button_label,
                    callback_data=callback(TelegramAnalysisChoice.COMPARE_ALL),
                )
            ],
        ]
    )


def build_full_report_keyboard(report_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Visa full analys",
                    callback_data=encode_full_report_callback_data(report_id),
                )
            ]
        ]
    )


def create_application(
    config: TelegramConfig,
    *,
    controller: TelegramBotController | None = None,
) -> Application[Any, Any, Any, Any, Any, Any]:
    active_controller = controller or TelegramBotController(config=config)
    application = (
        Application.builder()
        .token(config.bot_token)
        .concurrent_updates(False)
        .build()
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, active_controller.handle_ticker)
    )
    application.add_handler(CallbackQueryHandler(active_controller.handle_callback))
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    config = TelegramConfig.from_env()
    application = create_application(config)
    logger.info("Starting Trading Copilot Telegram bot with local long polling")
    application.run_polling(
        allowed_updates=("message", "callback_query"),
        drop_pending_updates=False,
    )


def _selection_text(snapshot: TechnicalSnapshot) -> str:
    close = f"{snapshot.current_candle.close:.2f}".replace(".", ",")
    currency = snapshot.instrument.market_currency or ""
    close_label = f"{close} {currency}".strip()
    return (
        f"✅ Snapshot klart för {snapshot.instrument.ticker}\n"
        f"Stängning: {close_label} · {snapshot.metadata.as_of_session.isoformat()}\n\n"
        "Välj analysmodell:"
    )


def _progress_text(
    pending: PendingRequest,
    choice: TelegramAnalysisChoice,
) -> str:
    return (
        f"⏳ {pending.snapshot.instrument.ticker} · {choice.button_label}\n"
        "Analys pågår…"
    )


def _complete_text(
    pending: PendingRequest,
    choice: TelegramAnalysisChoice,
) -> str:
    if choice is TelegramAnalysisChoice.COMPARE_ALL:
        return f"✅ {pending.snapshot.instrument.ticker} · 3 analyser klara"
    return f"✅ {pending.snapshot.instrument.ticker} · {choice.button_label} klar"
