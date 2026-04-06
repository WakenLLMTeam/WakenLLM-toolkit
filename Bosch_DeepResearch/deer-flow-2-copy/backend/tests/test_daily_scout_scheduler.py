import asyncio
import importlib
import sys
import time
import types
from datetime import UTC, datetime
from types import SimpleNamespace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage


def _install_module_stub(name: str, **attributes) -> None:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


try:
    import apscheduler.schedulers.asyncio  # noqa: F401
    import apscheduler.triggers.cron  # noqa: F401
except ModuleNotFoundError:
    _install_module_stub("apscheduler")
    _install_module_stub("apscheduler.schedulers")
    _install_module_stub("apscheduler.schedulers.asyncio", AsyncIOScheduler=object)
    _install_module_stub("apscheduler.triggers")
    _install_module_stub("apscheduler.triggers.cron", CronTrigger=object)

try:
    import tavily  # noqa: F401
    import tavily.errors  # noqa: F401
except ModuleNotFoundError:
    class _DummyTavilyClient:
        def search(self, *args, **kwargs):
            return {"results": []}


    class _DummyMissingAPIKeyError(Exception):
        pass


    _install_module_stub("tavily", TavilyClient=_DummyTavilyClient)
    _install_module_stub("tavily.errors", MissingAPIKeyError=_DummyMissingAPIKeyError)

from scripts.daily_scout_accounts import (
    load_user_scout_status,
    load_user_settings,
    save_user_scout_status,
    save_user_settings,
)


@pytest.fixture(autouse=True)
def _stub_hash_password(monkeypatch):
    monkeypatch.setattr("scripts.daily_scout_accounts.hash_password", lambda password: "stubbed-password-hash")
from scripts.daily_scout_teams_bot import (
    DailyScoutAlreadyAttemptedError,
    DailyScoutAlreadyRunningError,
    DailyScoutRunner,
    ScoutSettings,
    _apply_schedule,
    _sync_account_runners,
)
from deerflow.client import DeerFlowClient


def _make_settings(tmp_path: Path, email: str = "user@example.com") -> ScoutSettings:
    return ScoutSettings(
        owner_email=email,
        topic="AI and software engineering updates",
        rag_resources=[],
        model_name=None,
        thinking_enabled=True,
        run_hour=9,
        run_minute=0,
        timezone="UTC",
        report_dir=tmp_path / "reports",
        checkpoint_mode="none",
        checkpoint_path=tmp_path / "checkpoints" / "daily_scout.sqlite",
        scout_thread_id="daily-scout-test",
        run_on_startup=False,
        settings_path=tmp_path / "daily_scout_settings.json",
    )


def test_user_scout_status_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))

    initial = load_user_scout_status("user@example.com")
    assert initial["scheduled"]["last_status"] is None

    saved = save_user_scout_status(
        "user@example.com",
        {
            "scheduled": {
                "last_attempt_for_date": "2026-03-10",
                "last_status": "failed",
                "last_error": "boom",
            }
        },
    )

    assert saved["scheduled"]["last_attempt_for_date"] == "2026-03-10"
    assert saved["scheduled"]["last_status"] == "failed"
    assert saved["scheduled"]["last_error"] == "boom"


def test_save_user_settings_normalizes_timezone_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))

    saved = save_user_settings(
        "user@example.com",
        {
            "topic": "AI and software engineering updates",
            "timezone": "CST",
        },
    )

    assert saved["timezone"] == "Asia/Shanghai"
    assert load_user_settings("user@example.com")["timezone"] == "Asia/Shanghai"


def test_apply_schedule_uses_normalized_timezone_for_next_fire_time():
    scheduler = MagicMock()
    runner = MagicMock()
    runner.settings = MagicMock(run_hour=14, run_minute=38, timezone="UTC")

    hour, minute, timezone = _apply_schedule(
        scheduler,
        "user@example.com",
        runner,
        {"run_hour": 14, "run_minute": 38, "timezone": "CST"},
    )

    assert (hour, minute, timezone) == (14, 38, "Asia/Shanghai")
    trigger = scheduler.add_job.call_args.kwargs["trigger"]
    next_fire = trigger.get_next_fire_time(None, datetime(2026, 3, 10, 6, 37, tzinfo=UTC))
    assert next_fire.isoformat() == "2026-03-10T14:38:00+08:00"
    assert scheduler.add_job.call_args.kwargs["misfire_grace_time"] == 3600


def test_sync_account_runners_does_not_run_on_startup():
    scheduler = MagicMock()
    runner = MagicMock()
    runner._close_checkpointer = MagicMock()
    runner.run_once = AsyncMock()
    runner.settings = MagicMock(run_hour=9, run_minute=0, timezone="UTC")

    with (
        patch("scripts.daily_scout_teams_bot.list_active_accounts", return_value=[{"email": "user@example.com"}]),
        patch("scripts.daily_scout_teams_bot.build_account_scout_settings", return_value=MagicMock()),
        patch("scripts.daily_scout_teams_bot.DailyScoutRunner", return_value=runner),
        patch("scripts.daily_scout_teams_bot.load_user_settings", return_value={}),
        patch("scripts.daily_scout_teams_bot._apply_schedule", return_value=(9, 0, "UTC")),
    ):
        runners: dict[str, Any] = {}
        schedules: dict[str, tuple[int, int, str]] = {}
        asyncio.run(_sync_account_runners(scheduler, runners, schedules))

    runner.run_once.assert_not_called()


def test_scheduled_run_is_attempted_once_per_day_and_records_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))
    settings = _make_settings(tmp_path)
    fixed_now = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)
    client = MagicMock()
    client.chat = MagicMock(side_effect=RuntimeError("llm exploded"))

    with (
        patch.object(DailyScoutRunner, "_disable_summarization_for_scouting"),
        patch.object(DailyScoutRunner, "_install_tavily_recency_patch"),
        patch.object(DailyScoutRunner, "_build_checkpointer", return_value=None),
        patch.object(DailyScoutRunner, "_today_utc", return_value=fixed_now),
        patch("scripts.daily_scout_teams_bot.DeerFlowClient", return_value=client),
    ):
        runner = DailyScoutRunner(settings)

        with pytest.raises(RuntimeError, match="llm exploded"):
            asyncio.run(runner.run_once(trigger="scheduled"))

        status = load_user_scout_status(settings.owner_email)
        assert status["scheduled"]["last_attempt_for_date"] == "2026-03-10"
        assert status["scheduled"]["last_status"] == "failed"
        assert status["scheduled"]["last_error"] == "llm exploded"

        with pytest.raises(DailyScoutAlreadyAttemptedError):
            asyncio.run(runner.run_once(trigger="scheduled"))

    assert client.chat.call_count == 1


def test_duplicate_manual_runs_are_rejected_while_first_run_is_active(tmp_path, monkeypatch):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))
    settings = _make_settings(tmp_path)
    fixed_now = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)

    def slow_chat(*args, **kwargs):
        time.sleep(0.2)
        return "report body"

    client_one = MagicMock()
    client_one.chat = MagicMock(side_effect=slow_chat)
    client_two = MagicMock()
    client_two.chat = MagicMock(side_effect=slow_chat)

    with (
        patch.object(DailyScoutRunner, "_disable_summarization_for_scouting"),
        patch.object(DailyScoutRunner, "_install_tavily_recency_patch"),
        patch.object(DailyScoutRunner, "_build_checkpointer", return_value=None),
        patch.object(DailyScoutRunner, "_today_utc", return_value=fixed_now),
        patch("scripts.daily_scout_teams_bot.DeerFlowClient", side_effect=[client_one, client_two]),
    ):
        runner_one = DailyScoutRunner(settings)
        runner_two = DailyScoutRunner(settings)

        async def scenario() -> str:
            first_task = asyncio.create_task(
                runner_one.run_once(
                    trigger="web-user-report",
                    report_prefix="user-report",
                    report_dir_override=tmp_path / "user-reports",
                    report_filename_override="user-report-one.md",
                )
            )
            await asyncio.sleep(0.05)

            with pytest.raises(DailyScoutAlreadyRunningError):
                await runner_two.run_once(
                    trigger="web-user-report",
                    report_prefix="user-report",
                    report_dir_override=tmp_path / "user-reports",
                    report_filename_override="user-report-two.md",
                )

            return await first_task

        assert asyncio.run(scenario()) == "report body"


def test_run_once_uses_async_agent_path_for_deerflow_client(tmp_path, monkeypatch):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))
    settings = _make_settings(tmp_path)
    fixed_now = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)

    class _FakeAgent:
        async def astream(self, state, config=None, context=None, stream_mode=None):
            assert state["messages"][0].content.startswith("Run daily scouting for topic")
            yield {"messages": [AIMessage(content="intermediate text")]}
            yield {"messages": [AIMessage(content="final async report")]}

    client = object.__new__(DeerFlowClient)
    client_any: Any = client
    client_any._agent = _FakeAgent()
    client_any._get_runnable_config = MagicMock(return_value={"configurable": {"thread_id": settings.scout_thread_id}})
    client_any._ensure_agent = MagicMock()

    with (
        patch.object(DailyScoutRunner, "_disable_summarization_for_scouting"),
        patch.object(DailyScoutRunner, "_install_tavily_recency_patch"),
        patch.object(DailyScoutRunner, "_build_checkpointer", return_value=None),
        patch.object(DailyScoutRunner, "_today_utc", return_value=fixed_now),
        patch("scripts.daily_scout_teams_bot.DeerFlowClient", return_value=client),
    ):
        runner = DailyScoutRunner(settings)
        report = asyncio.run(
            runner.run_once(
                trigger="web-user-report",
                report_prefix="user-report",
                report_dir_override=tmp_path / "user-reports",
                report_filename_override="user-report-one.md",
            )
        )

    assert report == "final async report"
    client_any._ensure_agent.assert_called_once()


def test_run_once_emails_rendered_report_when_smtp_is_configured(tmp_path, monkeypatch):
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DAILY_SCOUT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("DAILY_SCOUT_EMAIL_SMTP_PORT", "2525")
    monkeypatch.setenv("DAILY_SCOUT_EMAIL_SMTP_USERNAME", "")
    monkeypatch.setenv("DAILY_SCOUT_EMAIL_SMTP_PASSWORD", "")
    monkeypatch.setenv("DAILY_SCOUT_EMAIL_FROM", "daily-scout@example.com")
    monkeypatch.setenv("DAILY_SCOUT_EMAIL_SMTP_USE_TLS", "false")
    settings = _make_settings(tmp_path)
    fixed_now = datetime(2026, 3, 10, 9, 0, tzinfo=UTC)
    client = MagicMock()
    client.chat = MagicMock(return_value="# Summary\n\n- Item one")
    sent_messages: list[Any] = []
    original_import_module = importlib.import_module

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            assert host == "smtp.example.com"
            assert port == 2525
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self):
            raise AssertionError("starttls should not be called")

        def login(self, username, password):
            raise AssertionError("login should not be called")

        def send_message(self, message):
            sent_messages.append(message)

    def _fake_import_module(name: str):
        if name == "markdown":
            return SimpleNamespace(markdown=lambda text, extensions=None: "<h1>Summary</h1><ul><li>Item one</li></ul>")
        if name == "smtplib":
            return SimpleNamespace(SMTP=_FakeSMTP, SMTP_SSL=_FakeSMTP)
        return original_import_module(name)

    with (
        patch.object(DailyScoutRunner, "_disable_summarization_for_scouting"),
        patch.object(DailyScoutRunner, "_install_tavily_recency_patch"),
        patch.object(DailyScoutRunner, "_build_checkpointer", return_value=None),
        patch.object(DailyScoutRunner, "_today_utc", return_value=fixed_now),
        patch("scripts.daily_scout_teams_bot.importlib.import_module", side_effect=_fake_import_module),
        patch("scripts.daily_scout_teams_bot.DeerFlowClient", return_value=client),
    ):
        runner = DailyScoutRunner(settings)
        report = asyncio.run(
            runner.run_once(
                trigger="scheduled",
                report_filename_override="daily-scout-2026-03-10.md",
            )
        )

    assert report == "# Summary\n\n- Item one"
    assert len(sent_messages) == 1
    message = sent_messages[0]
    assert message["To"] == settings.owner_email
    assert message["From"] == "daily-scout@example.com"
    assert "daily-scout-2026-03-10.md" in message["Subject"]
    html_part = message.get_body(preferencelist=("html",))
    assert html_part is not None
    assert "<h1>Summary</h1>" in html_part.get_content()
    assert "<li>Item one</li>" in html_part.get_content()