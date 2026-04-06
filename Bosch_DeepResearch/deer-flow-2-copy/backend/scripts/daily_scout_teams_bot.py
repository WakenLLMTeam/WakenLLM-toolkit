import asyncio
import importlib
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from langchain_core.messages import AIMessage, HumanMessage
from urllib.parse import urlparse
from tavily import TavilyClient
from tavily.errors import MissingAPIKeyError

from scripts.daily_scout_accounts import (
    DEFAULT_SCOUT_TIMEZONE,
    ensure_default_admin_account,
    list_active_accounts,
    load_user_scout_status,
    load_user_settings,
    normalize_scout_timezone,
    normalize_email,
    save_user_scout_status,
    user_checkpoint_path,
    user_reports_dir,
    user_settings_path,
    user_thread_id,
)
from scripts.daily_scout_email_template import (
    build_bosch_email_html,
    build_inline_asset_sources,
    load_bosch_inline_assets,
    render_markdown_html,
)
from deerflow.client import DeerFlowClient
from deerflow.config.paths import get_paths
from deerflow.config.summarization_config import SummarizationConfig, set_summarization_config

logger = logging.getLogger("daily_scout_teams_bot")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

_DEERFLOW_CLIENT_TYPE = DeerFlowClient
_TAVILY_PATCH_FLAG = "_deerflow_news_scout_patch_installed"
_TAVILY_CLIENT_PATCH_FLAG = "_deerflow_news_scout_client_patch_installed"
_REPORT_PATH_RE = re.compile(r"/mnt/user-data/outputs/[^\s`]+\.md")
_DEFAULT_PROMPT_PATH = Path(__file__).with_name("daily_scout_prompt.md")


def _load_default_prompt_template() -> str:
    try:
        template = _DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()
        if template:
            return template
        logger.warning("Daily scout prompt file %s is empty; using minimal fallback prompt", _DEFAULT_PROMPT_PATH)
    except Exception:
        logger.exception("Failed to read daily scout prompt file %s; using minimal fallback prompt", _DEFAULT_PROMPT_PATH)

    return (
        "You are the Daily Scout news analyst for topic: {topic}.\n\n"
        "Use only material published on or after {publication_date_floor}. Web search must stay within "
        "start_date={start_date} and end_date={end_date}.\n\n"
        "Return a concise, source-grounded report with publication dates, direct URLs, why each update matters, "
        "and one action recommendation per item. Do not fabricate facts."
    )

@dataclass
class ScoutSettings:
    owner_email: str
    topic: str
    rag_resources: list[dict]
    model_name: str | None
    thinking_enabled: bool
    run_hour: int
    run_minute: int
    timezone: str
    report_dir: Path
    checkpoint_mode: str
    checkpoint_path: Path
    scout_thread_id: str
    run_on_startup: bool
    settings_path: Path


class DailyScoutRunError(RuntimeError):
    pass


class DailyScoutAlreadyRunningError(DailyScoutRunError):
    pass


class DailyScoutAlreadyAttemptedError(DailyScoutRunError):
    pass


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _default_runtime_settings(base: ScoutSettings) -> dict[str, Any]:
    return {
        "topic": base.topic,
        "run_hour": base.run_hour,
        "run_minute": base.run_minute,
        "timezone": base.timezone,
        "lookback_days": 1,
        "system_prompt": "",
        "rag_resources": base.rag_resources,
    }


def _read_runtime_settings(base: ScoutSettings) -> dict[str, Any]:
    default_settings = _default_runtime_settings(base)
    path = base.settings_path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(default_settings, ensure_ascii=False, indent=2), encoding="utf-8")
        return default_settings

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse runtime settings file %s; using defaults", path)
        return default_settings

    if not isinstance(loaded, dict):
        return default_settings

    merged = default_settings.copy()
    merged.update({k: v for k, v in loaded.items() if v is not None})
    return merged


def build_account_scout_settings(email: str) -> ScoutSettings:
    runtime = load_user_settings(email)
    runtime_rag_resources = runtime.get("rag_resources")
    return ScoutSettings(
        owner_email=normalize_email(email),
        topic=str(runtime.get("topic") or os.getenv("DAILY_SCOUT_TOPIC", "AI and software engineering updates")),
        rag_resources=runtime_rag_resources if isinstance(runtime_rag_resources, list) else [],
        model_name=os.getenv("DAILY_SCOUT_MODEL"),
        thinking_enabled=os.getenv("DAILY_SCOUT_THINKING", "true").lower() == "true",
        run_hour=int(runtime.get("run_hour", os.getenv("DAILY_SCOUT_HOUR", "9"))),
        run_minute=int(runtime.get("run_minute", os.getenv("DAILY_SCOUT_MINUTE", "0"))),
        timezone=str(runtime.get("timezone", os.getenv("DAILY_SCOUT_TIMEZONE", DEFAULT_SCOUT_TIMEZONE))),
        report_dir=user_reports_dir(email),
        checkpoint_mode="none",
        checkpoint_path=user_checkpoint_path(email),
        scout_thread_id=user_thread_id(email),
        run_on_startup=os.getenv("DAILY_SCOUT_RUN_ON_STARTUP", "false").lower() == "true",
        settings_path=user_settings_path(email),
    )


class DailyScoutRunner:
    def __init__(self, settings: ScoutSettings):
        self.settings = settings
        self._checkpointer_cm = None
        self._disable_summarization_for_scouting()
        self._checkpointer = self._build_checkpointer()
        self.client = DeerFlowClient(
            model_name=settings.model_name,
            thinking_enabled=settings.thinking_enabled,
            checkpointer=self._checkpointer,
        )

    @staticmethod
    def _install_tavily_recency_patch() -> None:
        return None

    def _run_lock_path(self) -> Path:
        return self.settings.settings_path.parent / "daily_scout_run.lock"

    def _acquire_run_lock(self, *, trigger: str) -> Path:
        lock_path = self._run_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trigger": trigger,
            "owner_email": self.settings.owner_email,
            "created_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
        }
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise DailyScoutAlreadyRunningError("A scout run is already in progress for this account") from exc

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return lock_path

    @staticmethod
    def _release_run_lock(lock_path: Path | None) -> None:
        if lock_path is None:
            return
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to release daily scout run lock %s", lock_path)

    def _load_scout_status(self) -> dict[str, Any]:
        return load_user_scout_status(self.settings.owner_email)

    def _save_scout_status(self, status: dict[str, Any]) -> dict[str, Any]:
        return save_user_scout_status(self.settings.owner_email, status)

    def _thread_id_for_run(self, *, trigger: str, now: datetime) -> str:
        safe_trigger = re.sub(r"[^A-Za-z0-9_-]+", "-", trigger).strip("-") or "run"
        if trigger == "scheduled":
            return f"{self.settings.scout_thread_id}-scheduled-{now.date().isoformat()}"
        return f"{self.settings.scout_thread_id}-{safe_trigger}-{now.strftime('%Y%m%d%H%M%S%f')}"

    def _mark_scheduled_run_started(self, *, run_date: str, report_filename: str) -> None:
        status = self._load_scout_status()
        scheduled = status.get("scheduled", {})
        if scheduled.get("last_attempt_for_date") == run_date:
            raise DailyScoutAlreadyAttemptedError(
                f"Scheduled scout already attempted for {run_date}; skipping duplicate trigger"
            )

        status["scheduled"] = {
            **scheduled,
            "last_attempt_at": datetime.now(UTC).isoformat(),
            "last_attempt_for_date": run_date,
            "last_completed_at": None,
            "last_status": "running",
            "last_error": None,
            "last_report_filename": report_filename,
        }
        self._save_scout_status(status)

    def _mark_scheduled_run_finished(
        self,
        *,
        status_value: str,
        report_filename: str,
        error_message: str | None = None,
    ) -> None:
        status = self._load_scout_status()
        scheduled = status.get("scheduled", {})
        scheduled["last_completed_at"] = datetime.now(UTC).isoformat()
        scheduled["last_status"] = status_value
        scheduled["last_error"] = error_message
        scheduled["last_report_filename"] = report_filename
        if status_value == "succeeded":
            scheduled["last_success_at"] = scheduled["last_completed_at"]
        status["scheduled"] = scheduled
        self._save_scout_status(status)

    @staticmethod
    def _disable_summarization_for_scouting() -> None:
        set_summarization_config(SummarizationConfig(enabled=False))

    def _close_checkpointer(self) -> None:
        if self._checkpointer_cm is None:
            return
        try:
            self._checkpointer_cm.__exit__(None, None, None)
        except Exception:
            logger.exception("Failed to close checkpointer context")
        finally:
            self._checkpointer_cm = None

    def _build_checkpointer(self):
        mode = self.settings.checkpoint_mode.strip().lower()
        if mode != "none":
            logger.warning(
                "Daily scout ignores checkpoint mode '%s' and runs stateless to stay compatible with async execution",
                self.settings.checkpoint_mode,
            )
        else:
            logger.info("Daily scout checkpointer disabled; runs are stateless")
        return None

    @staticmethod
    def _today_utc() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _parse_iso_date(value: str) -> str:
        return datetime.fromisoformat(value).date().isoformat()

    @staticmethod
    def _dataset_id_from_resource_uri(uri: str) -> str | None:
        try:
            parsed = urlparse(uri)
        except Exception:
            return None
        if parsed.scheme != "rag":
            return None
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            return None
        return parts[1]

    @staticmethod
    def _normalize_metadata_condition(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        raw_conditions = value.get("conditions")
        if not isinstance(raw_conditions, list):
            return None

        conditions: list[dict[str, Any]] = []
        for item in raw_conditions:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            comparison_operator = str(item.get("comparison_operator") or "").strip()
            value_raw = item.get("value")
            if not name or not comparison_operator:
                continue
            if comparison_operator not in {"empty", "not empty"}:
                if value_raw is None:
                    continue
                if isinstance(value_raw, str):
                    value_raw = value_raw.strip()
                    if not value_raw:
                        continue
            condition: dict[str, Any] = {
                "name": name,
                "comparison_operator": comparison_operator,
            }
            if comparison_operator not in {"empty", "not empty"}:
                condition["value"] = value_raw
            conditions.append(condition)

        if not conditions:
            return None

        logic = "or" if str(value.get("logic") or "").lower() == "or" else "and"
        return {"logic": logic, "conditions": conditions}

    @classmethod
    def _normalize_rag_metadata_filters(cls, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for uri, condition in value.items():
            uri_text = str(uri or "").strip()
            if not uri_text:
                continue
            normalized_condition = cls._normalize_metadata_condition(condition)
            if normalized_condition is None:
                continue
            normalized[uri_text] = normalized_condition
        return normalized

    def _build_metadata_condition(
        self,
        publication_date_floor: str,
        rag_resources: list[dict],
        rag_metadata_filters: dict[str, dict[str, Any]] | None = None,
    ) -> dict:
        base_condition = {
            "logic": "and",
            "conditions": [
                {
                    "name": "Publication_Date",
                    "comparison_operator": "ge",
                    "value": publication_date_floor,
                }
            ],
        }

        normalized_filters = self._normalize_rag_metadata_filters(rag_metadata_filters)
        if not rag_resources:
            return base_condition

        per_dataset: dict[str, dict[str, Any]] = {}
        for resource in rag_resources:
            if not isinstance(resource, dict):
                continue
            uri = str(resource.get("uri") or "").strip()
            dataset_id = self._dataset_id_from_resource_uri(uri)
            if not dataset_id:
                continue
            dataset_condition = normalized_filters.get(uri)
            merged_conditions = list(base_condition["conditions"])
            if dataset_condition:
                merged_conditions.extend(dataset_condition.get("conditions", []))
            per_dataset[dataset_id] = {
                "logic": "and",
                "conditions": merged_conditions,
            }

        return {"per_dataset": per_dataset} if per_dataset else base_condition

    @classmethod
    def _set_tavily_date_window(cls, start_date: str, end_date: str, lookback_days: int) -> None:
        os.environ["DEER_FLOW_TAVILY_TOPIC"] = "general"
        os.environ["DEER_FLOW_TAVILY_START_DATE"] = start_date
        os.environ["DEER_FLOW_TAVILY_END_DATE"] = end_date
        os.environ.pop("DEER_FLOW_TAVILY_TIME_RANGE", None)
        os.environ["DEER_FLOW_TAVILY_SEARCH_DEPTH"] = "advanced"
        os.environ["DEER_FLOW_TAVILY_CHUNKS_PER_SOURCE"] = "3"
        os.environ["DEER_FLOW_TAVILY_MAX_RESULTS"] = "10"
        os.environ["DEER_FLOW_TAVILY_INCLUDE_ANSWER"] = "false"
        os.environ["DEER_FLOW_TAVILY_INCLUDE_RAW_CONTENT"] = "false"
        os.environ["DEER_FLOW_TAVILY_INCLUDE_IMAGES"] = "false"
        os.environ["DEER_FLOW_TAVILY_INCLUDE_IMAGE_DESCRIPTIONS"] = "false"
        os.environ["DEER_FLOW_TAVILY_INCLUDE_FAVICON"] = "false"
        os.environ["DEER_FLOW_TAVILY_AUTO_PARAMETERS"] = "false"
        os.environ["DEER_FLOW_TAVILY_EXACT_MATCH"] = "false"
        os.environ["DEER_FLOW_TAVILY_INCLUDE_USAGE"] = "false"

    @staticmethod
    def _normalize_rag_resources(value: Any, fallback: list[dict]) -> list[dict]:
        if not isinstance(value, list):
            return fallback
        normalized: list[dict] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            uri = str(item.get("uri") or "").strip()
            title = str(item.get("title") or uri).strip()
            description = str(item.get("description") or "").strip()
            if not uri:
                continue
            normalized.append({"uri": uri, "title": title, "description": description})
        return normalized or fallback

    def _candidate_output_dirs(self, thread_id: str) -> list[Path]:
        candidates = [get_paths().sandbox_outputs_dir(thread_id)]
        deduped: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            marker = str(path.resolve()) if path.exists() else str(path)
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(path)
        return deduped

    def _snapshot_output_files(self, thread_id: str) -> dict[str, float]:
        snapshot: dict[str, float] = {}
        for output_dir in self._candidate_output_dirs(thread_id):
            if not output_dir.exists():
                continue
            for path in output_dir.glob("*.md"):
                if path.is_file():
                    snapshot[str(path)] = path.stat().st_mtime
        return snapshot

    def _resolve_report_artifact_path(self, thread_id: str, response_text: str, before_snapshot: dict[str, float]) -> Path | None:
        matched_virtual_path = _REPORT_PATH_RE.search(response_text or "")
        if matched_virtual_path:
            filename = Path(matched_virtual_path.group(0)).name
            for output_dir in self._candidate_output_dirs(thread_id):
                candidate = output_dir / filename
                if candidate.exists() and candidate.is_file():
                    return candidate

        latest_path: Path | None = None
        latest_mtime = -1.0
        for output_dir in self._candidate_output_dirs(thread_id):
            if not output_dir.exists():
                continue
            for path in output_dir.glob("*.md"):
                if not path.is_file():
                    continue
                mtime = path.stat().st_mtime
                previous_mtime = before_snapshot.get(str(path))
                if previous_mtime is not None and mtime <= previous_mtime:
                    continue
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path
        return latest_path

    def _resolve_report_content(self, thread_id: str, response_text: str, before_snapshot: dict[str, float]) -> str:
        artifact_path = self._resolve_report_artifact_path(thread_id, response_text, before_snapshot)
        if artifact_path is not None:
            try:
                content = artifact_path.read_text(encoding="utf-8")
                if content.strip():
                    return content
            except Exception:
                logger.exception("Failed to read generated report artifact %s", artifact_path)
        return response_text

    def _build_prompt(
        self,
        *,
        topic: str,
        publication_date_floor: str,
        start_date: str,
        end_date: str,
        system_prompt: str,
    ) -> str:
        template = system_prompt.strip() or _load_default_prompt_template()
        if system_prompt.strip():
            try:
                return template.format(
                    topic=topic,
                    publication_date_floor=publication_date_floor,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception:
                logger.exception("Failed to format custom system prompt; falling back to default prompt")
                template = _load_default_prompt_template()

        return template.format(
            topic=topic,
            publication_date_floor=publication_date_floor,
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _email_delivery_config() -> dict[str, Any] | None:
        host = _env_text("DAILY_SCOUT_EMAIL_SMTP_HOST")
        from_address = _env_text("DAILY_SCOUT_EMAIL_FROM")
        if not host or not from_address:
            return None

        use_ssl = _env_flag("DAILY_SCOUT_EMAIL_SMTP_USE_SSL", False)
        use_tls = _env_flag("DAILY_SCOUT_EMAIL_SMTP_USE_TLS", not use_ssl)
        port_default = "465" if use_ssl else "587"
        username = _env_text("DAILY_SCOUT_EMAIL_SMTP_USERNAME") or None
        sender_address = _env_text("DAILY_SCOUT_EMAIL_SENDER") or username or parseaddr(from_address)[1]
        header_from = _env_text("DAILY_SCOUT_EMAIL_HEADER_FROM") or from_address
        return {
            "host": host,
            "port": int(_env_text("DAILY_SCOUT_EMAIL_SMTP_PORT", port_default)),
            "username": username,
            "password": os.getenv("DAILY_SCOUT_EMAIL_SMTP_PASSWORD", ""),
            "from_address": from_address,
            "header_from": header_from,
            "sender_address": sender_address or None,
            "reply_to": _env_text("DAILY_SCOUT_EMAIL_REPLY_TO") or None,
            "subject_prefix": _env_text("DAILY_SCOUT_EMAIL_SUBJECT_PREFIX", "[Daily Scout]") or "[Daily Scout]",
            "use_ssl": use_ssl,
            "use_tls": use_tls,
            "timeout": float(_env_text("DAILY_SCOUT_EMAIL_TIMEOUT_SECONDS", "30")),
        }

    def _build_email_message(
        self,
        *,
        report_text: str,
        report_name: str,
        topic: str,
        trigger: str,
    ) -> EmailMessage | None:
        config = self._email_delivery_config()
        if config is None:
            return None

        subject = f"{config['subject_prefix']} {topic} - {str(datetime.now(ZoneInfo(DEFAULT_SCOUT_TIMEZONE)).date())}"
        report_html = render_markdown_html(report_text)
        trigger_label = trigger.replace("-", " ").title()
        generated_at = datetime.now(ZoneInfo(DEFAULT_SCOUT_TIMEZONE)).strftime("%Y-%m-%d %H:%M %Z")
        assets = load_bosch_inline_assets()
        html_body = build_bosch_email_html(
            title=topic,
            recipient_email=self.settings.owner_email,
            trigger_label=trigger_label,
            generated_at=generated_at,
            report_html=report_html,
            report_name=report_name,
            asset_sources=build_inline_asset_sources(assets, mode="cid"),
        )

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = config["header_from"]
        sender_address = config["sender_address"]
        if sender_address and parseaddr(config["header_from"])[1].lower() != sender_address.lower():
            message["Sender"] = sender_address
        message["To"] = self.settings.owner_email
        if config["reply_to"]:
            message["Reply-To"] = config["reply_to"]
        message.set_content(report_text)
        message.add_alternative(html_body, subtype="html")
        html_part = message.get_body(preferencelist=("html",))
        if html_part is not None:
            for asset in assets.values():
                html_part.add_related(
                    asset.data,
                    maintype="image",
                    subtype=asset.mime_subtype,
                    cid=f"<{asset.content_id}>",
                    filename=asset.filename,
                    disposition="inline",
                )
        return message

    def _send_email_message(self, message: EmailMessage) -> None:
        config = self._email_delivery_config()
        if config is None:
            return

        smtplib_module = importlib.import_module("smtplib")
        smtp_factory = smtplib_module.SMTP_SSL if config["use_ssl"] else smtplib_module.SMTP
        with smtp_factory(config["host"], config["port"], timeout=config["timeout"]) as smtp_client:
            if config["use_tls"] and not config["use_ssl"]:
                smtp_client.starttls()
            username = config["username"]
            if username:
                smtp_client.login(username, config["password"])
            smtp_client.send_message(message)

    def _deliver_report_email(
        self,
        *,
        report_text: str,
        report_name: str,
        topic: str,
        trigger: str,
    ) -> None:
        message = self._build_email_message(
            report_text=report_text,
            report_name=report_name,
            topic=topic,
            trigger=trigger,
        )
        if message is None:
            return

        try:
            self._send_email_message(message)
            logger.info("Sent daily scout report email to %s for %s", self.settings.owner_email, report_name)
        except Exception:
            logger.exception("Failed to send daily scout report email to %s", self.settings.owner_email)

    async def _chat_with_deerflow_client(
        self,
        prompt: str,
        *,
        thread_id: str,
        rag_resources: list[dict],
        metadata_condition: dict[str, Any],
    ) -> str:
        config = self.client._get_runnable_config(
            thread_id,
            rag_resources=rag_resources,
            metadata_condition=metadata_condition,
        )
        self.client._ensure_agent(config)
        agent = self.client._agent
        if agent is None:
            raise RuntimeError("DeerFlow client agent failed to initialize")

        state: Any = {"messages": [HumanMessage(content=prompt)]}
        context = {"thread_id": thread_id}
        last_text = ""

        async for chunk in agent.astream(state, config=config, context=context, stream_mode="values"):
            for message in reversed(chunk.get("messages", [])):
                if not isinstance(message, AIMessage):
                    continue
                text = self.client._extract_text(message.content)
                if text:
                    last_text = text
                    break

        return last_text

    async def _invoke_client(
        self,
        prompt: str,
        *,
        thread_id: str,
        rag_resources: list[dict],
        metadata_condition: dict[str, Any],
    ) -> str:
        if isinstance(self.client, _DEERFLOW_CLIENT_TYPE):
            return await self._chat_with_deerflow_client(
                prompt,
                thread_id=thread_id,
                rag_resources=rag_resources,
                metadata_condition=metadata_condition,
            )

        return await asyncio.to_thread(
            self.client.chat,
            prompt,
            thread_id=thread_id,
            rag_resources=rag_resources,
            metadata_condition=metadata_condition,
        )

    async def run_once(
        self,
        *,
        trigger: str,
        topic_override: str | None = None,
        start_date_override: str | None = None,
        end_date_override: str | None = None,
        lookback_days_override: int | None = None,
        system_prompt_override: str | None = None,
        rag_resources_override: list[dict] | None = None,
        rag_metadata_filters_override: dict[str, dict[str, Any]] | None = None,
        report_prefix: str = "daily-scout",
        report_dir_override: Path | None = None,
        report_filename_override: str | None = None,
    ) -> str:
        runtime = _read_runtime_settings(self.settings)
        now = self._today_utc()
        today = now.date().isoformat()
        report_name = report_filename_override or f"{report_prefix}-{today}.md"
        run_thread_id = self._thread_id_for_run(trigger=trigger, now=now)
        lock_path: Path | None = None

        try:
            lock_path = self._acquire_run_lock(trigger=trigger)
            if trigger == "scheduled":
                self._mark_scheduled_run_started(run_date=today, report_filename=report_name)

            lookback_days = lookback_days_override
            if lookback_days is None:
                lookback_days = int(runtime.get("lookback_days", 1))
            lookback_days = max(1, int(lookback_days))

            computed_start = (now - timedelta(days=lookback_days)).date().isoformat()
            computed_end = now.date().isoformat()

            start_date = self._parse_iso_date(start_date_override) if start_date_override else computed_start
            end_date = self._parse_iso_date(end_date_override) if end_date_override else computed_end

            topic = (topic_override or str(runtime.get("topic", self.settings.topic))).strip() or self.settings.topic
            system_prompt = system_prompt_override if system_prompt_override is not None else str(runtime.get("system_prompt", ""))
            rag_resources = self._normalize_rag_resources(rag_resources_override, self.settings.rag_resources)
            if rag_resources_override is None:
                rag_resources = self._normalize_rag_resources(runtime.get("rag_resources"), self.settings.rag_resources)
            rag_metadata_filters = self._normalize_rag_metadata_filters(rag_metadata_filters_override)
            if rag_metadata_filters_override is None:
                rag_metadata_filters = self._normalize_rag_metadata_filters(runtime.get("rag_metadata_filters"))

            publication_date_floor = start_date

            self._set_tavily_date_window(start_date, end_date, lookback_days)
            metadata_condition = self._build_metadata_condition(publication_date_floor, rag_resources, rag_metadata_filters)
            prompt = self._build_prompt(
                topic=topic,
                publication_date_floor=publication_date_floor,
                start_date=start_date,
                end_date=end_date,
                system_prompt=system_prompt,
            )
            before_snapshot = self._snapshot_output_files(run_thread_id)

            logger.info(
                "Running scout trigger=%s topic=%s start_date=%s end_date=%s",
                trigger,
                topic,
                start_date,
                end_date,
            )
            report = await self._invoke_client(
                prompt,
                thread_id=run_thread_id,
                rag_resources=rag_resources,
                metadata_condition=metadata_condition,
            )
            report = self._resolve_report_content(run_thread_id, report, before_snapshot)

            report_dir = report_dir_override or self.settings.report_dir
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / report_name
            report_path.write_text(report, encoding="utf-8")
            self._deliver_report_email(
                report_text=report,
                report_name=report_name,
                topic=topic,
                trigger=trigger,
            )

            if trigger == "scheduled":
                self._mark_scheduled_run_finished(status_value="succeeded", report_filename=report_name)

            logger.info("Wrote daily scout report to %s", report_path)
            return report
        except Exception as exc:
            if trigger == "scheduled" and not isinstance(exc, (DailyScoutAlreadyAttemptedError, DailyScoutAlreadyRunningError)):
                self._mark_scheduled_run_finished(
                    status_value="failed",
                    report_filename=report_name,
                    error_message=str(exc),
                )
            raise
        finally:
            self._release_run_lock(lock_path)


def _job_id_for_email(email: str) -> str:
    digest = hashlib.sha1(email.encode("utf-8")).hexdigest()[:12]
    return f"daily_scout_job_{digest}"


def _apply_schedule(scheduler: AsyncIOScheduler, email: str, runner: DailyScoutRunner, settings_dict: dict[str, Any]) -> tuple[int, int, str]:
    hour = int(settings_dict.get("run_hour", runner.settings.run_hour))
    minute = int(settings_dict.get("run_minute", runner.settings.run_minute))
    timezone = normalize_scout_timezone(settings_dict.get("timezone", runner.settings.timezone))

    scheduler.add_job(
        _run_scheduled_job,
        trigger=CronTrigger(hour=hour, minute=minute, timezone=timezone),
        kwargs={"email": email, "runner": runner},
        id=_job_id_for_email(email),
        replace_existing=True,
        coalesce=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    return hour, minute, timezone


async def _run_scheduled_job(email: str, runner: DailyScoutRunner) -> None:
    try:
        await runner.run_once(trigger="scheduled")
    except DailyScoutAlreadyAttemptedError:
        logger.info("Scheduled scout already attempted today for %s; skipping", email)
    except DailyScoutAlreadyRunningError:
        logger.warning("Scheduled scout already running for %s; skipping duplicate trigger", email)
    except Exception:
        logger.exception("Scheduled scout run failed for %s", email)


async def _sync_account_runners(
    scheduler: AsyncIOScheduler,
    runners: dict[str, DailyScoutRunner],
    schedules: dict[str, tuple[int, int, str]],
) -> None:
    active_accounts = {item["email"]: item for item in list_active_accounts()}

    for email in list(runners):
        if email in active_accounts:
            continue
        try:
            scheduler.remove_job(_job_id_for_email(email))
        except Exception:
            pass
        runners[email]._close_checkpointer()
        del runners[email]
        schedules.pop(email, None)

    for email in sorted(active_accounts):
        if email not in runners:
            runners[email] = DailyScoutRunner(build_account_scout_settings(email))

        runner = runners[email]
        runtime = load_user_settings(email)
        desired_schedule = (
            int(runtime.get("run_hour", runner.settings.run_hour)),
            int(runtime.get("run_minute", runner.settings.run_minute)),
            normalize_scout_timezone(runtime.get("timezone", runner.settings.timezone)),
        )
        try:
            if schedules.get(email) != desired_schedule:
                schedules[email] = _apply_schedule(scheduler, email, runner, runtime)
                logger.info(
                    "Configured schedule for %s: daily %02d:%02d %s",
                    email,
                    schedules[email][0],
                    schedules[email][1],
                    schedules[email][2],
                )
        except Exception:
            logger.exception("Failed to configure schedule for %s", email)


async def main():
    ensure_default_admin_account()

    scheduler = AsyncIOScheduler(timezone=os.getenv("DAILY_SCOUT_TIMEZONE", DEFAULT_SCOUT_TIMEZONE))
    scheduler.start()
    runners: dict[str, DailyScoutRunner] = {}
    schedules: dict[str, tuple[int, int, str]] = {}

    logger.info("Starting scheduled news scouting service in multi-user mode")
    await _sync_account_runners(scheduler, runners, schedules)

    while True:
        await asyncio.sleep(30)
        await _sync_account_runners(scheduler, runners, schedules)


if __name__ == "__main__":
    asyncio.run(main())
