import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_LOCK = threading.Lock()
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_COOKIE_NAME = "daily_scout_session"
_SESSION_TTL_DAYS = 14
_PASSWORD_ITERATIONS = 200000
_ACCOUNTS_VERSION = 1
_DEFAULT_ADMIN_EMAIL = "uyn4szh@bosch.com"
_DEFAULT_ADMIN_PASSWORD = "Hyz046867"
_TIMEZONE_ALIASES = {
    "Z": "UTC",
    "GMT": "UTC",
    "UCT": "UTC",
    "PRC": "Asia/Shanghai",
    "CHINA": "Asia/Shanghai",
    "CST": "Asia/Shanghai",
}

DEFAULT_SCOUT_TIMEZONE = "Asia/Shanghai"


def storage_root() -> Path:
    env_home = os.getenv("DEER_FLOW_HOME")
    if env_home:
        return Path(env_home).resolve()
    return (Path.cwd() / ".deer-flow-2").resolve()


def daily_scout_root() -> Path:
    return storage_root() / "daily-scout"


def accounts_file_path() -> Path:
    return daily_scout_root() / "accounts.json"


def session_secret_path() -> Path:
    return daily_scout_root() / "session_secret"


def users_root() -> Path:
    return daily_scout_root() / "users"


def legacy_accounts_file_path() -> Path:
    return storage_root() / "daily_scout_accounts.json"


def legacy_session_secret_path() -> Path:
    return storage_root() / "daily_scout_session_secret"


def legacy_users_root() -> Path:
    return storage_root() / "daily-scout-users"


def legacy_settings_path() -> Path:
    return storage_root() / "daily_scout_settings.json"


def legacy_reports_dir() -> Path:
    return Path(os.getenv("DAILY_SCOUT_REPORT_DIR", "./reports")).resolve()


def legacy_user_reports_dir() -> Path:
    return Path(os.getenv("DAILY_SCOUT_USER_REPORT_DIR", "./user-reports")).resolve()


def legacy_checkpoint_path() -> Path:
    return Path(os.getenv("DAILY_SCOUT_CHECKPOINT_PATH", "./.deer-flow-2/checkpoints/daily_scout.sqlite")).resolve()


def _move_legacy_path(source: Path, target: Path) -> None:
    if target.exists() or not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _migrate_daily_scout_storage() -> None:
    _move_legacy_path(legacy_accounts_file_path(), accounts_file_path())
    _move_legacy_path(legacy_session_secret_path(), session_secret_path())
    _move_legacy_path(legacy_users_root(), users_root())


def default_runtime_settings() -> dict[str, Any]:
    rag_resources: list[dict[str, Any]] = []
    try:
        loaded = json.loads(os.getenv("DAILY_SCOUT_RAG_RESOURCES", "[]"))
        if isinstance(loaded, list):
            rag_resources = loaded
    except Exception:
        rag_resources = []

    return {
        "topic": os.getenv("DAILY_SCOUT_TOPIC", "AI and software engineering updates"),
        "run_hour": int(os.getenv("DAILY_SCOUT_HOUR", "9")),
        "run_minute": int(os.getenv("DAILY_SCOUT_MINUTE", "0")),
        "timezone": normalize_scout_timezone(os.getenv("DAILY_SCOUT_TIMEZONE", DEFAULT_SCOUT_TIMEZONE)),
        "lookback_days": 1,
        "system_prompt": "",
        "rag_resources": rag_resources,
    }


def normalize_scout_timezone(value: Any, *, fallback: str = DEFAULT_SCOUT_TIMEZONE) -> str:
    raw_value = str(value or "").strip()
    candidate = raw_value or fallback
    candidate = _TIMEZONE_ALIASES.get(candidate.upper(), candidate)
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError:
        return fallback
    return candidate


def _normalize_runtime_settings_dict(settings: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    normalized = defaults.copy()
    normalized.update({k: v for k, v in settings.items() if v is not None})
    normalized["run_hour"] = int(normalized.get("run_hour", defaults["run_hour"]))
    normalized["run_minute"] = int(normalized.get("run_minute", defaults["run_minute"]))
    normalized["lookback_days"] = max(1, int(normalized.get("lookback_days", defaults["lookback_days"])))
    normalized["topic"] = str(normalized.get("topic", defaults["topic"])).strip() or defaults["topic"]
    normalized["timezone"] = normalize_scout_timezone(normalized.get("timezone", defaults["timezone"]))
    normalized["system_prompt"] = str(normalized.get("system_prompt", ""))
    return normalized


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or not _EMAIL_RE.match(normalized):
        raise ValueError("Invalid email address")
    return normalized


def account_slug(email: str) -> str:
    normalized = validate_email(email)
    base = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "user"
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{suffix}"


def user_root(email: str) -> Path:
    return users_root() / account_slug(email)


def user_settings_path(email: str) -> Path:
    return user_root(email) / "daily_scout_settings.json"


def user_reports_dir(email: str) -> Path:
    return user_root(email) / "reports"


def user_generated_reports_dir(email: str) -> Path:
    return user_root(email) / "user-reports"


def user_checkpoint_path(email: str) -> Path:
    return user_root(email) / "checkpoints" / "daily_scout.sqlite"


def user_scout_status_path(email: str) -> Path:
    return user_root(email) / "daily_scout_status.json"


def user_thread_id(email: str) -> str:
    digest = hashlib.sha256(validate_email(email).encode("utf-8")).hexdigest()[:16]
    return f"daily-scout-{digest}"


def user_display_name(email: str) -> str:
    return validate_email(email)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return loaded if isinstance(loaded, dict) else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_user_scout_status() -> dict[str, Any]:
    return {
        "scheduled": {
            "last_attempt_at": None,
            "last_attempt_for_date": None,
            "last_completed_at": None,
            "last_success_at": None,
            "last_status": None,
            "last_error": None,
            "last_report_filename": None,
        }
    }


def load_user_scout_status(email: str) -> dict[str, Any]:
    ensure_user_storage(email)
    defaults = default_user_scout_status()
    loaded = _read_json(user_scout_status_path(email), defaults)

    merged = defaults.copy()
    merged.update({k: v for k, v in loaded.items() if v is not None})

    scheduled_defaults = defaults["scheduled"].copy()
    scheduled_raw = loaded.get("scheduled")
    scheduled_loaded = scheduled_raw if isinstance(scheduled_raw, dict) else {}
    scheduled_defaults.update({k: v for k, v in scheduled_loaded.items() if v is not None})
    merged["scheduled"] = scheduled_defaults
    return merged


def save_user_scout_status(email: str, status: dict[str, Any]) -> dict[str, Any]:
    ensure_user_storage(email)
    merged = load_user_scout_status(email)
    merged.update({k: v for k, v in status.items() if v is not None})

    scheduled = status.get("scheduled") if isinstance(status.get("scheduled"), dict) else None
    if scheduled is not None:
        merged_scheduled = merged.get("scheduled", {}).copy()
        merged_scheduled.update({k: v for k, v in scheduled.items()})
        merged["scheduled"] = merged_scheduled

    _write_json(user_scout_status_path(email), merged)
    return merged


def _accounts_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_accounts = payload.get("accounts")
    if not isinstance(raw_accounts, list):
        return []
    return [account for account in raw_accounts if isinstance(account, dict)]


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PASSWORD_ITERATIONS,
    )
    encoded = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${_PASSWORD_ITERATIONS}${salt}${encoded}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt, encoded = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
    except Exception:
        return False

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    expected = base64.b64encode(digest).decode("ascii")
    return hmac.compare_digest(expected, encoded)


def ensure_default_admin_account() -> None:
    admin_email = normalize_email(os.getenv("DAILY_SCOUT_ADMIN_EMAIL", _DEFAULT_ADMIN_EMAIL))
    admin_password = os.getenv("DAILY_SCOUT_ADMIN_PASSWORD", _DEFAULT_ADMIN_PASSWORD)

    with _LOCK:
        _migrate_daily_scout_storage()
        payload = _read_json(accounts_file_path(), {"version": _ACCOUNTS_VERSION, "accounts": []})
        accounts = _accounts_from_payload(payload)
        found = None
        for account in accounts:
            if normalize_email(account.get("email", "")) == admin_email:
                found = account
                break

        if found is None:
            accounts.append(
                {
                    "email": admin_email,
                    "password_hash": hash_password(admin_password),
                    "is_admin": True,
                    "is_active": True,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                }
            )
        else:
            found["is_admin"] = True
            found.setdefault("is_active", True)
            found["updated_at"] = _now_iso()

        payload["version"] = _ACCOUNTS_VERSION
        payload["accounts"] = accounts
        _write_json(accounts_file_path(), payload)


def load_accounts() -> list[dict[str, Any]]:
    ensure_default_admin_account()
    payload = _read_json(accounts_file_path(), {"version": _ACCOUNTS_VERSION, "accounts": []})
    accounts = _accounts_from_payload(payload)
    return sorted(accounts, key=lambda item: normalize_email(item.get("email", "")))


def save_accounts(accounts: list[dict[str, Any]]) -> None:
    with _LOCK:
        _write_json(accounts_file_path(), {"version": _ACCOUNTS_VERSION, "accounts": accounts})


def get_account(email: str) -> dict[str, Any] | None:
    normalized = normalize_email(email)
    for account in load_accounts():
        if normalize_email(account.get("email", "")) == normalized:
            return account
    return None


def list_active_accounts() -> list[dict[str, Any]]:
    accounts = []
    for account in load_accounts():
        if account.get("is_active", True):
            ensure_user_storage(account["email"])
            accounts.append(account)
    return accounts


def create_account(email: str, password: str, *, is_admin: bool = False, is_active: bool = True) -> dict[str, Any]:
    normalized = validate_email(email)
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters")

    accounts = load_accounts()
    if any(normalize_email(item.get("email", "")) == normalized for item in accounts):
        raise ValueError("Account already exists")

    account = {
        "email": normalized,
        "password_hash": hash_password(password),
        "is_admin": bool(is_admin),
        "is_active": bool(is_active),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    accounts.append(account)
    save_accounts(accounts)
    ensure_user_storage(normalized)
    return account


def update_account(
    email: str,
    *,
    password: str | None = None,
    is_admin: bool | None = None,
    is_active: bool | None = None,
) -> dict[str, Any]:
    normalized = validate_email(email)
    accounts = load_accounts()
    for account in accounts:
        if normalize_email(account.get("email", "")) != normalized:
            continue
        if password is not None:
            if len(password) < 8:
                raise ValueError("Password must be at least 8 characters")
            account["password_hash"] = hash_password(password)
        if is_admin is not None:
            account["is_admin"] = bool(is_admin)
        if is_active is not None:
            account["is_active"] = bool(is_active)
        account["updated_at"] = _now_iso()
        save_accounts(accounts)
        ensure_user_storage(normalized)
        return account
    raise ValueError("Account not found")


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    account = get_account(email)
    if not account or not account.get("is_active", True):
        return None
    if not verify_password(password, str(account.get("password_hash", ""))):
        return None
    ensure_user_storage(account["email"])
    return account


def ensure_user_storage(email: str) -> None:
    _migrate_daily_scout_storage()
    normalized = validate_email(email)
    root = user_root(normalized)
    user_reports_dir(normalized).mkdir(parents=True, exist_ok=True)
    user_generated_reports_dir(normalized).mkdir(parents=True, exist_ok=True)
    user_checkpoint_path(normalized).parent.mkdir(parents=True, exist_ok=True)

    settings_path = user_settings_path(normalized)
    if not settings_path.exists():
        default_settings = default_runtime_settings()
        legacy = legacy_settings_path()
        if legacy.exists():
            try:
                loaded = json.loads(legacy.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    default_settings.update({k: v for k, v in loaded.items() if v is not None})
            except Exception:
                pass
        _write_json(settings_path, default_settings)

    account = get_account(normalized)
    if account and account.get("is_admin"):
        _copy_legacy_reports_if_needed(normalized)
        _copy_legacy_checkpoint_if_needed(normalized)


def _copy_legacy_reports_if_needed(email: str) -> None:
    if not user_reports_dir(email).exists() or not any(user_reports_dir(email).glob("*.md")):
        for path in legacy_reports_dir().glob("daily-scout-*.md"):
            target = user_reports_dir(email) / path.name
            if path.is_file() and not target.exists():
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    if not user_generated_reports_dir(email).exists() or not any(user_generated_reports_dir(email).glob("*.md")):
        for path in legacy_user_reports_dir().glob("user-report-*.md"):
            target = user_generated_reports_dir(email) / path.name
            if path.is_file() and not target.exists():
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _copy_legacy_checkpoint_if_needed(email: str) -> None:
    legacy = legacy_checkpoint_path()
    target = user_checkpoint_path(email)
    if legacy.exists() and not target.exists():
        target.write_bytes(legacy.read_bytes())


def load_user_settings(email: str) -> dict[str, Any]:
    ensure_user_storage(email)
    defaults = default_runtime_settings()
    loaded = _read_json(user_settings_path(email), defaults)
    return _normalize_runtime_settings_dict(loaded, defaults)


def save_user_settings(email: str, settings: dict[str, Any]) -> dict[str, Any]:
    ensure_user_storage(email)
    defaults = default_runtime_settings()
    merged = _normalize_runtime_settings_dict(settings, defaults)
    _write_json(user_settings_path(email), merged)
    return merged


def _get_session_secret() -> bytes:
    _migrate_daily_scout_storage()
    path = session_secret_path()
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    path.write_bytes(secret)
    return secret


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode((raw + padding).encode("ascii"))


def session_cookie_name() -> str:
    return _COOKIE_NAME


def create_session_token(email: str) -> str:
    payload = {
        "email": validate_email(email),
        "exp": int((datetime.now(UTC) + timedelta(days=_SESSION_TTL_DAYS)).timestamp()),
        "nonce": secrets.token_hex(8),
    }
    payload_raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_token = _b64url_encode(payload_raw)
    signature = hmac.new(_get_session_secret(), payload_token.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_token}.{_b64url_encode(signature)}"


def verify_session_token(token: str | None) -> dict[str, Any] | None:
    if not token or "." not in token:
        return None
    payload_token, signature_token = token.split(".", 1)
    expected_signature = hmac.new(_get_session_secret(), payload_token.encode("ascii"), hashlib.sha256).digest()
    try:
        actual_signature = _b64url_decode(signature_token)
    except Exception:
        return None
    if not hmac.compare_digest(expected_signature, actual_signature):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_token).decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    exp = int(payload.get("exp", 0))
    if exp <= int(datetime.now(UTC).timestamp()):
        return None
    account = get_account(str(payload.get("email") or ""))
    if not account or not account.get("is_active", True):
        return None
    return account
