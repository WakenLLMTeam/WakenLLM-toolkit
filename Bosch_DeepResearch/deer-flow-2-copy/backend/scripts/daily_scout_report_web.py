import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from scripts.daily_scout_accounts import (
    authenticate,
    create_account,
    create_session_token,
  DEFAULT_SCOUT_TIMEZONE,
    list_active_accounts,
    load_accounts,
    load_user_scout_status,
    load_user_settings,
    normalize_email,
    save_user_settings,
    session_cookie_name,
    update_account,
    user_display_name,
    user_generated_reports_dir,
    user_reports_dir,
    verify_session_token,
    normalize_scout_timezone,
)
from scripts.daily_scout_teams_bot import (
  DailyScoutAlreadyRunningError,
  DailyScoutRunner,
  _load_default_prompt_template,
  build_account_scout_settings,
)
from deerflow.rag import Resource, build_retriever

app = FastAPI(title="Daily Scout Reports")


def _normalize_rag_resources(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("uri") or "").strip()
        title = str(item.get("title") or uri).strip()
        description = str(item.get("description") or "").strip()
        if not uri:
            continue
        normalized.append({"uri": uri, "title": title, "description": description})
    return normalized


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


def _normalize_rag_metadata_filters(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for uri, condition in value.items():
        uri_text = str(uri or "").strip()
        if not uri_text:
            continue
        normalized_condition = _normalize_metadata_condition(condition)
        if normalized_condition is None:
            continue
        normalized[uri_text] = normalized_condition
    return normalized


def _list_rag_resources() -> list[dict[str, str]]:
    retriever = build_retriever()
    if retriever is None:
        return []
    try:
        resources = retriever.list_resources()
    except Exception:
        return []
    return [Resource.model_validate(resource).model_dump() for resource in resources]


def _account_payload(account: dict[str, Any]) -> dict[str, Any]:
    return {
        "email": account["email"],
        "is_admin": bool(account.get("is_admin")),
        "is_active": bool(account.get("is_active", True)),
        "created_at": account.get("created_at"),
        "updated_at": account.get("updated_at"),
    }


def _current_account(request: Request) -> dict[str, Any]:
    token = request.cookies.get(session_cookie_name())
    account = verify_session_token(token)
    if account is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return account


def _require_admin(request: Request) -> dict[str, Any]:
    account = _current_account(request)
    if not account.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return account


def _report_dir_for(owner_email: str, report_type: str) -> Path:
    if report_type == "daily":
        return user_reports_dir(owner_email)
    if report_type == "user":
        return user_generated_reports_dir(owner_email)
    raise HTTPException(status_code=400, detail="Invalid report type")


def _parse_report_items(report_dir: Path, pattern: str, prefix: str, owner_email: str, report_type: str) -> list[dict[str, str]]:
    if not report_dir.exists():
        return []

    items: list[dict[str, str]] = []
    for fp in sorted(report_dir.glob(pattern), reverse=True):
        if not fp.is_file():
            continue
        date_label = fp.stem.replace(prefix, "")
        items.append(
            {
                "filename": fp.name,
                "date": date_label,
                "owner_email": owner_email,
                "owner_label": user_display_name(owner_email),
                "report_type": report_type,
            }
        )
    return items


def _reports_for_owner(owner_email: str) -> dict[str, Any]:
    normalized = normalize_email(owner_email)
    scout_status = load_user_scout_status(normalized)
    return {
        "owner_email": normalized,
        "owner_label": user_display_name(normalized),
        "daily_reports": _parse_report_items(user_reports_dir(normalized), "daily-scout-*.md", "daily-scout-", normalized, "daily"),
        "user_reports": _parse_report_items(user_generated_reports_dir(normalized), "user-report-*.md", "user-report-", normalized, "user"),
        "scheduled_status": scout_status.get("scheduled", {}),
    }


def _compute_next_run_preview(
    run_hour: int,
    run_minute: int,
    timezone_value: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not 0 <= run_hour <= 23:
        raise ValueError("Hour must be between 0 and 23")
    if not 0 <= run_minute <= 59:
        raise ValueError("Minute must be between 0 and 59")

    timezone = normalize_scout_timezone(timezone_value)
    current_utc = now or datetime.now(UTC)
    local_now = current_utc.astimezone(ZoneInfo(timezone))
    next_local = local_now.replace(hour=run_hour, minute=run_minute, second=0, microsecond=0)
    if next_local <= local_now:
        next_local += timedelta(days=1)

    next_utc = next_local.astimezone(UTC)
    return {
        "timezone": timezone,
        "next_run_at": next_utc.isoformat(),
        "seconds_until": max(0, int((next_utc - current_utc).total_seconds())),
    }


def _issue_auth_response(account: dict[str, Any]) -> JSONResponse:
    response = JSONResponse({"ok": True, "account": _account_payload(account)})
    response.set_cookie(
        session_cookie_name(),
        create_session_token(account["email"]),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=14 * 24 * 60 * 60,
        path="/",
    )
    return response


@app.post("/api/auth/signup")
def api_signup(payload: dict[str, Any]):
    account = create_account(str(payload.get("email") or ""), str(payload.get("password") or ""), is_admin=False, is_active=True)
    return _issue_auth_response(account)


@app.post("/api/auth/login")
def api_login(payload: dict[str, Any]):
    account = authenticate(str(payload.get("email") or ""), str(payload.get("password") or ""))
    if account is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return _issue_auth_response(account)


@app.post("/api/auth/logout")
def api_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(session_cookie_name(), path="/")
    return response


@app.get("/api/auth/me")
def api_me(request: Request):
    return JSONResponse({"account": _account_payload(_current_account(request))})


@app.get("/api/accounts")
def api_accounts(request: Request):
    _require_admin(request)
    return JSONResponse({"accounts": [_account_payload(account) for account in load_accounts()]})


@app.post("/api/accounts")
def api_create_account(payload: dict[str, Any], request: Request):
    _require_admin(request)
    account = create_account(
        str(payload.get("email") or ""),
        str(payload.get("password") or ""),
        is_admin=bool(payload.get("is_admin", False)),
        is_active=bool(payload.get("is_active", True)),
    )
    return JSONResponse({"ok": True, "account": _account_payload(account)})


@app.patch("/api/accounts/{email}")
def api_update_account(email: str, payload: dict[str, Any], request: Request):
    _require_admin(request)
    account = update_account(
        email,
        password=str(payload.get("password")) if payload.get("password") else None,
        is_admin=payload.get("is_admin") if "is_admin" in payload else None,
        is_active=payload.get("is_active") if "is_active" in payload else None,
    )
    return JSONResponse({"ok": True, "account": _account_payload(account)})


@app.get("/api/settings")
def api_get_settings(request: Request):
    account = _current_account(request)
    settings = load_user_settings(account["email"])
    default_system_prompt = _load_default_prompt_template()
    effective_system_prompt = str(settings.get("system_prompt") or "").strip() or default_system_prompt
    return JSONResponse(
      {
        **settings,
        "default_system_prompt": default_system_prompt,
        "effective_system_prompt": effective_system_prompt,
        "has_system_prompt_override": bool(str(settings.get("system_prompt") or "").strip()),
      }
    )


@app.post("/api/settings/next-run-preview")
def api_next_run_preview(payload: dict[str, Any], request: Request):
    _current_account(request)
    try:
        preview = _compute_next_run_preview(
            int(payload.get("run_hour", 0)),
            int(payload.get("run_minute", 0)),
          payload.get("timezone", DEFAULT_SCOUT_TIMEZONE),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(preview)


@app.put("/api/settings")
def api_update_settings(payload: dict[str, Any], request: Request):
    account = _current_account(request)
    current = load_user_settings(account["email"])
    allowed = {
        "topic",
        "run_hour",
        "run_minute",
        "timezone",
        "lookback_days",
        "system_prompt",
        "rag_resources",
        "rag_metadata_filters",
    }
    merged = current.copy()
    merged.update({k: v for k, v in payload.items() if k in allowed})

    merged["run_hour"] = int(merged.get("run_hour", 9))
    merged["run_minute"] = int(merged.get("run_minute", 0))
    merged["lookback_days"] = max(1, int(merged.get("lookback_days", 1)))
    merged["topic"] = str(merged.get("topic", "")).strip() or current.get("topic", "AI and software engineering updates")
    merged["timezone"] = str(merged.get("timezone", DEFAULT_SCOUT_TIMEZONE)).strip() or DEFAULT_SCOUT_TIMEZONE
    merged["system_prompt"] = str(merged.get("system_prompt", ""))
    merged["rag_resources"] = _normalize_rag_resources(merged.get("rag_resources"))
    merged["rag_metadata_filters"] = _normalize_rag_metadata_filters(merged.get("rag_metadata_filters"))

    saved = save_user_settings(account["email"], merged)
    return JSONResponse({"ok": True, "settings": saved})


@app.get("/api/rag/resources")
def api_rag_resources(request: Request):
    _current_account(request)
    return JSONResponse({"resources": _list_rag_resources()})


@app.get("/api/rag/metadata-fields")
def api_rag_metadata_fields(uri: str, request: Request):
    _current_account(request)
    retriever = build_retriever()
    if retriever is None:
        return JSONResponse({"fields": []})
    try:
        fields = retriever.list_metadata_fields(uri)
    except Exception:
        fields = []
    return JSONResponse({"fields": [field.model_dump() for field in fields]})


@app.get("/api/rag/metadata-field-values")
def api_rag_metadata_field_values(uri: str, field_name: str, request: Request, limit: int = 100):
    _current_account(request)
    retriever = build_retriever()
    if retriever is None:
        return JSONResponse({"values": []})
    try:
        values = retriever.list_metadata_field_values(uri, field_name, limit)
    except Exception:
        values = []
    return JSONResponse({"values": values})


@app.post("/api/rag/metadata-match-count")
def api_rag_metadata_match_count(payload: dict[str, Any], request: Request):
    _current_account(request)
    uri = str(payload.get("uri") or "").strip()
    if not uri:
        return JSONResponse({"count": 0})

    retriever = build_retriever()
    if retriever is None:
        return JSONResponse({"count": 0})

    try:
        count = retriever.count_matching_documents(uri, _normalize_metadata_condition(payload.get("metadata_condition")))
    except Exception:
        count = 0
    return JSONResponse({"count": max(0, int(count))})


@app.get("/api/reports")
def api_reports(request: Request):
    account = _current_account(request)
    own = _reports_for_owner(account["email"])
    others: list[dict[str, Any]] = []
    if account.get("is_admin"):
        for other in list_active_accounts():
            if normalize_email(other["email"]) == normalize_email(account["email"]):
                continue
            others.append(_reports_for_owner(other["email"]))
    return JSONResponse({"viewer": _account_payload(account), "own": own, "others": others})


@app.get("/api/reports/content")
def api_report_content(owner_email: str, report_type: str, filename: str, request: Request):
    account = _current_account(request)
    owner_email = normalize_email(owner_email)
    if owner_email != normalize_email(account["email"]) and not account.get("is_admin"):
        raise HTTPException(status_code=403, detail="Forbidden")
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = _report_dir_for(owner_email, report_type) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return PlainTextResponse(path.read_text(encoding="utf-8"))


@app.post("/api/reports/user")
async def api_generate_user_report(payload: dict[str, Any], request: Request):
    account = _current_account(request)
    topic = str(payload.get("topic") or "").strip() or None
    start_date = str(payload.get("start_date") or "").strip() or None
    end_date = str(payload.get("end_date") or "").strip() or None
    lookback_days = payload.get("lookback_days")
    lookback_days_value = int(lookback_days) if lookback_days not in (None, "") else None
    system_prompt = str(payload.get("system_prompt") or "")
    rag_resources = _normalize_rag_resources(payload.get("rag_resources")) or None
    rag_metadata_filters = _normalize_rag_metadata_filters(payload.get("rag_metadata_filters")) or None

    runner = DailyScoutRunner(build_account_scout_settings(account["email"]))
    filename = f"user-report-{datetime.now(UTC).strftime('%Y-%m-%d-%H%M%S')}.md"
    try:
      report = await runner.run_once(
        trigger="web-user-report",
        topic_override=topic,
        start_date_override=start_date,
        end_date_override=end_date,
        lookback_days_override=lookback_days_value,
        system_prompt_override=system_prompt,
        rag_resources_override=rag_resources,
        rag_metadata_filters_override=rag_metadata_filters,
        report_prefix="user-report",
        report_dir_override=user_generated_reports_dir(account["email"]),
        report_filename_override=filename,
      )
    except DailyScoutAlreadyRunningError as exc:
      raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
      raise HTTPException(status_code=500, detail=f"User report generation failed: {exc}") from exc
    return JSONResponse({"ok": True, "filename": filename, "preview": report[:200]})


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Daily Scout Reports</title>
  <style>
.user-report-item {
  font-size: 12px;
  padding: 5px 8px;
  border-radius: 8px;
}
#manualGenerateEntry {
  width: fit-content;
  display: block;
  margin: 8px auto;

  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;

  border-radius: 999px;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  border: none;
  color: white;

  cursor: pointer;
  transition: all 0.18s ease;
}

#manualGenerateEntry:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 10px rgba(99,102,241,0.4);
}
button[data-resource-filter] {
  font-size: 20px;      /* makes the gear bigger */
  padding: 6px 8px;     /* optional: increases clickable area */
  border-radius: 8px;   /* keeps rounded corners */
  background: rgba(18, 32, 59, 0.9); /* optional: match row style */
  color: var(--accent); /* optional: accent color */
  transition: all 0.2s ease;
}

button[data-resource-filter]:hover {
  transform: scale(1.2);
  box-shadow: 0 2px 6px rgba(109,200,255,0.5);
}
    :root {
      --bg: #09111f;
      --panel: #12203b;
      --panel-2: #0e1830;
      --text: #edf3ff;
      --muted: #9db0d3;
      --line: #28416d;
      --accent: #6dc8ff;
      --accent-2: #80f0b7;
      --danger: #ff8d8d;
      --ok: #80f0b7;
      --sidebar-width: 430px;
    }
    * { box-sizing: border-box; }
    html, body {
      height: 100%;
      overflow: hidden;
    }
    body {
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(109, 200, 255, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(128, 240, 183, 0.12), transparent 24%),
        linear-gradient(180deg, #09111f 0%, #08101d 100%);
      height: 100vh;
    }
    .hidden { display: none !important; }
    .auth-shell, .app-shell { height: 100vh; }
    .auth-shell { display: grid; place-items: center; padding: 24px; }
    .auth-card {
      width: min(460px, 100%);
      background: rgba(18, 32, 59, 0.9);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
    }
    .auth-title { margin: 0 0 6px; font-size: 28px; }
    .auth-subtitle { margin: 0 0 18px; color: var(--muted); line-height: 1.5; }
    .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
    .tab {
      flex: 1;
      border: 1px solid var(--line);
      background: #0d1731;
      color: var(--text);
      padding: 10px 12px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
    }
    .tab.active { border-color: var(--accent); background: rgba(109, 200, 255, 0.12); }
    .form-status, .status { font-size: 12px; min-height: 16px; margin-top: 8px; }
    .form-status.error { color: var(--danger); }
    .form-status.ok, .status { color: var(--ok); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    input, textarea, button, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #0d1731;
      color: var(--text);
      padding: 10px 12px;
      font-size: 13px;
    }
    textarea { min-height: 92px; resize: vertical; }
    button {
      cursor: pointer;
      background: linear-gradient(135deg, rgba(109, 200, 255, 0.24), rgba(128, 240, 183, 0.18));
      font-weight: 600;
    }
    button:hover { border-color: var(--accent); }
    .app-shell { display: grid; grid-template-rows: auto 1fr; overflow: hidden; }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(10, 18, 33, 0.85);
      backdrop-filter: blur(10px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .topbar-title { margin: 0; font-size: 20px; }
    .topbar-meta { color: var(--muted); font-size: 13px; }
    .badge { display: inline-flex; align-items: center; padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; font-size: 12px; }
    .badge.admin { border-color: var(--accent-2); color: var(--accent-2); }
    .layout { display: grid; grid-template-columns: var(--sidebar-width) 8px 1fr; height: calc(100vh - 72px); min-height: 0; overflow: hidden; }
.sidebar { 
  border-right: none; 
  background: rgba(14, 24, 48, 0.96); 
  padding: 14px; 
  
  /* Critical for full-panel scrolling */
  height: 100%;
  overflow-y: auto;          /* The entire panel scrolls */
  overflow-x: hidden;
  display: block;            /* Standard block flow */
}
    .sidebar-splitter {
      cursor: col-resize;
      background: var(--line);
      transition: background 0.15s ease;
    }
    .sidebar-splitter:hover { background: var(--accent); }
    body.resizing, body.resizing * { cursor: col-resize; user-select: none; }
.sidebar-footer {
  position: static;          /* Removes absolute/fixed placement */
  margin-top: 24px;          /* Space between last list and Settings */
  padding-bottom: 20px;      /* Extra padding so it's not hugged against the edge */
  width: 100%;
}
    .content { padding: 24px; overflow: hidden; min-height: 0; }
    #reportView {
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    #settingsView,
    #manualGenerateView {
      height: 100%;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      padding-right: 4px;
    }
    .card { border: 1px solid var(--line); border-radius: 14px; background: rgba(18, 32, 59, 0.9); padding: 14px; margin-bottom: 12px; }
    .card h3 { margin: 0 0 10px; font-size: 14px; }
    .hint { color: var(--muted); font-size: 11px; line-height: 1.45; margin-top: 6px; }
    .section-title { margin: 14px 0 8px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .list { display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow: auto; }
    #myDailyList {
      max-height: 300px;
      overflow-y: auto;
    }
    .item { text-align: left; border: 1px solid var(--line); border-radius: 10px; background: #111d3b; color: var(--text); padding: 10px 12px; cursor: pointer; }
    .item.active { border-color: var(--accent); box-shadow: inset 0 0 0 1px rgba(109, 200, 255, 0.25); }
    .item small { display: block; color: var(--muted); margin-top: 4px; }
    .report-header { margin-bottom: 12px; }
    .report-title { font-size: 24px; font-weight: 700; margin: 0; }
    .report-meta { color: var(--muted); font-size: 13px; margin-top: 6px; }
    .report-body {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(15, 23, 48, 0.95);
      padding: 20px;
      line-height: 1.65;
      font-size: 14px;
      flex: 1;
      min-height: 0;
      overflow: auto;
    }
    .report-body h1, .report-body h2, .report-body h3, .report-body h4 { margin: 1.1em 0 0.45em; line-height: 1.25; }
    .report-body h1:first-child, .report-body h2:first-child, .report-body h3:first-child { margin-top: 0; }
    .report-body p, .report-body ul, .report-body ol, .report-body blockquote { margin: 0.75em 0; }
    .report-body ul, .report-body ol { padding-left: 1.4em; }
    .report-body li { margin: 0.25em 0; }
    .report-body code { background: rgba(255, 255, 255, 0.08); padding: 0.1em 0.35em; border-radius: 6px; }
    .report-body pre { background: rgba(255, 255, 255, 0.05); padding: 12px; border-radius: 10px; overflow: auto; }
    .report-body pre code { background: transparent; padding: 0; }
    .report-body a { color: var(--accent); }
    .report-body strong { color: #ffffff; }
    .report-body hr { border: 0; border-top: 1px solid var(--line); margin: 1.2em 0; }
    .modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(5, 10, 20, 0.65);
      backdrop-filter: blur(6px);
      z-index: 40;
    }
    .modal-panel {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: min(880px, calc(100% - 48px));
      max-height: min(80vh, 760px);
      display: flex;
      flex-direction: column;
      background: rgba(14, 24, 48, 0.98);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      gap: 14px;
      z-index: 41;
      box-shadow: 0 24px 60px rgba(0, 0, 0, 0.4);
    }
    .modal-header { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .modal-title { font-size: 18px; margin: 0; }
    .modal-close { width: auto; padding: 6px 10px; }
    .metadata-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
    .metadata-tab {
      width: auto;
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #0d1731;
      color: var(--text);
      font-size: 12px;
      cursor: pointer;
    }
    .metadata-tab.active { border-color: var(--accent); background: rgba(109, 200, 255, 0.15); }
    .metadata-tab .badge-dot {
      display: inline-block;
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--accent-2);
      margin-left: 6px;
    }
    .modal-body { overflow: auto; min-height: 0; }
    .modal-footer { display: flex; gap: 8px; justify-content: flex-end; flex-wrap: wrap; }
    .modal-footer .primary { background: rgba(109, 200, 255, 0.24); }
    body.modal-open { overflow: hidden; }
    .account-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: start;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: rgba(13, 23, 49, 0.8);
      margin-bottom: 8px;
    }
    .account-actions { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
    .account-actions button { width: auto; padding: 8px 10px; }
    .inline-input { max-width: 180px; }
    select[multiple] { min-height: 120px; }
    @media (max-width: 1180px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .sidebar-splitter { display: none; }
    }
  </style>
</head>
<body>
  <section id=\"authShell\" class=\"auth-shell hidden\">
    <div class=\"auth-card\">
      <h1 class=\"auth-title\">Daily Scout</h1>
      <p class=\"auth-subtitle\">Sign in to manage your own scouting schedule, reports, and RAG datasets. New users can sign up directly here.</p>
      <div class=\"tabs\">
        <button id=\"loginTab\" class=\"tab active\">Login</button>
        <button id=\"signupTab\" class=\"tab\">Sign Up</button>
      </div>
      <div id=\"loginPanel\">
        <label>Email</label>
        <input id=\"loginEmail\" type=\"email\" placeholder=\"name@example.com\" />
        <label style=\"margin-top:10px\">Password</label>
        <input id=\"loginPassword\" type=\"password\" placeholder=\"Password\" />
        <button id=\"loginButton\" style=\"margin-top:14px\">Login</button>
        <div id=\"loginStatus\" class=\"form-status\"></div>
      </div>
      <div id=\"signupPanel\" class=\"hidden\">
        <label>Email</label>
        <input id=\"signupEmail\" type=\"email\" placeholder=\"name@example.com\" />
        <label style=\"margin-top:10px\">Password</label>
        <input id=\"signupPassword\" type=\"password\" placeholder=\"At least 8 characters\" />
        <button id=\"signupButton\" style=\"margin-top:14px\">Create Account</button>
        <div id=\"signupStatus\" class=\"form-status\"></div>
      </div>
    </div>
  </section>

  <section id=\"appShell\" class=\"app-shell hidden\">
    <header class=\"topbar\">
      <div>
        <h1 class=\"topbar-title\">Bosch XC Daily News Scouting Agent</h1>
        <div id=\"viewerMeta\" class=\"topbar-meta\"></div>
      </div>
      <div style=\"display:flex; gap:10px; align-items:center; flex-wrap:wrap\">
        <span id=\"viewerBadge\" class=\"badge\"></span>
        <button id=\"logoutButton\" style=\"width:auto\">Logout</button>
      </div>
    </header>

    <div class=\"layout\">
      <aside class=\"sidebar\">
        <div class=\"section-title\">Daily Reports</div>
        <div id=\"dailyScoutStatus\" class=\"hint\" style=\"margin-bottom:10px\"></div>
        <div id=\"myDailyList\" class=\"list\"></div>

        <div class=\"section-title\">User Reports</div>
        <button id=\"userFolderToggle\" class=\"item\"><div>User Reports</div><small id=\"userFolderHint\">Click to expand</small></button>
        <div id=\"userFolderChildren\" class=\"list hidden\">
          <button id="manualGenerateEntry" class="generate-btn">
            ⚡ Scout Now!
          </button>
          <div id=\"myUserList\" class=\"list\"></div>
        </div>

        <div id=\"adminReportsSection\" class=\"hidden\">
          <div class=\"section-title\">All Users</div>
          <div id=\"otherUsersReports\"></div>
        </div>

        <div class=\"sidebar-footer\">
          <button id=\"settingsEntry\" class=\"item\">Settings</button>
        </div>
      </aside>

      <div id=\"sidebarSplitter\" class=\"sidebar-splitter\" role=\"separator\" aria-orientation=\"vertical\"></div>

      <main class=\"content\">
        <section id=\"reportView\">
          <div class=\"report-header\">
            <h1 id=\"reportTitle\" class=\"report-title\">Select a report</h1>
            <div id=\"reportMeta\" class=\"report-meta\"></div>
          </div>
          <div id=\"reportBody\" class=\"report-body\">Loading...</div>
        </section>

        <section id=\"settingsView\" class=\"hidden\">
          <div id=\"settingsTabs\" class=\"tabs hidden\" style=\"max-width:560px; margin-bottom:16px;\">
            <button id=\"dailySettingsTab\" class=\"tab active\">Daily Settings</button>
            <button id=\"accountManagementTab\" class=\"tab hidden\">Account Management</button>
          </div>

          <div id=\"dailySettingsPanel\">
            <div class=\"card\">
              <h3>Daily Settings</h3>
              <label>Scouting Topic</label>
              <input id=\"topic\" placeholder=\"Topic\" />
              <div class=\"grid2\" style=\"margin-top:8px\">
                <div><label>Hour (0-23)</label><input id=\"runHour\" type=\"number\" min=\"0\" max=\"23\" /></div>
                <div><label>Minute (0-59)</label><input id=\"runMinute\" type=\"number\" min=\"0\" max=\"59\" /></div>
              </div>
              <div class=\"grid2\" style=\"margin-top:8px\">
                <div><label>Timezone</label><input id=\"timezone\" placeholder=\"Asia/Shanghai\" /></div>
                <div><label>Lookback Days</label><input id=\"lookbackDays\" type=\"number\" min=\"1\" /></div>
              </div>
              <div id=\"schedulePreview\" class=\"hint\" style=\"margin-top:8px\"></div>
              <label style=\"margin-top:8px\">RAG Datasets</label>
              <div id=\"ragResourcesList\" class=\"list\"></div>
              <div class=\"hint\">Select multiple datasets. Use the gear button beside each selected dataset to configure extra meta-data filtering.</div>
              <label style=\"margin-top:8px\">Prompt Override</label>
              <div class=\"hint\">Shows the effective agent prompt from the shared markdown template unless you already have a saved override. Saving changes here stores them only for your account.</div>
              <textarea id=\"systemPrompt\" placeholder=\"Use {topic}, {start_date}, {end_date}, {publication_date_floor}\"></textarea>
              <button id=\"saveSettings\" style=\"margin-top:8px\">Save Settings</button>
              <div id=\"settingsStatus\" class=\"status\"></div>
            </div>

          </div>
          <div id=\"accountManagementPanel\" class=\"hidden\">
            <div id=\"adminCard\" class=\"card\">
              <h3>Account Management</h3>
              <label>New Account Email</label>
              <input id=\"adminNewEmail\" type=\"email\" placeholder=\"user@example.com\" />
              <label style=\"margin-top:8px\">Password</label>
              <input id=\"adminNewPassword\" type=\"password\" placeholder=\"Temporary password\" />
              <label style=\"margin-top:8px; display:flex; gap:8px; align-items:center;\"><input id=\"adminNewIsAdmin\" type=\"checkbox\" style=\"width:auto\" /> Admin privileges</label>
              <button id=\"adminCreateAccount\" style=\"margin-top:8px\">Create Account</button>
              <div id=\"adminStatus\" class=\"status\"></div>
              <div class=\"section-title\">Accounts</div>
              <div id=\"accountsList\"></div>
            </div>
          </div>
        </section>

        <section id=\"manualGenerateView\" class=\"hidden\">
          <div id=\"userGenerateCard\" class=\"card\">
            <h3>Generate User Report</h3>
            <label>Topic Override</label>
            <input id=\"userTopic\" placeholder=\"Optional custom topic\" />
            <div class=\"grid2\" style=\"margin-top:8px\">
              <div><label>Start Date</label><input id=\"userStart\" placeholder=\"2026-03-01\" /></div>
              <div><label>End Date</label><input id=\"userEnd\" placeholder=\"2026-03-05\" /></div>
            </div>
            <label style=\"margin-top:8px\">Prompt Override</label>
            <div class=\"hint\">Starts from the effective agent prompt. Changes here apply only to this manually generated report.</div>
            <textarea id=\"userPrompt\" placeholder=\"Optional custom prompt template\"></textarea>
            <button id=\"generateUser\" style=\"margin-top:8px\">Generate User Report</button>
            <div id=\"userStatus\" class=\"status\"></div>
          </div>
        </section>
      </main>
    </div>

    <div id=\"metadataFilterModal\" class=\"hidden\">
      <div id=\"metadataFilterOverlay\" class=\"modal-overlay\"></div>
      <div class=\"modal-panel\" role=\"dialog\" aria-modal=\"true\">
        <div class=\"modal-header\">
          <div>
            <h3 id=\"metadataFilterTitle\" class=\"modal-title\">Meta-data Filters</h3>
            <div id=\"metadataFilterHint\" class=\"hint\"></div>
          </div>
          <button id=\"metadataFilterClose\" class=\"modal-close\">Close</button>
        </div>
        <div id=\"metadataFilterTabs\" class=\"metadata-tabs\"></div>
        <div class=\"modal-body\">
          <div id=\"metadataFilterFields\"></div>
        </div>
        <div class=\"modal-footer\">
          <div id=\"metadataFilterStatus\" class=\"status\" style=\"margin-right:auto;\"></div>
          <button id=\"applyAllMetadataFilter\" style=\"width:auto\">Apply to All</button>
          <button id=\"clearMetadataFilter\" style=\"width:auto\">Clear Filter</button>
          <button id=\"applyMetadataFilter\" class=\"primary\" style=\"width:auto\">Apply Filter</button>
        </div>
      </div>
    </div>
  </section>

  <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>
  <script>
    const authShell = document.getElementById('authShell');
    const appShell = document.getElementById('appShell');
    const loginTab = document.getElementById('loginTab');
    const signupTab = document.getElementById('signupTab');
    const loginPanel = document.getElementById('loginPanel');
    const signupPanel = document.getElementById('signupPanel');
    const loginStatus = document.getElementById('loginStatus');
    const signupStatus = document.getElementById('signupStatus');
    const viewerMeta = document.getElementById('viewerMeta');
    const viewerBadge = document.getElementById('viewerBadge');
    const adminReportsSection = document.getElementById('adminReportsSection');
    const accountsList = document.getElementById('accountsList');
    const adminStatus = document.getElementById('adminStatus');
    const dailyScoutStatus = document.getElementById('dailyScoutStatus');
    const myDailyList = document.getElementById('myDailyList');
    const myUserList = document.getElementById('myUserList');
    const otherUsersReports = document.getElementById('otherUsersReports');
    const userFolderToggle = document.getElementById('userFolderToggle');
    const userFolderChildren = document.getElementById('userFolderChildren');
    const manualGenerateEntry = document.getElementById('manualGenerateEntry');
    const settingsEntry = document.getElementById('settingsEntry');
    const reportView = document.getElementById('reportView');
    const settingsView = document.getElementById('settingsView');
    const manualGenerateView = document.getElementById('manualGenerateView');
    const settingsTabs = document.getElementById('settingsTabs');
    const dailySettingsPanel = document.getElementById('dailySettingsPanel');
    const accountManagementPanel = document.getElementById('accountManagementPanel');
    const dailySettingsTab = document.getElementById('dailySettingsTab');
    const accountManagementTab = document.getElementById('accountManagementTab');
    const reportTitle = document.getElementById('reportTitle');
    const reportMeta = document.getElementById('reportMeta');
    const reportBody = document.getElementById('reportBody');
    const topicEl = document.getElementById('topic');
    const runHourEl = document.getElementById('runHour');
    const runMinuteEl = document.getElementById('runMinute');
    const timezoneEl = document.getElementById('timezone');
    const lookbackDaysEl = document.getElementById('lookbackDays');
    const schedulePreviewEl = document.getElementById('schedulePreview');
    const ragResourcesList = document.getElementById('ragResourcesList');
    const systemPromptEl = document.getElementById('systemPrompt');
    const settingsStatus = document.getElementById('settingsStatus');
    const metadataFilterModal = document.getElementById('metadataFilterModal');
    const metadataFilterOverlay = document.getElementById('metadataFilterOverlay');
    const metadataFilterClose = document.getElementById('metadataFilterClose');
    const metadataFilterTabs = document.getElementById('metadataFilterTabs');
    const metadataFilterTitle = document.getElementById('metadataFilterTitle');
    const metadataFilterHint = document.getElementById('metadataFilterHint');
    const metadataFilterFields = document.getElementById('metadataFilterFields');
    const metadataFilterStatus = document.getElementById('metadataFilterStatus');
    const applyAllMetadataFilter = document.getElementById('applyAllMetadataFilter');
    const userGenerateCard = document.getElementById('userGenerateCard');
    const userFolderHint = document.getElementById('userFolderHint');
    const userTopicEl = document.getElementById('userTopic');
    const userStartEl = document.getElementById('userStart');
    const userEndEl = document.getElementById('userEnd');
    const userPromptEl = document.getElementById('userPrompt');
    const userStatus = document.getElementById('userStatus');
    const layoutEl = document.querySelector('.layout');
    const splitterEl = document.getElementById('sidebarSplitter');
    const LAST_METADATA_TAB_KEY = 'dailyScoutLastMetadataTab';
    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function setupSidebarResize() {
      if (!layoutEl || !splitterEl) return;
      const minWidth = 260;
      const maxPadding = 320;
      let isDragging = false;

      const onMove = (event) => {
        if (!isDragging) return;
        const rect = layoutEl.getBoundingClientRect();
        const maxWidth = rect.width - maxPadding;
        const nextWidth = clamp(event.clientX - rect.left, minWidth, maxWidth);
        document.documentElement.style.setProperty('--sidebar-width', `${nextWidth}px`);
      };

      const stopDrag = () => {
        if (!isDragging) return;
        isDragging = false;
        document.body.classList.remove('resizing');
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', stopDrag);
      };

      splitterEl.addEventListener('mousedown', (event) => {
        event.preventDefault();
        isDragging = true;
        document.body.classList.add('resizing');
        window.addEventListener('mousemove', onMove);
        window.addEventListener('mouseup', stopDrag);
      });
    }

    const STRING_OPERATORS = [
      { value: 'eq', label: '=' },
      { value: 'ne', label: '≠' },
      { value: 'contains', label: 'contains' },
      { value: 'not contains', label: 'not contains' },
      { value: 'start with', label: 'start with' },
      { value: 'empty', label: 'empty' },
      { value: 'not empty', label: 'not empty' },
    ];
    const NUMBER_OPERATORS = [
      { value: 'eq', label: '=' },
      { value: 'ne', label: '≠' },
      { value: 'gt', label: '>' },
      { value: 'lt', label: '<' },
      { value: 'ge', label: '≥' },
      { value: 'le', label: '≤' },
      { value: 'empty', label: 'empty' },
      { value: 'not empty', label: 'not empty' },
    ];

    let currentAccount = null;
    let availableRagResources = [];
    let ragMetadataFilters = {};
    let metadataFieldsByUri = {};
    let activeMetadataUri = null;
    let metadataCountTimer = null;
    let schedulePreviewState = null;
    let schedulePreviewFetchTimer = null;
    let schedulePreviewRenderTimer = null;
    let defaultSystemPrompt = '';
    let savedSystemPrompt = '';
    let hasSystemPromptOverride = false;

    function setStatus(element, message, tone = 'ok') {
      element.textContent = message || '';
      element.className = tone === 'error' ? 'form-status error' : tone === 'ok' ? 'form-status ok' : 'status';
    }

    function showAuth(mode = 'login') {
      authShell.classList.remove('hidden');
      appShell.classList.add('hidden');
      loginPanel.classList.toggle('hidden', mode !== 'login');
      signupPanel.classList.toggle('hidden', mode !== 'signup');
      loginTab.classList.toggle('active', mode === 'login');
      signupTab.classList.toggle('active', mode === 'signup');
    }

    function showApp(account) {
      currentAccount = account;
      authShell.classList.add('hidden');
      appShell.classList.remove('hidden');
      viewerMeta.textContent = account.email;
      viewerBadge.textContent = account.is_admin ? 'Admin' : 'User';
      viewerBadge.className = account.is_admin ? 'badge admin' : 'badge';
      adminReportsSection.classList.toggle('hidden', !account.is_admin);
      accountManagementTab.classList.toggle('hidden', !account.is_admin);
      dailySettingsTab.classList.toggle('hidden', !account.is_admin);
      if (!account.is_admin) {
        activateSettingsPanel('daily');
      }
      updateSettingsTabsVisibility();
    }

    function operatorNeedsValue(operator) {
      return operator !== 'empty' && operator !== 'not empty';
    }

    function escapeHtml(value) {
      return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function formatDuration(totalSeconds) {
      const seconds = Math.max(0, Math.floor(totalSeconds || 0));
      const days = Math.floor(seconds / 86400);
      const hours = Math.floor((seconds % 86400) / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      if (days > 0) return `${days}d ${hours}h ${minutes}m`;
      if (hours > 0) return `${hours}h ${minutes}m`;
      if (minutes > 0) return `${minutes}m`;
      return 'less than a minute';
    }

    function renderSchedulePreview() {
      if (!schedulePreviewState || !schedulePreviewState.next_run_at) {
        schedulePreviewEl.textContent = 'Enter a schedule to preview the next scouting run.';
        schedulePreviewEl.style.color = '';
        return;
      }

      const nextRun = new Date(schedulePreviewState.next_run_at);
      const remainingSeconds = Math.max(0, Math.floor((nextRun.getTime() - Date.now()) / 1000));
      const timezoneLabel = schedulePreviewState.timezone || 'Asia/Shanghai';
      const formattedRun = nextRun.toLocaleString(undefined, {
        dateStyle: 'medium',
        timeStyle: 'short',
        timeZone: timezoneLabel,
      });
      schedulePreviewEl.textContent = `Next scouting: ${formattedRun} (${timezoneLabel}), in ${formatDuration(remainingSeconds)}.`;
      schedulePreviewEl.style.color = 'var(--accent)';
    }

    function scheduleSchedulePreviewRefresh() {
      if (schedulePreviewRenderTimer) {
        clearInterval(schedulePreviewRenderTimer);
      }
      renderSchedulePreview();
      schedulePreviewRenderTimer = setInterval(renderSchedulePreview, 30000);
    }

    async function refreshSchedulePreview() {
      const runHour = Number(runHourEl.value);
      const runMinute = Number(runMinuteEl.value);
      const timezone = timezoneEl.value.trim() || 'Asia/Shanghai';

      if (!Number.isInteger(runHour) || runHour < 0 || runHour > 23) {
        schedulePreviewState = null;
        schedulePreviewEl.textContent = 'Hour must be between 0 and 23.';
        schedulePreviewEl.style.color = 'var(--danger)';
        return;
      }

      if (!Number.isInteger(runMinute) || runMinute < 0 || runMinute > 59) {
        schedulePreviewState = null;
        schedulePreviewEl.textContent = 'Minute must be between 0 and 59.';
        schedulePreviewEl.style.color = 'var(--danger)';
        return;
      }

      try {
        schedulePreviewState = await requestJson('/api/settings/next-run-preview', {
          method: 'POST',
          body: JSON.stringify({
            run_hour: runHour,
            run_minute: runMinute,
            timezone,
          }),
        });
        scheduleSchedulePreviewRefresh();
      } catch (error) {
        schedulePreviewState = null;
        schedulePreviewEl.textContent = error.message;
        schedulePreviewEl.style.color = 'var(--danger)';
      }
    }

    function queueSchedulePreviewRefresh() {
      if (schedulePreviewFetchTimer) {
        clearTimeout(schedulePreviewFetchTimer);
      }
      schedulePreviewFetchTimer = setTimeout(() => {
        schedulePreviewFetchTimer = null;
        void refreshSchedulePreview();
      }, 200);
    }

    function findResourceCheckbox(uri) {
      return Array.from(ragResourcesList.querySelectorAll('input[data-resource-uri]')).find((node) => node.dataset.resourceUri === uri) || null;
    }

    function fieldOperatorSelect(fieldName) {
      return Array.from(metadataFilterFields.querySelectorAll('select[data-filter-operator]')).find((node) => node.dataset.filterOperator === fieldName) || null;
    }

    function fieldValueInput(fieldName) {
      return Array.from(metadataFilterFields.querySelectorAll('input[data-filter-value]')).find((node) => node.dataset.filterValue === fieldName) || null;
    }

    function selectedRagResourceUris() {
      return Array.from(ragResourcesList.querySelectorAll('input[data-resource-uri]:checked')).map((node) => node.dataset.resourceUri);
    }

    function selectedRagResources() {
      const selectedUris = new Set(selectedRagResourceUris());
      return availableRagResources.filter((resource) => selectedUris.has(resource.uri));
    }

    function selectedRagMetadataFilters() {
      const selectedUris = new Set(selectedRagResourceUris());
      const selected = {};
      Object.entries(ragMetadataFilters).forEach(([uri, condition]) => {
        if (!selectedUris.has(uri)) return;
        if (!condition || !Array.isArray(condition.conditions) || !condition.conditions.length) return;
        selected[uri] = condition;
      });
      return selected;
    }

    function updateSettingsTabsVisibility() {
      const visibleTabs = [dailySettingsTab, accountManagementTab].filter((node) => !node.classList.contains('hidden'));
      settingsTabs.classList.toggle('hidden', visibleTabs.length === 0);
    }

    function setMainView(mode, panel = 'daily', scrollTarget = null) {
      reportView.classList.toggle('hidden', mode !== 'report');
      settingsView.classList.toggle('hidden', mode !== 'settings');
      manualGenerateView.classList.toggle('hidden', mode !== 'generate');
      if (mode === 'settings') {
        activateSettingsPanel(panel);
        if (scrollTarget) {
          scrollTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    }

    function activateSettingsPanel(panel) {
      dailySettingsPanel.classList.toggle('hidden', panel !== 'daily');
      accountManagementPanel.classList.toggle('hidden', panel !== 'account');
      dailySettingsTab.classList.toggle('active', panel === 'daily');
      accountManagementTab.classList.toggle('active', panel === 'account');
    }

    async function requestJson(url, options = {}) {
      const res = await fetch(url, {
        credentials: 'same-origin',
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {}),
        },
      });
      const isJson = (res.headers.get('content-type') || '').includes('application/json');
      const body = isJson ? await res.json() : await res.text();
      if (!res.ok) {
        const detail = isJson ? (body.detail || body.message || 'Request failed') : body;
        throw new Error(detail || 'Request failed');
      }
      return body;
    }

    async function login() {
      try {
        const body = await requestJson('/api/auth/login', {
          method: 'POST',
          body: JSON.stringify({
            email: document.getElementById('loginEmail').value,
            password: document.getElementById('loginPassword').value,
          }),
        });
        setStatus(loginStatus, 'Login succeeded.', 'ok');
        await boot(body.account);
      } catch (error) {
        setStatus(loginStatus, error.message, 'error');
      }
    }

    async function signup() {
      try {
        const body = await requestJson('/api/auth/signup', {
          method: 'POST',
          body: JSON.stringify({
            email: document.getElementById('signupEmail').value,
            password: document.getElementById('signupPassword').value,
          }),
        });
        setStatus(signupStatus, 'Account created.', 'ok');
        await boot(body.account);
      } catch (error) {
        setStatus(signupStatus, error.message, 'error');
      }
    }

    async function logout() {
      await requestJson('/api/auth/logout', { method: 'POST', body: JSON.stringify({}) });
      currentAccount = null;
      showAuth('login');
    }

    function renderRagResources(selectedUris = []) {
      const selected = new Set(selectedUris);
      ragResourcesList.innerHTML = '';
      if (!availableRagResources.length) {
        const empty = document.createElement('div');
        empty.className = 'item';
        empty.textContent = 'No RAG datasets available';
        ragResourcesList.appendChild(empty);
        return;
      }

      availableRagResources.forEach((resource) => {
        const row = document.createElement('div');
        row.className = 'account-row';
        const hasFilter = Boolean(ragMetadataFilters[resource.uri] && ragMetadataFilters[resource.uri].conditions && ragMetadataFilters[resource.uri].conditions.length);
        row.innerHTML = `
          <div>
            <label style=\"display:flex; gap:8px; align-items:center; margin-bottom:0; color:inherit;\">
              <input type=\"checkbox\" data-resource-uri=\"${escapeHtml(resource.uri)}\" style=\"width:auto\" ${selected.has(resource.uri) ? 'checked' : ''} />
              <span>${escapeHtml(resource.title || resource.uri)}</span>
            </label>
            <div class=\"hint\">${escapeHtml(resource.description || resource.uri)}${hasFilter ? ' · Meta-data filter active' : ''}</div>
          </div>
          <div class=\"account-actions\">
            <button type=\"button\" data-resource-filter=\"${escapeHtml(resource.uri)}\">⚙</button>
          </div>
        `;
        ragResourcesList.appendChild(row);
      });

      ragResourcesList.querySelectorAll('button[data-resource-filter]').forEach((button) => {
        button.addEventListener('click', async () => {
          const uri = button.dataset.resourceFilter;
          const resource = availableRagResources.find((item) => item.uri === uri);
          if (!resource) return;
          const checkbox = findResourceCheckbox(uri);
          if (checkbox) checkbox.checked = true;
          await openMetadataFilter(resource);
        });
      });
    }

    async function loadRagResources(selectedUris = []) {
      const data = await requestJson('/api/rag/resources', { method: 'GET' });
      availableRagResources = data.resources || [];
      renderRagResources(selectedUris);
    }

    async function loadSettings() {
      const settings = await requestJson('/api/settings', { method: 'GET' });
      ragMetadataFilters = settings.rag_metadata_filters || {};
      topicEl.value = settings.topic || '';
      runHourEl.value = settings.run_hour ?? 9;
      runMinuteEl.value = settings.run_minute ?? 0;
      timezoneEl.value = settings.timezone || 'Asia/Shanghai';
      lookbackDaysEl.value = settings.lookback_days ?? 1;
      defaultSystemPrompt = settings.default_system_prompt || '';
      hasSystemPromptOverride = Boolean(settings.has_system_prompt_override);
      savedSystemPrompt = settings.effective_system_prompt || defaultSystemPrompt;
      systemPromptEl.value = savedSystemPrompt;
      userPromptEl.value = savedSystemPrompt;
      await loadRagResources((settings.rag_resources || []).map((item) => item.uri));
      await refreshSchedulePreview();
    }

    async function saveSettings() {
      try {
        const previousSavedSystemPrompt = savedSystemPrompt;
        const nextSystemPrompt = systemPromptEl.value === defaultSystemPrompt ? '' : systemPromptEl.value;
        await requestJson('/api/settings', {
          method: 'PUT',
          body: JSON.stringify({
            topic: topicEl.value,
            run_hour: Number(runHourEl.value),
            run_minute: Number(runMinuteEl.value),
            timezone: timezoneEl.value,
            lookback_days: Number(lookbackDaysEl.value),
            rag_resources: selectedRagResources(),
            rag_metadata_filters: selectedRagMetadataFilters(),
            system_prompt: nextSystemPrompt,
          }),
        });
        hasSystemPromptOverride = Boolean(nextSystemPrompt.trim());
        savedSystemPrompt = systemPromptEl.value;
        if (!userPromptEl.value || userPromptEl.value === previousSavedSystemPrompt) {
          userPromptEl.value = savedSystemPrompt;
        }
        settingsStatus.textContent = 'Saved. Scheduler will auto-apply.';
      } catch (error) {
        settingsStatus.textContent = error.message;
      }
    }

    async function generateUserReport() {
      try {
        setMainView('generate');
        userStatus.textContent = 'Generating...';
        await requestJson('/api/reports/user', {
          method: 'POST',
          body: JSON.stringify({
            topic: userTopicEl.value,
            start_date: userStartEl.value,
            end_date: userEndEl.value,
            lookback_days: null,
            rag_resources: selectedRagResources(),
            rag_metadata_filters: selectedRagMetadataFilters(),
            system_prompt: userPromptEl.value,
          }),
        });
        userStatus.textContent = 'User report generated.';
        await loadReports();
      } catch (error) {
        userStatus.textContent = error.message;
      }
    }

    function buildReportButton(item) {
      const button = document.createElement('button');
      button.className = item.report_type === 'user'
        ? 'item user-report-item'
        : 'item';
      button.innerHTML = `<div>${escapeHtml(item.date)}</div><small>${escapeHtml(item.owner_label)} · ${escapeHtml(item.filename)}</small>`;
      button.addEventListener('click', async () => {
        document.querySelectorAll('.item').forEach((node) => node.classList.remove('active'));
        button.classList.add('active');
        await loadReport(item);
      });
      return button;
    }

    function renderFlatList(container, items) {
      container.innerHTML = '';
      if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'item';
        empty.textContent = 'No reports yet';
        empty.style.opacity = '0.7';
        container.appendChild(empty);
        return;
      }
      items.forEach((item) => container.appendChild(buildReportButton(item)));
    }

    function renderAdminReports(groups) {
      otherUsersReports.innerHTML = '';
      if (!currentAccount || !currentAccount.is_admin) return;
      if (!groups.length) {
        const empty = document.createElement('div');
        empty.className = 'hint';
        empty.textContent = 'No other active users yet.';
        otherUsersReports.appendChild(empty);
        return;
      }

      groups.forEach((group) => {
        const folder = document.createElement('button');
        folder.className = 'item';
        folder.textContent = group.owner_label;
        const children = document.createElement('div');
        children.className = 'list hidden';
        renderFlatList(children, [...(group.daily_reports || []), ...(group.user_reports || [])]);
        folder.addEventListener('click', () => {
          children.classList.toggle('hidden');
        });
        otherUsersReports.appendChild(folder);
        otherUsersReports.appendChild(children);
      });
    }

    function renderScheduledStatus(status) {
      if (!status || !status.last_status) {
        dailyScoutStatus.textContent = 'Scheduled scout has not run yet.';
        dailyScoutStatus.style.color = '';
        return;
      }

      if (status.last_status === 'failed') {
        const completedAt = status.last_completed_at || status.last_attempt_at || 'unknown time';
        const detail = status.last_error || 'Unknown error';
        dailyScoutStatus.textContent = `Last scheduled run failed at ${completedAt}: ${detail}`;
        dailyScoutStatus.style.color = 'var(--danger)';
        return;
      }

      if (status.last_status === 'running') {
        dailyScoutStatus.textContent = `Scheduled scout is currently running. Started at ${status.last_attempt_at || 'unknown time'}.`;
        dailyScoutStatus.style.color = 'var(--accent)';
        return;
      }

      dailyScoutStatus.textContent = `Last scheduled run succeeded at ${status.last_success_at || status.last_completed_at || 'unknown time'}.`;
      dailyScoutStatus.style.color = 'var(--ok)';
    }

    async function loadReports() {
      const data = await requestJson('/api/reports', { method: 'GET' });
      renderScheduledStatus(data.own.scheduled_status || null);
      renderFlatList(myDailyList, data.own.daily_reports || []);
      renderFlatList(myUserList, data.own.user_reports || []);
      renderAdminReports(data.others || []);
      const firstOwn = (data.own.daily_reports || [])[0] || (data.own.user_reports || [])[0];
      if (firstOwn) {
        await loadReport(firstOwn);
      } else {
        reportTitle.textContent = 'No reports available';
        reportMeta.textContent = '';
        reportBody.textContent = 'Run the scout or generate a user report first.';
      }
    }

    async function loadReport(item) {
      setMainView('report');
      reportTitle.textContent = `${item.owner_label} · ${item.report_type === 'daily' ? 'Daily' : 'User'} Report · ${item.date}`;
      reportMeta.textContent = item.filename;
      reportBody.textContent = 'Loading report...';
      const query = `/api/reports/content?owner_email=${encodeURIComponent(item.owner_email)}&report_type=${encodeURIComponent(item.report_type)}&filename=${encodeURIComponent(item.filename)}`;
      try {
        const res = await fetch(query, { credentials: 'same-origin' });
        if (!res.ok) {
          reportBody.textContent = 'Failed to load report.';
          return;
        }
        const markdown = await res.text();
        reportBody.innerHTML = window.marked ? window.marked.parse(markdown, { breaks: true, gfm: true }) : markdown;
      } catch (error) {
        reportBody.textContent = error.message;
      }
    }

    async function loadAccounts() {
      if (!currentAccount || !currentAccount.is_admin) return;
      const data = await requestJson('/api/accounts', { method: 'GET' });
      accountsList.innerHTML = '';
      (data.accounts || []).forEach((account) => {
        const row = document.createElement('div');
        row.className = 'account-row';
        row.innerHTML = `
          <div>
            <div><strong>${escapeHtml(account.email)}</strong></div>
            <div class=\"hint\">${account.is_admin ? 'Admin' : 'User'} · ${account.is_active ? 'Active' : 'Disabled'}</div>
          </div>
          <div class=\"account-actions\">
            <button data-email=\"${escapeHtml(account.email)}\" data-action=\"toggle-admin\">${account.is_admin ? 'Revoke Admin' : 'Grant Admin'}</button>
            <button data-email=\"${escapeHtml(account.email)}\" data-action=\"toggle-active\">${account.is_active ? 'Disable' : 'Enable'}</button>
            <input data-email=\"${escapeHtml(account.email)}\" class=\"inline-input\" type=\"password\" placeholder=\"New password\" />
            <button data-email=\"${escapeHtml(account.email)}\" data-action=\"reset-password\">Reset Password</button>
          </div>
        `;
        accountsList.appendChild(row);
      });

      accountsList.querySelectorAll('button[data-action]').forEach((button) => {
        button.addEventListener('click', async () => {
          const email = button.dataset.email;
          const action = button.dataset.action;
          const row = button.closest('.account-row');
          const passwordInput = row.querySelector('input[type="password"]');
          const account = (data.accounts || []).find((item) => item.email === email);
          const payload = {};
          if (action === 'toggle-admin') payload.is_admin = !account.is_admin;
          if (action === 'toggle-active') payload.is_active = !account.is_active;
          if (action === 'reset-password') payload.password = passwordInput.value;
          try {
            await requestJson(`/api/accounts/${encodeURIComponent(email)}`, {
              method: 'PATCH',
              body: JSON.stringify(payload),
            });
            adminStatus.textContent = 'Account updated.';
            await loadAccounts();
            await loadReports();
          } catch (error) {
            adminStatus.textContent = error.message;
          }
        });
      });
    }

    async function createManagedAccount() {
      try {
        await requestJson('/api/accounts', {
          method: 'POST',
          body: JSON.stringify({
            email: document.getElementById('adminNewEmail').value,
            password: document.getElementById('adminNewPassword').value,
            is_admin: document.getElementById('adminNewIsAdmin').checked,
            is_active: true,
          }),
        });
        adminStatus.textContent = 'Account created.';
        document.getElementById('adminNewEmail').value = '';
        document.getElementById('adminNewPassword').value = '';
        document.getElementById('adminNewIsAdmin').checked = false;
        await loadAccounts();
      } catch (error) {
        adminStatus.textContent = error.message;
      }
    }

    async function fetchMetadataFields(uri) {
      if (metadataFieldsByUri[uri]) return metadataFieldsByUri[uri];
      const data = await requestJson(`/api/rag/metadata-fields?uri=${encodeURIComponent(uri)}`, { method: 'GET' });
      metadataFieldsByUri[uri] = data.fields || [];
      return metadataFieldsByUri[uri];
    }

    function operatorOptionsForField(field) {
      return field.type === 'number' ? NUMBER_OPERATORS : STRING_OPERATORS;
    }

    function findCondition(uri, fieldName) {
      const condition = ragMetadataFilters[uri];
      if (!condition || !Array.isArray(condition.conditions)) return null;
      return condition.conditions.find((item) => item.name === fieldName) || null;
    }

    function renderMetadataFilterEditor(uri) {
      const resource = availableRagResources.find((item) => item.uri === uri);
      const fields = metadataFieldsByUri[uri] || [];
      metadataFilterTitle.textContent = resource ? `Meta-data Filter · ${resource.title || resource.uri}` : 'Meta-data Filter';
      metadataFilterHint.textContent = fields.length ? 'These filters are applied in addition to the mandatory publication-date cutoff.' : 'No metadata fields are available for this dataset.';
      metadataFilterFields.innerHTML = '';
      if (!fields.length) return;

      fields.forEach((field) => {
        const existing = findCondition(uri, field.name);
        const options = operatorOptionsForField(field)
          .map((option) => `<option value=\"${escapeHtml(option.value)}\" ${existing && existing.comparison_operator === option.value ? 'selected' : ''}>${escapeHtml(option.label)}</option>`)
          .join('');
        const value = existing && existing.value != null ? String(existing.value) : '';
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <h3>${escapeHtml(field.name)}</h3>
          <div class=\"hint\">Type: ${escapeHtml(field.type || 'string')}</div>
          <div class=\"grid2\" style=\"margin-top:8px\">
            <div>
              <label>Operator</label>
              <select data-filter-operator=\"${escapeHtml(field.name)}\">${options}</select>
            </div>
            <div>
              <label>Value</label>
              <input data-filter-value=\"${escapeHtml(field.name)}\" value=\"${escapeHtml(value)}\" placeholder=\"Optional value\" />
            </div>
          </div>
        `;
        metadataFilterFields.appendChild(card);
      });

      Array.from(metadataFilterFields.querySelectorAll('select[data-filter-operator]')).forEach((select) => {
        const fieldName = select.dataset.filterOperator;
        const input = fieldValueInput(fieldName);
        const sync = () => {
          if (!input) return;
          input.disabled = !operatorNeedsValue(select.value);
          if (input.disabled) input.value = '';
          scheduleMetadataCount();
        };
        select.addEventListener('change', sync);
        sync();
      });

      Array.from(metadataFilterFields.querySelectorAll('input[data-filter-value]')).forEach((input) => {
        input.addEventListener('input', () => {
          scheduleMetadataCount();
        });
      });
    }

    function collectMetadataFilterEditorState() {
      if (!activeMetadataUri) return null;
      const fields = metadataFieldsByUri[activeMetadataUri] || [];
      const conditions = [];
      fields.forEach((field) => {
        const operator = fieldOperatorSelect(field.name)?.value || 'eq';
        const input = fieldValueInput(field.name);
        const rawValue = input ? input.value.trim() : '';
        if (operatorNeedsValue(operator) && !rawValue) return;
        const condition = {
          name: field.name,
          comparison_operator: operator,
        };
        if (operatorNeedsValue(operator)) {
          const normalizedValue = field.type === 'number' ? Number(rawValue) : rawValue;
          if (field.type === 'number' && Number.isNaN(normalizedValue)) {
            return;
          }
          condition.value = normalizedValue;
        }
        conditions.push(condition);
      });
      return conditions.length ? { logic: 'and', conditions } : null;
    }

    function renderMetadataTabs(resources) {
      metadataFilterTabs.innerHTML = '';
      resources.forEach((resource) => {
        const hasFilter = Boolean(ragMetadataFilters[resource.uri]?.conditions?.length);
        const button = document.createElement('button');
        button.className = 'metadata-tab' + (resource.uri === activeMetadataUri ? ' active' : '');
        button.innerHTML = `${escapeHtml(resource.title || resource.uri)}${hasFilter ? '<span class="badge-dot"></span>' : ''}`;
        button.addEventListener('click', async () => {
          activeMetadataUri = resource.uri;
          localStorage.setItem(LAST_METADATA_TAB_KEY, resource.uri);
          renderMetadataTabs(resources);
          await fetchMetadataFields(resource.uri);
          renderMetadataFilterEditor(resource.uri);
          scheduleMetadataCount();
        });
        metadataFilterTabs.appendChild(button);
      });
    }

    function closeMetadataFilter() {
      metadataFilterStatus.textContent = '';
      metadataFilterModal.classList.add('hidden');
      document.body.classList.remove('modal-open');
    }

    async function openMetadataFilter(resource) {
      const selected = selectedRagResources();
      if (!selected.length) return;
      const storedUri = localStorage.getItem(LAST_METADATA_TAB_KEY);
      const storedMatch = storedUri && selected.some((item) => item.uri === storedUri);
      activeMetadataUri = storedMatch ? storedUri : resource.uri;
      metadataFilterModal.classList.remove('hidden');
      document.body.classList.add('modal-open');
      metadataFilterStatus.textContent = '';
      renderMetadataTabs(selected);
      await fetchMetadataFields(activeMetadataUri);
      renderMetadataFilterEditor(activeMetadataUri);
      scheduleMetadataCount();
    }

    async function updateMetadataCount() {
      if (!activeMetadataUri) return;
      try {
        metadataFilterStatus.textContent = 'Counting matching documents...';
        const metadataCondition = collectMetadataFilterEditorState();
        const data = await requestJson('/api/rag/metadata-match-count', {
          method: 'POST',
          body: JSON.stringify({ uri: activeMetadataUri, metadata_condition: metadataCondition }),
        });
        metadataFilterStatus.textContent = `Matching files: ${data.count}`;
      } catch (error) {
        metadataFilterStatus.textContent = error.message;
      }
    }

    function scheduleMetadataCount() {
      if (metadataCountTimer) {
        clearTimeout(metadataCountTimer);
      }
      metadataCountTimer = setTimeout(() => {
        metadataCountTimer = null;
        void updateMetadataCount();
      }, 250);
    }

    function applyMetadataFilter() {
      if (!activeMetadataUri) return;
      const metadataCondition = collectMetadataFilterEditorState();
      if (metadataCondition) {
        ragMetadataFilters[activeMetadataUri] = metadataCondition;
        metadataFilterStatus.textContent = 'Filter applied. Save settings to persist it.';
      } else {
        delete ragMetadataFilters[activeMetadataUri];
        metadataFilterStatus.textContent = 'Filter cleared for this dataset.';
      }
      renderMetadataTabs(selectedRagResources());
      renderRagResources(selectedRagResourceUris());
      scheduleMetadataCount();
    }

    function applyMetadataFilterToAll() {
      if (!activeMetadataUri) return;
      const metadataCondition = collectMetadataFilterEditorState();
      const selected = selectedRagResources();
      selected.forEach((resource) => {
        if (metadataCondition) {
          ragMetadataFilters[resource.uri] = metadataCondition;
        } else {
          delete ragMetadataFilters[resource.uri];
        }
      });
      metadataFilterStatus.textContent = metadataCondition
        ? 'Filter applied to all selected datasets. Save settings to persist it.'
        : 'Filter cleared for all selected datasets.';
      renderMetadataTabs(selected);
      renderRagResources(selectedRagResourceUris());
      scheduleMetadataCount();
    }

    function clearMetadataFilter() {
      if (!activeMetadataUri) return;
      delete ragMetadataFilters[activeMetadataUri];
      renderMetadataFilterEditor(activeMetadataUri);
      renderMetadataTabs(selectedRagResources());
      renderRagResources(selectedRagResourceUris());
      metadataFilterStatus.textContent = 'Filter cleared for this dataset.';
      scheduleMetadataCount();
    }

    async function boot(account = null) {
      if (!account) {
        try {
          const me = await requestJson('/api/auth/me', { method: 'GET' });
          account = me.account;
        } catch (_) {
          showAuth('login');
          return;
        }
      }
      showApp(account);
      await loadSettings();
      await loadReports();
      if (account.is_admin) await loadAccounts();
    }

    loginTab.addEventListener('click', () => showAuth('login'));
    signupTab.addEventListener('click', () => showAuth('signup'));
    document.getElementById('loginButton').addEventListener('click', login);
    document.getElementById('signupButton').addEventListener('click', signup);
    document.getElementById('logoutButton').addEventListener('click', logout);
    document.getElementById('saveSettings').addEventListener('click', saveSettings);
    document.getElementById('generateUser').addEventListener('click', generateUserReport);
    document.getElementById('adminCreateAccount').addEventListener('click', createManagedAccount);
    document.getElementById('applyMetadataFilter').addEventListener('click', applyMetadataFilter);
    document.getElementById('clearMetadataFilter').addEventListener('click', clearMetadataFilter);
    runHourEl.addEventListener('input', queueSchedulePreviewRefresh);
    runMinuteEl.addEventListener('input', queueSchedulePreviewRefresh);
    timezoneEl.addEventListener('input', queueSchedulePreviewRefresh);
    applyAllMetadataFilter.addEventListener('click', applyMetadataFilterToAll);
    metadataFilterOverlay.addEventListener('click', closeMetadataFilter);
    metadataFilterClose.addEventListener('click', closeMetadataFilter);
    userFolderToggle.addEventListener('click', () => {
      const isHidden = userFolderChildren.classList.toggle('hidden');
      userFolderHint.textContent = isHidden ? 'Click to expand' : 'Click to collapse';
    });
    manualGenerateEntry.addEventListener('click', () => {
      if (!userPromptEl.value) {
        userPromptEl.value = savedSystemPrompt || defaultSystemPrompt;
      }
      setMainView('generate');
    });
    settingsEntry.addEventListener('click', () => setMainView('settings', 'daily'));
    dailySettingsTab.addEventListener('click', () => activateSettingsPanel('daily'));
    accountManagementTab.addEventListener('click', () => activateSettingsPanel('account'));
    setupSidebarResize();
    boot();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("DAILY_SCOUT_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("DAILY_SCOUT_WEB_PORT", "8088"))
    reload_enabled = os.getenv("DAILY_SCOUT_WEB_RELOAD", "false").lower() in {"1", "true", "yes", "on"}
    uvicorn.run("scripts.daily_scout_report_web:app", host=host, port=port, reload=reload_enabled)
