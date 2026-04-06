"""Live Tavily bulk check: Search → collect up to 50 URLs → Extract → infer ``YYYY-MM-DD``.

Uses real API credits. Skips when ``TAVILY_API_KEY`` / config key is missing.

Run (requires opt-in; charges API credits)::

    cd backend && PYTHONPATH=packages/harness TAVILY_API_KEY=tvly-... TAVILY_LIVE_PUBLICATION_BULK=1 \\
        uv run pytest tests/test_tavily_live_publication_dates_bulk.py -v --tb=short -m \"tavily_live and slow\"

Expectation: a **majority** of generic web results yield a non-empty calendar date after
``/extract`` (text). Some hosts (login walls, JS-only shells, paywalls) will legitimately
fail — the threshold is set conservatively and can be raised as extraction improves.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from deerflow.utils.publication_date import infer_publication_calendar_date


def _tavily_api_key() -> str | None:
    k = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if k:
        return k
    root = Path(__file__).resolve().parents[2]
    cfg = root / "config.yaml"
    if not cfg.is_file():
        return None
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    for t in data.get("tools") or []:
        if not isinstance(t, dict):
            continue
        if t.get("name") != "web_search":
            continue
        use = str(t.get("use") or "")
        if "tavily" not in use:
            continue
        key = t.get("api_key")
        if key is None:
            continue
        ks = str(key).strip()
        if ks.startswith("$"):
            continue
        return ks
    return None


requires_tavily = pytest.mark.skipif(
    not _tavily_api_key(),
    reason="Set TAVILY_API_KEY or add web_search tavily api_key in project config.yaml",
)

# Live corpus + Tavily extract quality drift; run only when explicitly opted in.
requires_tavily_bulk_opt_in = pytest.mark.skipif(
    (os.environ.get("TAVILY_LIVE_PUBLICATION_BULK") or "").strip().lower() not in ("1", "true", "yes"),
    reason="Set TAVILY_LIVE_PUBLICATION_BULK=1 to run this slow, API-costly bulk check",
)

# Tavily Search ``max_results`` is capped (commonly 20); Extract accepts up to 20 URLs/request.
_SEARCH_PER_QUERY = 10
_EXTRACT_BATCH = 20
_TARGET_UNIQUE_URLS = 50
_MIN_UNIQUE_URLS = 35
# After ``/extract``, require at least this fraction to yield ``YYYY-MM-DD``.
_MIN_NONEMPTY_RATE = 0.48

# Diverse English + one Chinese query to surface CSDN / regional blogs when indexed.
_SEARCH_QUERIES: list[str] = [
    "Tesla FSD full self driving explained 2024",
    "Python 3.13 new features official",
    "transformer neural network paper explained",
    "IPCC climate report summary site:un.org OR site:ipcc.ch",
    "Kubernetes getting started tutorial",
    "Rust programming language ownership",
    "TypeScript 5 handbook",
    "欧盟 AI 法案 解读 博客",
    "自动驾驶 激光雷达 技术 CSDN",
    "SpaceX Starship launch news site:nasa.gov OR site:space.com",
]


def _gather_unique_urls(client, *, target: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in _SEARCH_QUERIES:
        if len(out) >= target:
            break
        res = client.search(
            q,
            max_results=_SEARCH_PER_QUERY,
            include_raw_content="text",
        )
        for hit in res.get("results") or []:
            if not isinstance(hit, dict):
                continue
            u = str(hit.get("url") or "").strip()
            if not (u.startswith("http://") or u.startswith("https://")):
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
    return out[:target]


def _chunks(xs: list[str], n: int) -> list[list[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


@requires_tavily
@requires_tavily_bulk_opt_in
@pytest.mark.tavily_live
@pytest.mark.slow
def test_live_bulk_search_extract_publication_date_coverage() -> None:
    from tavily import TavilyClient

    key = _tavily_api_key()
    assert key
    client = TavilyClient(api_key=key)

    urls = _gather_unique_urls(client, target=_TARGET_UNIQUE_URLS)
    assert len(urls) >= _MIN_UNIQUE_URLS, (
        f"Tavily returned only {len(urls)} unique URLs (need >= {_MIN_UNIQUE_URLS}); "
        "try again later or add queries."
    )

    results: list[tuple[str, str]] = []
    for batch in _chunks(urls, _EXTRACT_BATCH):
        ext = client.extract(
            batch,
            format="text",
            extract_depth="basic",
            timeout=120,
        )
        failed_raw = ext.get("failed_results") or []
        failed_by_url: dict[str, str] = {}
        if isinstance(failed_raw, dict):
            for k, v in failed_raw.items():
                failed_by_url[str(k).rstrip("/")] = str(v)
                failed_by_url[str(k)] = str(v)
        else:
            for entry in failed_raw:
                if isinstance(entry, dict):
                    fu = (entry.get("url") or "").strip()
                    if fu:
                        msg = entry.get("error") or entry.get("message") or str(entry)
                        failed_by_url[fu] = str(msg)
                        failed_by_url[fu.rstrip("/")] = str(msg)
        by_url: dict[str, dict] = {}
        for item in ext.get("results") or []:
            if not isinstance(item, dict):
                continue
            u = (item.get("url") or "").strip()
            if u:
                by_url[u.rstrip("/")] = item
                by_url[u] = item
        for u in batch:
            item = by_url.get(u.rstrip("/")) or by_url.get(u)
            if not item:
                err = failed_by_url.get(u) or failed_by_url.get(u.rstrip("/")) or "missing"
                results.append((u, f"(extract failed: {err})"))
                continue
            raw = item.get("raw_content") or ""
            cal = (
                infer_publication_calendar_date(raw, source_url=u) if raw.strip() else ""
            )
            results.append((u, cal))

    nonempty = sum(1 for _, cal in results if cal and not cal.startswith("("))
    rate = nonempty / len(results)
    missing = [u for u, cal in results if not cal or cal.startswith("(")]

    assert rate >= _MIN_NONEMPTY_RATE, (
        f"infer_publication_calendar_date non-empty rate {rate:.1%} ({nonempty}/{len(results)}) "
        f"is below {_MIN_NONEMPTY_RATE:.0%}. First URLs without a date: {missing[:15]}"
    )
