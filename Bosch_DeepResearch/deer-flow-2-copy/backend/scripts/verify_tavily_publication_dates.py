#!/usr/bin/env python3
"""Live Tavily check: search + publication-date extraction (YYYY-MM-DD).

Requires ``TAVILY_API_KEY`` in the environment.

Usage (from ``backend/``)::

    PYTHONPATH=packages/harness TAVILY_API_KEY=tvly-... \\
        uv run python scripts/verify_tavily_publication_dates.py

Optional args: query string and max_results as argv.

Bulk mode (same logic as ``tests/test_tavily_live_publication_dates_bulk.py``)::

    uv run python scripts/verify_tavily_publication_dates.py bulk 50
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# backend/ → packages/harness on PYTHONPATH (see usage above)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _BACKEND_ROOT / "packages" / "harness"
if _HARNESS.is_dir() and str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))


def _enrich_missing(
    client,
    rows: list[dict],
    *,
    max_urls: int,
    extract_format: str,
    timeout: float,
) -> None:
    from deerflow.utils.publication_date import infer_publication_calendar_date

    if max_urls <= 0:
        return
    pending: list[tuple[int, str]] = []
    for i, row in enumerate(rows):
        if (row.get("published_date") or "").strip():
            continue
        u = (row.get("url") or "").strip()
        if u.startswith("http://") or u.startswith("https://"):
            pending.append((i, u))
    if not pending:
        return
    batch = pending[:max_urls]
    urls = [u for _, u in batch]
    ext = client.extract(
        urls,
        format=extract_format if extract_format in ("markdown", "text") else "text",
        extract_depth="basic",
        timeout=timeout,
    )
    by_norm: dict[str, dict] = {}
    for item in ext.get("results") or []:
        if not isinstance(item, dict):
            continue
        u = (item.get("url") or "").strip()
        if not u:
            continue
        by_norm[u.rstrip("/")] = item
        by_norm[u] = item
    for idx, url in batch:
        item = by_norm.get(url.rstrip("/")) or by_norm.get(url)
        if not item:
            continue
        raw = item.get("raw_content") or ""
        pub = infer_publication_calendar_date(raw, source_url=url) if raw else ""
        if pub:
            rows[idx]["published_date"] = pub


def _resolve_api_key() -> str:
    k = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if k:
        return k
    cfg = _BACKEND_ROOT.parent / "config.yaml"
    if cfg.is_file():
        os.environ.setdefault("DEER_FLOW_CONFIG_PATH", str(cfg))
        try:
            from deerflow.config import get_app_config

            tc = get_app_config().get_tool_config("web_search")
            if tc and tc.model_extra:
                return str(tc.model_extra.get("api_key") or "").strip()
        except Exception:
            pass
    return ""


def _run_bulk(client, target: int, min_rate: float) -> int:
    from deerflow.utils.publication_date import infer_publication_calendar_date

    queries = [
        "Tesla FSD full self driving explained 2024",
        "Python 3.13 new features official",
        "transformer neural network paper explained",
        "Kubernetes getting started tutorial",
        "Rust programming language ownership",
        "TypeScript 5 handbook",
        "自动驾驶 技术 博客 CSDN",
        "SpaceX Starship news site:nasa.gov",
    ]
    seen: set[str] = set()
    urls: list[str] = []
    for q in queries:
        if len(urls) >= target:
            break
        res = client.search(q, max_results=10, include_raw_content="text")
        for hit in res.get("results") or []:
            if not isinstance(hit, dict):
                continue
            u = str(hit.get("url") or "").strip()
            if u.startswith("http") and u not in seen:
                seen.add(u)
                urls.append(u)
    urls = urls[:target]
    print(f"Collected {len(urls)} unique URLs (target {target})\n")

    batch_size = 20
    rows: list[tuple[str, str]] = []
    for i in range(0, len(urls), batch_size):
        batch = urls[i : i + batch_size]
        ext = client.extract(batch, format="text", extract_depth="basic", timeout=120)
        failed_raw = ext.get("failed_results") or []
        failed_by_url: dict[str, str] = {}
        if isinstance(failed_raw, dict):
            for k, v in failed_raw.items():
                failed_by_url[str(k).rstrip("/")] = str(v)
        else:
            for entry in failed_raw:
                if isinstance(entry, dict) and entry.get("url"):
                    fu = str(entry["url"]).strip()
                    failed_by_url[fu] = str(entry.get("error", entry))
        by_u: dict[str, dict] = {}
        for item in ext.get("results") or []:
            if isinstance(item, dict) and item.get("url"):
                u = str(item["url"]).strip()
                by_u[u] = item
                by_u[u.rstrip("/")] = item
        for u in batch:
            it = by_u.get(u) or by_u.get(u.rstrip("/"))
            if not it:
                rows.append((u, ""))
                continue
            raw = it.get("raw_content") or ""
            rows.append(
                (u, infer_publication_calendar_date(raw, source_url=u) if raw.strip() else "")
            )

    ok = sum(1 for _, d in rows if d)
    rate = ok / len(rows) if rows else 0.0
    for i, (u, d) in enumerate(rows, 1):
        print(f"[{i:2}] {d or '(none)':12}  {u[:90]}")
    print(f"\nWith YYYY-MM-DD: {ok}/{len(rows)} ({rate:.1%})  (min required {min_rate:.0%})")
    return 0 if rate >= min_rate else 1


def main() -> int:
    api_key = _resolve_api_key()
    if not api_key:
        print(
            "Error: set TAVILY_API_KEY or web_search.api_key in config.yaml (project root).",
            file=sys.stderr,
        )
        return 1

    from tavily import TavilyClient

    from deerflow.utils.publication_date import infer_publication_calendar_date

    argv = list(sys.argv[1:])
    if argv and argv[0] == "bulk":
        target = 50
        min_rate = 0.48
        rest = argv[1:]
        if rest and rest[0].isdigit():
            target = max(10, min(100, int(rest[0])))
            rest = rest[1:]
        if rest and rest[0].replace(".", "", 1).isdigit():
            min_rate = float(rest[0])
        client = TavilyClient(api_key=api_key)
        return _run_bulk(client, target, min_rate)

    argv = list(sys.argv[1:])
    max_results = 5
    if argv and argv[-1].isdigit():
        max_results = max(1, int(argv.pop()))
    query = " ".join(argv).strip() or "Python 3.12 release site:python.org OR site:docs.python.org"

    client = TavilyClient(api_key=api_key)
    print(f"Query: {query!r}\nmax_results={max_results}\n")

    res = client.search(
        query,
        max_results=max_results,
        include_raw_content="text",
    )

    rows: list[dict] = []
    for hit in res.get("results") or []:
        if not isinstance(hit, dict):
            continue
        raw = hit.get("raw_content") or ""
        u_hit = str(hit.get("url") or "").strip()
        pub = infer_publication_calendar_date(raw, source_url=u_hit) if raw else ""
        rows.append(
            {
                "title": hit.get("title", "")[:70],
                "url": hit.get("url", ""),
                "published_date": pub,
                "stage": "search_raw",
            }
        )

    had_from_search = [bool((r.get("published_date") or "").strip()) for r in rows]
    before = sum(had_from_search)
    _enrich_missing(client, rows, max_urls=min(8, max_results), extract_format="text", timeout=45.0)
    after = sum(1 for r in rows if (r.get("published_date") or "").strip())
    for i, r in enumerate(rows):
        pub = (r.get("published_date") or "").strip()
        if not pub:
            r["stage"] = "none"
        elif had_from_search[i]:
            r["stage"] = "search_raw"
        else:
            r["stage"] = "extract"

    for i, r in enumerate(rows, 1):
        print(f"[{i}] {r.get('published_date') or '(none)':12}  ({r.get('stage')})")
        print(f"    {r.get('title')}")
        print(f"    {r.get('url')}")
        print()

    print(f"With date after search raw: {before}/{len(rows)}")
    print(f"With date after +extract:   {after}/{len(rows)}")
    print(
        json.dumps(
            [{k: v for k, v in x.items() if k != "stage"} for x in rows],
            indent=2,
            ensure_ascii=False,
        )
    )

    # Single known URL sanity (Wikipedia stable article; date in page metadata)
    check_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    print("\n--- Single-page extract sanity ---")
    print(f"URL: {check_url}")
    ex = client.extract([check_url], format="text", extract_depth="basic", timeout=45.0)
    if ex.get("results"):
        raw0 = ex["results"][0].get("raw_content") or ""
        p = infer_publication_calendar_date(raw0, source_url=check_url)
        print(f"inferred published_date (calendar): {p or '(none)'}")
        print("(Wikipedia infobox/first revision is complex; non-empty ISO-like date usually OK.)")
    else:
        print("extract failed:", ex.get("failed_results"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
