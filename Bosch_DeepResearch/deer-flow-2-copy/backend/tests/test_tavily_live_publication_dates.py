"""Live Tavily ``/extract`` + publication date + IEEE reference line (optional).

Runs only when ``TAVILY_API_KEY`` is set or project-root ``config.yaml`` has a Tavily
``web_search`` tool ``api_key`` (non-interpolated). Requires network.

Run::

    cd backend && PYTHONPATH=packages/harness TAVILY_API_KEY=tvly-... \\
        uv run pytest tests/test_tavily_live_publication_dates.py -v --tb=short

Bulk Search→Extract (up to 50 URLs, minimum success rate)::

    uv run pytest tests/test_tavily_live_publication_dates_bulk.py -v -m \"tavily_live and slow\"
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from deerflow.agents.middlewares.citation_middleware import CitationMiddleware
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


LIVE_CASES: list[tuple[str, str, str]] = [
    (
        "https://www.python.org/downloads/release/python-3131/",
        "2024-12-03",
        "python.org release page: Release date in body",
    ),
    (
        "https://peps.python.org/pep-0719/",
        "2023-05-26",
        "PEP Created: (document creation date)",
    ),
]

# Tavily ``/extract`` often blocks or strips metadata for these; full-page behaviour is
# covered in ``tests/test_publication_date.py``. Optional live checks (best-effort):
_LIVE_OPTIONAL_CASES: list[tuple[str, str, str]] = [
    (
        "https://www.reddit.com/r/Python/comments/1fybncq/python_313_released/",
        "2024-10-07",
        "Reddit: prefer created_utc JSON when present; no footer aggregate date",
    ),
    (
        "https://www.youtube.com/watch?v=6gN6p-Iw_x4",
        "2024-08-25",
        "YouTube: needs rich HTML (often absent in Tavily extract)",
    ),
    (
        "https://blog.csdn.net/weixin_43336281/article/details/123609878",
        "2022-03-20",
        "CSDN: Tavily may fail fetch from datacenter IP",
    ),
    (
        "https://arxiv.org/abs/1706.03762",
        "2017-06-12",
        "arXiv: citation_date when raw HTML includes meta",
    ),
]


requires_tavily = pytest.mark.skipif(
    not _tavily_api_key(),
    reason="Set TAVILY_API_KEY or add web_search tavily api_key in project config.yaml",
)


def _live_case_id(case: tuple[str, str, str]) -> str:
    url, _, _ = case
    try:
        host = url.split("/")[2]
    except IndexError:
        host = "url"
    tail = url.rstrip("/").rsplit("/", 1)[-1][:24]
    return f"{host}:{tail}"


@requires_tavily
@pytest.mark.parametrize("url,expected_date,note", LIVE_CASES, ids=[_live_case_id(c) for c in LIVE_CASES])
def test_live_extract_infer_publication_calendar_date(url: str, expected_date: str, note: str) -> None:
    from tavily import TavilyClient

    client = TavilyClient(api_key=_tavily_api_key())
    res = client.extract([url], format="text", extract_depth="basic", timeout=90)
    assert not res.get("failed_results"), res.get("failed_results")
    assert res.get("results"), "no extract results"
    raw = (res["results"][0].get("raw_content") or "")[:200000]
    got = infer_publication_calendar_date(raw, source_url=url)
    assert got == expected_date, f"{note}: got {got!r} expected {expected_date!r}"


@requires_tavily
@pytest.mark.parametrize(
    "url,expected_date,note",
    _LIVE_OPTIONAL_CASES,
    ids=[_live_case_id(c) for c in _LIVE_OPTIONAL_CASES],
)
def test_live_extract_optional_hosts_best_effort(url: str, expected_date: str, note: str) -> None:
    """Soft check: pass if Tavily returns parseable body; skip if extract fails or date missing."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=_tavily_api_key())
    res = client.extract([url], format="text", extract_depth="basic", timeout=90)
    failed = res.get("failed_results") or []
    if failed:
        pytest.skip(f"Tavily extract failed for {url}: {failed}")
    if not res.get("results"):
        pytest.skip("no extract results")
    raw = (res["results"][0].get("raw_content") or "")[:200000]
    if not raw.strip():
        pytest.skip("empty raw_content from Tavily")
    got = infer_publication_calendar_date(raw, source_url=url)
    if got != expected_date:
        pytest.skip(f"{note}: inferred {got!r}, expected {expected_date!r} (Tavily body differs from browser)")


@requires_tavily
def test_live_search_json_pipeline_reference_line_has_year_after_domain() -> None:
    """End-to-end: synthetic JSON like ``web_search_tool`` + IEEE refs include YYYY after domain."""
    from tavily import TavilyClient

    key = _tavily_api_key()
    client = TavilyClient(api_key=key)
    url = "https://www.python.org/downloads/release/python-3131/"
    res = client.extract([url], format="text", extract_depth="basic", timeout=90)
    raw = res["results"][0].get("raw_content") or ""
    pub = infer_publication_calendar_date(raw, source_url=url)
    assert pub == "2024-12-03"

    payload = json.dumps(
        [
            {
                "title": "Python 3.13.1",
                "url": url,
                "snippet": "Test snippet.",
                "published_date": pub,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    refs = CitationMiddleware._format_references_ieee(cites)
    assert "python.org" in refs
    assert "2024-12-03" not in refs
    assert "python.org. 2024" in refs
