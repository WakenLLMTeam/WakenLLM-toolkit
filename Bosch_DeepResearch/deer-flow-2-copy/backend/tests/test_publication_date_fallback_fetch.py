"""Tests for HTTP fallback when search payloads lack a parseable publication date."""

import json
from unittest.mock import MagicMock, patch

import pytest

from deerflow.agents.middlewares.citation_middleware import CitationMiddleware
from deerflow.utils.publication_date_fallback_fetch import (
    infer_publication_calendar_from_url_http,
    infer_publication_calendar_via_jina_reader,
    url_allowed_for_ssrf_guard,
)


def test_url_allowed_for_ssrf_guard_blocks_loopback_and_private() -> None:
    assert url_allowed_for_ssrf_guard("http://127.0.0.1/") is False
    assert url_allowed_for_ssrf_guard("http://localhost/foo") is False
    assert url_allowed_for_ssrf_guard("http://192.168.1.1/") is False
    assert url_allowed_for_ssrf_guard("file:///etc/passwd") is False
    assert url_allowed_for_ssrf_guard("https://example.com/article") is True


def test_infer_publication_calendar_via_jina_reader_non_social_empty() -> None:
    assert infer_publication_calendar_via_jina_reader("https://example.com/a", timeout=5.0) == ""


def test_infer_publication_calendar_via_jina_reader_facebook_html() -> None:
    url = "https://www.facebook.com/groups/1242309506256283/posts/1995955427558350"

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def post(self, url_post, headers=None, json=None):
            r = MagicMock()
            r.status_code = 200
            assert json.get("url") == url
            r.text = (
                '{"story_fbid":"1995955427558350","creation_time":1735689600}'
            )
            return r

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_via_jina_reader(url, timeout=10.0)
    assert got == "2025-01-01"


def test_infer_publication_calendar_via_jina_reader_uses_jina_markdown() -> None:
    url = "https://zhuanlan.zhihu.com/p/572974435"

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def post(self, url_post, headers=None, json=None):
            r = MagicMock()
            r.status_code = 200
            assert json.get("url") == url
            r.text = "作者 · 发布于 2022年10月20日 · 上海\n\n正文"
            return r

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_via_jina_reader(url, timeout=10.0)
    assert got == "2022-10-20"


def test_infer_publication_calendar_from_url_http_zhihu_prefers_jina() -> None:
    """Direct GET is not used when Jina returns a calendar date."""

    url = "https://zhuanlan.zhihu.com/p/572974435"

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def post(self, url_post, headers=None, json=None):
            r = MagicMock()
            r.status_code = 200
            r.text = "发表于：2022-10-20\n"
            return r

        def get(self, url_get, headers=None):
            raise AssertionError("Zhihu path should not GET when Jina succeeds")

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_from_url_http(url, timeout=8.0)
    assert got == "2022-10-20"


def test_infer_publication_calendar_from_url_http_facebook_jina_before_get() -> None:
    """Facebook: Jina succeeds → no direct GET to facebook.com."""

    url = "https://www.facebook.com/groups/x/posts/1995955427558350"

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def post(self, url_post, headers=None, json=None):
            r = MagicMock()
            r.status_code = 200
            r.text = '{"1995955427558350":{"creation_time":1735689600}}'
            return r

        def get(self, url_get, headers=None):
            raise AssertionError("Facebook path should not GET when Jina succeeds")

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_from_url_http(url, timeout=8.0)
    assert got == "2025-01-01"


def test_infer_publication_calendar_from_url_http_facebook_400_then_mobile() -> None:
    """www.facebook.com 4xx → still try m.facebook.com before giving up."""

    url = "https://www.facebook.com/groups/g/posts/1995955427558350"
    mobile_html = '{"story_fbid":"1995955427558350","creation_time":1735689600}'

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def post(self, url_post, headers=None, json=None):
            r = MagicMock()
            r.status_code = 204
            r.text = ""
            return r

        def get(self, url_get, headers=None):
            r = MagicMock()
            u = str(url_get)
            if "m.facebook.com" in u:
                r.status_code = 200
                r.url = url_get
                r.text = mobile_html
            else:
                r.status_code = 400
                r.url = url_get
                r.text = ""
            return r

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_from_url_http(url, timeout=8.0)
    assert got == "2025-01-01"


def test_infer_publication_calendar_from_url_http_parses_meta() -> None:
    html = '<html><meta property="article:published_time" content="2024-06-15T12:00:00Z" /></html>'

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, headers=None):
            r = MagicMock()
            r.status_code = 200
            r.url = url
            r.text = html
            return r

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        assert infer_publication_calendar_from_url_http("https://blog.example.com/p") == "2024-06-15"


def test_infer_publication_calendar_from_url_http_path_merge_when_body_has_no_meta() -> None:
    """Electrek-style /YYYY/MM/DD/ must still yield a date when HTML is empty or stripped."""

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, headers=None):
            r = MagicMock()
            r.status_code = 200
            r.url = url
            r.text = "<html><body></body></html>"
            return r

    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_from_url_http(
            "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review"
        )
    assert got == "2025-12-16"


def test_infer_publication_calendar_from_url_http_path_overrides_conflicting_meta() -> None:
    """WordPress path date must beat a wrong syndicated ``article:published_time`` in HTML."""

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, headers=None):
            r = MagicMock()
            r.status_code = 200
            r.url = url
            r.text = (
                '<meta property="article:published_time" content="2020-01-01T00:00:00Z" />'
                "<p>Article body.</p>"
            )
            return r

    u = "https://cleantechnica.com/2026/03/20/teslas-camera-weather-problem-is-serious"
    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_from_url_http(u)
    assert got == "2026-03-20"


def test_infer_publication_calendar_from_url_http_reddit_json_fallback() -> None:
    """Thread HTML without embed JSON → follow-up GET to ``…/comments/{id}/….json``."""

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, headers=None):
            r = MagicMock()
            r.status_code = 200
            r.url = url
            if str(url).rstrip("/").endswith(".json"):
                r.text = '{"created_utc": 1700000000.0}'
            else:
                r.text = "<html><body>Reddit</body></html>"
            return r

    u = "https://www.reddit.com/r/TeslaFSD/comments/abc123/title_here/"
    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        got = infer_publication_calendar_from_url_http(u)
    assert got == "2023-11-14"


@pytest.fixture
def fetch_fallback_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CitationMiddleware,
        "_publication_date_http_fetch_config",
        staticmethod(
            lambda: {
                "enabled": True,
                "max_urls": 4,
                "timeout": 5.0,
                "max_bytes": 500_000,
            }
        ),
    )


def test_citations_json_triggers_fetch_when_enabled(fetch_fallback_enabled) -> None:
    html = '<meta property="article:published_time" content="2022-03-20T00:00:00Z" />'

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, headers=None):
            r = MagicMock()
            r.status_code = 200
            r.url = url
            r.text = html
            return r

    payload = json.dumps(
        [
            {
                "title": "No date in search",
                "url": "https://blog.example.com/posts/no-date-here",
                "snippet": "Summary without structured date.",
            }
        ],
        ensure_ascii=False,
    )
    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2022-03-20"


def test_citations_json_skips_fetch_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CitationMiddleware,
        "_publication_date_http_fetch_config",
        staticmethod(lambda: {"enabled": False, "max_urls": 8, "timeout": 5.0, "max_bytes": 100_000}),
    )
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url, headers=None):
            calls.append(url)
            r = MagicMock()
            r.status_code = 200
            r.url = url
            r.text = "<html></html>"
            return r

    payload = json.dumps(
        [{"title": "x", "url": "https://example.com/a", "snippet": "y"}],
        ensure_ascii=False,
    )
    with patch("deerflow.utils.publication_date_fallback_fetch.httpx.Client", FakeClient):
        CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert calls == []
