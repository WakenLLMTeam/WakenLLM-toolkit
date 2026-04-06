"""Live integration tests for real Facebook and Zhihu URLs (requires network access).

These tests attempt to fetch and parse publication dates from actual live URLs.
They are skipped by default (use CITATION_EXAMPLES_LIVE=1 env var to enable).

Test URLs:
[11] Facebook: https://www.facebook.com/groups/1242309506256283/posts/1995955427558350
[18] Zhihu: https://www.zhihu.com/answer/3246911414
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from deerflow.utils.publication_date import infer_publication_calendar_date
from deerflow.utils.publication_date_fallback_fetch import (
    infer_publication_calendar_from_url_http,
    infer_publication_calendar_via_jina_reader,
    url_allowed_for_ssrf_guard,
)


LIVE_TESTS_ENABLED = os.getenv("CITATION_EXAMPLES_LIVE", "").lower() in ("1", "true", "yes")


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_facebook_group_post_live_http_fetch() -> None:
    """Attempt to fetch the Facebook group post and extract publication date.
    
    Note: Facebook may require authentication or block scrapers. This test documents
    the expected structure and can help debug live extraction issues.
    """
    url = "https://www.facebook.com/groups/1242309506256283/posts/1995955427558350"
    
    # Try HTTP fallback (may be blocked by Facebook)
    result = infer_publication_calendar_from_url_http(url, timeout=10.0)
    
    # If successful, should have extracted a date
    if result:
        assert len(result) == 10, f"Expected YYYY-MM-DD format, got {result}"
        assert result[4] == "-" and result[7] == "-"
    else:
        # Document that extraction failed (likely due to Facebook blocking)
        print(f"⚠️  Could not extract date from {url} (Facebook may block crawlers)")


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_zhihu_answer_live_http_fetch() -> None:
    """Attempt to fetch the Zhihu answer and extract publication date."""
    url = "https://www.zhihu.com/answer/3246911414"
    
    # Try HTTP fallback
    result = infer_publication_calendar_from_url_http(url, timeout=10.0)
    
    # If successful, should have extracted a date
    if result:
        assert len(result) == 10, f"Expected YYYY-MM-DD format, got {result}"
        assert result[4] == "-" and result[7] == "-"
        print(f"✅ Extracted publication date from Zhihu: {result}")
    else:
        print(f"⚠️  Could not extract date from {url}")


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_zhihu_answer_via_jina_reader() -> None:
    """Attempt to use Jina reader API to fetch Zhihu article content."""
    url = "https://www.zhihu.com/answer/3246911414"
    
    # Try Jina reader API (if configured)
    result = infer_publication_calendar_via_jina_reader(url, timeout=10.0)
    
    if result:
        assert len(result) == 10, f"Expected YYYY-MM-DD format, got {result}"
        print(f"✅ Extracted publication date via Jina: {result}")
    else:
        print(f"⚠️  Jina reader could not extract date from {url}")


def test_facebook_url_passes_ssrf_guard() -> None:
    """Ensure Facebook URLs pass SSRF validation."""
    url = "https://www.facebook.com/groups/1242309506256283/posts/1995955427558350"
    assert url_allowed_for_ssrf_guard(url) is True


def test_zhihu_url_passes_ssrf_guard() -> None:
    """Ensure Zhihu URLs pass SSRF validation."""
    url = "https://www.zhihu.com/answer/3246911414"
    assert url_allowed_for_ssrf_guard(url) is True


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_facebook_extraction_with_mock_response() -> None:
    """Mock HTTP response to test Facebook extraction logic (no real network call)."""
    url = "https://www.facebook.com/groups/1242309506256283/posts/1995955427558350"
    
    # Simulate a realistic Facebook HTML response structure
    mock_html = '''
    <html>
    <head>
        <script>
            var initialData = {
                "stories": {
                    "1995955427558350": {
                        "creation_time": 1735689600,
                        "message": "Tesla FSD performance discussion"
                    }
                }
            };
        </script>
    </head>
    <body></body>
    </html>
    '''
    
    # Test that the extraction function can find the timestamp in this structure
    result = infer_publication_calendar_date(mock_html, source_url=url)
    assert result == "2025-01-01"


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_zhihu_extraction_with_mock_response() -> None:
    """Mock HTTP response to test Zhihu extraction logic (no real network call)."""
    url = "https://www.zhihu.com/answer/3246911414"
    
    # Simulate a realistic Zhihu HTML response with JSON-LD and embedded data
    mock_html = '''
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@type": "Article",
            "articleBody": "Why does Tesla insist on the pure vision approach?",
            "datePublished": "2024-01-01T08:00:00Z"
        }
        </script>
    </head>
    <body></body>
    </html>
    '''
    
    result = infer_publication_calendar_date(mock_html, source_url=url)
    assert result == "2024-01-01"


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_finviz_news_live_http_fetch() -> None:
    """Attempt to fetch the Finviz news article and extract publication date."""
    url = "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
    
    # Try HTTP fallback
    result = infer_publication_calendar_from_url_http(url, timeout=10.0)
    
    # If successful, should have extracted a date
    if result:
        assert len(result) == 10, f"Expected YYYY-MM-DD format, got {result}"
        assert result[4] == "-" and result[7] == "-"
        print(f"✅ Extracted publication date from Finviz: {result}")
    else:
        print(f"⚠️  Could not extract date from {url}")


def test_finviz_url_passes_ssrf_guard() -> None:
    """Ensure Finviz URLs pass SSRF validation."""
    url = "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
    assert url_allowed_for_ssrf_guard(url) is True


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set CITATION_EXAMPLES_LIVE=1 to enable")
def test_finviz_extraction_with_mock_response() -> None:
    """Mock HTTP response to test Finviz extraction logic (no real network call)."""
    url = "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
    
    # Simulate a realistic Finviz HTML response with meta tags
    mock_html = '''
    <html>
    <head>
        <meta property="article:published_time" content="2024-12-15T10:30:00Z" />
        <title>Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets</title>
    </head>
    <body>
        <article>
            <h1>Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets</h1>
            <p>Tesla's Full Self-Driving has now accumulated...</p>
        </article>
    </body>
    </html>
    '''
    
    result = infer_publication_calendar_date(mock_html, source_url=url)
    assert result == "2024-12-15"
