"""Offline regression tests shaped like real web_search citations (no network).

URLs and patterns mirror user-reported citation rows (Electrek, Fredpope, Facebook, Yahoo,
Reddit, YouTube, path-only blogs, etc.). CI always runs these.
"""

from __future__ import annotations

import json

import pytest

from deerflow.agents.middlewares.citation_middleware import CitationMiddleware
from deerflow.utils.publication_date import infer_publication_calendar_date


@pytest.fixture
def no_http_publication_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        CitationMiddleware,
        "_publication_date_http_fetch_config",
        staticmethod(
            lambda: {
                "enabled": False,
                "max_urls": 8,
                "timeout": 5.0,
                "max_bytes": 100_000,
            }
        ),
    )


def test_electrek_path_merge_fills_date_without_body() -> None:
    url = "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review"
    merged, fb = CitationMiddleware._merge_publication_date_with_url("", url)
    assert merged == "2025-12-16"
    assert fb == ""


def test_cleantechnica_wordpress_path_merge() -> None:
    url = "https://cleantechnica.com/2026/03/20/teslas-camera-weather-problem-is-serious"
    merged, _ = CitationMiddleware._merge_publication_date_with_url("", url)
    assert merged == "2026-03-20"


def test_fredpope_next_meta_article_published_time() -> None:
    html = '<meta property="article:published_time" content="2025-06-24T00:00:00.000Z"/>'
    url = "https://www.fredpope.com/blog/machine-learning/tesla-fsd-12"
    assert infer_publication_calendar_date(html, source_url=url) == "2025-06-24"


def test_yahoo_news_portal_embedded_iso() -> None:
    blob = '{"stream":{"contentPublishedDate":"2024-11-05T18:30:00.000Z"}}'
    url = "https://www.yahoo.com/news/articles/example-story-id"
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-11-05"


def test_msn_host_uses_yahoo_style_portal_keys() -> None:
    blob = '{"gem":{"firstPublishedMillis":1735689600000}}'
    url = "https://www.msn.com/en-us/news/insight/example"
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-01-01"


def test_facebook_post_scoped_creation_beats_group_chrome() -> None:
    decoy = '{"creation_time":1600000000}'
    pid = "1166359048819502"
    story = f'"feedbackTarget":{{"id":"{pid}","creation_time":1735689600}}'
    blob = decoy + story
    url = f"https://www.facebook.com/groups/teslaownersaustralia/posts/{pid}"
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-01-01"


def test_reddit_thread_url_created_utc() -> None:
    blob = '{"created_utc": 1700000000.0}'
    url = "https://www.reddit.com/r/TeslaFSD/comments/1pa8pqz/fsd_fail_example/"
    assert infer_publication_calendar_date(blob, source_url=url) == "2023-11-14"


def test_youtube_watch_player_response_publish_date() -> None:
    html = (
        "<script>var ytInitialPlayerResponse = "
        '{"microformat":{"playerMicroformatRenderer":'
        '{"publishDate":"2024-09-01T10:00:00-07:00"}}};</script>'
    )
    url = "https://www.youtube.com/watch?v=aEfPq33PH1s"
    assert infer_publication_calendar_date(html, source_url=url) == "2024-09-01"


def test_notateslaapp_style_wordpress_meta() -> None:
    html = '<meta property="article:published_time" content="2025-03-10T12:00:00+00:00" />'
    url = "https://www.notateslaapp.com/news/2571/example-slug"
    assert infer_publication_calendar_date(html, source_url=url) == "2025-03-10"


def test_recharged_style_article_meta() -> None:
    html = '<meta property="og:published_time" content="2025-02-14T08:00:00Z" />'
    url = "https://recharged.com/articles/fsd-13-tesla-full-self-driving-guide"
    assert infer_publication_calendar_date(html, source_url=url) == "2025-02-14"


def test_shopify_blog_slug_year_when_no_calendar() -> None:
    url = "https://www.teslaacessories.com/blogs/news/fsd-regulatory-opening-in-2026"
    merged, fb = CitationMiddleware._merge_publication_date_with_url(
        "",
        url,
        title="Tesla FSD at a Crossroads in 2026",
    )
    assert merged == ""
    assert fb == "2026"


def test_citations_json_electrek_no_raw_uses_path_merge(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "Electrek FSD review",
                "url": "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review",
                "snippet": "Mind blowing FSD v14.",
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-12-16"
    assert cites[0].get("year") == "2025"


def test_citations_json_fredpope_raw_meta(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "Neural Network Revolution",
                "url": "https://www.fredpope.com/blog/machine-learning/tesla-fsd-12",
                "snippet": "Tesla FSD history 2016 and 2025.",
                "raw_content": (
                    '<meta property="article:published_time" content="2025-06-24T00:00:00.000Z"/>'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-06-24"
    assert cites[0].get("year") == "2025"


def test_citations_json_zhihu_zhuanlan_markdown_publish_line(no_http_publication_fetch) -> None:
    """知乎专栏：无 boot JSON 时，Reader 类正文里的「发布于」须进 ``published_date``。"""
    payload = json.dumps(
        [
            {
                "title": "《2022年Tesla AI Day——FSD技术进展分析总结》",
                "url": "https://zhuanlan.zhihu.com/p/572974435",
                "snippet": "FSD 技术分析。",
                "raw_content": (
                    "# 摘要\n\n"
                    "作者 · 发布于 2022年10月20日 · 著作权归作者所有\n\n"
                    "正文……"
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2022-10-20"
    assert cites[0].get("year") == "2022"
