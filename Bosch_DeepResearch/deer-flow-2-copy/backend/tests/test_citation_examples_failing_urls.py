"""Test cases for failing Facebook and Zhihu URLs from citations (待爬取的失败案例).

These are real URLs that previously failed to extract publication dates:
[11] Facebook Group: https://www.facebook.com/groups/1242309506256283/posts/1995955427558350
[18] Zhihu Answer: https://www.zhihu.com/en/answer/3246911414
[5] Medium: jakubjirak.medium.com/.../35551ba6bbde (JSON ``firstPublishedAt`` or body year)
[3] Instagram reel: instagram.com/reel/C-5egDmutbr (``og:description`` ``on Month D, YYYY:`` or ``taken_at``)
[12] Facebook photo: ``/photos/…/884784524246026`` (embedded ``creation_time`` near post id)
[4] Sina mobile finance: cj.sina.cn (``<meta name="weibo: article:create_at" …>``)
Tesla accessories blog: teslaacessories.com ``/blogs/news/…`` (Shopify ``published_at`` / ``<time itemprop>``; merge **raw first** + 1.5M scan; ``srsltid`` query OK; NHTSA FSD visibility post with ``3.2`` in slug)
X / x.com status: GraphQL-style ``legacy.created_at`` (RFC2822) or ISO near ``rest_id`` / status id — before shell JSON-LD
[8] Sina cj.sina.cn: ``weibo: article:create_at`` + ``<span class=…date…>`` (ignore ``pagepubtime`` JSON)
[4] eet-china.com ``/mp/…``: ``发布时间：…年…月…日`` + ``article:published_time``
[4] 51fusa.com ``…/id/2045.html``: id must not become year 2045; with ``article:published_time`` expect **2021**
[3] wallstreetcn.com ``/articles/…``: fixture HTML with ``<article>…<time>`` or boot JSON; bare SPA shell has no date
[1] eeworld.com.cn ``/qcdz/eic….html``: byline ``最新更新时间：YYYY-MM-DD`` (often not in ``<head>`` meta; WAF on bare GET)

Each real URL is paired with a **fixture ground-truth calendar date** (synthetic HTML/JSON
matching that site). Tests pass only when ``published_date`` / ``year`` match that date—not
merely “non-empty” or “not wrong year”.
"""

from __future__ import annotations

import json

import pytest

from deerflow.agents.middlewares.citation_middleware import CitationMiddleware
from deerflow.utils.publication_date import (
    calendar_date_from_raw_string,
    infer_publication_calendar_date,
)


@pytest.fixture
def no_http_publication_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable HTTP fallback for offline testing."""
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


def test_facebook_group_post_with_story_id_extraction() -> None:
    """Facebook group post: Extract creation_time near the post ID (1995955427558350).
    
    Facebook embeds many creation_time values; we anchor on the specific post ID from the URL.
    """
    group_id = "1242309506256283"
    post_id = "1995955427558350"
    url = f"https://www.facebook.com/groups/{group_id}/posts/{post_id}"
    
    # Mock a realistic Facebook payload with decoy creation_time + actual post creation_time
    decoy = '{"creation_time":1600000000}'  # Decoy: Oct 2020
    actual_post = f'{{"story_fbid":"{post_id}","creation_time":1735689600}}'  # Jan 2025
    blob = decoy + actual_post
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2025-01-01", f"Expected 2025-01-01 but got {result}"


def test_facebook_group_post_with_legacy_id_format() -> None:
    """Facebook group post with legacyId field (alternative format)."""
    group_id = "1242309506256283"
    post_id = "1995955427558350"
    url = f"https://www.facebook.com/groups/{group_id}/posts/{post_id}"
    
    # Alternative Facebook payload format with legacyId
    blob = f'{{"legacyId":"{post_id}","creation_time":1735689600}}'
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2025-01-01", f"Expected 2025-01-01 but got {result}"


def test_facebook_group_post_milliseconds_epoch() -> None:
    """Facebook may use 13-digit millisecond timestamps (creation_time in ms)."""
    group_id = "1242309506256283"
    post_id = "1995955427558350"
    url = f"https://www.facebook.com/groups/{group_id}/posts/{post_id}"
    
    # 13-digit millisecond epoch
    blob = f'{{"story_fbid":"{post_id}","creation_time":1735689600000}}'  # Jan 2025 in ms
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2025-01-01", f"Expected 2025-01-01 but got {result}"


def test_zhihu_answer_with_published_time_json() -> None:
    """Zhihu answer: Extract publishedTime near the answer ID (3246911414).
    
    Zhihu embeds publishedTime in nested entities; anchor on the URL-extracted ID.
    """
    answer_id = "3246911414"
    url = f"https://www.zhihu.com/answer/{answer_id}"
    
    # Mock Zhihu payload with answer-scoped publishedTime
    blob = (
        '{"feed":[{"publishedTime":1600000000000}],'  # Decoy: Sept 2020
        '"entities":{"answers":{"' + answer_id + '":'
        '{"title":"Why does Tesla insist on the pure vision approach?",'
        '"publishedTime":1704067200000}}}}'  # Jan 2024
    )
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-01-01", f"Expected 2024-01-01 but got {result}"


def test_zhihu_answer_with_created_time_field() -> None:
    """Zhihu may use createdTime instead of publishedTime."""
    answer_id = "3246911414"
    url = f"https://www.zhihu.com/answer/{answer_id}"
    
    blob = (
        '"entities":{"answers":{"' + answer_id + '":'
        '{"title":"Why does Tesla insist on the pure vision approach?",'
        '"createdTime":1704067200000}}}'
    )
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-01-01", f"Expected 2024-01-01 but got {result}"


def test_zhihu_answer_published_time_as_string() -> None:
    """Zhihu sometimes wraps publishedTime as a JSON string (not integer)."""
    answer_id = "3246911414"
    url = f"https://www.zhihu.com/answer/{answer_id}"
    
    blob = (
        '"entities":{"answers":{"' + answer_id + '":'
        '{"publishedTime":"1704067200000"}}}'
    )
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-01-01", f"Expected 2024-01-01 but got {result}"


def test_zhihu_answer_iso_published_time() -> None:
    """Zhihu may embed ISO 8601 publishedTime strings."""
    answer_id = "3246911414"
    url = f"https://www.zhihu.com/answer/{answer_id}"
    
    blob = (
        '"entities":{"answers":{"' + answer_id + '":'
        '{"publishedTime":"2024-01-01T08:00:00Z"}}}'
    )
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-01-01"


def test_citations_facebook_group_post_raw_content(no_http_publication_fetch) -> None:
    """Citation middleware test: Facebook group post with raw_content payload."""
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD performance in complex city scenarios",
                "url": "https://www.facebook.com/groups/1242309506256283/posts/1995955427558350",
                "snippet": "Discussion on FSD performance in urban environments.",
                "raw_content": (
                    '{"story_fbid":"1995955427558350",'
                    '"creation_time":1735689600}'  # 2025-01-01
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-01-01", f"Got {cites[0].get('published_date')}"
    assert cites[0].get("year") == "2025"


def test_citations_zhihu_answer_raw_content(no_http_publication_fetch) -> None:
    """Citation middleware test: Zhihu answer with raw_content payload."""
    payload = json.dumps(
        [
            {
                "title": "Why does Tesla insist on the pure vision approach?",
                "url": "https://www.zhihu.com/answer/3246911414",
                "snippet": "Analysis of Tesla's vision-only approach to FSD.",
                "raw_content": (
                    '{"entities":{"answers":{"3246911414":'
                    '{"title":"Why does Tesla insist on the pure vision approach?",'
                    '"publishedTime":1704067200000}}}}'  # 2024-01-01
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-01-01", f"Got {cites[0].get('published_date')}"
    assert cites[0].get("year") == "2024"


def test_zhihu_answer_with_question_id_alternative_path() -> None:
    """Some Zhihu URLs use /question/{id} instead of /answer/{id}."""
    question_id = "3246911414"
    url = f"https://www.zhihu.com/question/{question_id}"
    
    blob = (
        '"entities":{"questions":{"' + question_id + '":'
        '{"title":"Why does Tesla insist on the pure vision approach?",'
        '"publishedTime":1704067200000}}}'
    )
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-01-01", f"Expected 2024-01-01 but got {result}"


def test_facebook_post_with_permalink_id() -> None:
    """Facebook permalinks may use different ID extraction patterns."""
    post_id = "1995955427558350"
    url = f"https://www.facebook.com/permalink.php?story_fbid={post_id}"
    
    blob = f'{{"story_fbid":"{post_id}","creation_time":1735689600}}'
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2025-01-01", f"Expected 2025-01-01 but got {result}"


def test_finviz_news_article_with_article_published_time() -> None:
    """Finviz news article: Extract article:published_time from meta tags.
    
    Finviz uses standard meta tags for published dates. The news ID is in the URL path.
    URL: https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets
    """
    news_id = "264338"
    url = f"https://finviz.com/news/{news_id}/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
    
    # Finviz HTML with meta article:published_time
    blob = '<meta property="article:published_time" content="2024-12-15T10:30:00Z" />'
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-12-15", f"Expected 2024-12-15 but got {result}"


def test_finviz_news_article_with_og_published_time() -> None:
    """Finviz may use og:published_time instead of article:published_time."""
    news_id = "264338"
    url = f"https://finviz.com/news/{news_id}/tesla-fsd-approaches-7b-miles"
    
    blob = '<meta property="og:published_time" content="2024-12-15T10:30:00Z" />'
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-12-15", f"Expected 2024-12-15 but got {result}"


def test_finviz_news_article_json_ld_datetime() -> None:
    """Finviz with JSON-LD NewsArticle schema."""
    url = "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles"
    
    blob = '''<script type="application/ld+json">
    {
        "@type": "NewsArticle",
        "headline": "Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets",
        "datePublished": "2024-12-15T10:30:00Z",
        "dateModified": "2024-12-15T11:00:00Z"
    }
    </script>'''
    
    result = infer_publication_calendar_date(blob, source_url=url)
    assert result == "2024-12-15", f"Expected 2024-12-15 but got {result}"


def test_finviz_path_date_extraction_from_url() -> None:
    """Finviz /news/… often has no meta; byline is ``</a> | Month D, YYYY, h:mm AM``."""
    url = "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
    blob = (
        '<h1 class="title m-0">Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets</h1>'
        '<div class="news-publish-info"><div>By '
        '<a class="tab-link" href="/news/zacks/zacks-equity-research">Zacks Equity Research</a>'
        "                        | December 30, 2025, 9:32 AM"
        "</div></div>"
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-12-30"


def test_finviz_news_byline_date_without_time_suffix() -> None:
    """Byline may omit clock time (still parseable)."""
    url = "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
    blob = (
        '<a class="tab-link" href="/news/foo">Reuters</a> | March 15, 2024'
        "<p>Body</p>"
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-03-15"


def test_citations_finviz_news_raw_content(no_http_publication_fetch) -> None:
    """Citation middleware test: Finviz news article with raw_content."""
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets",
                "url": "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets",
                "snippet": "Tesla Full Self-Driving continues to accumulate miles across various driving conditions.",
                "raw_content": (
                    '<meta property="article:published_time" content="2024-12-15T10:30:00Z" />'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-12-15", f"Got {cites[0].get('published_date')}"
    assert cites[0].get("year") == "2024"


def test_citations_finviz_news_byline_only_raw_content(no_http_publication_fetch) -> None:
    """Tavily-style extract: real Finviz shape without article:published_time meta."""
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets - Finviz",
                "url": "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets",
                "snippet": "Tesla FSD fleet reaching 7 billion miles.",
                "raw_content": (
                    '<div class="news-publish-info">By '
                    '<a class="tab-link" href="/news/zacks/zacks-equity-research">Zacks Equity Research</a>'
                    " | December 30, 2025, 9:32 AM</div>"
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-12-30", f"Got {cites[0].get('published_date')}"
    assert cites[0].get("year") == "2025"


# User-reported: Medium story year / date missing though body or JSON carries signals.
MEDIUM_JAKUB_URL = (
    "https://jakubjirak.medium.com/how-tesla-rewrote-the-driving-rules-the-revolutionary-"
    "move-that-will-change-everything-35551ba6bbde"
)


def test_medium_jakubjirak_first_published_at_ms_calendar() -> None:
    """Medium boot JSON: ``firstPublishedAt`` ms epoch near story hex id."""
    blob = (
        '{"postId":"35551ba6bbde","firstPublishedAt":1704067200000,'
        '"title":"How Tesla Rewrote The Driving Rules"}'
    )
    assert infer_publication_calendar_date(blob, source_url=MEDIUM_JAKUB_URL) == "2024-01-01"


def test_citations_medium_jakubjirak_embedded_json(no_http_publication_fetch) -> None:
    """Fixture ground truth: ``firstPublishedAt`` ms → calendar ``2024-01-01``."""
    payload = json.dumps(
        [
            {
                "title": (
                    "How Tesla Rewrote The Driving Rules: The Revolutionary Move. "
                    "jakubjirak.medium.com"
                ),
                "url": MEDIUM_JAKUB_URL,
                "snippet": "The revolutionary move that will change everything in autonomous driving.",
                "raw_content": '{"slug":"x-35551ba6bbde","firstPublishedAt":1704067200000}',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-01-01"
    assert cites[0].get("year") == "2024"


def test_citations_medium_jakubjirak_merged_title_snippet_raw_full_calendar(
    no_http_publication_fetch,
) -> None:
    """Tavily-style merged title+snippet+raw; same fixture date as ``firstPublishedAt``."""
    payload = json.dumps(
        [
            {
                "title": (
                    "How Tesla Rewrote The Driving Rules: The Revolutionary Move. "
                    "jakubjirak.medium.com"
                ),
                "url": MEDIUM_JAKUB_URL,
                "snippet": "In 2024, Tesla continued to reshape expectations around autonomy.",
                "raw_content": '{"firstPublishedAt":1704067200000,"postId":"35551ba6bbde"}',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-01-01", cites[0]
    assert cites[0].get("year") == "2024", cites[0]


# User-reported: Zacks / blockchain.news article dates missing in citations.
ZACKS_TESLA_FSD_URL = (
    "https://www.zacks.com/stock/news/2810171/"
    "tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets"
)
BLOCKCHAIN_TESLA_FSD_URL = (
    "https://blockchain.news/ainews/"
    "tesla-fsd-v14-2-2-1-with-grok-navigation-excels-in-snowy-night-driving-"
    "real-world-ai-performance-analysis"
)


def test_calendar_date_us_mdy_with_am_pm() -> None:
    assert calendar_date_from_raw_string("12/27/2025 6:47:00 AM") == "2025-12-27"


def test_zacks_embedded_publish_date_dmy() -> None:
    blob = "var x = { publish_date : '30/12/2025' };"
    assert infer_publication_calendar_date(blob, source_url=ZACKS_TESLA_FSD_URL) == "2025-12-30"


def test_zacks_embedded_placement_date_iso() -> None:
    blob = "articleCta.placementDate = '2025-12-30';"
    assert infer_publication_calendar_date(blob, source_url=ZACKS_TESLA_FSD_URL) == "2025-12-30"


def test_zacks_title_trailing_month_day_year() -> None:
    blob = (
        "<title>Tesla FSD Approaches 7B Miles With 2.5B on Urban Streets - "
        "December 30, 2025 - Zacks.com</title>"
    )
    assert infer_publication_calendar_date(blob, source_url=ZACKS_TESLA_FSD_URL) == "2025-12-30"


def test_blockchain_news_timestamp_div() -> None:
    blob = (
        '<div class="timestamp"><span>12/27/2025 6:47:00 AM</span></div>'
        "<p>Tesla FSD V14.2.2.1 with Grok Navigation</p>"
    )
    assert infer_publication_calendar_date(blob, source_url=BLOCKCHAIN_TESLA_FSD_URL) == "2025-12-27"


def test_citations_zacks_merged_snippet_publish_date(no_http_publication_fetch) -> None:
    """Tavily-style row: no API date; DMY lives only in merged title/snippet/raw."""
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD Approaches 7B Miles. zacks.com",
                "url": ZACKS_TESLA_FSD_URL,
                "snippet": "publish_date : '30/12/2025'",
                "raw_content": "",
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-12-30"
    assert cites[0].get("year") == "2025"


def test_citations_blockchain_news_timestamp_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD V14.2.2.1 with Grok Navigation. blockchain.news",
                "url": BLOCKCHAIN_TESLA_FSD_URL,
                "snippet": "Real-world AI performance analysis.",
                "raw_content": (
                    '<div class="timestamp"><span>12/27/2025 6:47:00 AM</span></div>'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-12-27"
    assert cites[0].get("year") == "2025"


INSTAGRAM_REEL_FSD_URL = "https://www.instagram.com/reel/C-5egDmutbr/"


def test_instagram_reel_infer_from_og_description_shape() -> None:
    """Shape from logged-out Instagram HTML (see ``og:description``)."""
    blob = (
        '<meta property="og:description" content="4,115 likes, 104 comments - teslapro '
        'on August 20, 2024: &quot;Tesla Full Self Driving (FSD) handling the heavy rain'
    )
    assert infer_publication_calendar_date(blob, source_url=INSTAGRAM_REEL_FSD_URL) == "2024-08-20"


def test_citations_instagram_reel_og_description_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": (
                    "FSD 13: Tesla Full Self‑Driving Guide for 2025 - Recharged. instagram.com"
                ),
                "url": INSTAGRAM_REEL_FSD_URL,
                "snippet": "Tesla FSD reel from teslapro.",
                "raw_content": (
                    '<meta property="og:description" content="teslapro on August 20, 2024: '
                    '&quot;FSD in heavy rain&quot;" />'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-08-20", cites[0]
    assert cites[0].get("year") == "2024"


def test_instagram_post_tavily_snippet_date_prefix(no_http_publication_fetch) -> None:
    """Instagram post: Tavily snippet with 'Month D, YYYY —' extracts date without raw_content."""
    from datetime import date, timedelta

    url = "https://www.instagram.com/p/DVuNwuBgeUP"
    payload = json.dumps(
        [
            {
                "title": "The AI Computer Era: How Tesla FSD v14 Solved the Perception. instagram.com",
                "url": url,
                "snippet": "Apr 15, 2025 — The AI Computer Era: How Tesla FSD v14 Solved the Perception problem with end-to-end learning.",
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-04-15", f"got {cites[0].get('published_date')!r}"
    assert cites[0].get("year") == "2025"


def test_instagram_post_tavily_snippet_relative_date(no_http_publication_fetch) -> None:
    """Instagram post: Tavily snippet with 'N months ago' approximates date from today."""
    from datetime import date, timedelta

    today = date.today()
    url = "https://www.instagram.com/p/DVuNwuBgeUP"
    payload = json.dumps(
        [
            {
                "title": "The AI Computer Era: How Tesla FSD v14. instagram.com",
                "url": url,
                "snippet": "3 months ago — Tesla FSD v14 solved the perception problem completely.",
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    expected = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    assert cites[0].get("published_date") == expected, f"got {cites[0].get('published_date')!r}"


# [12] Photo permalink: trailing story id after ``/photos/{slug}/``.
FB_TESLAHK_PHOTO_URL = (
    "https://www.facebook.com/teslahkowners/photos/"
    "tesla-fsd-supervised-motortrend-2026/884784524246026"
)
SINA_CJ_ARTICLE_URL = (
    "https://cj.sina.cn/articles/view/7879848900/1d5acf3c401902tn6a?froms=ggmp"
)


def test_facebook_photo_url_infer_near_post_id() -> None:
    post_id = "884784524246026"
    url = FB_TESLAHK_PHOTO_URL
    blob = (
        '{"creation_time":1500000000}'
        f'{{"story_fbid":"{post_id}","creation_time":1775000000}}'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2026-03-31"


def test_sina_cj_infer_from_weibo_article_create_at() -> None:
    blob = '<meta name="weibo: article:create_at" content="2026-03-20 16:44:11" />'
    assert infer_publication_calendar_date(blob, source_url=SINA_CJ_ARTICLE_URL) == "2026-03-20"


def test_citations_facebook_photo_post_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "将2026 年最佳科技駕駛輔助獎頒發給Tesla FSD. Facebook",
                "url": FB_TESLAHK_PHOTO_URL,
                "snippet": "Tesla FSD Supervised MotorTrend 2026 award.",
                "raw_content": (
                    '{"creation_time":1500000000}'
                    '{"story_fbid":"884784524246026","creation_time":1775000000}'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-03-31", cites[0]
    assert cites[0].get("year") == "2026"


def test_citations_sina_cj_article_meta_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "特斯拉智驾与国内智驾技术对比及竞争格局分析. cj.sina.cn",
                "url": SINA_CJ_ARTICLE_URL,
                "snippet": "截至2026年初，特斯拉FSD已正式引入中国市场。",
                "raw_content": (
                    '<meta name="weibo: article:create_at" content="2026-03-20 16:44:11" />'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-03-20", cites[0]
    assert cites[0].get("year") == "2026"


TESLAACCESSORIES_FSD_V124_BLOG_URL = (
    "https://www.teslaacessories.com/blogs/news/"
    "the-fsd-v12.4-paradigm-shift-unpacking-the-end-to-end-ai-architecture-impact-on-"
    "urban-driving-and-safety-metrics"
)
# Same article with Google Ads ``srsltid`` (long query); path still ``/blogs/news/…``.
TESLAACCESSORIES_FSD_V124_BLOG_URL_WITH_SRSLTID = (
    TESLAACCESSORIES_FSD_V124_BLOG_URL
    + "?srsltid=AfmBOoo9MpbikG09voah0eYUUtDcTlMkULsWulWzuXEJKcpTXBUGXms7"
)
TESLAACCESSORIES_NHTSA_FSD_VISIBILITY_BLOG_URL = (
    "https://www.teslaacessories.com/blogs/news/"
    "nhtsa-escalates-fsd-visibility-investigation-what-it-means-for-3.2-million-us-tesla-owners-"
    "and-how-to-stay-safe?srsltid=AfmBOooPOADMPuWPU_bvikeWIK7bWnJl7yOqZCQVMkQ1g8eUesLThI0g"
)


def test_shopify_blog_published_at_json_beyond_default_head() -> None:
    """``published_at`` often appears after ~100k+ of HTML/JSON; ``/blogs/`` widens head + scans 900k."""
    pad = " " * 120_000
    blob = (
        pad
        + '{"title":"FSD V12.4","published_at":"2024-06-15T14:22:00-05:00","author":"x"}'
    )
    assert (
        infer_publication_calendar_date(blob, source_url=TESLAACCESSORIES_FSD_V124_BLOG_URL)
        == "2024-06-15"
    )


def test_shopify_blog_time_itemprop_datepublished() -> None:
    blob = (
        '<article><h1>FSD</h1>'
        '<time datetime="2024-06-15T09:00:00Z" itemprop="datePublished">June 15, 2024</time>'
        "</article>"
    )
    assert (
        infer_publication_calendar_date(blob, source_url=TESLAACCESSORIES_FSD_V124_BLOG_URL)
        == "2024-06-15"
    )


def test_citations_teslaacessories_blog_published_at_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": (
                    "The FSD V12.4 Paradigm Shift: Unpacking the End-to-End AI. "
                    "teslaacessories.com"
                ),
                "url": TESLAACCESSORIES_FSD_V124_BLOG_URL,
                "snippet": "End-to-end AI architecture and urban driving safety metrics.",
                "raw_content": (
                    '{"article":{"published_at":"2024-06-15T10:00:00-05:00","handle":"fsd-v12-4"}}'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-06-15", cites[0]
    assert cites[0].get("year") == "2024"


def test_citations_teslaacessories_blog_srsltid_url_and_raw_first_merge(
    no_http_publication_fetch,
) -> None:
    """Live SERP URLs carry ``srsltid=…``; merged infer must put *raw* first so ``published_at`` is found."""
    pad = "S" * 400_000
    raw = pad + '{"published_at":"2025-05-10T08:00:00+00:00"}'
    payload = json.dumps(
        [
            {
                "title": "The FSD V12.4 Paradigm Shift. teslaacessories.com",
                "url": TESLAACCESSORIES_FSD_V124_BLOG_URL_WITH_SRSLTID,
                "snippet": "End-to-end AI and safety metrics.",
                "raw_content": raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-05-10", cites[0]
    assert cites[0].get("year") == "2025", cites[0]


def test_citations_x_com_legacy_created_at_merged_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "FSD just hits different. The drive experience. x.com",
                "url": "https://x.com/pbeisel/status/1988637059158863880",
                "snippet": "Full Self-Driving",
                "raw_content": (
                    '{"rest_id":"1988637059158863880",'
                    '"legacy":{"created_at":"Wed Nov 12 16:47:05 +0000 2025"}}'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-11-12", cites[0]
    assert cites[0].get("year") == "2025", cites[0]


def test_teslaacessories_nhtsa_blog_webpage_ld_json_decoy_infer_publication() -> None:
    """Stale ``WebPage`` JSON-LD must not hide Shopify ``published_at`` (user [11])."""
    blob = (
        '<script type="application/ld+json">'
        '{"@type":"WebPage","datePublished":"2020-01-01T00:00:00Z"}'
        "</script>"
        '{"published_at":"2026-03-19T15:30:00-05:00"}'
    )
    assert (
        infer_publication_calendar_date(blob, source_url=TESLAACCESSORIES_NHTSA_FSD_VISIBILITY_BLOG_URL)
        == "2026-03-19"
    )


def test_citations_teslaacessories_nhtsa_blog_merged_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": (
                    "NHTSA Escalates FSD Visibility Investigation What It Means for 3.2. "
                    "teslaacessories.com"
                ),
                "url": TESLAACCESSORIES_NHTSA_FSD_VISIBILITY_BLOG_URL,
                "snippet": "NHTSA investigation and Tesla FSD visibility.",
                "raw_content": (
                    '<script type="application/ld+json">'
                    '{"@type":"WebPage","datePublished":"2020-01-01T00:00:00Z"}'
                    "</script>"
                    '{"published_at":"2026-03-19T15:30:00-05:00"}'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-03-19", cites[0]
    assert cites[0].get("year") == "2026", cites[0]


EET_CHINA_FSD_MP_URL = "https://www.eet-china.com/mp/a472559.html"


def test_sina_cj_span_date_when_head_meta_missing() -> None:
    """Tavily may drop ``<head>``; visible ``<span class=…date…>YYYY年MM月DD日`` matches live cj.sina.cn."""
    blob = (
        '<div class="main"><p>截至2026年初，特斯拉FSD已引入中国市场。</p>'
        '<span class="date">2026年03月20日 16:44</span></div>'
    )
    assert (
        infer_publication_calendar_date(blob, source_url=SINA_CJ_ARTICLE_URL) == "2026-03-20"
    )


def test_sina_cj_json_pagepubtime_string_not_used_as_publish_date() -> None:
    """Shell JS may embed ``"pagepubtime":"2016-…"`` (template); real row is weibo meta or span."""
    blob = (
        '{"pagepubtime":"2016-05-30","noise":1}'
        '<meta name="weibo: article:create_at" content="2026-03-20 16:44:11" />'
    )
    assert infer_publication_calendar_date(blob, source_url=SINA_CJ_ARTICLE_URL) == "2026-03-20"


def test_eet_china_mp_publish_time_cn_line() -> None:
    blob = (
        '<div class="article-meta">发布时间：2026年01月22日</div>'
        "<p>揭秘特斯拉FSD 核心：端到端算法…</p>"
    )
    assert infer_publication_calendar_date(blob, source_url=EET_CHINA_FSD_MP_URL) == "2026-01-22"


def test_eet_china_mp_article_published_time_meta() -> None:
    blob = (
        '<head><meta property="article:published_time" content="2026-01-22T10:30:00+08:00" /></head>'
        "<body><p>正文</p></body>"
    )
    assert infer_publication_calendar_date(blob, source_url=EET_CHINA_FSD_MP_URL) == "2026-01-22"


def test_citations_eet_china_mp_merged_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "揭秘特斯拉FSD 核心：端到端算法的“三大难点”. eet-china.com",
                "url": EET_CHINA_FSD_MP_URL,
                "snippet": "端到端算法与智驾方案对比。",
                "raw_content": "<p>发布时间：2026年02月10日</p>",
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-02-10", cites[0]
    assert cites[0].get("year") == "2026"


def test_citations_sina_cj_merged_span_date_only(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "特斯拉智驾与国内智驾技术对比及竞争格局分析. cj.sina.cn",
                "url": SINA_CJ_ARTICLE_URL,
                "snippet": "中美智驾技术竞争。",
                "raw_content": '<span class="date">2026年03月20日 16:44</span>',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-03-20", cites[0]
    assert cites[0].get("year") == "2026"


EEWORLD_FSD_ARTICLE_URL = "https://www.eeworld.com.cn/qcdz/eic682225.html"

FUSA_51_INFORMATION_URL = (
    "https://www.51fusa.com/client/information/informationdetail/id/2045.html"
)
WALLSTREETCN_TESLA_ARTICLE_URL = "https://wallstreetcn.com/articles/3724247"


def test_citations_eeworld_fsd_article_merged_update_line(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "特斯拉FSD，从全栈自研到智能驾驶的未来 - 电子工程世界. eeworld.com.cn",
                "url": EEWORLD_FSD_ARTICLE_URL,
                "snippet": "FSD 全栈自研与智能驾驶。",
                "raw_content": (
                    "<p>发布者：SparklingRiver</p>"
                    "<p>最新更新时间：2024-10-29 来源: 智驾最前沿</p>"
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-10-29", cites[0]
    assert cites[0].get("year") == "2024", cites[0]


def test_citations_51fusa_raw_article_published_time_resolves_2021(
    no_http_publication_fetch,
) -> None:
    """With Tavily-style ``raw_content`` (real meta), must get 2021 — not 2045 from the URL."""
    raw = '<meta property="article:published_time" content="2021-08-20T08:00:00+08:00" />'
    payload = json.dumps(
        [
            {
                "title": "Tesla AI Day决策规划部分解析 - 功能安全. 51fusa.com",
                "url": FUSA_51_INFORMATION_URL,
                "snippet": "解析",
                "raw_content": raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2021-08-20", cites[0]
    assert cites[0].get("year") == "2021", cites[0]


FUSA_51_KNOWLEDGE_URL = (
    "https://www.51fusa.com/client/knowledge/knowledgedetail/id/2045.html"
)


def test_citations_51fusa_knowledge_url_title_echo_id_not_year_2021(
    no_http_publication_fetch,
) -> None:
    """SERP title may end with ``. 2045`` (article id); year must still be **2021** from meta."""
    raw = '<meta property="article:published_time" content="2021-08-20T08:00:00+08:00" />'
    payload = json.dumps(
        [
            {
                "title": "Tesla AI Day决策规划部分解析. 51fusa.com. 2045",
                "url": FUSA_51_KNOWLEDGE_URL,
                "snippet": "功能安全 决策规划",
                "raw_content": raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2021-08-20", cites[0]
    assert cites[0].get("year") == "2021", cites[0]
    assert "2045" not in (cites[0].get("title") or ""), cites[0]


def test_citations_xueqiu_created_at_in_merged_raw(no_http_publication_fetch) -> None:
    payload = json.dumps(
        [
            {
                "title": "特斯拉FSD自动驾驶的信息更新. xueqiu.com",
                "url": "https://xueqiu.com/319838879/321301428",
                "snippet": "FSD 进展",
                "raw_content": r'{"status":{"created_at":1609459200000}}',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2021-01-01", cites[0]
    assert cites[0].get("year") == "2021", cites[0]


def test_citations_wallstreetcn_article_time_in_merged_raw(
    no_http_publication_fetch,
) -> None:
    """Uses **synthetic** article DOM (full render / rich extract), not empty SPA document."""
    raw = (
        "<article><header>"
        '<time datetime="2024-08-15T12:25:49.000Z">2024-08-15</time>'
        "</header><p>FSD V12</p></article>"
    )
    payload = json.dumps(
        [
            {
                "title": "万字硬核解读：端到端与特斯拉FSD V12. wallstreetcn.com",
                "url": WALLSTREETCN_TESLA_ARTICLE_URL,
                "snippet": "特斯拉",
                "raw_content": raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2024-08-15", cites[0]
    assert cites[0].get("year") == "2024"


# User-reported: teslaacessories.com FSD v14.2.2.5 blog - JSON-LD beyond 900K head.
# The real page has datePublished at ~1.07MB in a JSON-LD block; /blogs/ path must
# expand the JSON-LD scan window to 1.5M to reach it.
TESLAACCESSORIES_FSD_V1422_BLOG_URL = (
    "https://www.teslaacessories.com/blogs/news/"
    "tesla-fsd-supervised-v14.2.2.5-latest-safety-breakthroughs-transforming-drives-for-us-and-european-owners"
    "?srsltid=AfmBOoq-gOVFKjcdjDREIRFPQGwc0bawJsUnVQo-JHmdU2RdcGZoY6IH"
)


def test_shopify_blog_json_ld_beyond_900k_head() -> None:
    """`datePublished` in JSON-LD at >1MB must be found for ``/blogs/`` paths (1.5M scan cap)."""
    pad = "x" * 1_000_000  # 1MB padding to push JSON-LD past default 900K cap
    blob = (
        pad
        + '<script type="application/ld+json">'
        '{"@type":"BlogPosting","headline":"Tesla FSD v14.2.2.5",'
        '"datePublished":"2026-03-12T09:58:00Z","dateCreated":"2026-03-12T09:44:19Z"}'
        "</script>"
    )
    assert (
        infer_publication_calendar_date(blob, source_url=TESLAACCESSORIES_FSD_V1422_BLOG_URL)
        == "2026-03-12"
    )


def test_citations_teslaacessories_v1422_blog_json_ld_deep_raw(no_http_publication_fetch) -> None:
    """Tavily raw_content: JSON-LD at >1MB in /blogs/ page must resolve to calendar date."""
    pad = "x" * 1_000_000
    raw = (
        pad
        + '<script type="application/ld+json">'
        '{"@type":"BlogPosting","datePublished":"2026-03-12T09:58:00Z"}'
        "</script>"
    )
    payload = json.dumps(
        [
            {
                "title": (
                    "Tesla FSD Supervised v14.2.2.5 Latest Safety Breakthroughs. "
                    "teslaacessories.com"
                ),
                "url": TESLAACCESSORIES_FSD_V1422_BLOG_URL,
                "snippet": "Tesla FSD Supervised v14.2.2.5 latest safety breakthroughs.",
                "raw_content": raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-03-12", cites[0]
    assert cites[0].get("year") == "2026", cites[0]
