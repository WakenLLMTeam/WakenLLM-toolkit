"""Tests for publication-date extraction (calendar YYYY-MM-DD)."""

import pytest

from deerflow.utils.publication_date import (
    calendar_date_from_raw_string,
    infer_publication_calendar_date,
    infer_publication_date_from_text,
    infer_reddit_post_calendar_date,
)


def test_empty():
    assert infer_publication_calendar_date("") == ""
    assert infer_publication_calendar_date("   ") == ""


def test_json_ld_date_published_calendar():
    html = '<script type="application/ld+json">{"@type":"Article","datePublished":"2023-11-08T12:00:00Z"}</script>'
    assert infer_publication_calendar_date(html) == "2023-11-08"


def test_meta_article_published_time_content_first():
    html = '<meta content="2024-05-01" property="article:published_time" />'
    assert infer_publication_calendar_date(html) == "2024-05-01"


def test_meta_article_published_time_property_first():
    html = '<meta property="article:published_time" content="2022-03-15T08:30:00+00:00" />'
    assert infer_publication_calendar_date(html) == "2022-03-15"


def test_bare_time_element_not_used_as_publish():
    """Plain ``<time datetime>`` is not treated as article publish (use meta / JSON-LD)."""
    html = '<article><time datetime="2019-07-20">July 20, 2019</time></article>'
    assert infer_publication_calendar_date(html) == ""


def test_wallstreetcn_article_time_in_first_article_block_calendar():
    """**Synthetic** full-render HTML: first ``<article>`` + ``<time datetime>`` → 2024-08-15.

    This is *not* what a plain HTTP GET of the live PC page returns (see SPA shell test below).
    """
    html = (
        "<article><header>"
        '<time datetime="2024-08-15T12:25:49.000Z">2024-08-15</time>'
        "</header><p>Body</p></article>"
        '<div class="related"><time datetime="2020-01-01">old</time></div>'
    )
    assert (
        infer_publication_calendar_date(
            html, source_url="https://wallstreetcn.com/articles/3724247"
        )
        == "2024-08-15"
    )


def test_wallstreetcn_live_pc_document_is_spa_shell_without_publish_in_html() -> None:
    """Live ``wallstreetcn.com/articles/…`` is a JS app: initial HTML has no article/time/JSON date.

    Search snippets that only contain this shell cannot yield 2024 — tests that assert 2024 use
    **fixture** HTML/JSON, not this shape.
    """
    blob = (
        "<!doctype html><html><head><meta charset=\"utf-8\"></head>"
        '<body><div id="app"></div>'
        '<script src="https://static.wscn.net/ivanka-pc/17e6206e52813a2fd977.js"></script>'
        "</body></html>"
    )
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://wallstreetcn.com/articles/3724247"
        )
        == ""
    )


def test_wallstreetcn_boot_json_published_at_when_no_article_tag() -> None:
    """When inline boot JSON includes ``published_at`` / ``display_time``, extract calendar date."""
    blob = (
        "<script>window.__INITIAL_STATE__="
        '{"article":{"published_at":"2024-08-15T12:25:49.000Z","title":"x"}}'
        "</script>"
    )
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://wallstreetcn.com/articles/3724247"
        )
        == "2024-08-15"
    )


def test_51fusa_informationdetail_meta_not_confused_with_id_2045():
    """Correct publish calendar is 2021 (Tesla AI Day 材料); URL ``2045`` is id only.

    ``article:published_time`` must win — never treat ``2045`` in the path as the year.
    """
    url = "https://www.51fusa.com/client/information/informationdetail/id/2045.html"
    blob = '<meta property="article:published_time" content="2021-08-20T08:00:00+08:00" />'
    assert infer_publication_calendar_date(blob, source_url=url) == "2021-08-20"


def test_51fusa_knowledgedetail_path_same_meta_resolution():
    """``/client/knowledge/knowledgedetail/id/2045.html`` uses the same id pattern as *information*."""
    url = "https://www.51fusa.com/client/knowledge/knowledgedetail/id/2045.html"
    blob = '<meta property="article:published_time" content="2021-08-20T08:00:00+08:00" />'
    assert infer_publication_calendar_date(blob, source_url=url) == "2021-08-20"


def test_xueqiu_status_created_at_json_ms():
    """Snowball status pages embed ``created_at`` as Unix ms (e.g. ``/319838879/321301428``)."""
    url = "https://xueqiu.com/319838879/321301428"
    # 2021-01-01 00:00:00 UTC
    blob = r'"created_at":1609459200000,"title":"特斯拉FSD"'
    assert infer_publication_calendar_date(blob, source_url=url) == "2021-01-01"


def test_eet_china_publish_time_beats_generic_webpage_json_ld():
    """Visible ``发布时间`` must win over a shell ``WebPage`` ``datePublished`` (often wrong)."""
    url = "https://www.eet-china.com/mp/a472559.html"
    blob = (
        '<script type="application/ld+json">'
        '{"@type":"WebPage","datePublished":"2045-01-01T00:00:00Z"}'
        "</script>"
        '<div class="meta">发布时间：<strong>2024</strong>年<strong>03</strong>月<strong>15</strong>日</div>'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-03-15"


def test_x_com_legacy_created_at_near_status_id():
    """X / Twitter GraphQL-style JSON: ``legacy.created_at`` uses RFC2822-style strings."""
    url = "https://x.com/pbeisel/status/1988637059158863880"
    blob = (
        '{"rest_id":"1988637059158863880","legacy":'
        '{"created_at":"Wed Nov 12 16:47:05 +0000 2025","full_text":"FSD"}}'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-11-12"


def test_x_com_iso_created_at_beats_stale_webpage_json_ld():
    """Tweet ``created_at`` (ISO) is parsed before generic ``WebPage`` JSON-LD."""
    url = "https://x.com/user/status/1234567890123456789"
    blob = (
        '<script type="application/ld+json">'
        '{"@type":"WebPage","datePublished":"2010-01-01T00:00:00Z"}</script>'
        '{"rest_id":"1234567890123456789",'
        '"created_at":"2025-06-01T10:00:00.000Z"}'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-06-01"


def test_shopify_blog_published_at_lost_when_only_at_end_after_scan_cap_padding():
    """``_try_cms_blog_article_publish`` scans 1.5M: ``published_at`` after that prefix of junk is invisible."""
    url = "https://www.teslaacessories.com/blogs/news/fsd-v12-4-paradigm?srsltid=test"
    raw = '{"published_at":"2025-07-20T12:00:00+00:00"}'
    pad = "P" * 1_550_000
    assert infer_publication_calendar_date(pad + "\n" + raw, source_url=url) == ""
    assert infer_publication_calendar_date(raw + "\n" + pad, source_url=url) == "2025-07-20"


def test_rejects_url_like_match():
    blob = '"datePublished": "https://evil.example/phishing"'
    assert infer_publication_calendar_date(blob) == ""


def test_created_utc_json_calendar():
    blob = '{"data": {"children": [{"data": {"created_utc": 1700000000.0}}]}}'
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://www.reddit.com/r/x/comments/y/z/"
        )
        == "2023-11-14"
    )


def test_infer_reddit_ignores_visible_year_prefers_created_utc():
    """Reddit-only helper must not use prose/snippet years (sidebar chrome)."""
    blob = (
        'Posted on January 15, 2026\n<body>'
        '{"created_utc": 1700000000.0}\n'
        "</body>"
    )
    assert infer_reddit_post_calendar_date(blob) == "2023-11-14"


def test_infer_publication_prefers_created_utc_before_heading():
    blob = (
        'Posted on January 15, 2026\n<body>'
        '{"created_utc": 1700000000.0}\n'
        "</body>"
    )
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://old.reddit.com/r/x/comments/y/z/"
        )
        == "2023-11-14"
    )


def test_blog_json_ld_upload_date_not_used_as_article_publish():
    """Embedded VideoObject ``uploadDate`` must not beat absence of article date."""
    html = '<script type="application/ld+json">{"@type":"VideoObject","uploadDate":"2016-03-01T12:00:00Z"}</script>'
    assert infer_publication_calendar_date(html) == ""


def test_data_time_attr_not_used_as_publish():
    blob = '<a href="#" class="timestamp" data-time="1700000000">link</a>'
    assert infer_publication_calendar_date(blob) == ""


def test_posted_on_line_calendar():
    blob = "Some intro\nPosted on January 15, 2022\n\nBody here"
    assert infer_publication_calendar_date(blob) == "2022-01-15"


def test_pubdate_rss_calendar():
    blob = "<channel><item><pubDate>Mon, 03 Jun 2024 08:00:00 GMT</pubDate></item></channel>"
    assert infer_publication_calendar_date(blob) == "2024-06-03"


def test_generic_timestamp_json_not_used():
    """Generic ``timestamp`` must not be treated as publication time."""
    blob = '{"timestamp": 1700000000, "updated": true}'
    assert infer_publication_calendar_date(blob) == ""


def test_calendar_date_from_raw_helpers():
    assert calendar_date_from_raw_string("2024-01-02T00:00:00Z") == "2024-01-02"
    assert calendar_date_from_raw_string("Mon, 15 Jan 2024 12:00:00 GMT") == "2024-01-15"


def test_modified_meta_used_as_fallback():
    """``article:modified_time`` alone is accepted as a fallback when no publish date exists."""
    html = '<meta property="article:modified_time" content="2025-01-01T00:00:00Z" />'
    # publish date takes priority; when absent, modified_time is the best-effort result
    assert infer_publication_calendar_date(html) == "2025-01-01"

    # published_time must beat modified_time
    html2 = (
        '<meta property="article:published_time" content="2022-06-01T00:00:00Z" />'
        '<meta property="article:modified_time" content="2025-01-01T00:00:00Z" />'
    )
    assert infer_publication_calendar_date(html2) == "2022-06-01"


def test_pep_created_block_calendar():
    blob = "PEP 719\nCreated:\n:   26-May-2023\nStatus:\n:   Active\n"
    assert infer_publication_calendar_date(blob) == "2023-05-26"


def test_reddit_top_posts_footer_not_used_as_publish_date():
    """Sidebar/footer aggregates like ``Top posts of …`` are not the thread publish date."""
    blob = "Post body\nreReddit: Top posts of October 7, 2024\n"
    assert infer_publication_calendar_date(blob) == ""


def test_csdn_visible_first_publish_banner_calendar():
    """CSDN keeps this in HTML even when ``<script>`` / JSON-LD is stripped by fetchers."""
    blob = (
        '<div class="up-time"><span>于&nbsp;2022-03-20 14:23:49&nbsp;首次发布</span></div>'
    )
    assert infer_publication_calendar_date(blob) == "2022-03-20"


def test_csdn_post_time_var_calendar():
    blob = '       var postTime = "2022-03-20 14:23:49"\n'
    assert infer_publication_calendar_date(blob) == "2022-03-20"


def test_youtube_json_publish_date_calendar():
    assert (
        infer_publication_calendar_date(
            '"publishDate":"2024-08-25T02:00:09-07:00"',
            source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        )
        == "2024-08-25"
    )


def test_youtube_meta_deep_in_large_html_calendar():
    """YouTube puts itemprop / JSON dates far past the default 96k scan window."""
    pad = "x" * 100_000
    tail = '<meta itemprop="datePublished" content="2024-08-25T02:00:09-07:00">'
    assert (
        infer_publication_calendar_date(
            pad + tail, source_url="https://www.youtube.com/watch?v=test"
        )
        == "2024-08-25"
    )


def test_youtube_ytinitial_player_response_calendar():
    """``ytInitialPlayerResponse`` blob (common when extractors keep scripts, not meta)."""
    html = (
        "<script>var ytInitialPlayerResponse = "
        '{"microformat":{"playerMicroformatRenderer":'
        '{"publishDate":"2024-03-18T14:22:05-07:00"}}};</script>'
    )
    assert (
        infer_publication_calendar_date(
            html, source_url="https://www.youtube.com/watch?v=hlb5WRN0SGg"
        )
        == "2024-03-18"
    )


def test_youtube_watch_player_publish_beats_json_ld_webpage():
    """WebPage JSON-LD on watch pages can predate the real upload; player blob wins."""
    url = "https://www.youtube.com/watch?v=w43uWn6Zhcg"
    html = (
        '<script type="application/ld+json">'
        '{"@type":"WebPage","datePublished":"2024-06-01T12:00:00Z"}'
        "</script>"
        "<script>var ytInitialPlayerResponse = "
        '{"microformat":{"playerMicroformatRenderer":'
        '{"publishDate":"2025-01-01T08:00:00Z"}}};</script>'
    )
    assert infer_publication_calendar_date(html, source_url=url) == "2025-01-01"


def test_youtube_watch_player_publish_beats_og_meta():
    url = "https://www.youtube.com/watch?v=w43uWn6Zhcg"
    html = (
        '<meta property="og:published_time" content="2024-12-01T00:00:00Z" />'
        "<script>var ytInitialPlayerResponse = "
        '{"microformat":{"playerMicroformatRenderer":'
        '{"publishDate":"2025-01-01T08:00:00Z"}}};</script>'
    )
    assert infer_publication_calendar_date(html, source_url=url) == "2025-01-01"


def test_reddit_graphql_created_at_iso_calendar():
    blob = '{"__typename":"Post","id":"t3_abc","createdAt":"2025-04-02T18:30:00.000Z"}'
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://www.reddit.com/r/TeslaFSD/comments/1jwxbfd/x"
        )
        == "2025-04-02"
    )


def test_infer_reddit_created_at_when_no_created_utc():
    assert infer_reddit_post_calendar_date(
        '{"post":{"createdAt":"2024-01-20T12:00:00+00:00"}}'
    ) == "2024-01-20"


def test_facebook_embedded_creation_time_calendar():
    blob = '<script>{"creation_time":1700000000}</script>'
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://www.facebook.com/groups/3258771700819965/posts/25635668892703593"
        )
        == "2023-11-14"
    )


def test_facebook_prefers_creation_time_near_url_post_id():
    """Global-first ``creation_time`` is often group/chrome; the post id anchors the real story."""
    decoy = '{"creation_time":1600000000}'
    block = '{"legacyId":"1166359048819502","creation_time":1735689600}'
    blob = decoy + block
    url = "https://www.facebook.com/groups/teslaownersaustralia/posts/1166359048819502"
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-01-01"


def test_facebook_embedded_creation_time_milliseconds_calendar():
    """Graph-style payloads often use 13-digit epoch milliseconds."""
    blob = '<script>{"creation_time":1700000000000}</script>'
    assert (
        infer_publication_calendar_date(
            blob, source_url="https://www.facebook.com/groups/3258771700819965/posts/25635668892703593"
        )
        == "2023-11-14"
    )


def test_citation_date_meta_slash_format_calendar():
    blob = '<meta name="citation_date" content="2017/06/12" />'
    assert infer_publication_calendar_date(blob) == "2017-06-12"


def test_calendar_date_slash_ymd_string():
    from deerflow.utils.publication_date import calendar_date_from_raw_string

    assert calendar_date_from_raw_string("2017/06/12") == "2017-06-12"


def test_zhihu_zhuanlan_published_time_scoped_to_article_id():
    """知乎专栏: ``publishedTime`` must pair with the ``/p/{id}`` from *source_url*."""
    url = "https://zhuanlan.zhihu.com/p/572974435"
    blob = (
        '{"feed":[{"publishedTime":1600000000000}],'
        '"entities":{"articles":{"572974435":'
        '{"title":"《2022年Tesla AI Day——FSD技术进展分析总结》",'
        '"publishedTime":1666262400000}}}}'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2022-10-20"


def test_zhihu_zhuanlan_published_time_quoted_string_epoch():
    """Some builds quote the epoch in JSON strings."""
    url = "https://zhuanlan.zhihu.com/p/572974435"
    blob = (
        '{"entities":{"articles":{"572974435":'
        '{"publishedTime":"1666262400000","title":"x"}}}}'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2022-10-20"


def test_calendar_date_chinese_ymd():
    from deerflow.utils.publication_date import calendar_date_from_raw_string

    assert calendar_date_from_raw_string("2022年10月20日") == "2022-10-20"
    assert calendar_date_from_raw_string("2022年1月5日") == "2022-01-05"
    assert calendar_date_from_raw_string("正文 2022年12月31日 结尾") == "2022-12-31"
    assert calendar_date_from_raw_string("2022 年 10 月 20 日") == "2022-10-20"


def test_zhihu_markdown_publish_inline_chinese_and_iso():
    url = "https://zhuanlan.zhihu.com/p/572974435"
    md_cn = "# 标题\n\n某作者 · 发布于 2022年10月20日 · 上海\n\n正文"
    assert infer_publication_calendar_date(md_cn, source_url=url) == "2022-10-20"
    md_cn_spaced = "作者 · 发布于 2022 年 10 月 20 日 · 著作权\n"
    assert infer_publication_calendar_date(md_cn_spaced, source_url=url) == "2022-10-20"
    md_iso = "发表于：2022-10-20 18:40\n\n段落"
    assert infer_publication_calendar_date(md_iso, source_url=url) == "2022-10-20"


def test_zhihu_en_answer_dom_edit_line_calendar():
    """``/en/answer`` Web UI embeds ``Edit <!-- -->YYYY-MM-DD HH:MM`` (not 发布于)."""
    url = "https://www.zhihu.com/en/answer/3246911414"
    html = '<div class="meta">Edit <!-- -->2023-11-08 19:26</div><p>Body.</p>'
    assert infer_publication_calendar_date(html, source_url=url) == "2023-11-08"


def test_facebook_creation_time_beyond_legacy_8kb_window():
    """Widen JSON scan so ``creation_time`` after ``story_fbid`` is still found."""
    post_id = "1995955427558350"
    url = f"https://www.facebook.com/groups/1242309506256283/posts/{post_id}"
    pad = "x" * 15000
    blob = (
        '{"story_fbid":"'
        + post_id
        + '","padding":"'
        + pad
        + '","creation_time":1735689600}'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-01-01"


def test_facebook_photo_permalink_extracts_trailing_post_id():
    """``/USER/photos/SLUG/POST_ID`` — anchor ``creation_time`` on trailing numeric id."""
    post_id = "884784524246026"
    url = (
        "https://www.facebook.com/teslahkowners/photos/"
        "tesla-fsd-supervised-motortrend-2026/884784524246026"
    )
    decoy = '{"creation_time":1600000000}'
    real = f'{{"story_fbid":"{post_id}","creation_time":1775000000}}'
    assert infer_publication_calendar_date(decoy + real, source_url=url) == "2026-03-31"


def test_sina_cj_weibo_article_create_at_meta():
    """cj.sina.cn articles expose publish time via ``weibo: article:create_at``."""
    url = "https://cj.sina.cn/articles/view/7879848900/1d5acf3c401902tn6a?froms=ggmp"
    blob = '<meta name="weibo: article:create_at" content="2026-03-20 16:44:11" />'
    assert infer_publication_calendar_date(blob, source_url=url) == "2026-03-20"


def test_infer_heading_line_发表于_calendar():
    blob = "发表于 2022-11-03\n\n正文"
    assert (
        infer_publication_calendar_date(blob, source_url="https://zhuanlan.zhihu.com/p/1")
        == "2022-11-03"
    )


def test_cn_portal_publish_time_unix_ms_calendar():
    """36氪等常用 ``publishTime`` 毫秒时间戳（JSON 内嵌）。"""
    blob = '{"detailWidget":{"publishTime":1704067200000,"title":"x"}}'
    assert infer_publication_calendar_date(blob) == "2024-01-01"


def test_36kr_mobile_byline_beats_meta_footer_and_sidebar_iso():
    """``m.36kr.com/p/…``: visible ``·2025年MM月DD日`` beats shell meta, ``© … 2026``, and sidebar ISO."""
    url = "https://m.36kr.com/p/3519984081869697"
    pad_nav = "x" * 8000
    meta_bad = '<meta property="article:published_time" content="2026-03-30T00:00:00Z" />'
    footer = "© 2011~ 2026 北京多氪信息科技有限公司"
    sidebar = "最近内容 2026-03-25"
    byline = "智能车参考·2025年10月22日 15:58"
    blob = pad_nav + meta_bad + footer + sidebar + byline
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-10-22"


def test_36kr_byline_beats_publish_time_json_when_host_matches():
    """With 36kr host, author middot byline wins over earlier ``publishTime`` millis (wrong shell)."""
    url = "https://www.36kr.com/p/2492318105786505"
    bad_ms = "1767225600000"  # 2026-01-01T00:00:00Z in ms
    blob = f'{{"publishTime":{bad_ms}}}' + "x" * 200 + "专栏作者·2023年08月10日"
    assert infer_publication_calendar_date(blob, source_url=url) == "2023-08-10"


def test_36kr_falls_back_to_publish_time_when_no_byline():
    """Without ``·年…月…日`` row, 36kr pages still use embedded ``publishTime`` like other CN portals."""
    url = "https://m.36kr.com/p/3519984081869697"
    blob = '{"detailWidget":{"publishTime":1704067200000,"title":"x"}}'
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-01-01"


def test_cn_portal_publish_time_iso_string_calendar():
    blob = '{"article":{"published_at":"2023-06-15"}}'
    assert infer_publication_calendar_date(blob) == "2023-06-15"


def test_cn_portal_publish_time_beyond_default_scan_window():
    pad = "x" * 100_000
    tail = '"widgetData":{"publishTime":1704067200000}'
    assert infer_publication_calendar_date(pad + tail, max_scan_chars=96_000) == "2024-01-01"


def test_yahoo_msn_portal_embedded_iso_calendar():
    blob = '{"boot":{"contentPublishedDate":"2024-08-02T14:00:00.000Z"}}'
    assert (
        infer_publication_calendar_date(blob, source_url="https://www.yahoo.com/news/test-article")
        == "2024-08-02"
    )


def test_json_ld_deep_in_page_for_yahoo_host():
    """Yahoo/MSN shells may place ``application/ld+json`` after large inline boot payloads."""
    ld = (
        '<script type="application/ld+json">'
        '{"@type":"NewsArticle","datePublished":"2021-06-01T00:00:00Z"}'
        "</script>"
    )
    pad = "p" * 950_000
    assert (
        infer_publication_calendar_date(
            pad + ld,
            source_url="https://news.yahoo.com/articles/example-slug",
        )
        == "2021-06-01"
    )


def test_x_created_at_near_status_id_beats_earlier_timeline_field():
    noise = '{"created_at":"2015-01-01T00:00:00.000Z"}'
    story = '"rest_id":"1999888777666555444","created_at":"2024-07-15T12:00:00.000Z"'
    blob = noise + story
    url = "https://x.com/someuser/status/1999888777666555444"
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-07-15"


def test_instagram_taken_at_timestamp_calendar():
    blob = '<script>window._sharedData={"taken_at_timestamp":1700000000}</script>'
    assert (
        infer_publication_calendar_date(blob, source_url="https://www.instagram.com/p/AbCdEfGh/")
        == "2023-11-14"
    )


def test_instagram_reel_og_description_on_month_date():
    """Logged-out reel HTML: post date in ``og:description`` (``on August 20, 2024:``)."""
    url = "https://www.instagram.com/reel/C-5egDmutbr/"
    blob = (
        '<meta property="og:description" content="4,115 likes - teslapro on August 20, 2024: '
        '&quot;Tesla Full Self Driving (FSD) handling the heavy rain&quot;" />'
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-08-20"


def test_instagram_taken_at_ms_epoch_near_shortcode():
    """Prefer ``taken_at`` in a window around the URL shortcode when multiple exist."""
    url = "https://www.instagram.com/reel/C-5egDmutbr/"
    decoy = '{"shortcode":"OTHER","taken_at_timestamp":1600000000}'
    near = (
        '{"permalink":"/reel/C-5egDmutbr/","taken_at_timestamp":1724112000000}'
    )  # 2024-08-20 UTC
    blob = decoy + near
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-08-20"


def test_tiktok_create_time_near_video_id():
    vid = "7123456789012345678"
    decoy = '{"createTime":1600000000}'
    block = f'"id":"{vid}","createTime":1735689600'
    blob = decoy + block
    url = f"https://www.tiktok.com/@user/video/{vid}"
    assert infer_publication_calendar_date(blob, source_url=url) == "2025-01-01"


TESLAACCESSORIES_NHTSA_BLOG_URL = (
    "https://www.teslaacessories.com/blogs/news/"
    "nhtsa-escalates-fsd-visibility-investigation-what-it-means-for-3.2-million-us-tesla-owners-"
    "and-how-to-stay-safe?srsltid=AfmBOooPOADMPuWPU_bvikeWIK7bWnJl7yOqZCQVMkQ1g8eUesLThI0g"
)


def test_teslaacessories_nhtsa_blog_json_ld_webpage_decoy_does_not_hide_published_at() -> None:
    """Shopify themes emit ``WebPage`` JSON-LD with a non-article date; ``published_at`` must win.

    Slug contains ``3.2`` (million owners) — must not be confused with a year.
    """
    blob = (
        '<script type="application/ld+json">'
        '{"@type":"WebPage","datePublished":"2020-01-01T00:00:00Z"}'
        "</script>"
        "<p>3.2 million US Tesla owners — not a date token.</p>"
        '{"published_at":"2026-03-19T14:00:00-05:00"}'
    )
    assert (
        infer_publication_calendar_date(blob, source_url=TESLAACCESSORIES_NHTSA_BLOG_URL)
        == "2026-03-19"
    )


def test_eeworld_article_latest_update_time_line_calendar() -> None:
    """eeworld.com.cn byline (reader/Tavily shape): ``最新更新时间：YYYY-MM-DD``."""
    url = "https://www.eeworld.com.cn/qcdz/eic682225.html"
    blob = (
        "<p>发布者：<a href=\"/u\">SparklingRiver</a>"
        "最新更新时间：2024-10-29 来源: 智驾最前沿</p>"
    )
    assert infer_publication_calendar_date(blob, source_url=url) == "2024-10-29"


def test_teslaacessories_blog_created_at_json_fallback() -> None:
    """Some Shopify payloads expose ``created_at`` when ``published_at`` is absent."""
    url = (
        "https://www.teslaacessories.com/blogs/news/"
        "nhtsa-escalates-fsd-visibility-investigation-what-it-means-for-3.2-million-us-tesla-owners-"
        "and-how-to-stay-safe"
    )
    blob = '{"article":{"created_at":"2026-03-19T19:00:00.000Z"}}'
    assert infer_publication_calendar_date(blob, source_url=url) == "2026-03-19"


_USER_REPORTED_FIXTURE_MATRIX = [
    (
        "teslaacc_nhtsa_meta",
        TESLAACCESSORIES_NHTSA_BLOG_URL,
        '<meta property="article:published_time" content="2026-03-19T12:00:00Z" />',
        "2026-03-19",
    ),
    (
        "teslaacc_nhtsa_published_at",
        TESLAACCESSORIES_NHTSA_BLOG_URL,
        '{"published_at":"2026-03-19T08:00:00+00:00"}',
        "2026-03-19",
    ),
    (
        "51fusa_id2045_meta",
        "https://www.51fusa.com/client/information/informationdetail/id/2045.html",
        '<meta property="article:published_time" content="2021-08-20T08:00:00+08:00" />',
        "2021-08-20",
    ),
    (
        "wallstreetcn_article_time",
        "https://wallstreetcn.com/articles/3724247",
        (
            '<article><time datetime="2024-08-15T12:25:49.000Z">'
            "2024-08-15</time></article>"
        ),
        "2024-08-15",
    ),
    (
        "finviz_meta",
        "https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets",
        '<meta property="article:published_time" content="2024-12-15T10:30:00Z" />',
        "2024-12-15",
    ),
    (
        "zacks_publish_date",
        "https://www.zacks.com/stock/news/2810171/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets",
        "var x = { publish_date : '30/12/2025' };",
        "2025-12-30",
    ),
    (
        "blockchain_timestamp_div",
        "https://blockchain.news/ainews/"
        "tesla-fsd-v14-2-2-1-with-grok-navigation-excels-in-snowy-night-driving-"
        "real-world-ai-performance-analysis",
        '<div class="timestamp"><span>12/27/2025 6:47:00 AM</span></div>',
        "2025-12-27",
    ),
    (
        "instagram_og_description",
        "https://www.instagram.com/reel/C-5egDmutbr/",
        (
            '<meta property="og:description" content="teslapro on August 20, 2024: '
            '&quot;FSD&quot;" />'
        ),
        "2024-08-20",
    ),
    (
        "zhihu_answer_entity",
        "https://www.zhihu.com/answer/3246911414",
        '"entities":{"answers":{"3246911414":{"publishedTime":1704067200000}}}',
        "2024-01-01",
    ),
    (
        "facebook_group_post",
        "https://www.facebook.com/groups/1242309506256283/posts/1995955427558350",
        '{"story_fbid":"1995955427558350","creation_time":1735689600}',
        "2025-01-01",
    ),
    (
        "sina_cj_weibo_meta",
        "https://cj.sina.cn/articles/view/7879848900/1d5acf3c401902tn6a?froms=ggmp",
        '<meta name="weibo: article:create_at" content="2026-03-20 16:44:11" />',
        "2026-03-20",
    ),
    (
        "eet_china_publish_line",
        "https://www.eet-china.com/mp/a472559.html",
        '<div class="article-meta">发布时间：2026年01月22日</div>',
        "2026-01-22",
    ),
    (
        "medium_first_published_at",
        "https://jakubjirak.medium.com/how-tesla-rewrote-the-driving-rules-the-revolutionary-move-that-will-change-everything-35551ba6bbde",
        '{"postId":"35551ba6bbde","firstPublishedAt":1704067200000}',
        "2024-01-01",
    ),
    (
        "eeworld_qcdz_update_line",
        "https://www.eeworld.com.cn/qcdz/eic682225.html",
        "发布者：SparklingRiver最新更新时间：2024-10-29 来源: 智驾最前沿",
        "2024-10-29",
    ),
]


@pytest.mark.parametrize(
    "url,blob,expected",
    [(row[1], row[2], row[3]) for row in _USER_REPORTED_FIXTURE_MATRIX],
    ids=[row[0] for row in _USER_REPORTED_FIXTURE_MATRIX],
)
def test_user_reported_url_fixture_calendar_matrix(url: str, blob: str, expected: str) -> None:
    """User-batch URL shapes × synthetic HTML/JSON; each must yield the fixture calendar date.

    Case ids and count: ``_USER_REPORTED_FIXTURE_MATRIX``. Use ``pytest -v`` for per-case output.
    """
    got = infer_publication_calendar_date(blob, source_url=url)
    assert got == expected, f"expected {expected!r}, got {got!r}"
