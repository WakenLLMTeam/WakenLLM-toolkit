"""Heuristic extraction of **publication** dates from HTML or markdown snippets.

We intentionally avoid ``article:modified_time`` and generic ``timestamp`` keys so the
result represents *first publish* where possible. Output is normalized to ``YYYY-MM-DD``
when parsing succeeds.

**Platform-aware scanning** (when ``source_url`` is set): Yahoo / MSN, Facebook (early embedded
unix; photo permalinks ``/photos/…/POST_ID``), ``/blogs/`` paths (Shopify ``published_at`` / ``<time itemprop=datePublished>`` scanned **before**
JSON-LD so generic ``WebPage`` schema dates do not override article inline timestamps; up to ~900k),
YouTube,
LinkedIn, X/Twitter (tweet ``created_at``: ISO, legacy ``Wed Nov 12 … +0000 2025``, or unix — **before** JSON-LD),
Instagram (early: ``taken_at`` / ``taken_at_timestamp`` ms, then
``og:description`` / meta ``on Month D, YYYY:`` for reels), TikTok, Zhihu (专栏 ``/p/{id}``, 问答 ``/answer/{id}``,
英文站 ``Edit <!-- -->YYYY-MM-DD`` DOM), **eeworld.com.cn** (visible ``最新更新时间：YYYY-MM-DD`` byline),
and **Finviz** ``/news/`` pages (author link +
``| Month D, YYYY, h:mm AM/PM`` byline when meta/JSON-LD is absent) use wider ``<head>`` /
``application/ld+json`` windows and site-specific embed rules.

After strict meta / JSON-LD / site-specific passes, a **loose** scan walks every
``<meta>`` tag and many ``"key": "value"`` JSON pairs whose *names* look like publish
semantics (Parsely, DC.*, ``post_date``, …) so alternate CMS naming still yields a date.
That pass is skipped for ``reddit.com`` HTML where sidebar JSON is untrustworthy.
"""

from __future__ import annotations

import json
import re
from html import unescape
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, unquote, urlparse

_EN_MONTHS: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Explicit HTML meta / RSS — all are *publish* semantics (not modified_time).
_META_PUBLISH_PATTERNS: list[tuple[str, int]] = [
    (
        r'<meta[^>]+(?:property|name)\s*=\s*["\']article:published_time["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]*(?:property|name)\s*=\s*["\']article:published_time["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+(?:property|name)\s*=\s*["\']og:published_time["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]*(?:property|name)\s*=\s*["\']og:published_time["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+name\s*=\s*["\']citation_date["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]+name\s*=\s*["\']citation_date["\']',
        re.I | re.DOTALL,
    ),
    # Sina mobile finance (cj.sina.cn): ``<meta name="weibo: article:create_at" content="…" />``
    (
        r'<meta[^>]+name\s*=\s*["\']weibo:\s*article:create_at["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]+name\s*=\s*["\']weibo:\s*article:create_at["\']',
        re.I | re.DOTALL,
    ),
]

# RSS / CMS JSON keys that mean *published* (not generic ``datePublished`` on whole HTML —
# that is parsed inside ``application/ld+json`` only to avoid VideoObject stealing the date).
_EXTRA_PUBLISH_KEY_PATTERNS: list[tuple[str, int]] = [
    (r'"publication_date"\s*:\s*"([^"]+)"', re.I),
    (r'"publicationDate"\s*:\s*"([^"]+)"', 0),
    (r'"pubDate"\s*:\s*"([^"]+)"', re.I),
    (r"<pubDate>([^<]+)</pubDate>", re.I),
]

_JSON_LD_SCRIPT_RE = re.compile(
    r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.I | re.DOTALL,
)

# Types that are never used alone for “article publish” (VideoObject handled on YouTube path).
_JSONLD_SKIP_SOLO_TYPES = frozenset(
    {
        "organization",
        "person",
        "product",
        "website",
        "breadcrumblist",
        "faqpage",
        "searchresultspage",
        "offer",
        "rating",
        "imageobject",
        "audience",
        "place",
        "brand",
        "howto",
        "recipe",
    }
)

_JSONLD_ARTICLE_TYPES = frozenset(
    {
        "article",
        "newsarticle",
        "blogposting",
        "webpage",
        "scholarlyarticle",
        "techarticle",
        "report",
        "blog",
        "liveblogposting",
        "socialmediaposting",
        "discussionforumposting",
    }
)


def calendar_date_from_raw_string(raw: str) -> str:
    """Normalize a date string to ``YYYY-MM-DD``, or ``""`` if unknown."""
    s = (raw or "").strip()
    if not s:
        return ""

    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        y, mo, d = m.group(1).split("-")
        if _valid_ymd(int(y), int(mo), int(d)):
            return m.group(1)

    iso_try = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_try)
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC)
        return dt.date().isoformat()
    except ValueError:
        pass

    try:
        dt2 = parsedate_to_datetime(s)
        if dt2 is not None:
            if dt2.tzinfo is not None:
                dt2 = dt2.astimezone(UTC)
            return dt2.date().isoformat()
    except (TypeError, ValueError):
        pass

    m2 = re.search(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b", s)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"

    m_slash_ymd = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})\s*$", s.strip())
    if m_slash_ymd:
        y, mo, d = int(m_slash_ymd.group(1)), int(m_slash_ymd.group(2)), int(m_slash_ymd.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"

    # US news sites: ``12/27/2025 6:47:00 AM`` (e.g. blockchain.news ``.timestamp`` rows).
    m_mdy_ampm = re.search(
        r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)\b",
        s,
        re.I,
    )
    if m_mdy_ampm:
        mo, d, y = int(m_mdy_ampm.group(1)), int(m_mdy_ampm.group(2)), int(m_mdy_ampm.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31 and _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"

    m_us = re.search(
        r"(?i)\b([A-Za-z]{3,9})\.\s+(\d{1,2}),\s*((?:19|20)\d{2})\b",
        s,
    )
    if not m_us:
        m_us = re.search(
            r"(?i)\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*((?:19|20)\d{2})\b",
            s,
        )
    if m_us:
        mon_s, day_s, year_s = m_us.group(1), m_us.group(2), m_us.group(3)
        mi = _EN_MONTHS.get(mon_s.lower())
        if mi:
            y, d = int(year_s), int(day_s)
            if _valid_ymd(y, mi, d):
                return f"{y:04d}-{mi:02d}-{d:02d}"

    m3 = re.search(
        r"(?i)\b(\d{1,2})\s+([A-Za-z]{3,9})\s+((?:19|20)\d{2})\b",
        s,
    )
    if m3:
        day_s, mon_s, year_s = m3.group(1), m3.group(2), m3.group(3)
        mi = _EN_MONTHS.get(mon_s.lower())
        if mi:
            y, d = int(year_s), int(day_s)
            if _valid_ymd(y, mi, d):
                return f"{y:04d}-{mi:02d}-{d:02d}"

    m4 = re.match(r"(?i)^(\d{1,2})-([A-Za-z]{3})-((?:19|20)\d{2})\s*$", s.strip())
    if m4:
        d_s, mon_s, y_s = m4.group(1), m4.group(2), m4.group(3)
        mi = _EN_MONTHS.get(mon_s.lower())
        if mi:
            y, d = int(y_s), int(d_s)
            if _valid_ymd(y, mi, d):
                return f"{y:04d}-{mi:02d}-{d:02d}"

    m_cn = re.search(
        r"(?u)((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?",
        s,
    )
    if m_cn:
        y, mo, d = int(m_cn.group(1)), int(m_cn.group(2)), int(m_cn.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"

    return ""


def _valid_ymd(y: int, m: int, d: int) -> bool:
    if y < 1900 or y > 2100 or m < 1 or m > 12 or d < 1 or d > 31:
        return False
    try:
        datetime(y, m, d)
    except ValueError:
        return False
    return True


def _hostname_from_url(url: str) -> str:
    try:
        return (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return ""


def _url_path_contains_blog_segment(source_url: str) -> bool:
    """True when URL path looks like a CMS blog article (often Shopify ``/blogs/…``).

    These pages often put ``published_at`` / ``<time itemprop=datePublished>`` in mid-file JSON,
    so default 96k head scans miss them; see ``infer_publication_date_from_text`` head widening.
    """
    try:
        pth = unquote(urlparse((source_url or "").strip()).path or "").lower()
    except Exception:
        return False
    return "/blogs/" in pth


def _hostname_is_reddit(host: str) -> bool:
    h = (host or "").lower()
    return h == "reddit.com" or h.endswith(".reddit.com")


def _hostname_is_facebook(host: str) -> bool:
    h = (host or "").lower()
    return h == "facebook.com" or h.endswith(".facebook.com")


def _hostname_is_yahoo_or_msn(host: str) -> bool:
    h = (host or "").lower()
    if not h:
        return False
    if h == "msn.com" or h.endswith(".msn.com"):
        return True
    if h == "yahoo.com" or h.endswith(".yahoo.com"):
        return True
    return False


def _hostname_is_x_or_twitter(host: str) -> bool:
    h = (host or "").lower()
    return h in ("x.com", "twitter.com") or h.endswith(".x.com") or h.endswith(".twitter.com")


def _hostname_is_instagram(host: str) -> bool:
    h = (host or "").lower()
    return h == "instagram.com" or h.endswith(".instagram.com")


def _hostname_is_tiktok(host: str) -> bool:
    h = (host or "").lower()
    return h == "tiktok.com" or h.endswith(".tiktok.com")


def _hostname_is_linkedin(host: str) -> bool:
    h = (host or "").lower()
    return h == "linkedin.com" or h.endswith(".linkedin.com")


def _hostname_is_zhihu(host: str) -> bool:
    h = (host or "").lower()
    return h == "zhihu.com" or h.endswith(".zhihu.com")


def _hostname_is_finviz(host: str) -> bool:
    h = (host or "").lower()
    return h == "finviz.com" or h.endswith(".finviz.com")


def _hostname_is_medium(host: str) -> bool:
    h = (host or "").lower()
    return h == "medium.com" or h.endswith(".medium.com")


def _hostname_is_zacks(host: str) -> bool:
    h = (host or "").lower()
    return h == "zacks.com" or h.endswith(".zacks.com")


def _hostname_is_blockchain_news(host: str) -> bool:
    h = (host or "").lower()
    return h == "blockchain.news" or h.endswith(".blockchain.news")


def _hostname_is_sina_cj(host: str) -> bool:
    """新浪财经移动 / 财经头条（``cj.sina.cn``、``cj.sina.com.cn``）。"""
    h = (host or "").lower()
    return (
        h == "cj.sina.cn"
        or h.endswith(".cj.sina.cn")
        or h == "cj.sina.com.cn"
        or h.endswith(".cj.sina.com.cn")
    )


def _hostname_is_eet_china(host: str) -> bool:
    h = (host or "").lower()
    return h == "eet-china.com" or h.endswith(".eet-china.com")


def _hostname_is_xueqiu(host: str) -> bool:
    h = (host or "").lower()
    return h == "xueqiu.com" or h.endswith(".xueqiu.com")


def _hostname_is_eeworld(host: str) -> bool:
    h = (host or "").lower()
    return h == "eeworld.com.cn" or h.endswith(".eeworld.com.cn")


def _hostname_is_wallstreetcn(host: str) -> bool:
    h = (host or "").lower()
    return h == "wallstreetcn.com" or h.endswith(".wallstreetcn.com")


def _hostname_is_36kr(host: str) -> bool:
    """36氪主站 / 移动站（``m.36kr.com``、``www.36kr.com``）。"""
    h = (host or "").lower()
    return h == "36kr.com" or h.endswith(".36kr.com")


def _try_36kr_article_publish(blob: str, source_url: str) -> str:
    """Prefer visible author byline ``·YYYY年MM月DD日`` over shell JSON-LD / footer ``© … 2026``.

    Mobile shells embed ``datePublished`` / ``publishTime`` that may disagree with the byline,
    and page footers carry ``2011~ 2026`` plus sidebar ISO dates — all of which can win if
    parsed before the article row.
    """
    if not blob or not _hostname_is_36kr(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 500_000)]
    # Author row: ``名称·2025年10月22日 15:58`` (U+00B7 middot; common on 36kr).
    m = re.search(
        r"(?u)·\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日(?:\s+[\d:]+)?",
        head,
    )
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    labeled = re.compile(
        r"(?is)(?:发布(?:时间)?|发表(?:时间)?)\s*[:：]\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
    )
    m2 = labeled.search(head)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


# Visible row like blockchain.news: ``<div class="timestamp"><span>MM/DD/YYYY h:mm:ss AM</span>``.
_TIMESTAMP_DIV_US_RE = re.compile(
    r'(?is)class\s*=\s*["\'][^"\']*\btimestamp\b[^"\']*["\'][^>]*>'
    r'\s*<span[^>]*>\s*'
    r'(\d{1,2}/\d{1,2}/(?:19|20)\d{2}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM))\s*</span>',
)


def _try_timestamp_div_us_mdy_am_pm(blob: str) -> str:
    if not blob:
        return ""
    head = blob[: min(len(blob), 800_000)]
    m = _TIMESTAMP_DIV_US_RE.search(head)
    if not m:
        return ""
    raw = (m.group(1) or "").strip()
    return raw if raw and _looks_like_date_token(raw) else ""


def _try_zacks_embedded_publish(blob: str, source_url: str) -> str:
    """Zacks article shell: inline ``placementDate``, ``publish_date`` (D/M/Y), JSON-LD-ish keys, ``<title>`` tail."""
    if not blob or not _hostname_is_zacks(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 800_000)]

    m = re.search(
        r"placementDate\s*=\s*['\"]((?:19|20)\d{2}-\d{2}-\d{2})['\"]",
        head,
        re.I,
    )
    if m:
        frag = m.group(1).strip()
        if frag and _looks_like_date_token(frag):
            return frag

    m2 = re.search(
        r'"datePublished"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2})"',
        head,
        re.I,
    )
    if m2:
        frag = m2.group(1).strip()
        if frag and _looks_like_date_token(frag):
            return frag

    m3 = re.search(
        r"publish_date\s*:\s*['\"](\d{1,2})/(\d{1,2})/((?:19|20)\d{2})['\"]",
        head,
        re.I,
    )
    if m3:
        day_s, month_s, year_s = m3.group(1), m3.group(2), m3.group(3)
        day_i, month_i, y = int(day_s), int(month_s), int(year_s)
        if _valid_ymd(y, month_i, day_i):
            return f"{y:04d}-{month_i:02d}-{day_i:02d}"

    m4 = re.search(
        r"<title>[^<]*-\s*"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},\s*(?:19|20)\d{2})\s*-\s*Zacks",
        head[:16_000],
        re.I,
    )
    if m4:
        frag = (m4.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag

    return ""


def _medium_slug_hex_tail(url: str) -> str:
    """Hex story id at end of Medium slug: ``...-35551ba6bbde``."""
    try:
        path = unquote(urlparse((url or "").strip()).path or "")
    except Exception:
        return ""
    segs = [x for x in path.rstrip("/").split("/") if x]
    if not segs:
        return ""
    last = segs[-1]
    m = re.search(r"-([0-9a-f]{8,16})$", last, re.I)
    return (m.group(1) or "").lower() if m else ""


def _try_medium_embedded_publish(blob: str, source_url: str) -> str:
    """Medium story JSON: ``firstPublishedAt`` / ``publishedAt`` (ms or ISO), optionally near slug hex id."""
    if not blob or not _hostname_is_medium(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 1_500_000)]
    iso_pat = re.compile(
        r'"(?:firstPublishedAt|publishedAt|latestPublishedAt|mediumPublishedAt)"\s*:\s*'
        r'"((?:19|20)\d{2}-\d{2}-\d{2}[^"]*)"',
        re.I,
    )
    m0 = iso_pat.search(head)
    if m0:
        frag = (m0.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag
    ms_pat = re.compile(
        r'"(?:firstPublishedAt|publishedAt|latestPublishedAt|mediumPublishedAt)"\s*:\s*(\d{10,16})\b',
        re.I,
    )
    hid = _medium_slug_hex_tail(source_url)
    if hid:
        pos = 0
        hl = head.lower()
        while True:
            i = hl.find(hid, pos)
            if i < 0:
                break
            win = head[i : i + 14_000]
            m2 = ms_pat.search(win)
            if m2:
                val = int(m2.group(1))
                if val > 10_000_000_000:
                    val //= 1000
                if val > 0:
                    got = _utc_iso_from_unix(val)
                    if got:
                        return got
            pos = i + max(1, len(hid))
    m3 = ms_pat.search(head)
    if m3:
        val = int(m3.group(1))
        if val > 10_000_000_000:
            val //= 1000
        if val > 0:
            return _utc_iso_from_unix(val)
    return ""


def _zhihu_content_id_from_url(url: str) -> str:
    """Numeric id from zhuanlan ``/p/{id}`` or ``www.zhihu.com/.../{id}`` paths."""
    try:
        p = urlparse((url or "").strip())
    except Exception:
        return ""
    path = unquote(p.path or "")
    m = re.search(r"/p/(\d{5,22})\b", path, re.I)
    if m:
        return m.group(1)
    m = re.search(r"/(?:question|answer|zvideo)/(\d{5,22})\b", path, re.I)
    if m:
        return m.group(1)
    return ""


def _try_zhihu_article_publish(blob: str, source_url: str) -> str:
    """Zhihu / 知乎专栏: ``publishedTime`` / ``createdTime`` sit near the article id in boot JSON.

    A generic ``publishedTime`` scan (see ``_infer_cn_portal_json_publish``) can pick another
    article in the same HTML shell; anchor on the URL id when possible.
    """
    if not blob or not _hostname_is_zhihu(_hostname_from_url(source_url)):
        return ""
    aid = _zhihu_content_id_from_url(source_url)
    if not aid:
        return ""
    head = blob[: min(len(blob), 2_000_000)]

    def _normalize_ts(val: int) -> str:
        if val > 10_000_000_000:
            val //= 1000
        if val <= 0:
            return ""
        return _utc_iso_from_unix(val)

    ts_patterns: tuple[re.Pattern[str], ...] = (
        re.compile(r'"publishedTime"\s*:\s*(\d{10,13})\b'),
        re.compile(r'"createdTime"\s*:\s*(\d{10,13})\b'),
        # Some Zhihu payloads quote the epoch (string) or use ISO in JSON.
        re.compile(r'"publishedTime"\s*:\s*"(\d{10,13})"'),
        re.compile(r'"createdTime"\s*:\s*"(\d{10,13})"'),
        re.compile(
            r'"publishedTime"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}[^"]*)"'
        ),
        re.compile(r'"createdTime"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}[^"]*)"'),
    )

    markers: list[str] = []
    for m in (
        f'"/p/{aid}"',
        f"/p/{aid}",
        f'"answers":{{"{aid}"',
        f'"answers":{{"{aid}":',
        f'"/answer/{aid}"',
        f"/answer/{aid}",
        f'"{aid}"',
        f'/{aid}"',
        f'/{aid}?',
        f":{aid},",
        f": {aid},",
    ):
        if m not in markers:
            markers.append(m)

    seen_anchor: set[int] = set()
    for marker in markers:
        pos = 0
        while True:
            i = head.find(marker, pos)
            if i < 0:
                break
            if i not in seen_anchor:
                seen_anchor.add(i)
                win = head[i : i + 36_000]
                for pat in ts_patterns:
                    mm = pat.search(win)
                    if mm:
                        g1 = (mm.group(1) or "").strip()
                        if g1.isdigit():
                            got = _normalize_ts(int(g1))
                        else:
                            got = g1 if _looks_like_date_token(g1) else ""
                        if got:
                            return got
            pos = i + max(1, len(marker))

    return ""


_ZHIHU_PUBLISH_INLINE_RE: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?u)(?:发布|发表)于[：:\s]*((?:19|20)\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日?)"
    ),
    re.compile(
        r"(?u)(?:发布|发表)于[：:\s]*((?:19|20)\d{2}-\d{1,2}-\d{1,2})(?:\s+[\d:]+)?"
    ),
    # Web EN UI (React): ``Edit <!-- -->2023-11-08 19:26``
    re.compile(
        r"(?i)Edit\s*(?:<!--\s*-->)?\s*((?:19|20)\d{2}-\d{1,2}-\d{1,2})\s+[\d:]+"
    ),
    # Same pattern, Chinese labels (DOM often mirrors EN with HTML comments).
    re.compile(
        r"(?u)(?:编辑于|修改于)\s*(?:<!--\s*-->)?\s*((?:19|20)\d{2}-\d{1,2}-\d{1,2})\s+[\d:]+"
    ),
    re.compile(
        r"(?i)Published\s*(?:<!--\s*-->)?\s*((?:19|20)\d{2}-\d{1,2}-\d{1,2})\s+[\d:]+"
    ),
)


def _try_zhihu_markdown_publish(blob: str, source_url: str) -> str:
    """Reader / Tavily markdown and Zhihu HTML: 发布于 / 发表于, or EN ``Edit <!-- -->YYYY-MM-DD``."""
    if not blob or not _hostname_is_zhihu(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 600_000)]
    for pat in _ZHIHU_PUBLISH_INLINE_RE:
        m = pat.search(head)
        if not m:
            continue
        frag = (m.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag
    return ""


# Finviz news article: ``<a>Source</a> | December 30, 2025, 9:32 AM`` inside ``news-publish-info`` (often no OG/meta).
_FINVIZ_PIPE_BYLINE_RE = re.compile(
    r"(?is)</a>\s*\|\s*"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s*(?:19|20)\d{2}"
    r"(?:,\s*\d{1,2}:\d{2}\s*(?:AM|PM))?)"
)


def _try_finviz_news_pipe_byline(blob: str, source_url: str) -> str:
    """``finviz.com/news/...`` byline datetime after author link (pipe-separated)."""
    if not blob or not _hostname_is_finviz(_hostname_from_url(source_url)):
        return ""
    try:
        pth = unquote(urlparse((source_url or "").strip()).path or "").lower()
    except Exception:
        pth = ""
    if "/news/" not in pth:
        return ""
    head = blob[: min(len(blob), 400_000)]
    m = _FINVIZ_PIPE_BYLINE_RE.search(head)
    if not m:
        return ""
    frag = (m.group(1) or "").strip()
    return frag if frag and _looks_like_date_token(frag) else ""


# Shopify / similar themes: ``published_at`` and microdata ``<time>`` often sit outside the first
# screen of HTML; scan up to 1.5M of *full* text (long themes + JSON bundles).
_BLOG_CMS_PUBLISH_SCAN_CAP = 1_500_000

_BLOG_TIME_DATEPUBLISHED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r'<time[^>]+itemprop\s*=\s*["\']datePublished["\'][^>]*\bdatetime\s*=\s*["\']([^"\']+)["\']',
        re.I,
    ),
    re.compile(
        r'<time[^>]+datetime\s*=\s*["\']([^"\']+)["\'][^>]+itemprop\s*=\s*["\']datePublished["\']',
        re.I,
    ),
)
_BLOG_PUBLISHED_AT_JSON_RE = re.compile(
    r'"(?:published_at|publishedAt)"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}(?:T[^"\\]+)?)"',
    re.I,
)
# Some Shopify / Liquid dumps use a space between date and time instead of ``T``.
_BLOG_PUBLISHED_AT_JSON_SPACE_RE = re.compile(
    r'"(?:published_at|publishedAt)"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[+-]\d{2}:?\d{2,4})?)"',
    re.I,
)
_BLOG_CREATED_AT_JSON_RE = re.compile(
    r'"(?:created_at|createdAt)"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}(?:T[^"\\]+)?)"',
    re.I,
)


def _try_cms_blog_article_publish(blob: str, source_url: str) -> str:
    """Shopify-style blog: ``published_at`` ISO string or ``<time itemprop=datePublished>``."""
    if not blob or not _url_path_contains_blog_segment(source_url):
        return ""
    head = blob[: min(len(blob), _BLOG_CMS_PUBLISH_SCAN_CAP)]
    for tp in _BLOG_TIME_DATEPUBLISHED_PATTERNS:
        m = tp.search(head)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw
    for pub_rx in (_BLOG_PUBLISHED_AT_JSON_RE, _BLOG_PUBLISHED_AT_JSON_SPACE_RE):
        m = pub_rx.search(head)
        if m:
            raw = (m.group(1) or "").strip()
            if raw and _looks_like_date_token(raw):
                return raw
    m = _BLOG_CREATED_AT_JSON_RE.search(head)
    if m:
        raw = (m.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw
    return ""


# eet-china / similar CMS: optional HTML between colon / 年 / 月 / 日 (e.g. ``<strong>2024</strong>年``).
_PUBLISH_TIME_INLINE_CN_RE = re.compile(
    r"(?is)发布时间\s*[:：]\s*(?:<[^>]+>\s*|&nbsp;|\s)*"
    r"((?:19|20)\d{2})(?:\s*</[^>]+>)?\s*年\s*(?:<[^>]+>\s*)*(\d{1,2})(?:\s*</[^>]+>)?\s*月\s*(?:<[^>]+>\s*)*(\d{1,2})\s*日?",
)
_EET_META_ARTICLE_PUBLISHED_RE = re.compile(
    r'<meta[^>]+property\s*=\s*["\']article:published_time["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
    re.I | re.DOTALL,
)
_EET_META_ARTICLE_PUBLISHED_RE2 = re.compile(
    r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]+property\s*=\s*["\']article:published_time["\']',
    re.I | re.DOTALL,
)

_SINA_WEIBO_CREATE_AT_META_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        r'<meta[^>]+name\s*=\s*["\']weibo:\s*article:create_at["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    re.compile(
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]+name\s*=\s*["\']weibo:\s*article:create_at["\']',
        re.I | re.DOTALL,
    ),
)
_SINA_SPAN_DATE_CN_RE = re.compile(
    r"(?is)<span[^>]*\bclass\s*=\s*[\"'][^\"']*\bdate\b[^\"']*[\"'][^>]*>\s*"
    r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
)


def _try_sina_cj_article_publish(blob: str, source_url: str) -> str:
    """Sina finance mobile: ``weibo: article:create_at`` meta or ``<span class=…date…>YYYY年MM月DD日``.

    Page JS may contain misleading ``pagepubtime`` (template); do not use loose ``pubtime`` keys
    that match ``pagepubtime`` (see ``_infer_cn_portal_json_publish``).
    """
    if not blob or not _hostname_is_sina_cj(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 900_000)]
    for rx in _SINA_WEIBO_CREATE_AT_META_RES:
        m = rx.search(head)
        if m:
            raw = (m.group(1) or "").strip()
            if raw and _looks_like_date_token(raw):
                return raw
    sm = _SINA_SPAN_DATE_CN_RE.search(head)
    if sm:
        y, mo, d = int(sm.group(1)), int(sm.group(2)), int(sm.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


def _try_xueqiu_status_publish(blob: str, source_url: str) -> str:
    """xueqiu.com status / post: inline JSON often has ``created_at`` (seconds or ms)."""
    if not blob or not _hostname_is_xueqiu(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 900_000)]
    m = re.search(
        r'"(?:created_at|createdAt|edit_at)"\s*:\s*(\d{13})\b',
        head,
    )
    if m:
        ms = int(m.group(1))
        if ms > 10**12:
            sec = ms // 1000
            if sec > 946_684_800:
                raw = _utc_iso_from_unix(sec)
                return raw if raw and _looks_like_date_token(raw) else ""
    m2 = re.search(r'"(?:created_at|createdAt)"\s*:\s*(\d{10})\b', head)
    if m2:
        sec = int(m2.group(1))
        if sec > 946_684_800:
            raw = _utc_iso_from_unix(sec)
            return raw if raw and _looks_like_date_token(raw) else ""
    return ""


def _try_eet_china_mp_article_publish(blob: str, source_url: str) -> str:
    """eet-china ``/mp/…``: visible ``发布时间：…年…月…日`` or ``article:published_time`` mid-page."""
    if not blob or not _hostname_is_eet_china(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 900_000)]
    m = _PUBLISH_TIME_INLINE_CN_RE.search(head)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    for rx in (_EET_META_ARTICLE_PUBLISHED_RE, _EET_META_ARTICLE_PUBLISHED_RE2):
        mm = rx.search(head)
        if mm:
            raw = (mm.group(1) or "").strip()
            if raw and _looks_like_date_token(raw):
                return raw
    return ""


def _try_eeworld_article_publish(blob: str, source_url: str) -> str:
    """eeworld.com.cn tech articles: byline ``最新更新时间：YYYY-MM-DD`` (often mid-page, not meta).

    Live pages use WAF on bare GET; search extracts / reader HTML still carry this prose line.
    Prefer *更新时间* over a generic ``发布时间`` elsewhere on the page when both exist.
    """
    if not blob or not _hostname_is_eeworld(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 900_000)]
    # ``最新更新时间：2024-10-29 来源: …`` (Markdown or HTML-stripped)
    iso_update = re.compile(
        r"(?u)最新更新时间\s*[:：]\s*((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})"
        r'(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?',
    )
    m = iso_update.search(head)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    iso_pub = re.compile(
        r"(?u)发布时间\s*[:：]\s*((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})",
    )
    m2 = iso_pub.search(head)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    cn_update = re.compile(
        r"(?u)最新更新时间\s*[:：]\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?",
    )
    m3 = cn_update.search(head)
    if m3:
        y, mo, d = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


# PC 站多为 SPA：首屏 HTML 常无 ``<article>``，但内联 boot JSON 里可能有发布时间字段。
_WSCN_BOOT_PUBLISH_ISO_RE = re.compile(
    r'"(?:published_at|publishedAt|display_time|displayTime|publish_time|publishTime|created_at|createdAt)"\s*:\s*'
    r'"((?:19|20)\d{2}-\d{2}-\d{2}(?:T[^"\\]+)?)"',
    re.I,
)


def _try_wallstreetcn_article_publish(blob: str, source_url: str) -> str:
    """wallstreetcn.com: SSR ``<article>…<time datetime>``; else boot JSON publish keys (SPA shell).

    Live ``/articles/{id}`` often returns a minimal document (``<div id="app"></div>`` + bundles)
    with **no** article DOM in *raw* HTML; search APIs then cannot infer a date unless the
    extract includes hydrated JSON or a full render. The regex below covers common inline keys
    when present in the same blob.
    """
    if not blob or not _hostname_is_wallstreetcn(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 900_000)]
    am = re.search(r"(?is)<article\b[^>]*>(.*?)</article>", head)
    if am:
        art = am.group(1) or ""
        tm = re.search(
            r'(?is)<time\b[^>]*\bdatetime\s*=\s*["\']([^"\']+)["\']',
            art,
        )
        if tm:
            raw = (tm.group(1) or "").strip()
            if raw and _looks_like_date_token(raw):
                return raw
    bm = _WSCN_BOOT_PUBLISH_ISO_RE.search(head)
    if bm:
        raw = (bm.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw
    jp = _infer_json_ld_publish_date(head, max_chars=len(head), max_scripts=48)
    if jp:
        return jp
    return ""


def _hostname_is_youtube_host(host: str) -> bool:
    h = (host or "").lower()
    return (
        h == "youtu.be"
        or h.endswith(".youtube.com")
        or h.endswith("youtube-nocookie.com")
        or h == "youtube.com"
    )


def _hostname_needs_wide_head_and_json_ld(host: str) -> bool:
    """Larger `<head>` / JSON-LD windows — SPAs and news shells defer structured data."""
    return bool(
        _hostname_is_yahoo_or_msn(host)
        or _hostname_is_facebook(host)
        or _hostname_is_linkedin(host)
        or _hostname_is_x_or_twitter(host)
        or _hostname_is_instagram(host)
        or _hostname_is_tiktok(host)
        or _hostname_is_youtube_host(host)
        or _hostname_is_zhihu(host)
        or _hostname_is_medium(host)
        or _hostname_is_zacks(host)
        or _hostname_is_blockchain_news(host)
        or _hostname_is_sina_cj(host)
        or _hostname_is_eet_china(host)
        or _hostname_is_xueqiu(host)
        or _hostname_is_eeworld(host)
        or _hostname_is_wallstreetcn(host)
        or _hostname_is_36kr(host)
    )


def _facebook_story_ids_from_url(url: str) -> list[str]:
    """Numeric post / video / reel ids from a facebook.com URL for JSON proximity matching."""
    out: list[str] = []
    try:
        p = urlparse((url or "").strip())
    except Exception:
        return out
    path = unquote(p.path or "")
    for m in re.finditer(r"/(?:posts|videos|reel|reels)/(\d{6,22})\b", path, re.I):
        out.append(m.group(1))
    pm = re.search(r"/permalink/(\d{6,22})\b", path, re.I)
    if pm:
        out.append(pm.group(1))
    # Photo permalink: ``/PAGE/photos/SLUG/POST_ID`` (trailing numeric story/media id).
    path_trim = path.rstrip("/")
    photo_tail = re.search(r"/photos/.+/(\d{6,22})$", path_trim, re.I)
    if photo_tail:
        out.append(photo_tail.group(1))
    qs = parse_qs(p.query or "")
    for key in ("story_fbid", "fbid", "v", "id"):
        for val in qs.get(key, []):
            v = (val or "").strip()
            if v.isdigit() and len(v) >= 8:
                out.append(v)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _youtube_video_context(source_url: str, full_text: str) -> bool:
    h = _hostname_from_url(source_url)
    if any(
        s in h
        for s in ("youtube.com", "youtube-nocookie.com", "youtu.be", "m.youtube.com")
    ):
        return True
    low = full_text[: min(len(full_text), 400_000)].lower()
    return (
        "youtube.com/watch" in low
        or "youtu.be/" in low
        or '"@type":"videoobject"' in low.replace(" ", "")
        or '"@type": "videoobject"' in low
    )


def _youtube_primary_video_url(source_url: str) -> bool:
    """True when the URL is a single watch / shorts / live / youtu.be video page.

    On these pages ``ytInitialPlayerResponse`` (and watch-page itemprop) reflect the
    actual upload time. JSON-LD ``WebPage`` / ``og:published_time`` often lags or
    disagrees, which previously produced the wrong calendar year.
    """
    try:
        p = urlparse((source_url or "").strip())
    except Exception:
        return False
    host = (p.netloc or "").lower()
    if not _hostname_is_youtube_host(host):
        return False
    path = unquote((p.path or "").rstrip("/"))
    qs = parse_qs(p.query or "")

    if host == "youtu.be" or host.endswith(".youtu.be"):
        segs = [s for s in path.split("/") if s]
        return len(segs) >= 1 and bool(segs[0])

    if re.search(r"/(?:shorts|live)/[A-Za-z0-9_-]{6,}", path, re.I):
        return True

    if path == "/watch" or path.endswith("/watch"):
        return bool((qs.get("v") or [""])[0].strip())

    return False


def _json_ld_type_set(obj: dict) -> set[str]:
    t = obj.get("@type")
    if isinstance(t, str):
        return {t.lower()}
    if isinstance(t, list):
        return {str(x).lower() for x in t}
    return set()


def _json_ld_collect_objects(node: object, acc: list[dict]) -> None:
    if isinstance(node, dict):
        acc.append(node)
        g = node.get("@graph")
        if isinstance(g, list):
            for x in g:
                _json_ld_collect_objects(x, acc)
        for k, v in node.items():
            if k == "@graph" or v is node:
                continue
            if isinstance(v, (dict, list)):
                _json_ld_collect_objects(v, acc)
    elif isinstance(node, list):
        for x in node:
            _json_ld_collect_objects(x, acc)


def _json_ld_object_uses_publish_date(obj: dict) -> bool:
    types = _json_ld_type_set(obj)
    if not types:
        return False
    if types <= {"videoobject"}:
        return False
    if "videoobject" in types and not (types & _JSONLD_ARTICLE_TYPES):
        return False
    if types & _JSONLD_SKIP_SOLO_TYPES and not (types & _JSONLD_ARTICLE_TYPES):
        return False
    if types <= {"website"}:
        return False
    return bool(types & _JSONLD_ARTICLE_TYPES or types == {"webpage"})


def _infer_json_ld_publish_date(
    blob: str,
    *,
    max_scripts: int = 48,
    max_chars: int = 900_000,
) -> str:
    """``datePublished`` inside ``application/ld+json`` for article-like objects only."""
    window = blob[:max_chars] if len(blob) > max_chars else blob
    n = 0
    for m in _JSON_LD_SCRIPT_RE.finditer(window):
        n += 1
        if n > max_scripts:
            break
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        objs: list[dict] = []
        _json_ld_collect_objects(data, objs)
        for obj in objs:
            if not _json_ld_object_uses_publish_date(obj):
                continue
            dp = obj.get("datePublished") or obj.get("datepublished")
            if isinstance(dp, str):
                frag = dp.strip()
                if frag and _looks_like_date_token(frag):
                    return frag
    return ""


_YAHOO_MSN_PORTAL_ISO_RE = re.compile(
    r'"(?:contentPublishedDate|firstPublishedDate|timePublished|originalPublishDate|'
    r"articlePublishedDate|publishDateTime|originalPublishTimeIso|datePublishedUtc)"
    r'"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}[^"]*)"',
    re.I,
)
_YAHOO_MSN_PORTAL_ALT_ISO_RE = re.compile(
    r'"(?:publishDate|publishedDate|datePosted|dateCreated|postedDate)"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}[^"]*)"',
    re.I,
)
_YAHOO_MSN_PUBLISH_MS_RE = re.compile(
    r'"(?:publishedTime|publishTime|firstPublishedMillis|pubTimestamp|published_at_ms)"\s*:\s*(\d{10,13})\b',
    re.I,
)


def _infer_yahoo_msn_portal_json(blob: str) -> str:
    """Yahoo News / MSN often embed publish fields in non–JSON-LD boot payloads (Fusion, Gemini, etc.)."""
    if not blob:
        return ""
    head = blob[: min(len(blob), 2_000_000)]
    for rx in (_YAHOO_MSN_PORTAL_ISO_RE, _YAHOO_MSN_PORTAL_ALT_ISO_RE):
        m = rx.search(head)
        if not m:
            continue
        frag = (m.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag
    m = _YAHOO_MSN_PUBLISH_MS_RE.search(head)
    if m:
        val = int(m.group(1))
        if val > 10_000_000_000:
            val //= 1000
        if val > 0:
            return _utc_iso_from_unix(val)
    return ""


def _x_status_id_from_url(url: str) -> str:
    try:
        p = urlparse((url or "").strip())
    except Exception:
        return ""
    path = unquote(p.path or "")
    m = re.search(r"/(?:status|statuses)/(\d{10,25})\b", path, re.I)
    return m.group(1) if m else ""


_TWITTER_CREATED_AT_LEGACY_RE = re.compile(
    r'"(?:created_at|createdAt)"\s*:\s*"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+'
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+[+-]\d{4}\s+(?:19|20)\d{2})\"",
    re.I,
)

_X_CREATED_AT_UNIX_RE = re.compile(
    r'"(?:created_at|createdAt|created_at_ms|createdAtMs)"\s*:\s*(\d{10,16})\b',
    re.I,
)


def _twitter_legacy_created_at_to_utc_iso(raw: str) -> str:
    """Parse Twitter / X API string ``Wed Nov 12 16:47:05 +0000 2025`` → UTC ISO."""
    s = (raw or "").strip()
    if not s:
        return ""
    try:
        dt = datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return ""


def _x_try_created_at_in_window(win: str) -> str:
    """Return a raw publish token from *win* (ISO, Twitter legacy string, or unix)."""
    iso_pat = re.compile(
        r'"(?:created_at|createdAt)"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}T[^"]+)"',
        re.I,
    )
    m = iso_pat.search(win)
    if m:
        frag = (m.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag
    lm = _TWITTER_CREATED_AT_LEGACY_RE.search(win)
    if lm:
        legacy = (lm.group(1) or "").strip()
        iso = _twitter_legacy_created_at_to_utc_iso(legacy)
        if iso and _looks_like_date_token(iso):
            return iso
    um = _X_CREATED_AT_UNIX_RE.search(win)
    if um:
        val = int(um.group(1))
        # 13-digit Unix ms (common in embedded X / GraphQL payloads)
        if val > 10**12:
            val //= 1000
        if val > 946_684_800:
            got = _utc_iso_from_unix(val)
            if got and _looks_like_date_token(got):
                return got
    return ""


def _try_x_embedded_created_at(blob: str, source_url: str) -> str:
    """X / Twitter: ``created_at`` near the status id (ISO, legacy API string, or unix)."""
    if not blob or not _hostname_is_x_or_twitter(_hostname_from_url(source_url)):
        return ""
    head = blob[: min(len(blob), 1_200_000)]
    sid = _x_status_id_from_url(source_url)
    if sid:
        markers: list[str] = [f'"{sid}"', f':"{sid}"']
        if len(sid) >= 15:
            markers.append(sid)
        seen_i: set[int] = set()
        for marker in markers:
            pos = 0
            while True:
                i = head.find(marker, pos)
                if i < 0:
                    break
                if i in seen_i:
                    pos = i + max(1, len(marker))
                    continue
                seen_i.add(i)
                win = head[i : i + 24_000]
                got = _x_try_created_at_in_window(win)
                if got:
                    return got
                pos = i + max(1, len(marker))
    return _x_try_created_at_in_window(head)


# Logged-out HTML often omits ``taken_at`` JSON but repeats ``og:description`` / ``meta description``:
# ``… username on August 20, 2024: "caption…"``.
_IG_ON_MONTH_DAY_YEAR_RE = re.compile(
    r"(?i)\bon\s+"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s*(?:19|20)\d{2})\s*:\s*",
)


def _instagram_shortcode_from_url(url: str) -> str:
    try:
        path = unquote(urlparse((url or "").strip()).path or "")
    except Exception:
        return ""
    m = re.search(r"/(?:p|reel|tv)/([^/?#]+)/?", path, re.I)
    return (m.group(1) or "").strip() if m else ""


def _try_instagram_meta_on_month_date(blob: str) -> str:
    if not blob:
        return ""
    head = unescape(blob[: min(len(blob), 600_000)])
    m = _IG_ON_MONTH_DAY_YEAR_RE.search(head)
    if not m:
        return ""
    frag = (m.group(1) or "").strip()
    return frag if frag and _looks_like_date_token(frag) else ""


def _try_instagram_taken_at(blob: str, source_url: str = "") -> str:
    """``taken_at`` / ``taken_at_timestamp`` (10–13 digit); optional window near URL shortcode."""
    if not blob:
        return ""
    head = blob[: min(len(blob), 1_200_000)]
    ts_patterns: tuple[re.Pattern[str], ...] = (
        re.compile(
            r'"(?:taken_at_timestamp|taken_at)"\s*:\s*(\d{10,13})\b',
            re.I,
        ),
        re.compile(
            r'\\"(?:taken_at_timestamp|taken_at)\\"\s*:\s*(\d{10,13})\b',
            re.I,
        ),
    )

    def _consume_ts(m: re.Match[str]) -> str:
        val = int(m.group(1))
        if val > 10_000_000_000:
            val //= 1000
        if val <= 0:
            return ""
        return _utc_iso_from_unix(val)

    code = _instagram_shortcode_from_url(source_url)
    if code:
        pos = 0
        hl = head.lower()
        cl = code.lower()
        seen: set[int] = set()
        while True:
            i = hl.find(cl, pos)
            if i < 0:
                break
            if i not in seen:
                seen.add(i)
                win = head[i : i + 28_000]
                for ts_pat in ts_patterns:
                    mm = ts_pat.search(win)
                    if mm:
                        got = _consume_ts(mm)
                        if got:
                            return got
            pos = i + max(1, len(cl))

    for ts_pat in ts_patterns:
        mm = ts_pat.search(head)
        if mm:
            got = _consume_ts(mm)
            if got:
                return got
    return ""


def _try_instagram_publish(blob: str, source_url: str) -> str:
    """Reels/posts: embedded unix first, then ``on Month D, YYYY:`` in OG/meta description,
    then Tavily/SERP snippet date patterns (Month D YYYY prefix, ISO, relative)."""
    if not blob or not _hostname_is_instagram(_hostname_from_url(source_url)):
        return ""
    full = blob[: min(len(blob), 2_000_000)]
    taken = _try_instagram_taken_at(full, source_url=source_url)
    if taken:
        return taken
    desc = _try_instagram_meta_on_month_date(full)
    if desc:
        return desc
    # Tavily/SERP snippets: short text with date prefix or relative "N ago"
    snip = _infer_serp_snippet_date(blob)
    if snip:
        return snip
    return ""


def _tiktok_video_id_from_url(url: str) -> str:
    try:
        p = urlparse((url or "").strip())
    except Exception:
        return ""
    path = unquote(p.path or "")
    m = re.search(r"/video/(\d{10,25})\b", path, re.I)
    return m.group(1) if m else ""


def _try_tiktok_create_time(blob: str, source_url: str) -> str:
    if not blob:
        return ""
    head = blob[: min(len(blob), 1_200_000)]
    vid = _tiktok_video_id_from_url(source_url)
    ts_pat = re.compile(
        r'"(?:createTime|create_time|createTimeUTC|video_create_time)"\s*:\s*(\d{10})\b',
        re.I,
    )
    if vid:
        markers = [f'"{vid}"', f':"{vid}"']
        if len(vid) >= 12:
            markers.append(vid)
        seen_i: set[int] = set()
        for marker in markers:
            pos = 0
            while True:
                i = head.find(marker, pos)
                if i < 0:
                    break
                if i in seen_i:
                    pos = i + max(1, len(marker))
                    continue
                seen_i.add(i)
                win = head[i : i + 20_000]
                m = ts_pat.search(win)
                if m:
                    val = int(m.group(1))
                    if val > 0:
                        return _utc_iso_from_unix(val)
                pos = i + max(1, len(marker))
    m = ts_pat.search(head)
    if m:
        val = int(m.group(1))
        if val > 0:
            return _utc_iso_from_unix(val)
    return ""


def infer_publication_date_from_text(
    text: str,
    *,
    max_scan_chars: int = 96000,
    source_url: str = "",
) -> str:
    """Extract a *publish* time string (then normalize via ``calendar_date_from_raw_string``).

    Skips ``data-time`` and bare ``<time>`` without article microdata; for ``/blogs/`` paths,
    ``<time itemprop="datePublished">`` and ``"published_at"`` JSON (Shopify-style) are scanned
    in a wide window. Pass *source_url*
    so Reddit (``created_utc``) and YouTube (watch-page player JSON / itemprop before
    JSON-LD ``WebPage``) paths apply correctly.
    """
    if not text or not str(text).strip():
        return ""
    full_text = str(text)
    host = _hostname_from_url(source_url)
    head_cap = max_scan_chars
    if _hostname_needs_wide_head_and_json_ld(host) or _url_path_contains_blog_segment(source_url):
        head_cap = max(max_scan_chars, min(500_000, len(full_text)))
    blob = full_text[:head_cap]

    if _hostname_is_reddit(host):
        raw_unix = _try_unix_from_post_created_json(full_text[:500_000])
        if raw_unix:
            return raw_unix

    if _youtube_primary_video_url(source_url):
        yt_video = _infer_video_publish_metadata(full_text)
        if yt_video:
            return yt_video
        # Tavily/SERP snippets: short text with "Month D, YYYY — ..." prefix; no HTML.
        yt_snip = _infer_serp_snippet_date(full_text)
        if yt_snip:
            return yt_snip

    # Shopify ``/blogs/``: article JSON ``published_at`` must beat generic ``<head>`` meta — themes
    # often emit stale ``article:published_time`` / shop defaults before real article timestamps
    # (Tavily and other extracts keep both in one blob). Run **before** ``_META_PUBLISH_PATTERNS``.
    if _url_path_contains_blog_segment(source_url):
        blog_pub = _try_cms_blog_article_publish(full_text, source_url)
        if blog_pub:
            return blog_pub

    # 36氪: visible ``·YYYY年MM月DD日`` byline must beat ``© … 2026`` / sidebar dates / shell JSON-LD.
    if _hostname_is_36kr(host):
        kr_pub = _try_36kr_article_publish(full_text, source_url)
        if kr_pub:
            return kr_pub

    for pat, flags in _META_PUBLISH_PATTERNS:
        m = re.search(pat, blob, flags)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw

    if _hostname_is_wallstreetcn(host):
        ws = _try_wallstreetcn_article_publish(full_text, source_url)
        if ws:
            return ws

    if _hostname_is_sina_cj(host):
        sina_pub = _try_sina_cj_article_publish(full_text, source_url)
        if sina_pub:
            return sina_pub

    # Before JSON-LD: generic ``WebPage`` dates must not beat visible 发布时间 / meta on these hosts.
    if _hostname_is_eet_china(host):
        eet_pub = _try_eet_china_mp_article_publish(full_text, source_url)
        if eet_pub:
            return eet_pub

    if _hostname_is_xueqiu(host):
        xq = _try_xueqiu_status_publish(full_text, source_url)
        if xq:
            return xq

    if _hostname_is_eeworld(host):
        ee_pub = _try_eeworld_article_publish(full_text, source_url)
        if ee_pub:
            return ee_pub

    if _hostname_is_instagram(host):
        ig = _try_instagram_publish(full_text, source_url)
        if ig:
            return ig

    # Facebook: same ``_try_facebook_embedded_publish_unix`` as before (posts / videos / reels /
    # permalink / query ids / ``/photos/…/POST_ID``). Run **before** JSON-LD so embedded
    # story unix beats generic WebPage dates; a second pass remains below after headings/labels.
    if _hostname_is_facebook(host):
        fb_u = _try_facebook_embedded_publish_unix(
            full_text[:1_000_000], source_url=source_url
        )
        if fb_u:
            return fb_u

    # X / Twitter: hydrate JSON uses ISO or legacy ``created_at``; must run **before** JSON-LD —
    # otherwise a shell ``WebPage`` / unrelated script can win and skip tweet timestamps.
    if _hostname_is_x_or_twitter(host):
        xt = _try_x_embedded_created_at(full_text[:1_200_000], source_url)
        if xt:
            return xt

    json_ld_cap = 900_000
    json_ld_scripts = 48
    if _hostname_is_yahoo_or_msn(host) or _hostname_is_linkedin(host) or _hostname_is_youtube_host(host):
        json_ld_cap = min(len(full_text), 2_000_000)
        json_ld_scripts = 96
    if _url_path_contains_blog_segment(source_url):
        json_ld_scripts = max(json_ld_scripts, 96)
        json_ld_cap = max(json_ld_cap, min(len(full_text), 1_500_000))

    jp = _infer_json_ld_publish_date(blob, max_chars=min(len(blob), json_ld_cap), max_scripts=json_ld_scripts)
    if jp:
        return jp
    if len(full_text) > len(blob) or json_ld_cap > len(blob):
        jp2 = _infer_json_ld_publish_date(
            full_text,
            max_chars=json_ld_cap,
            max_scripts=json_ld_scripts,
        )
        if jp2:
            return jp2

    ts_div = _try_timestamp_div_us_mdy_am_pm(blob)
    if ts_div:
        return ts_div
    if len(full_text) > len(blob):
        ts2 = _try_timestamp_div_us_mdy_am_pm(
            full_text[: min(len(full_text), max(head_cap, 500_000))]
        )
        if ts2:
            return ts2

    if _hostname_is_zhihu(host):
        zh = _try_zhihu_article_publish(full_text, source_url)
        if zh:
            return zh
        zh_md = _try_zhihu_markdown_publish(full_text, source_url)
        if zh_md:
            return zh_md

    if _hostname_is_finviz(host):
        try:
            pth = unquote(urlparse((source_url or "").strip()).path or "").lower()
        except Exception:
            pth = ""
        if "/news/" in pth:
            fv = _try_finviz_news_pipe_byline(full_text, source_url)
            if fv:
                return fv

    if _hostname_is_medium(host):
        mp = _try_medium_embedded_publish(full_text, source_url)
        if mp:
            return mp

    if _hostname_is_zacks(host):
        zk = _try_zacks_embedded_publish(full_text, source_url)
        if zk:
            return zk

    if _hostname_is_yahoo_or_msn(host) and not _hostname_is_reddit(host):
        yp = _infer_yahoo_msn_portal_json(full_text[: min(len(full_text), 2_000_000)])
        if yp:
            return yp

    for pat, flags in _EXTRA_PUBLISH_KEY_PATTERNS:
        m = re.search(pat, blob, flags)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw

    csdn = _infer_csdn_style_publish(blob)
    if csdn:
        return csdn

    cn_portal = _infer_cn_portal_json_publish(blob)
    if not cn_portal and len(full_text) > len(blob):
        cn_portal = _infer_cn_portal_json_publish(full_text[:400_000])
    if cn_portal:
        return cn_portal

    loose = _infer_from_heading_lines(blob)
    if loose:
        return loose

    created = _infer_created_label_multiline(blob)
    if created:
        return created

    if _hostname_is_facebook(host):
        fb_u2 = _try_facebook_embedded_publish_unix(
            full_text[:1_000_000], source_url=source_url
        )
        if fb_u2:
            return fb_u2

    if _youtube_video_context(source_url, full_text):
        video = _infer_video_publish_metadata(full_text)
        if video:
            return video

    if _hostname_is_tiktok(host):
        tt = _try_tiktok_create_time(full_text[:1_200_000], source_url)
        if tt:
            return tt

    # Alternate naming (Parsely, DC.*, itemprop, CMS JSON) — last resort before giving up.
    if not _hostname_is_reddit(host):
        loose_cap = 900_000 if _url_path_contains_blog_segment(source_url) else 600_000
        scan_loose = full_text[: min(len(full_text), loose_cap)]
        alt_m = _infer_loose_meta_publish(scan_loose)
        if alt_m:
            return alt_m
        alt_j = _infer_loose_json_publish_kv(scan_loose)
        if alt_j:
            return alt_j

    # Fallback: no publish date found — accept a *modified / edited* date as a best effort.
    # This covers pages that only expose ``article:modified_time``, ``dateModified``,
    # ``og:updated_time``, or visible "Updated:", "Edited by:", "最近更新：" labels.
    edited = _infer_edited_or_modified_date(full_text, source_url=source_url)
    if edited:
        return edited

    return ""


def infer_reddit_post_calendar_date(text: str, *, max_scan_chars: int = 1_000_000) -> str:
    """Best-effort post date for ``reddit.com`` threads only.

    Uses embedded ``created_utc`` (or escaped variants), then ISO ``createdAt`` from GraphQL
    payloads. **Ignores** visible page text, headings, and ``<time>`` nodes — those often
    reflect “top posts this month” or UI chrome and are wrong for the thread.
    """
    if not text or not str(text).strip():
        return ""
    window = str(text)[:max_scan_chars]
    raw_unix = _try_unix_from_post_created_json(window)
    if raw_unix:
        cal = calendar_date_from_raw_string(raw_unix)
        if cal:
            return cal
        if re.match(r"^\d{4}-\d{2}-\d{2}T", raw_unix):
            return raw_unix[:10]
    raw_iso = _try_reddit_created_at_iso(window)
    if raw_iso:
        cal = calendar_date_from_raw_string(raw_iso)
        if cal:
            return cal
        if re.match(r"^\d{4}-\d{2}-\d{2}T", raw_iso):
            return raw_iso[:10]
    return ""


def infer_publication_calendar_date(
    text: str,
    *,
    max_scan_chars: int = 96000,
    source_url: str = "",
) -> str:
    """Return publication date as ``YYYY-MM-DD``, or empty string.

    Pass *source_url* (final URL after redirects) so Reddit ``created_utc`` and YouTube
    deep-page ``datePublished`` / ``publishDate`` paths apply; Tavily and HTTP fallback
    should pass the result URL here.
    """
    if _hostname_is_reddit(_hostname_from_url(source_url)):
        r = infer_reddit_post_calendar_date(text, max_scan_chars=max(max_scan_chars, 500_000))
        if r:
            return r
    raw = infer_publication_date_from_text(
        text, max_scan_chars=max_scan_chars, source_url=source_url
    )
    if not raw:
        return ""
    cal = calendar_date_from_raw_string(raw)
    if cal:
        return cal
    # Unix ISO from _utc_iso_from_unix: 2023-11-14T22:13:20Z
    if re.match(r"^\d{4}-\d{2}-\d{2}T", raw):
        return raw[:10]
    return ""


def _try_unix_from_post_created_json(blob: str) -> str:
    """Forum/post ``created_utc`` style (e.g. Reddit), not generic ``timestamp``."""
    head = blob[:500_000]
    patterns = (
        r'"(?:created_utc|createdUtc)"\s*:\s*(\d{10,16}(?:\.\d+)?)\b',
        # Escaped JSON (some embedded payloads)
        r'\\"(?:created_utc|createdUtc)\\"\s*:\s*(\d{10,16}(?:\.\d+)?)\b',
    )
    for pat in patterns:
        m = re.search(pat, head, re.I)
        if not m:
            continue
        val = int(float(m.group(1)))
        if val > 10_000_000_000:
            val //= 1000
        if val <= 0:
            continue
        return _utc_iso_from_unix(val)
    return ""


def _utc_iso_from_unix(ts: int) -> str:
    try:
        dt = datetime.fromtimestamp(ts, tz=UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, ValueError, OverflowError):
        return ""


def _infer_created_label_multiline(blob: str) -> str:
    """PEP-style ``Created:`` / ``:   26-May-2023`` blocks (document creation ≈ publish)."""
    m = re.search(
        r"(?is)\bCreated\s*:\s*(?:\n\s*:\s*)?(\d{1,2}-[A-Za-z]{3}-(?:19|20)\d{2})\b",
        blob[:16000],
    )
    if not m:
        return ""
    frag = m.group(1).strip()
    return frag if frag and _looks_like_date_token(frag) else ""


# Cap for a second-pass scan (YouTube watch HTML is often ~1MB+; metadata is mid-file).
_VIDEO_METADATA_SCAN_CAP = 2_000_000

_YT_INITIAL_PLAYER_MARKER = "ytInitialPlayerResponse"


def _infer_youtube_player_response_publish(text: str) -> str:
    """Parse ``publishDate`` / ``uploadDate`` inside ``ytInitialPlayerResponse`` (watch-page JSON).

    Tavily and other extractors often drop ``<meta itemprop>`` but keep large inline scripts;
    the player blob is the most reliable *upload = publish* signal for the primary video.
    """
    if not text or _YT_INITIAL_PLAYER_MARKER not in text:
        return ""
    idx = text.find(_YT_INITIAL_PLAYER_MARKER)
    window = text[idx : idx + min(1_500_000, len(text) - idx)]
    for pat in (
        r'"publishDate"\s*:\s*"([^"]+)"',
        r'"uploadDate"\s*:\s*"([^"]+)"',
    ):
        m = re.search(pat, window)
        if not m:
            continue
        frag = (m.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag
    return ""


def _infer_serp_snippet_date(text: str) -> str:
    """Extract publication date from a Tavily/Google SERP snippet (YouTube, Instagram, etc.).

    Handles three formats that HTML-only extractors miss when only a short snippet is provided:
    1. ``"Month D, YYYY — ..."``  — Google/Tavily snippet date prefix
    2. ``"YYYY-MM-DD"``  — ISO date anywhere in short snippet
    3. ``"N days/weeks/months/years ago"``  — relative date (approximated from today)
    """
    if not text:
        return ""
    # Only scan short snippets — if the text is large HTML it has its own path.
    scan = text[: min(len(text), 4000)]
    # Pattern 1: "Mar 27, 2025 —" or "March 27, 2025 —" (Google/Tavily snippet date prefix)
    m = re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2}),?\s+((?:19|20)\d{2})(?:\s*[\u2014\-]|\s|$)",
        scan,
        re.I,
    )
    if m:
        frag = f"{m.group(1)} {m.group(2)}, {m.group(3)}"
        if _looks_like_date_token(frag):
            return frag
    # Pattern 2: ISO date anywhere in short snippet
    m2 = re.search(r"\b((?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b", scan)
    if m2:
        return m2.group(1)
    # Pattern 3: relative "N unit(s) ago" — approximate from today's date
    rel = re.search(
        r"\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b",
        scan,
        re.I,
    )
    if rel:
        try:
            from datetime import date, timedelta
            n = int(rel.group(1))
            unit = rel.group(2).lower()
            today = date.today()
            if unit in ("second", "minute", "hour"):
                approx = today
            elif unit == "day":
                approx = today - timedelta(days=n)
            elif unit == "week":
                approx = today - timedelta(weeks=n)
            elif unit == "month":
                # Approximate: 30 days per month
                approx = today - timedelta(days=n * 30)
            else:  # year
                approx = today.replace(year=today.year - n)
            return approx.strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""


def _try_reddit_created_at_iso(blob: str) -> str:
    """GraphQL / newer Reddit HTML: ISO ``createdAt`` on posts (after ``created_utc`` pass)."""
    head = blob[:800_000]
    patterns = (
        r'"(?:createdAt|created_at)"\s*:\s*"((?:19|20)\d{2}-\d{2}-\d{2}T[^"]+)"',
        r'\\"(?:createdAt|created_at)\\"\s*:\s*\\"((?:19|20)\d{2}-\d{2}-\d{2}T[^"\\]+)\\"',
    )
    for pat in patterns:
        m = re.search(pat, head, re.I)
        if not m:
            continue
        frag = (m.group(1) or "").strip()
        if frag and _looks_like_date_token(frag):
            return frag
    return ""


# Story JSON can be sparse; 8KB was too small for some group-post shells.
_FB_STORY_TIMESTAMP_WINDOW = 32_000


def _try_facebook_embedded_publish_unix(blob: str, *, source_url: str = "") -> str:
    """Embedded JSON on ``facebook.com`` HTML (only safe to run when host is Facebook).

    Pages embed many ``creation_time`` values (group creation, unrelated stories, etc.). When the
    URL contains a post/video id, prefer a timestamp in JSON **near that id** instead of the
    document-global first match.
    """
    # Callers pass up to 1M chars; scan the full passed window (was 900k, which dropped tail).
    head = blob[: min(len(blob), 1_000_000)]
    ids = _facebook_story_ids_from_url(source_url)
    ts_pat = re.compile(
        r'"(?:creation_time|creationTime|publish_time|publishTime|story_publish_time|storyPublishTime|'
        r"created_time|post_publish_time|feed_creation_time|feedCreationTime)"
        r'"\s*:\s*(\d{10,13})\b',
        re.I,
    )
    for sid in ids:
        markers = [
            f'"{sid}"',
            f"'{sid}'",
            f'"story_fbid":"{sid}"',
            f'"story_fbid": "{sid}"',
            f'"legacyId":"{sid}"',
            f'"legacyId": "{sid}"',
            f'"post_id":"{sid}"',
            f'"postId":"{sid}"',
        ]
        if len(sid) >= 12:
            markers.append(sid)
        seen_pos: set[int] = set()
        for marker in markers:
            start = 0
            while True:
                pos = head.find(marker, start)
                if pos < 0:
                    break
                if pos in seen_pos:
                    start = pos + max(1, len(marker))
                    continue
                seen_pos.add(pos)
                chunk = head[pos : pos + _FB_STORY_TIMESTAMP_WINDOW]
                m = ts_pat.search(chunk)
                if m:
                    val = int(m.group(1))
                    if val > 10_000_000_000:
                        val //= 1000
                    if val > 0:
                        got = _utc_iso_from_unix(val)
                        if got:
                            return got
                start = pos + max(1, len(marker))
    m = ts_pat.search(head)
    if not m:
        return ""
    val = int(m.group(1))
    if val > 10_000_000_000:
        val //= 1000
    if val <= 0:
        return ""
    return _utc_iso_from_unix(val)


# Only used when ``_youtube_video_context`` is true: real video *publish* signals first,
# then ``uploadDate`` (YouTube uses both; avoids scanning these on arbitrary pages).
_VIDEO_PUBLISH_ITEMPROP_PATTERNS: list[tuple[str, int]] = [
    (
        r'<meta[^>]+itemprop\s*=\s*["\']datePublished["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]+itemprop\s*=\s*["\']datePublished["\']',
        re.I | re.DOTALL,
    ),
]

_VIDEO_PUBLISH_FALLBACK_PATTERNS: list[tuple[str, int]] = [
    (r'"publishDate"\s*:\s*"([^"]+)"', re.I),
    (
        r'<meta[^>]+itemprop\s*=\s*["\']uploadDate["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        re.I | re.DOTALL,
    ),
    (
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]+itemprop\s*=\s*["\']uploadDate["\']',
        re.I | re.DOTALL,
    ),
]


def _infer_video_publish_metadata(text: str) -> str:
    """Extract video *publish* times from large HTML (YouTube watch pages)."""
    if not text:
        return ""
    scan = text[:_VIDEO_METADATA_SCAN_CAP] if len(text) > _VIDEO_METADATA_SCAN_CAP else text
    ytr = _infer_youtube_player_response_publish(scan)
    if ytr:
        return ytr
    for pat, flags in _VIDEO_PUBLISH_ITEMPROP_PATTERNS:
        m = re.search(pat, scan, flags)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw
    for pat, flags in _VIDEO_PUBLISH_FALLBACK_PATTERNS:
        m = re.search(pat, scan, flags)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if raw and _looks_like_date_token(raw):
            return raw
    return ""


def _infer_csdn_style_publish(blob: str) -> str:
    """Best-effort first-publish date for CSDN-style pages when JSON-LD is missing.

    CSDN embeds ``"pubDate"`` inside ``application/ld+json`` (already handled above).
    Search snippets and reader HTML often drop ``<script>`` but keep the banner HTML
    ``<span>于&nbsp;2022-03-20 14:23:49&nbsp;首次发布</span>`` or the ``postTime`` JS var.
    """
    head = blob[:120000]

    m = re.search(
        r"(?is)于(?:&nbsp;|\u00a0|\s)+(\d{4}-\d{2}-\d{2})(?:\s+[\d:]+)?(?:&nbsp;|\u00a0|\s)*首次发布",
        head,
    )
    if m:
        y, mo, d = (int(x) for x in m.group(1).split("-"))
        if _valid_ymd(y, mo, d):
            return m.group(1)

    m = re.search(
        r'(?is)\b(?:var\s+)?postTime\s*=\s*["\'](\d{4}-\d{2}-\d{2})\s',
        head,
    )
    if m:
        y, mo, d = (int(x) for x in m.group(1).split("-"))
        if _valid_ymd(y, mo, d):
            return m.group(1)

    return ""


def _infer_cn_portal_json_publish(blob: str) -> str:
    """Unix publish time embedded in JSON (36氪、少数派、国内 CMS 等常见 ``publishTime`` 毫秒字段)."""
    head = blob[:240_000]
    m = re.search(
        r'"(?:publishTime|publishedTime|publish_time|ptime|first_publish_time|firstPublishTime)"\s*:\s*(\d{10,13})\b',
        head,
        re.I,
    )
    if m:
        val = int(m.group(1))
        if val > 10_000_000_000:
            val //= 1000
        if val > 0:
            return _utc_iso_from_unix(val)
    # Avoid matching ``"pagepubtime"`` (Sina shell template); require full key names only.
    m2 = re.search(
        r'"(?:publishTime|published_at|publishedAt|pub_date)"\s*:\s*"(\d{4}-\d{2}-\d{2})',
        head,
        re.I,
    )
    if m2:
        frag = m2.group(1).strip()
        return frag if frag and _looks_like_date_token(frag) else ""
    return ""


def _infer_edited_or_modified_date(full_text: str, *, source_url: str = "") -> str:
    """Last-resort fallback: extract a *modified / edited* date when no publish date exists.

    Covers:
    - HTML meta ``article:modified_time`` / ``og:updated_time`` / ``dateModified``
    - JSON-LD ``dateModified``
    - JSON keys ``updated_at`` / ``modified_at`` / ``last_modified``
    - Visible labels: "Updated:", "Last updated:", "Edited by:", "最近更新：", "编辑于", etc.
    """
    if not full_text:
        return ""
    blob = full_text[:600_000]

    # --- HTML meta tags ---
    _MODIFIED_META_PATS = (
        r'<meta[^>]+(?:property|name)\s*=\s*["\']article:modified_time["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]*(?:property|name)\s*=\s*["\']article:modified_time["\']',
        r'<meta[^>]+(?:property|name)\s*=\s*["\']og:updated_time["\'][^>]*\bcontent\s*=\s*["\']([^"\']+)["\']',
        r'<meta[^>]+\bcontent\s*=\s*["\']([^"\']+)["\'][^>]*(?:property|name)\s*=\s*["\']og:updated_time["\']',
    )
    for pat in _MODIFIED_META_PATS:
        m = re.search(pat, blob, re.I | re.DOTALL)
        if m:
            raw = (m.group(1) or "").strip()
            if raw and _looks_like_date_token(raw):
                return raw

    # --- JSON-LD dateModified ---
    for script_m in re.finditer(r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>', blob, re.I | re.DOTALL):
        chunk = script_m.group(1)
        dm = re.search(r'"dateModified"\s*:\s*"([^"]+)"', chunk)
        if dm:
            raw = dm.group(1).strip()
            if raw and _looks_like_date_token(raw):
                return raw

    # --- JSON key-value: updated_at, modified_at, last_modified, lastModified, etc. ---
    for pat in (
        r'"(?:updated_at|updatedAt|modified_at|modifiedAt|last_modified|lastModified|date_updated|dateUpdated|update_time|updateTime)"\s*:\s*"([^"\\]{4,80})"',
        r'"(?:updated_at|updatedAt|modified_at|modifiedAt|last_modified|lastModified|date_updated|dateUpdated|update_time|updateTime)"\s*:\s*(\d{10,16})\b',
    ):
        m = re.search(pat, blob, re.I)
        if not m:
            continue
        raw = (m.group(1) or "").strip()
        if re.fullmatch(r"\d{10,16}", raw):
            ts = _utc_iso_from_unix(int(raw))
            if ts:
                return ts
        if raw and _looks_like_date_token(raw):
            return raw

    # --- Visible heading labels ---
    head = full_text[:32_000]
    label_re = re.compile(
        r"(?im)^\s*(?:"
        r"Updated|Last\s+updated|Last\s+modified|Modified|Edited|Edited\s+by"
        r"|更新(?:时间|于|日期)?|最近更新|编辑(?:时间|于|日期)?|修改(?:时间|于|日期)?"
        r")\s*(?:by\s+\S+\s*)?(?:on|at|[:：]|\s)+\s*(.+?)\s*$"
    )
    for line in head.splitlines():
        line = line.strip()
        m = label_re.match(line)
        if not m:
            continue
        frag = m.group(1).strip()
        frag = re.split(r"\s*[·|•]\s*", frag, maxsplit=1)[0].strip()
        if len(frag) > 120:
            frag = frag[:120]
        if frag and _looks_like_date_token(frag):
            return frag

    # --- <time> element with datetime= in modified context ---
    for m in re.finditer(r'<time[^>]+datetime\s*=\s*["\']([^"\']+)["\'][^>]*>', blob[:200_000], re.I):
        ctx = blob[max(0, m.start() - 120) : m.start()].lower()
        if re.search(r"(update|modif|edit|最近|更新|编辑)", ctx):
            raw = (m.group(1) or "").strip()
            if raw and _looks_like_date_token(raw):
                return raw

    return ""


def _infer_from_heading_lines(blob: str) -> str:
    """Match visible 'Posted on …' / 'Release date:' / '发布于…' lines."""
    head = blob[:16000]
    line_re = re.compile(
        r"(?im)^\s*(?:"
        r"Posted|Publishing date|Date posted|Published|Release date|Published date|Date released"
        r"|发布(?:时间|于|日期)|发表(?:时间|于|日期)|发布时间"
        r")\s*(?:on|at|[:：]|\s)+\s*(.+?)\s*$"
    )
    for line in head.splitlines():
        line = line.strip()
        m = line_re.match(line)
        if not m:
            continue
        frag = m.group(1).strip()
        frag = re.split(r"\s*[·|•]\s*", frag, maxsplit=1)[0].strip()
        frag = re.split(r"\s+by\s+", frag, maxsplit=1, flags=re.I)[0].strip()
        if len(frag) > 120:
            frag = frag[:120]
        if frag and _looks_like_date_token(frag):
            return frag
    return ""


_LOOSE_META_KEY_SKIP = re.compile(
    r"(?i)(modified|updated|revision|last[-_]?mod|expires|refresh|etag|valid[-_]?until)",
)


def _loose_meta_key_looks_publish(key: str) -> bool:
    k = (key or "").strip().lower()
    if not k or _LOOSE_META_KEY_SKIP.search(k):
        return False
    if k in (
        "article:published_time",
        "og:published_time",
        "citation_date",
        "citation_publication_date",
    ):
        return True
    if "parsely" in k and "pub" in k:
        return True
    if k.startswith("dc.") or k.startswith("dcterms."):
        if any(x in k for x in ("modified", "updated", "valid")):
            return False
        if any(x in k for x in ("issued", "created", "date")):
            return True
    if k in ("pubdate", "publication-date", "original-publish-date", "sailthru.date"):
        return True
    if "datepublished" in k.replace(" ", "") or "publishdate" in k.replace(" ", ""):
        return True
    if re.search(r"(?i)(^|[-_])(publish|pub_date|post_date|postdate|firstpublished|originalpublish)", k):
        return True
    return False


def _infer_loose_meta_publish(blob: str) -> str:
    """Alternate CMS ``<meta>`` names (Parsely, DC.*, ``pubdate``, …) — strict passes already ran."""
    if not blob:
        return ""
    head = blob[: min(len(blob), 900_000)]
    for m in re.finditer(r"<meta\b[^>]*>", head, re.I):
        tag = m.group(0)
        low = tag.lower()
        if "article:modified_time" in low or "og:updated_time" in low:
            continue
        if _LOOSE_META_KEY_SKIP.search(tag) and "published" not in low:
            continue
        km = re.search(r'(?:property|name|itemprop)\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        if not km:
            continue
        k = km.group(1).strip()
        if not _loose_meta_key_looks_publish(k):
            continue
        cm = re.search(r'\bcontent\s*=\s*["\']([^"\']*)["\']', tag, re.I)
        if not cm:
            continue
        val = (cm.group(1) or "").strip()
        if val and _looks_like_date_token(val):
            return val
    return ""


_LOOSE_JSON_KEY_SKIP = re.compile(r"(?i)(modified|updated|revision|last_?edit|expire|valid_until)")
_LOOSE_JSON_KEY_PUBLISH = re.compile(
    r"(?i)(datepublished|date_published|published_at|publishedtime|"
    r"publish(ed)?(time|date|at)?|pub_?date|post_?date|first_?publish|article_?date|issued|parsely)",
)


def _infer_loose_json_publish_kv(blob: str) -> str:
    """``"post_date": "…"``-style pairs when not caught by stricter passes."""
    if not blob:
        return ""
    head = blob[: min(len(blob), 900_000)]
    for m in re.finditer(r'"([^"\\]{1,96})"\s*:\s*"([^"\\]{4,120})"', head):
        k = m.group(1).lower()
        v = m.group(2).strip()
        if re.fullmatch(r"(?i)pagepubtime|pageshowpubtime|pagetemplatepubtime", k):
            continue
        if _LOOSE_JSON_KEY_SKIP.search(k) and "publish" not in k:
            continue
        if not _LOOSE_JSON_KEY_PUBLISH.search(k):
            continue
        if v and _looks_like_date_token(v):
            return v
    return ""


def _looks_like_date_token(s: str) -> bool:
    """Reject obvious junk (URLs, long prose) while accepting ISO-like and English dates."""
    if len(s) > 120:
        return False
    if s.startswith("http://") or s.startswith("https://"):
        return False
    if not re.search(r"\d", s):
        return False
    if re.search(
        r"^\d{4}-\d{2}-\d{2}",
        s,
    ) or re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", s):
        return True
    if re.search(
        r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        s,
    ):
        return True
    if re.search(
        r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
        s,
    ):
        return True
    if re.search(r"(年|月|日)", s):
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}T", s):
        return True
    if re.search(r"(?i)^[a-z]{3},\s+\d{1,2}\s+[a-z]{3}\s+\d{4}", s.strip()):
        return True
    return False
