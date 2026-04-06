"""Normalize search-hit titles for IEEE-style reference lines."""

from __future__ import annotations

import re
from urllib.parse import urlparse

# SERP / snippet noise
_ELLIPSIS_RUN = re.compile(r"\.{2,}|…+")
_TRAIL_JUNK = re.compile(r"[\s\.…\-–—]+$")


def strip_all_ellipsis(text: str) -> str:
    """Remove ellipsis runs and collapse whitespace."""
    if not text:
        return ""
    t = _ELLIPSIS_RUN.sub(" ", text)
    return re.sub(r"\s+", " ", t).strip()


def strip_trailing_ellipsis(text: str) -> str:
    """Drop trailing ellipsis / dots."""
    if not text:
        return ""
    return _ELLIPSIS_RUN.sub("", text).rstrip()


def strip_trailing_punctuation(text: str) -> str:
    """Strip common trailing punctuation from a title fragment."""
    if not text:
        return ""
    return text.rstrip(" \t.,;:!?…-–—")


def strip_trailing_article_id_from_title(title: str, url: str) -> str:
    """Drop a trailing numeric echo of ``…/id/{id}`` (e.g. ``… 51fusa.com. 2045``) — *id* is not a year."""
    t = (title or "").strip()
    u = (url or "").strip()
    if not t or not u:
        return t
    try:
        path = urlparse(u).path or ""
    except Exception:
        return t
    m = re.search(r"/id/(\d{1,12})(?:\.html?)?$", path.rstrip("/"), re.I)
    if not m:
        return t
    aid = m.group(1)
    if not aid or len(aid) < 2:
        return t
    t2 = re.sub(rf"(?:\s*[.．]\s*|\s+){re.escape(aid)}\s*$", "", t).strip()
    return t2 if t2 else t


def display_domain_for_reference(url: str, site_fallback: str = "") -> str:
    """Return host for bibliography (lowercase, no ``www.``, no path).

    ``site_fallback`` may be a Tavily ``site`` field or bare hostname.
    """
    raw = (site_fallback or "").strip().lower()
    host = ""
    if raw and "://" not in raw:
        host = raw.split("/")[0].split(":")[0]
    if not host and (url or "").strip():
        try:
            host = (urlparse(url.strip()).netloc or "").lower()
        except Exception:
            host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def polish_search_hit_title(title: str, snippet: str, url: str = "") -> str:
    """Produce a short display title from SERP fields (fallback: host from URL)."""
    t = strip_all_ellipsis((title or "").strip())
    if not t and snippet:
        t = strip_all_ellipsis((snippet or "").strip()[:200])
    if not t and url:
        try:
            host = urlparse(url).netloc or ""
            t = host.removeprefix("www.") or url
        except Exception:
            t = url
    t = strip_trailing_ellipsis(t)
    t = strip_trailing_punctuation(t)
    # Facebook SERP / index noise: trailing host, `` - Facebook`` label, or ``Facebook. facebook.com``.
    if url and "facebook.com" in url.lower():
        t = re.sub(r"(?i)\.?\s*facebook\.com\s*$", "", t).strip()
        t = strip_trailing_punctuation(t)
        t = re.sub(r"(?i)\s*facebook\s*\.\s*facebook\.com\s*$", "", t).strip()
        t = strip_trailing_punctuation(t)
        # ``Question - Facebook`` is redundant when the citation URL is already a Facebook post.
        t = re.sub(r"(?i)\s+[-–—]\s*facebook\s*$", "", t).strip()
        t = strip_trailing_punctuation(t)
    if len(t) > 400:
        t = t[:397].rsplit(" ", 1)[0] + "…"
    return t or "Source"


def polish_fetched_page_title(
    api_title: str,
    body_text_or_markdown: str,
    *,
    url: str,
    raw_html: str | None = None,
) -> str:
    """Best-effort title after ``web_fetch`` (metadata + body lead); API-compatible with legacy ``src.utils``."""
    _ = raw_html
    snippet = (body_text_or_markdown or "")[:800]
    return polish_search_hit_title((api_title or "").strip(), snippet, url)

