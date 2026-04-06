"""Direct HTTP fetch to infer ``YYYY-MM-DD`` when search/extract payloads lack a date.

Used when merge/heuristics leave no calendar date — see
``CitationMiddleware._apply_http_fetch_publication_date_fallback`` (enabled by default with a
per-response URL budget; set ``fetch_publication_date_fallback: false`` in ``web_search``
``model_extra`` to disable).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from urllib.parse import urlparse, urlunparse

import httpx

from deerflow.utils.publication_date import infer_publication_calendar_date, infer_reddit_post_calendar_date

logger = logging.getLogger(__name__)

# Browser-like headers: many news/social CDNs throttle or strip payloads for bot UAs.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}


def _hostname_is_reddit(host: str) -> bool:
    h = (host or "").lower()
    return h == "reddit.com" or h.endswith(".reddit.com")


def _hostname_is_facebook(host: str) -> bool:
    h = (host or "").lower()
    return h == "facebook.com" or h.endswith(".facebook.com")


def _reddit_thread_json_url(page_url: str) -> str:
    """``…/comments/{id}/slug/`` → ``….json`` (public thread JSON with ``created_utc``)."""
    try:
        p = urlparse((page_url or "").strip())
    except Exception:
        return ""
    if (p.scheme or "").lower() not in ("http", "https"):
        return ""
    path = (p.path or "").rstrip("/")
    if not path or "/comments/" not in path:
        return ""
    if path.endswith(".json"):
        return page_url.strip()
    return urlunparse((p.scheme, p.netloc, path + ".json", "", p.query, p.fragment))


def _facebook_m_dot_url(page_url: str) -> str:
    """Mobile host sometimes returns a lighter HTML shell (still often login-gated)."""
    u = (page_url or "").strip()
    if not u or "facebook.com" not in u.lower():
        return ""
    if "m.facebook.com" in u.lower():
        return ""
    u2 = re.sub(r"(?i)^(https?://)www\.facebook\.com\b", r"\1m.facebook.com", u, count=1)
    if u2 != u:
        return u2
    u2 = re.sub(r"(?i)^(https?://)facebook\.com\b", r"\1m.facebook.com", u, count=1)
    return u2 if u2 != u else ""


def _infer_merge_calendar(text: str, source_url: str) -> str:
    """Run HTML heuristics then apply URL path / slug merge (same order as citation middleware)."""
    inferred = infer_publication_calendar_date(text, source_url=source_url).strip()
    try:
        from deerflow.agents.middlewares.citation_middleware import CitationMiddleware

        merged, _ = CitationMiddleware._merge_publication_date_with_url(inferred, source_url)
        return (merged or inferred).strip()
    except Exception:
        return inferred


def _url_host_is_zhihu(url: str) -> bool:
    try:
        h = (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return False
    return h == "zhihu.com" or h.endswith(".zhihu.com")


def _url_host_is_facebook(url: str) -> bool:
    try:
        h = (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return False
    return h == "facebook.com" or h.endswith(".facebook.com")


def _url_host_is_x_or_twitter(url: str) -> bool:
    try:
        h = (urlparse((url or "").strip()).hostname or "").lower()
    except Exception:
        return False
    return h in ("x.com", "twitter.com") or h.endswith(".x.com") or h.endswith(".twitter.com")


def resolve_jina_reader_api_key() -> str:
    """Prefer ``JINA_API_KEY``; else ``tools.web_search`` ``model_extra`` (for ``$VAR`` in YAML)."""
    env = (os.getenv("JINA_API_KEY") or "").strip()
    if env:
        return env
    try:
        from deerflow.config import get_app_config

        tc = get_app_config().get_tool_config("web_search")
        extra = getattr(tc, "model_extra", None) if tc is not None else None
        if isinstance(extra, dict):
            for k in ("jina_reader_api_key", "jina_api_key"):
                v = extra.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass
    return ""


def _jina_reader_fetch(target_url: str, *, return_format: str, timeout: float) -> str:
    """POST to Jina Reader (same service as ``jina_ai``); returns body text or ``\"\"``."""
    u = (target_url or "").strip()
    if not u or not url_allowed_for_ssrf_guard(u):
        return ""
    fmt = (return_format or "markdown").strip().lower()
    if fmt not in ("markdown", "html", "text"):
        fmt = "markdown"
    headers = {
        "Content-Type": "application/json",
        "X-Return-Format": fmt,
        "X-Timeout": str(int(max(1, min(timeout, 120)))),
    }
    key = resolve_jina_reader_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.post("https://r.jina.ai/", headers=headers, json={"url": u})
    except Exception as exc:
        logger.debug("Jina reader POST failed for %s: %s", u[:80], exc)
        return ""
    if r.status_code != 200:
        logger.debug("Jina reader %s returned %s for %s", fmt, r.status_code, u[:80])
        return ""
    return (r.text or "").strip()


def infer_publication_calendar_via_jina_reader(url: str, *, timeout: float = 28.0) -> str:
    """Use Jina Reader to fetch Zhihu, Facebook, or X.com URLs and infer ``YYYY-MM-DD``.

    **Zhihu:** datacenter IPs often get 403; Reader keeps ``发布于`` / ``发表于`` or
    EN ``Edit <!-- -->YYYY-MM-DD`` lines.

    **Facebook:** direct GET often returns 4xx or a login shell; Reader sometimes returns
    enough HTML/markdown for post-id–scoped ``creation_time`` heuristics.

    **X.com:** client-side rendering returns empty shell; Reader extracts hydration JSON
    with ``created_at`` timestamps.
    """
    u = (url or "").strip()
    if not u or not url_allowed_for_ssrf_guard(u):
        return ""
    if not (_url_host_is_zhihu(u) or _url_host_is_facebook(u) or _url_host_is_x_or_twitter(u)):
        return ""
    cap = 1_200_000
    # Two formats: markdown (byline line) then html (may retain more structure).
    slice_timeout = max(5.0, float(timeout) / 2.0)
    for fmt in ("markdown", "html"):
        body = _jina_reader_fetch(u, return_format=fmt, timeout=slice_timeout)
        if not body:
            continue
        if len(body) > cap:
            body = body[:cap]
        cal = _infer_merge_calendar(body, u).strip()
        if cal:
            return cal
    return ""


def url_allowed_for_ssrf_guard(url: str) -> bool:
    """Reject schemes/hosts that must not be hit from server-side fetch (basic SSRF hygiene)."""
    try:
        p = urlparse((url or "").strip())
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost",) or host.endswith(".localhost"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return False
    except ValueError:
        pass
    return True


def infer_publication_calendar_from_url_http(
    url: str,
    *,
    timeout: float = 12.0,
    max_response_bytes: int = 2_097_152,
) -> str:
    """GET *url*, scan HTML/text with existing heuristics, return ``YYYY-MM-DD`` or ``""``.

    - **Path override:** strong ``/YYYY/MM/DD/`` in the URL wins over a conflicting
      ``article:published_time`` in HTML (fixes syndicated / wrong meta vs WordPress path).
    - **Reddit:** if the HTML shell lacks ``created_utc``, a second GET to the same thread
      ``.json`` endpoint is tried.
    - **Facebook:** optional second GET to ``m.facebook.com`` when the first body yields
      no date (still often login-gated).
    """
    u = (url or "").strip()
    if not u or not url_allowed_for_ssrf_guard(u):
        return ""
    # Zhihu / Facebook / X.com: try Reader first (avoids 403/400 shells from bare datacenter GET).
    slice_timeout = min(float(timeout), 35.0)
    if _url_host_is_zhihu(u):
        jcal = infer_publication_calendar_via_jina_reader(u, timeout=slice_timeout)
        if jcal:
            return jcal
    if _url_host_is_facebook(u):
        jcal_fb = infer_publication_calendar_via_jina_reader(u, timeout=slice_timeout)
        if jcal_fb:
            return jcal_fb
    if _url_host_is_x_or_twitter(u):
        jcal_x = infer_publication_calendar_via_jina_reader(u, timeout=slice_timeout)
        if jcal_x:
            return jcal_x
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(u, headers=dict(_BROWSER_HEADERS))
            final = str(r.url)
            if not url_allowed_for_ssrf_guard(final):
                return ""
            try:
                orig_host = (urlparse(u).hostname or "").lower()
            except Exception:
                orig_host = ""

            if r.status_code >= 400:
                if _hostname_is_facebook(orig_host):
                    murl = _facebook_m_dot_url(u)
                    if murl and url_allowed_for_ssrf_guard(murl):
                        try:
                            rm = client.get(murl, headers=dict(_BROWSER_HEADERS))
                            if rm.status_code < 400:
                                tm = rm.text
                                if len(tm) > max_response_bytes:
                                    tm = tm[:max_response_bytes]
                                got = _infer_merge_calendar(tm, str(rm.url))
                                if got:
                                    return got
                        except Exception as exc:
                            logger.debug("facebook m. fetch after %s for %s: %s", r.status_code, u[:96], exc)
                return ""

            text = r.text
            if len(text) > max_response_bytes:
                text = text[:max_response_bytes]
            try:
                host = urlparse(final).hostname or ""
            except Exception:
                host = ""

            if _hostname_is_reddit(host):
                cal = infer_reddit_post_calendar_date(text)
                if not cal:
                    jurl = _reddit_thread_json_url(final)
                    if jurl and url_allowed_for_ssrf_guard(jurl):
                        try:
                            rj = client.get(jurl, headers=dict(_BROWSER_HEADERS))
                            if rj.status_code < 400:
                                tj = rj.text
                                if len(tj) > max_response_bytes:
                                    tj = tj[:max_response_bytes]
                                cal = infer_reddit_post_calendar_date(tj)
                        except Exception as exc:
                            logger.debug("reddit .json fetch failed for %s: %s", jurl[:96], exc)
                return (cal or "").strip()

            result = _infer_merge_calendar(text, final)
            if result:
                return result

            if _hostname_is_facebook(host):
                murl = _facebook_m_dot_url(final)
                if murl and url_allowed_for_ssrf_guard(murl):
                    try:
                        rm = client.get(murl, headers=dict(_BROWSER_HEADERS))
                        if rm.status_code < 400:
                            tm = rm.text
                            if len(tm) > max_response_bytes:
                                tm = tm[:max_response_bytes]
                            result = _infer_merge_calendar(tm, str(rm.url))
                            if result:
                                return result
                    except Exception as exc:
                        logger.debug("facebook m. fetch failed for %s: %s", murl[:96], exc)
            return ""
    except Exception as exc:
        logger.debug("publication_date HTTP fetch failed for %s: %s", u[:96], exc)
        return ""
