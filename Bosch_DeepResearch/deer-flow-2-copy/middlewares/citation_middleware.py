"""Middleware for automatic citation injection using IEEE numeric format.

This middleware acts as a post-processing step after the model generates a response.
It implements strict IEEE [1][2][3]... numeric citation format:

Pipeline:
1. Extracts all citation tags and content snippets from tool results (web_search, web_fetch, etc.)
2. Validates URLs come from tool results (prevent hallucination)
3. Intelligently matches content snippets to sentences in the AI response
4. Inserts inline citations after matching sentences
5. Converts [citation:Title](URL) format to numeric [1][2][3]...
6. Compacts adjacent citations: [1] [2] → [1][2]
7. Appends IEEE-formatted "## 参考文献" section with web-style rows
   ``[n] "Title". URL``
8. Ensures reference section only contains URLs actually cited in body

Output format (IEEE numeric):
- Body: "Text here[1][2]. More text[3]."
- References: ``[n] "Title (may include org and 4-digit year)". https://…`` — year only in the
  quoted title; full calendar dates stay on citation metadata internally.

The middleware modifies the last AIMessage in-place using the LangGraph
add_messages reducer (same `id` → update).
"""

import json
import logging
import re
from collections.abc import Iterator
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.utils.publication_date import (
    calendar_date_from_raw_string,
    infer_publication_calendar_date,
    infer_reddit_post_calendar_date,
)
from deerflow.utils.reference_titles import (
    display_domain_for_reference,
    polish_search_hit_title,
    strip_all_ellipsis,
    strip_trailing_article_id_from_title,
    strip_trailing_ellipsis,
    strip_trailing_punctuation,
)

logger = logging.getLogger(__name__)


class CitationMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    pass


# ──────────────────────────────────────────────────────────────────────────────
# IEEE Citation Format Constants and Validators
# ──────────────────────────────────────────────────────────────────────────────

class IEEE_CITATION_FORMAT:
    """IEEE numeric citation format specification.
    
    Examples:
    - Body: "Text[1][2]. More[3]."
    - Reference: [n] "Title". https://… (title in quotes, period, then URL)
    """
    
    NUMERIC_PATTERN = re.compile(r"\[\d+\]")
    """Pattern for numeric citations like [1], [2], etc."""
    
    COMPACT_PATTERN = re.compile(r"(\[\d+\])+")
    """Pattern for compact citation clusters like [1][2][3]."""
    
    REFERENCE_LINE_PATTERN = re.compile(
        r"^\[\d+\]\s+\"[^\"]*\"\.\s+https?://"
        r"|^\[\d+\]\s+\"[^\"]*,\"\s+https?://"
        r"|^\[\d+\]\s+\"[^\"]*\"\.\s+\[Online\]\.\s+Available:\s+https?://"
    )
    """Bibliography row: ``[n] "Title". URL``, legacy ``"Title," URL``, or legacy ``[Online]. Available:``."""
    
    @staticmethod
    def validate_numeric_citation(text: str) -> bool:
        """Check if text uses only numeric [1][2]... format (no [citation:...])."""
        has_legacy = "[citation:" in text
        has_numeric = bool(IEEE_CITATION_FORMAT.NUMERIC_PATTERN.search(text))
        return not has_legacy and has_numeric
    
    @staticmethod
    def count_citations(text: str) -> int:
        """Count total numeric citations in text."""
        return len(IEEE_CITATION_FORMAT.NUMERIC_PATTERN.findall(text))
    
    @staticmethod
    def get_citation_numbers(text: str) -> set[int]:
        """Extract all unique citation numbers from text."""
        matches = IEEE_CITATION_FORMAT.NUMERIC_PATTERN.findall(text)
        return {int(m[1:-1]) for m in matches}  # Remove [ and ]


class CitationMiddleware(AgentMiddleware[CitationMiddlewareState]):
    """Post-process AI responses to ensure inline citations and a reference list.

    Pipeline
    --------
    1.  Scan all ToolMessages for citation tags and content snippets.
    2.  Get the last AIMessage (the final, non-tool-calling response).
    3.  Match content snippets to sentences in the response.
    4.  Insert inline citations after matching sentences.
    5.  Append a well-formatted reference section listing all sources.
    """

    state_schema = CitationMiddlewareState

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _reference_heading_regex() -> str:
        """Reference heading matcher (Markdown and plain-text).

        Supports:
        - `## 参考文献` / `## References`
        - numbered report sections such as `## 6. References` / `6. References` (consulting-style outlines)
        - plain lines like `参考文献` / `References`

        NOTE: Do **not** treat a lone Chinese word ``引用`` as a bibliography heading.
        It appears frequently in normal prose (e.g. “引用如下”) and caused false
        splits where the entire message body was treated as “before references” and
        then discarded by pruning when no ``[n]`` appeared in that (empty) prefix.
        """
        # Match heading either as Markdown (optionally `#` / `##`) or as a plain line.
        # Optional numbered prefix (e.g. ``6.`` / ``2.3.``) matches "## 6. References"
        # and outline-style ``6. References`` so the
        # model's final section is not missed (otherwise we append ``## 参考文献`` twice).
        # IMPORTANT: anchor to end-of-line so we don't match strings like "参考文献如下".
        return (
            r"(?:^|\n)\s*(?:#{1,2}\s+)?(?:\d+(?:\.\d+)*\.?\s+)?"
            r"(?:References|参考文献|Bibliography|参考资料)\s*(?:$|\n)"
        )

    @staticmethod
    def _strip_legacy_h3_h4_source_blocks(text: str) -> str:
        """Remove ``###`` / ``####`` blocks that duplicate the bibliography.

        Models often emit ``### Primary Sources`` (under ``## 参考文献`` or mid-report) with
        the same URLs as inline citations. Those are removed; the canonical list is **参考文献**
        only.
        """
        if not (text or "").strip():
            return text
        legacy_h3 = (
            r"Primary\s+Sources|Secondary\s+Sources|Media\s+Coverage|"
            r"Academic\s*(?:/\s*)?\s*Technical\s+Sources|Community\s+Sources|"
            r"Official\s+Sources|Government\s+Sources|Additional\s+Sources|"
            r"Web\s+Sources|Key\s+Sources|External\s+Links|"
            r"主要来源|次要来源|媒体报道|学术来源|社区来源|官方来源|外部链接|来源列表"
        )
        block = re.compile(
            rf"(?:^|\n)\s*#{{3,4}}\s+(?:{legacy_h3})\s*(?:\n|$)"
            rf".*?(?=(?:^|\n)\s*#{{1,6}}\s+\S|\Z)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        out = text
        for _ in range(12):
            nxt = block.sub("\n", out)
            if nxt == out:
                break
            out = nxt
        return out

    @staticmethod
    def _strip_legacy_sources_sections(text: str) -> str:
        """Remove model-written ``## Sources`` / ``## Primary Sources`` / 来源 blocks.

        Inline ``[citation:…](URL)`` is the single source of truth; the middleware appends
        one IEEE-style **参考文献** list. Legacy categorized link lists duplicate that and
        are stripped so the final document has a single bibliography.
        """
        if not (text or "").strip():
            return text
        # H2 headings only (## not ###). Subsections under ## Sources are removed with the block.
        legacy_h2 = (
            r"Sources|Primary\s+Sources|Secondary\s+Sources|Media\s+Coverage|"
            r"Academic\s*(?:/\s*)?\s*Technical\s+Sources|Community\s+Sources|"
            r"Official\s+Sources|Government\s+Sources|Web\s+Sources|Key\s+Sources|"
            r"Further\s+Reading|Works\s+Cited|Additional\s+Sources|External\s+Links|"
            r"Related\s+Links|Link\s+List|Citation\s+Sources|"
            r"资料来源|引用来源|来源|主要来源|参考链接|引用列表|文献来源|网页来源"
        )
        # Allow blank lines before the H2 so "\n\n## Sources" matches (not only "\n## Sources").
        # End at the next peer ``##`` heading (not ``###``) so subsections under Sources are
        # removed with the block; use ``\Z`` for Sources sections at EOF.
        # NOTE: use literal ``##`` here — ``rf"#{2}"`` is an f-string and becomes ``#2``.
        block = re.compile(
            rf"(?:^|\n)\s*##\s+(?:{legacy_h2})\s*(?:\n|$)"
            rf".*?(?=(?:^|\n)\s*##(?![#])\s+\S|\Z)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        out = text
        for _ in range(12):
            nxt = block.sub("\n", out)
            if nxt == out:
                break
            out = nxt
        out = CitationMiddleware._strip_legacy_h3_h4_source_blocks(out)
        return re.sub(r"\n{3,}", "\n\n", out)

    @staticmethod
    def _ieee_web_reference_line(index: int, title: str, display_url: str) -> str:
        """One bibliography row: ``[n] "Title". URL`` (period only between quoted title and URL)."""
        t = (title or "").strip() or "Unspecified Source"
        u = (display_url or "").strip()
        return f'[{index}] "{t}". {u}'

    @staticmethod
    def _escape_title_for_reference_line(title: str) -> str:
        """Titles from SERP/citations may contain `"`, which breaks ``[n] "Title".`` rendering."""
        t = (title or "").strip()
        return t.replace('"', "'")

    @staticmethod
    def _sanitize_citation_tag_title(title: str) -> str:
        """Make *title* safe inside ``[citation:title](url)`` (parser stops at first ``]``)."""
        t = (title or "").strip() or "Source"
        t = t.replace("]", "").replace("\n", " ")
        if len(t) > 240:
            t = t[:237] + "…"
        return t

    @staticmethod
    def _tool_text_window_around_url(content: str, url: str, radius: int = 6000) -> str:
        """Return a slice of *content* around *url* so nearby metadata lines are included.

        Clips to the enclosing ====...==== result block when present so that
        Organization/Published lines from adjacent results are never included.
        """
        if not content or not url:
            return content or ""
        pos = content.find(url)
        if pos == -1:
            return content

        # Attempt to identify the boundary of the result block that contains *url*.
        # Tavily tool output separates results with lines of 20+ '=' characters.
        SEP = re.compile(r"\n={20,}")

        # Last separator BEFORE the URL position marks the start of this block.
        block_start = 0
        for m in SEP.finditer(content, 0, pos):
            block_start = m.end()  # character just after the separator line

        # First separator AFTER the URL position marks the end of this block.
        block_end = len(content)
        m_after = SEP.search(content, pos + len(url))
        if m_after:
            block_end = m_after.start()

        # Combine block boundaries with the original radius limit.
        a = max(max(0, pos - radius), block_start)
        b = min(min(len(content), pos + len(url) + radius), block_end)
        return content[a:b]

    @staticmethod
    def _parse_metadata_from_tool_text(blob: str) -> dict[str, str]:
        """Parse Published / Author / Organization lines from tool output.

        Uses the LAST match for each field so that, when the text window
        accidentally includes an adjacent result's metadata at the top, the
        metadata belonging to the actual result (which appears later in the
        block) takes precedence.
        """
        out: dict[str, str] = {"published_date": "", "author_org": ""}
        if not blob:
            return out

        # findall returns all matches; pick the last one to prefer the nearest
        # entry to the URL rather than one from a preceding result block.
        pub_matches = re.findall(r"(?im)^\s*Published:\s*(.+)$", blob)
        if pub_matches:
            out["published_date"] = pub_matches[-1].strip()

        auth_matches = re.findall(r"(?im)^\s*Author:\s*(.+)$", blob)
        org_matches = re.findall(r"(?im)^\s*Organization:\s*(.+)$", blob)
        author = auth_matches[-1].strip() if auth_matches else ""
        org = org_matches[-1].strip() if org_matches else ""
        parts: list[str] = []
        if author:
            parts.append(author)
        if org and org.lower() != author.lower():
            parts.append(org)
        out["author_org"] = ", ".join(parts)
        return out

    @staticmethod
    def _is_reddit_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).netloc or "").lower()
        except Exception:
            return False
        return host == "reddit.com" or host.endswith(".reddit.com")

    @staticmethod
    def _is_facebook_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "facebook.com" or host.endswith(".facebook.com")

    @staticmethod
    def _is_medium_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "medium.com" or host.endswith(".medium.com")

    @staticmethod
    def _is_zacks_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "zacks.com" or host.endswith(".zacks.com")

    @staticmethod
    def _is_blockchain_news_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "blockchain.news" or host.endswith(".blockchain.news")

    @staticmethod
    def _is_instagram_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "instagram.com" or host.endswith(".instagram.com")

    @staticmethod
    def _is_sina_cj_url(url: str) -> bool:
        """Sina mobile finance articles (``weibo: article:create_at`` often in ``<head>``)."""
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return (
            host == "cj.sina.cn"
            or host.endswith(".cj.sina.cn")
            or host == "cj.sina.com.cn"
            or host.endswith(".cj.sina.com.cn")
        )

    @staticmethod
    def _is_eet_china_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "eet-china.com" or host.endswith(".eet-china.com")

    @staticmethod
    def _is_wallstreetcn_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "wallstreetcn.com" or host.endswith(".wallstreetcn.com")

    @staticmethod
    def _is_eeworld_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "eeworld.com.cn" or host.endswith(".eeworld.com.cn")

    @staticmethod
    def _is_51fusa_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host == "51fusa.com" or host.endswith(".51fusa.com")

    @staticmethod
    def _url_path_contains_blogs_segment(url: str) -> bool:
        """Shopify / many CMS article URLs use ``/blogs/…``; merge fields so ``published_at`` can match."""
        try:
            from urllib.parse import unquote, urlparse

            pth = unquote(urlparse((url or "").strip()).path or "").lower()
        except Exception:
            return False
        return "/blogs/" in pth

    @staticmethod
    def _is_x_or_twitter_url(url: str) -> bool:
        try:
            from urllib.parse import urlparse

            host = (urlparse((url or "").strip()).hostname or "").lower()
        except Exception:
            return False
        return host in ("x.com", "twitter.com") or host.endswith(".x.com") or host.endswith(".twitter.com")

    @staticmethod
    def _merged_blob_for_publication_infer(
        raw_blob: str,
        display_title: str,
        snippet: str,
    ) -> str:
        """Prefer *raw* first so long SERP title/snippet does not push HTML/JSON out of scan windows."""
        parts: list[str] = []
        for p in (raw_blob, display_title, snippet):
            if isinstance(p, str) and p.strip():
                parts.append(p.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _merge_title_snippet_raw_for_publication_infer(url: str) -> bool:
        """Tavily hits often lack JSON-LD in *raw*; merge title + snippet + raw for these hosts."""
        return (
            CitationMiddleware._is_medium_url(url)
            or CitationMiddleware._is_zacks_url(url)
            or CitationMiddleware._is_blockchain_news_url(url)
            or CitationMiddleware._is_instagram_url(url)
            or CitationMiddleware._is_facebook_url(url)
            or CitationMiddleware._is_x_or_twitter_url(url)
            or CitationMiddleware._is_sina_cj_url(url)
            or CitationMiddleware._is_eet_china_url(url)
            or CitationMiddleware._is_wallstreetcn_url(url)
            or CitationMiddleware._is_eeworld_url(url)
            or CitationMiddleware._is_51fusa_url(url)
            or CitationMiddleware._url_path_contains_blogs_segment(url)
        )

    @staticmethod
    def _publication_date_http_fetch_config() -> dict[str, bool | int | float]:
        """``web_search`` tool ``model_extra`` keys for optional HTML fetch fallback."""
        try:
            from deerflow.config import get_app_config

            tc = get_app_config().get_tool_config("web_search")
        except Exception:
            tc = None
        extra = getattr(tc, "model_extra", None) or {}
        if not isinstance(extra, dict):
            extra = {}
        return {
            # Default on: Tavily bodies often omit <head>/JSON-LD; a bounded GET improves dates.
            "enabled": bool(extra.get("fetch_publication_date_fallback", True)),
            "max_urls": int(extra.get("fetch_publication_date_fallback_max_urls", 8)),
            "timeout": float(extra.get("fetch_publication_date_fallback_timeout", 12)),
            "max_bytes": int(extra.get("fetch_publication_date_fallback_max_bytes", 3_145_728)),
        }

    @staticmethod
    def _apply_http_fetch_publication_date_fallback(
        normalized_url: str,
        pub_merged: str,
        cache: dict[str, str],
        budget: list[int],
        cfg: dict[str, bool | int | float],
    ) -> str:
        """When merge left no calendar date, GET the page and re-run publication heuristics."""
        if (pub_merged or "").strip():
            return pub_merged
        if not cfg.get("enabled") or budget[0] <= 0:
            return pub_merged
        if normalized_url not in cache:
            from deerflow.utils.publication_date_fallback_fetch import infer_publication_calendar_from_url_http

            cal = infer_publication_calendar_from_url_http(
                normalized_url,
                timeout=float(cfg["timeout"]),
                max_response_bytes=int(cfg["max_bytes"]),
            )
            cache[normalized_url] = cal
            budget[0] -= 1
        got = cache.get(normalized_url, "")
        return got if got else pub_merged

    @staticmethod
    def _url_path_calendar_year_plausible(year: int) -> bool:
        """Reject ``2045``-style numeric *ids* that match ``(19|20)\\\\d{2}`` but are not years."""
        from datetime import datetime

        y = int(year)
        cap = datetime.now().year + 2
        return 1900 <= y <= cap

    @staticmethod
    def _date_hints_from_url_path(url: str) -> tuple[str, str, str]:
        """Infer publication calendar date or year from the URL path (not query).

        Returns ``(yyyy_mm_dd_or_empty, four_digit_year_or_empty, strength)`` where
        *strength* is ``path_ymd``, ``path_ym``, ``slug_year``, or ``""``.

        WordPress-style ``/2025/12/16/slug`` and ISO-ish ``/.../2025-09-02-id`` segments
        are treated as strong signals. Slug suffix ``...-best-tech-2026`` supplies a year
        when no full date exists in the path.

        Four-digit segments that look like ``20xx`` but exceed ``current_year + 2`` (e.g.
        ``…/id/2045.html``) are **not** treated as publication years. ``…/id/{id}.html`` is
        never interpreted as a slug-year when the filename equals the id.
        """
        from datetime import datetime
        from urllib.parse import urlparse

        if not url or not isinstance(url, str):
            return ("", "", "")
        try:
            parsed = urlparse(url.strip())
            path = (parsed.path or "").strip("/")
            netloc = (parsed.netloc or "").lower()
        except Exception:
            return ("", "", "")
        if not path:
            return ("", "", "")
        full = "/" + path + "/"

        # 大纪元等：路径 ``/b5/25/10/7/n….htm`` 表示西元 2025-10-07（两位年份 + 月 + 日）
        if "epochtimes.com" in netloc:
            em = re.search(
                r"/(?:[a-z]{1,4}\d?)/(\d{2})/(\d{1,2})/(\d{1,2})/",
                full,
                re.I,
            )
            if em:
                yy, mo, d = int(em.group(1)), int(em.group(2)), int(em.group(3))
                full_y = 2000 + yy if yy < 70 else 1900 + yy
                if CitationMiddleware._url_path_calendar_year_plausible(full_y):
                    try:
                        datetime(full_y, mo, d)
                        return (f"{full_y:04d}-{mo:02d}-{d:02d}", f"{full_y:04d}", "path_ymd")
                    except ValueError:
                        pass

        m = re.search(r"/((?:19|20)\d{2})/(\d{1,2})/(\d{1,2})/", full)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if CitationMiddleware._url_path_calendar_year_plausible(y):
                try:
                    datetime(y, mo, d)
                    return (f"{y:04d}-{mo:02d}-{d:02d}", f"{y:04d}", "path_ymd")
                except ValueError:
                    pass

        m = re.search(r"/((?:19|20)\d{2})-(\d{2})-(\d{2})/", full)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if CitationMiddleware._url_path_calendar_year_plausible(y):
                try:
                    datetime(y, mo, d)
                    return (f"{y:04d}-{mo:02d}-{d:02d}", f"{y:04d}", "path_ymd")
                except ValueError:
                    pass

        for seg in path.split("/"):
            if not seg or re.match(r"^10\.\d+", seg):
                continue
            mseg = re.match(r"^((?:19|20)\d{2})-(\d{2})-(\d{2})(?:$|[-_].+)", seg)
            if mseg:
                y, mo, d = int(mseg.group(1)), int(mseg.group(2)), int(mseg.group(3))
                if CitationMiddleware._url_path_calendar_year_plausible(y):
                    try:
                        datetime(y, mo, d)
                        return (f"{y:04d}-{mo:02d}-{d:02d}", f"{y:04d}", "path_ymd")
                    except ValueError:
                        pass

        m = re.search(r"/((?:19|20)\d{2})/(\d{1,2})/", full)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and CitationMiddleware._url_path_calendar_year_plausible(y):
                return ("", f"{y:04d}", "path_ym")

        last_seg = path.split("/")[-1]
        base = re.sub(r"\.[a-zA-Z0-9]{1,12}$", "", last_seg)
        # Skip long hex/paper-id slugs (Semantic Scholar, arXiv-style paths) — avoid junk years.
        if len(base) >= 24 and re.fullmatch(r"[0-9a-f]{24,}", base, re.I):
            return ("", "", "")
        # ``51fusa.com/…/id/2045`` or ``…/id/2045.html`` — trailing segment is article id, not ``YYYY``.
        id_tail = re.search(r"/id/(\d{1,12})(?:\.html?)?$", "/" + path.rstrip("/"), re.I)
        if id_tail and id_tail.group(1) == base:
            return ("", "", "")
        m = re.search(r"(?:^|[-_/])((?:19|20)\d{2})$", base)
        if m:
            y = int(m.group(1))
            if CitationMiddleware._url_path_calendar_year_plausible(y):
                return ("", m.group(1), "slug_year")

        return ("", "", "")

    @staticmethod
    def _four_digit_article_id_strings_from_url(url: str) -> set[str]:
        """``/id/2045`` segments: four-digit ids matching ``19xx``/``20xx`` are not publication years."""
        out: set[str] = set()
        if not url or not isinstance(url, str):
            return out
        try:
            from urllib.parse import urlparse

            p = (urlparse(url.strip()).path or "").strip("/")
        except Exception:
            return out
        if not p:
            return out
        wrapped = f"/{p}/"
        for m in re.finditer(r"/id/(\d{4})(?:\.html?)?/", wrapped, re.I):
            digits = m.group(1)
            if digits.startswith(("19", "20")):
                out.add(digits)
        return out

    @staticmethod
    def _merge_publication_date_with_url(
        published_date: str,
        url: str,
        *,
        title: str = "",
    ) -> tuple[str, str]:
        """Merge heuristic ``published_date`` with URL path hints.

        Explicit ``/YYYY/MM/DD/`` (or ISO-ish segment) in the URL overrides conflicting
        inferred dates from HTML/snippet heuristics.

        Returns ``(merged_published_date, url_year_fallback)`` where *url_year_fallback*
        is set for ``path_ym`` / ``slug_year`` when no full calendar string is chosen.
        """
        cal, uy, strength = CitationMiddleware._date_hints_from_url_path(url)
        pub = (published_date or "").strip()
        if strength == "path_ymd" and cal:
            return (cal, "")
        if strength == "path_ym" and uy:
            if not pub:
                return ("", uy)
            return (pub, "")
        if strength == "slug_year" and uy:
            if not pub:
                return ("", uy)
            pub_cal = calendar_date_from_raw_string(pub)
            pub_y = pub_cal[:4] if len(pub_cal) >= 4 else ""
            if not pub_y:
                py_m = re.search(r"\b(19|20)\d{2}\b", pub)
                pub_y = py_m.group(0) if py_m else ""
            tit = (title or "").strip()
            if (
                pub_y
                and pub_y != uy
                and tit
                and re.search(rf"(?<!\d){re.escape(uy)}(?!\d)", tit)
            ):
                return ("", uy)
            return (pub, "")
        return (pub, "")

    @staticmethod
    def _year_from_published_or_snippet(
        published_date: str,
        snippet: str,
        url_year_fallback: str,
        normalized_url: str,
        *,
        allow_snippet_year: bool = True,
    ) -> str:
        """Resolve a 4-digit year with URL hints before snippet text."""
        from datetime import datetime

        cap_y = datetime.now().year + 2
        id_years = CitationMiddleware._four_digit_article_id_strings_from_url(normalized_url)

        pub = (published_date or "").strip()
        if pub:
            for ym in re.finditer(r"\b(19|20)\d{2}\b", pub):
                ys = ym.group(0)
                y = int(ys)
                if ys in id_years:
                    continue
                if 1900 <= y <= cap_y:
                    return ys
        ufb = (url_year_fallback or "").strip()
        if ufb and re.fullmatch(r"\d{4}", ufb) and ufb not in id_years:
            yf = int(ufb)
            if 1900 <= yf <= cap_y:
                return ufb
        sn = (snippet or "").strip() if allow_snippet_year else ""
        if sn:
            # Snippets mix (a) real publish context, (b) historical years, (c) sidebar / "2026 model"
            # spam. Taking ``max`` alone misreads (b)+(c); taking ``min`` misreads (b) alone.
            # Heuristic: keep years in the "upper band" near the snippet's max year (within 5y),
            # then take the **minimum** of that band — usually the article date among related-news years.
            years = sorted(
                {
                    int(x)
                    for x in re.findall(r"\b((?:19|20)\d{2})\b", sn)
                    if x not in id_years
                }
            )
            plausible = [y for y in years if 1900 <= y <= cap_y]
            if plausible:
                ymax = max(plausible)
                floor_y = ymax - 5
                upper_band = [y for y in plausible if y >= floor_y]
                if upper_band:
                    return str(min(upper_band))
                return str(ymax)
        if not ufb:
            _, uy2, _st = CitationMiddleware._date_hints_from_url_path(normalized_url)
            if (
                uy2
                and re.fullmatch(r"\d{4}", uy2)
                and uy2 not in id_years
                and 1900 <= int(uy2) <= cap_y
            ):
                return uy2
        return ""

    @staticmethod
    def _compose_ieee_reference_title(
        base_title: str,
        author_org: str,
        published_date: str,
        year_fallback: str = "",
        domain: str = "",
    ) -> str:
        """
        Compose reference title components: Title. organization. domain. publication date.

        Caller wraps the result in quotes for ``[n] "…". URL``.

        - **domain**: registrable host (e.g. ``electrek.co``), omitted if redundant with title/org.
        - **date**: full ``YYYY-MM-DD`` when a calendar day is known; otherwise a four-digit year
          from ``published_date`` / ``year_fallback``.
        """
        base = (base_title or "").strip()
        if not base:
            base = "Unspecified Source"
        
        parts: list[str] = [base]
        
        pd = (published_date or "").strip()
        if pd:
            norm_cal = calendar_date_from_raw_string(pd)
            if norm_cal:
                pd = norm_cal
        if not pd and (year_fallback or "").strip():
            yf = year_fallback.strip()
            if re.fullmatch(r"\d{4}", yf):
                pd = yf
        
        ao = (author_org or "").strip().replace('"', "'")
        
        org = ""
        if ao:
            if "," in ao:
                parts_split = [p.strip() for p in ao.split(",", 1)]
                org = parts_split[1] if len(parts_split) > 1 else ""
            else:
                org = ao

        if org:
            parts.append(org)

        dom = (domain or "").strip().lower()
        if dom.startswith("www."):
            dom = dom[4:]
        if dom:

            def _host_key(h: str) -> str:
                x = (h or "").strip().lower()
                return x[4:] if x.startswith("www.") else x

            org_lower = (org or "").strip().lower()
            org_host = _host_key(org)
            # Skip appending domain when:
            # 1. org IS the domain (exact match), or
            # 2. org is a platform-branded name that already implies the domain
            #    (e.g. "Facebook Group · slug" implies facebook.com).
            _platform_domain_prefixes = ("facebook", "youtube", "instagram", "linkedin", "tiktok", "twitter", "reddit")
            _org_implies_dom = any(
                org_lower.startswith(p) for p in _platform_domain_prefixes
            ) and dom.split(".")[0] in _platform_domain_prefixes
            if org_host != dom and not _org_implies_dom:
                parts.append(dom)
        
        if pd:
            # Show full YYYY-MM-DD when a calendar day is known, else the 4-digit year.
            year_match = re.search(r"\b(19|20)\d{2}\b", pd)
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", pd):
                parts.append(pd)
            elif year_match:
                parts.append(year_match.group(0))
            else:
                parts.append(pd)
        
        return ". ".join(parts)

    @staticmethod
    def _expand_truncated_search_title(title: str, snippet: str | None) -> str:
        """Backward-compatible wrapper; full logic lives in ``reference_titles``."""
        return polish_search_hit_title(title or "", (snippet or "").strip(), url="")

    @staticmethod
    def _unwrap_outer_prose_fence_only(text: str) -> str:
        """If *text* is entirely wrapped in a prose code fence, return inner markdown.

        Aligns with frontend ``unwrapOuterProseMarkdownFence``: strip a single outer
        `` ```markdown `` / `` ``` `` wrapper so Streamdown parses headings/tables
        instead of rendering the whole report as a code block.
        """
        if not text or not str(text).strip():
            return text
        lines = str(text).replace("\r\n", "\n").split("\n")
        lo, hi = 0, len(lines) - 1
        while lo < len(lines) and lines[lo].strip() == "":
            lo += 1
        while hi >= 0 and lines[hi].strip() == "":
            hi -= 1
        if hi - lo < 2:
            return text
        open_line = lines[lo].strip()
        close_line = lines[hi].strip()
        open_m = re.match(r"^(`{3,}|~{3,})\s*([\w-]*)\s*$", open_line)
        if not open_m:
            return text
        fence_open = open_m[1]
        fence_char = fence_open[0]
        open_len = len(fence_open)
        lang = (open_m.group(2) or "").lower()
        prose_langs = {"", "markdown", "md", "text", "txt", "plaintext"}
        if lang not in prose_langs:
            return text
        close_m = re.match(r"^(`{3,}|~{3,})\s*$", close_line)
        if not close_m:
            return text
        fence_close = close_m[1]
        if fence_close[0] != fence_char or len(fence_close) < open_len:
            return text
        return "\n".join(lines[lo + 1 : hi]).strip("\n")

    @staticmethod
    def _merge_fenced_body_with_reference_section(text: str) -> str:
        """Keep the main report and 参考文献 as one markdown document for the UI.

        Typical failure mode: model outputs::

            ```markdown
            ## Report ...
            ```

        and the middleware appends ``## 参考文献`` *after* the closing fence. Renderers
        then show the report as a code block and references as unrelated paragraphs.
        Here we strip the outer fence from the main segment only and rejoin.
        """
        if not text:
            return text
        ref_m = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if not ref_m:
            return CitationMiddleware._unwrap_outer_prose_fence_only(text)
        main = text[: ref_m.start()].rstrip()
        refs = text[ref_m.start() :].strip()
        main_u = CitationMiddleware._unwrap_outer_prose_fence_only(main)
        return f"{main_u.rstrip()}\n\n{refs}\n"

    @staticmethod
    def _iter_citation_markdown_link_spans(text: str) -> Iterator[tuple[str, str, int, int]]:
        """Yield (title, url, start, end_exclusive) for each ``[citation:title](url)``.

        URLs may contain unescaped ``)`` (e.g. Wikipedia disambiguation paths). The naive
        regex ``https?://[^)]+`` truncates there and breaks tool extraction + inline parsing.
        """
        needle = "[citation:"
        pos = 0
        n = len(text)
        while pos < n:
            i = text.find(needle, pos)
            if i == -1:
                break
            title_start = i + len(needle)
            title_end = text.find("]", title_start)
            if title_end == -1:
                pos = title_start
                continue
            title = text[title_start:title_end].strip()
            if title_end + 1 >= n or text[title_end + 1] != "(":
                pos = title_end + 1
                continue
            url_start = title_end + 2
            k = url_start
            depth = 0
            while k < n:
                ch = text[k]
                if ch == "\\" and k + 1 < n:
                    k += 2
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        url = text[url_start:k].strip()
                        if url.startswith("http://") or url.startswith("https://"):
                            yield (title, url, i, k + 1)
                        pos = k + 1  # always advance past closing )
                        break
                    depth -= 1
                k += 1
            else:
                pos = url_start

    @staticmethod
    def _iter_citation_markdown_links(text: str) -> Iterator[tuple[str, str]]:
        for title, url, _, _ in CitationMiddleware._iter_citation_markdown_link_spans(text):
            yield (title, url)

    @staticmethod
    def _canonicalize_urls_in_citation_markdown(text: str) -> str:
        """Rewrite each ``[citation:title](url)`` so *url* matches ``_normalize_url`` output.

        LLMs often insert ASCII/Unicode spaces inside the URL (e.g. ``v14 - review``). We
        normalize **before** ``_strip_ungrounded_inline_citations`` so grounding compares the
        same canonical string as tool results, and Markdown is not left half-broken for
        ``_clean_broken_urls`` / renderers.

        Uses the same balanced-paren scan as ``_iter_citation_markdown_link_spans`` (not
        ``[^)]+``), so paths containing ``)`` still parse.
        """
        if not text or "[citation:" not in text:
            return text
        spans = list(CitationMiddleware._iter_citation_markdown_link_spans(text))
        if not spans:
            return text
        out = text
        for title, url, start, end in sorted(spans, key=lambda s: s[2], reverse=True):
            old = out[start:end]
            try:
                cleaned = CitationMiddleware._clean_url_for_reference(url)
                canon = CitationMiddleware._normalize_url(cleaned) if cleaned else ""
            except Exception:
                canon = ""
            if not canon:
                continue
            new_span = f"[citation:{title}]({canon})"
            if new_span == old:
                continue
            out = out[:start] + new_span + out[end:]
        return out

    @staticmethod
    def _strip_all_citation_markdown_links(text: str) -> str:
        """Remove every well-formed ``[citation:...](http...)`` span from *text*."""
        if not text:
            return text
        spans = list(CitationMiddleware._iter_citation_markdown_link_spans(text))
        if not spans:
            return text
        out: list[str] = []
        last = 0
        for _t, _u, start, end in spans:
            out.append(text[last:start])
            last = end
        out.append(text[last:])
        return "".join(out)

    @staticmethod
    def _strip_ungrounded_inline_citations(text: str, allowed_normalized_urls: set[str]) -> str:
        """Drop ``[citation:...](url)`` unless *url* normalizes to a tool-returned URL.

        LLMs often invent plausible article URLs (same shape as real news links). Pattern-based
        `_is_valid_url` cannot catch those; grounding to search/fetch tool output does.
        """
        if not text:
            return text
        if not allowed_normalized_urls:
            return CitationMiddleware._strip_all_citation_markdown_links(text)

        spans = list(CitationMiddleware._iter_citation_markdown_link_spans(text))
        if not spans:
            return text

        out: list[str] = []
        last = 0
        for _title, url, start, end in spans:
            out.append(text[last:start])
            try:
                clean_url = CitationMiddleware._clean_url_for_reference(url)
                norm = (
                    CitationMiddleware._normalize_url(clean_url)
                    if clean_url
                    else ""
                )
                if norm and norm in allowed_normalized_urls:
                    out.append(text[start:end])
                else:
                    logger.warning(
                        "CitationMiddleware: stripping ungrounded inline citation "
                        "(URL not from tool results): %s",
                        (url[:100] if url else ""),
                    )
            except Exception as e:
                logger.warning(
                    "CitationMiddleware: stripping inline citation (URL handling failed): %s — %s",
                    (url[:80] if url else ""),
                    e,
                )
            last = end
        out.append(text[last:])
        return "".join(out)

    @staticmethod
    def _clean_broken_urls(text: str) -> str:
        """Best-effort cleanup for明显残缺的 URL 片段，避免在最终报告中表现为“乱码”.

        典型情况：
        - 只剩下域名尾巴和路径，例如：
          - ``net/boyedu/article/details/149260127)。``
          - ``jiqizhixin.io/pages/fe/block/history.``
        - 这些通常不是用户可点击的完整链接，而是清洗/截断后的残片。

        清理策略（保守）：
        - 仅删除**不带协议头**且单独出现的尾巴式 URL 片段；
        - 不触碰正常的 Markdown 链接 `[text](https://...)` 或 `http(s)://` 开头的 URL。
        """
        import re

        # 先保护正常的完整 URL 和 Markdown 链接，避免误删
        # 标记所有正常的 http(s):// URL 和 [text](url) 格式的链接
        protected_placeholders: dict[str, str] = {}
        placeholder_counter = 0

        def protect_url(match: re.Match[str]) -> str:
            nonlocal placeholder_counter
            placeholder = f"__PROTECTED_URL_{placeholder_counter}__"
            protected_placeholders[placeholder] = match.group(0)
            placeholder_counter += 1
            return placeholder

        def protect_markdown_link(match: re.Match[str]) -> str:
            nonlocal placeholder_counter
            placeholder = f"__PROTECTED_MD_{placeholder_counter}__"
            protected_placeholders[placeholder] = match.group(0)
            placeholder_counter += 1
            return placeholder

        # CRITICAL: ``[citation:…](https://… v14 - review/)`` — LLMs insert spaces in the URL.
        # A naive ``https?://…[^\s]+`` match runs *before* Markdown protection and stops at the
        # first space, splitting the link and leaving `` - review/)`` as garbage ("乱码").
        cleaned = text
        cite_spans = list(CitationMiddleware._iter_citation_markdown_link_spans(cleaned))
        for _t, _u, start, end in sorted(cite_spans, key=lambda s: s[2], reverse=True):
            chunk = cleaned[start:end]
            ph = f"__PROTECTED_CITATION_MD_{placeholder_counter}__"
            protected_placeholders[ph] = chunk
            placeholder_counter += 1
            cleaned = cleaned[:start] + ph + cleaned[end:]

        # 裸 http(s)（不含已在 citation 块内的 URL）
        cleaned = re.sub(r"https?://[^\s\)\]\u4e00-\u9fff]+", protect_url, cleaned)
        cleaned = re.sub(r"\[[^\]]+\]\(https?://[^\)]+\)", protect_markdown_link, cleaned)
        
        # 删除常见尾巴式残缺链接（不以 http 开头，且前后是空白或标点）
        # 匹配前面可能有空格/括号/中文标点/行首，后面可能有右括号和句号的情况
        patterns = [
            # CSDN 类：net/boyedu/article/details/149260127) 或 net/boyedu/article/details/149260127)。
            r"(?:[\s\(（，。、]|^)net/[a-zA-Z0-9/._-]+[)）]?[。．\.]?",
            # 机器之心类：jiqizhixin.io/pages/fe/block/history.
            r"(?:[\s\(（，。、]|^)jiqizhixin\.io/[a-zA-Z0-9/._-]+[)）]?[。．\.]?",
            # 通用尾巴式片段：com/xxx、io/xxx、org/xxx、cn/xxx
            r"(?:[\s\(（，。、]|^)(?:com|io|org|cn|edu|gov)/[a-zA-Z0-9/._-]+[)）]?[。．\.]?",
            # 域名+路径但缺协议（非 Markdown 链接场景）
            # 例如：blog.csdn.net/boyedu/article/details/149260127)
            r"(?:[\s\(（，。、]|^)[a-zA-Z0-9.-]+\.(?:com|net|io|org|cn|edu|gov)/[a-zA-Z0-9/._-]+[)）]?[。．\.]?",
            # 常见异常拼接：区块链2.jiqizhixin.io/... 或 0.io/...
            r"(?:[\s\(（，。、]|^)\d+\.[a-zA-Z0-9.-]+\.(?:com|net|io|org|cn)/[a-zA-Z0-9/._-]+[)）]?[。．\.]?",
        ]

        for pat in patterns:
            cleaned = re.sub(pat, "", cleaned)
        
        # 额外：在还原受保护链接之前，清理长串百分号编码的 URL 残片（正文中常见的“乱码”部分）。
        # 由于此时正常 http(s) 链接已经被替换为占位符，所以这个规则只会命中那些
        # 非 http(s) 开头的残缺片段，不会误删参考文献里的正式 URL。
        cleaned = re.sub(
            r"(?:[\s\(（]|^)[^\s]*%[0-9A-Fa-f]{2}(?:[^\s]*%[0-9A-Fa-f]{2}){5,}[^\s]*",
            "",
            cleaned,
        )

        # 恢复被保护的正常链接
        for placeholder, original in protected_placeholders.items():
            cleaned = cleaned.replace(placeholder, original)

        # 压缩多余空白
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    @staticmethod
    def _citations_from_plain_json_search_results(
        content: str,
        tool_name: str,
    ) -> list[dict]:
        """Build citation dicts when ``web_search`` returns JSON (Firecrawl, InfoQuest, …).

        Those providers return a JSON array of objects with ``title`` / ``url`` / ``snippet``
        (or ``desc``) and **no** ``[citation:](url)`` markers. Without this fallback, the
        middleware finds zero tool citations and skips all post-processing.
        """
        raw = (content or "").strip()
        if not raw.startswith("["):
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        cfg_fb = CitationMiddleware._publication_date_http_fetch_config()
        fetch_cache: dict[str, str] = {}
        fetch_budget = [int(cfg_fb["max_urls"])] if cfg_fb.get("enabled") else [0]

        out: list[dict] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            url = (
                item.get("url")
                or item.get("link")
                or item.get("href")
                or ""
            )
            url = str(url).strip()
            if not url or not (
                url.startswith("http://") or url.startswith("https://")
            ):
                continue

            normalized_url = CitationMiddleware._normalize_url(url)
            title = (
                item.get("title")
                or item.get("name")
                or item.get("source")
                or ""
            )
            title = str(title).strip()

            snippet_val = (
                item.get("snippet")
                or item.get("desc")
                or item.get("description")
                or item.get("content")
                or ""
            )
            if isinstance(snippet_val, (dict, list)):
                snippet_val = json.dumps(snippet_val, ensure_ascii=False)
            snippet = str(snippet_val).strip()
            if len(snippet) > 500:
                snippet = snippet[:500]

            display_title = polish_search_hit_title(
                title, snippet, url=normalized_url
            )
            display_title = strip_trailing_article_id_from_title(
                display_title, normalized_url
            )
            citation_tag = f"[citation:{display_title}]({normalized_url})"

            # Extract metadata: site, year, author_org, and published_date
            # Must be done early so we have all necessary data
            
            # Extract published_date FIRST (needed for year extraction priority).
            # Search APIs (e.g. Tavily) often omit structured dates; optional raw HTML/markdown
            # in the payload is parsed heuristically (see ``publication_date`` util).
            raw_blob = item.get("raw_content") or item.get("rawContent") or ""
            raw_blob = raw_blob.strip() if isinstance(raw_blob, str) else ""

            if CitationMiddleware._is_reddit_url(normalized_url):
                pub_item = infer_reddit_post_calendar_date(raw_blob) if raw_blob else ""
            else:
                pub_item = (
                    item.get("published_date")
                    or item.get("publishedDate")
                    or item.get("publish_date")
                    or item.get("pubdate")
                    or item.get("date")
                    or ""
                )
                pub_item = str(pub_item).strip() if pub_item else ""
                if not pub_item:
                    infer_src = raw_blob
                    if CitationMiddleware._merge_title_snippet_raw_for_publication_infer(
                        normalized_url
                    ):
                        merged = CitationMiddleware._merged_blob_for_publication_infer(
                            raw_blob,
                            display_title,
                            snippet,
                        )
                        if merged:
                            infer_src = merged
                    if infer_src:
                        pub_item = infer_publication_calendar_date(
                            infer_src, source_url=normalized_url
                        )

            pub_item, url_year_fb = CitationMiddleware._merge_publication_date_with_url(
                pub_item, normalized_url, title=display_title
            )
            pub_item = CitationMiddleware._apply_http_fetch_publication_date_fallback(
                normalized_url,
                pub_item,
                fetch_cache,
                fetch_budget,
                cfg_fb,
            )
            pub_item, url_year_fb = CitationMiddleware._merge_publication_date_with_url(
                pub_item, normalized_url, title=display_title
            )

            # Extract domain from URL
            # For each URL, we need to extract its ACTUAL domain, not reuse previous values
            site = ""
            try:
                from urllib.parse import urlparse

                parsed = urlparse(normalized_url)
                netloc = parsed.netloc.lower() if parsed.netloc else ""
                
                # Remove 'www.' prefix and 'm.' (mobile) prefix for cleaner domain display
                # Keep the actual domain, not just first part
                if netloc:
                    if netloc.startswith("www."):
                        site = netloc[4:]
                    elif netloc.startswith("m."):
                        site = netloc[2:]
                    else:
                        site = netloc
                        
            except Exception:
                site = ""

            year_src = snippet
            if (
                not (pub_item or "").strip()
                and CitationMiddleware._merge_title_snippet_raw_for_publication_infer(
                    normalized_url
                )
            ):
                rb = (raw_blob or "").strip()
                parts: list[str] = []
                if rb:
                    parts.append(rb[:12_000])
                if snippet:
                    parts.append(snippet)
                if display_title:
                    parts.append(display_title)
                year_src = "\n".join(p for p in parts if p).strip()

            year = CitationMiddleware._year_from_published_or_snippet(
                pub_item,
                year_src,
                url_year_fb,
                normalized_url,
                allow_snippet_year=not CitationMiddleware._is_reddit_url(normalized_url),
            )

            # Extract author and organization information
            author_parts: list[str] = []
            if item.get("author"):
                author_parts.append(str(item["author"]).strip())
            elif item.get("authors"):
                ar = item["authors"]
                if isinstance(ar, (list, tuple)):
                    author_parts.extend(str(x).strip() for x in ar if str(x).strip())
                else:
                    author_parts.append(str(ar).strip())
            org_val = item.get("organization") or item.get("publisher") or item.get("site_name") or ""
            org_val = str(org_val).strip() if org_val else ""
            ao_parts = [p for p in author_parts if p]
            if org_val and org_val.lower() not in {p.lower() for p in ao_parts}:
                ao_parts.append(org_val)
            author_org_joined = ", ".join(ao_parts)

            raw_blob = item.get("raw_content") or item.get("rawContent") or ""
            raw_blob = raw_blob.strip() if isinstance(raw_blob, str) else ""

            out.append(
                {
                    "title": display_title,
                    "url": normalized_url,
                    "citation_tag": citation_tag,
                    "snippet": snippet,
                    "tool": tool_name or "",
                    "site": site,
                    "year": year,
                    "author_org": author_org_joined,
                    "published_date": pub_item,
                    "raw_content": raw_blob[:80_000],
                }
            )

        return out

    @staticmethod
    def _extract_tool_citations_with_snippets(messages: list) -> list[dict]:
        """Extract citations with their associated content snippets from ToolMessages.
        
        Returns list of citation dicts with title, url, citation_tag, snippet, tool, site, year,
        and optional author_org / published_date (from tool output or JSON providers).
        
        - title:   来源标题（来自 Tavily）
        - url:     规范化后的完整 URL
        - site:    站点名称（URL 域名，如 `www.bbc.com`）
        - year:    尝试从 snippet / URL 中解析出的年份（如 2024），若无法判断则为空字符串
        """
        seen_urls: set[str] = set()
        citations: list[dict] = []
        cfg_fb = CitationMiddleware._publication_date_http_fetch_config()
        fetch_cache: dict[str, str] = {}
        fetch_budget = [int(cfg_fb["max_urls"])] if cfg_fb.get("enabled") else [0]

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_name = getattr(msg, "name", "") or ""

            # Extract citation tags (balanced-paren URLs)
            citation_matches = list(CitationMiddleware._iter_citation_markdown_links(content))
            if not citation_matches:
                # Firecrawl / InfoQuest web_search returns JSON only — no [citation:] tags.
                for jc in CitationMiddleware._citations_from_plain_json_search_results(
                    content, tool_name
                ):
                    ju = jc.get("url", "").strip()
                    if not ju or ju in seen_urls:
                        continue
                    seen_urls.add(ju)
                    citations.append(jc)
                continue

            # Normalize URLs in extracted citations
            citation_matches = [(title, CitationMiddleware._normalize_url(url)) for title, url in citation_matches]

            # Legacy split: --- Result i --- (older prompts); Tavily now uses banner lines.
            result_blocks = re.split(r"---\s*Result \d+\s*---", content, flags=re.IGNORECASE)
            tavily_chunks = re.split(r"\n={20,}\n", content)

            # Try to extract snippets for each citation
            for title, url in citation_matches:
                url = url.strip()
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                # Strip all ellipsis from title before creating citation tag
                clean_title = strip_all_ellipsis(title.strip())
                citation_tag = f"[citation:{clean_title}]({url})"
                snippet = ""

                # Tavily / current web_search: block is between ===== lines; body follows
                # ``--- Content from this source ---`` (not ``Content:``).
                for chunk in tavily_chunks:
                    if not chunk.strip():
                        continue
                    if (
                        url in chunk
                        or (title.strip() and title.strip() in chunk)
                        or citation_tag in chunk
                    ):
                        content_m = re.search(
                            r"---\s*Content from this source\s*---\s*\n(.+?)(?:\n\n💡|\n={20,}|\Z)",
                            chunk,
                            re.DOTALL | re.IGNORECASE,
                        )
                        if content_m:
                            snippet = content_m.group(1).strip()
                            snippet = re.sub(
                                r"\[citation:[^\]]+\]\([^\)]+\)", "", snippet
                            )
                            snippet = re.sub(
                                r"💡\s*EXAMPLE.*?$",
                                "",
                                snippet,
                                flags=re.DOTALL | re.IGNORECASE,
                            )
                            snippet = re.sub(r"\s+", " ", snippet).strip()
                            if len(snippet) >= 20:
                                if len(snippet) > 500:
                                    snippet = snippet[:500]
                                break
                        break

                # Search in result blocks (skip first empty block)
                if not snippet:
                    for block in result_blocks[1:]:
                        # Check if this block contains our citation
                        if citation_tag in block or url in block or title.strip() in block:
                            # Try multiple patterns to extract content
                            patterns = [
                                r"Content:\s*(.+?)(?:\n\n|⚠️|💡|EXAMPLE|---|$)",  # Standard format
                                r"Content:\s*(.+?)(?:\n[^\s]|$)",  # Until next non-whitespace line
                                r"(?:Content|Snippet):\s*(.+?)(?:\n\n|$)",  # Alternative format
                            ]

                            for pattern in patterns:
                                content_match = re.search(
                                    pattern,
                                    block,
                                    re.DOTALL | re.IGNORECASE,
                                )
                                if content_match:
                                    snippet = content_match.group(1).strip()
                                    # Clean up: remove citation tags, example text, and extra whitespace
                                    snippet = re.sub(
                                        r"\[citation:[^\]]+\]\([^\)]+\)",
                                        "",
                                        snippet,
                                    )
                                    snippet = re.sub(
                                        r"💡\s*EXAMPLE.*?$",
                                        "",
                                        snippet,
                                        flags=re.DOTALL | re.IGNORECASE,
                                    )
                                    snippet = re.sub(r"\s+", " ", snippet)  # Normalize whitespace
                                    snippet = snippet.strip()

                                    # Only use if we got meaningful content
                                    if len(snippet) >= 20:
                                        if len(snippet) > 500:
                                            snippet = snippet[:500]
                                        break

                            if snippet:
                                break
                
                # If still no snippet, try to extract from web_fetch format
                if not snippet and "WEB PAGE" in content:
                    # web_fetch format: Title: ...\n\n--- Page Content ---\n<content>
                    page_match = re.search(
                        r"Title:\s*" + re.escape(title.strip()) + r".*?---\s*Page\s*Content\s*---\s*(.+?)(?:\n\n|$)",
                        content,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if page_match:
                        snippet = page_match.group(1).strip()[:500]
                
                # Ensure URL is normalized before storing
                normalized_url = CitationMiddleware._normalize_url(url)
                # Update citation_tag with normalized URL if it changed
                normalized_citation_tag = citation_tag.replace(url, normalized_url) if url != normalized_url else citation_tag

                blob = CitationMiddleware._tool_text_window_around_url(content, normalized_url)
                meta = CitationMiddleware._parse_metadata_from_tool_text(blob)

                # Derive published_date first (needed for year extraction priority)
                if CitationMiddleware._is_reddit_url(normalized_url):
                    wide = CitationMiddleware._tool_text_window_around_url(
                        content, normalized_url, radius=250_000
                    )
                    pub_date = infer_reddit_post_calendar_date(wide)
                else:
                    pub_date = (meta.get("published_date") or "").strip()
                pub_date, url_year_fb = CitationMiddleware._merge_publication_date_with_url(
                    pub_date, normalized_url, title=clean_title
                )
                pub_date = CitationMiddleware._apply_http_fetch_publication_date_fallback(
                    normalized_url,
                    pub_date,
                    fetch_cache,
                    fetch_budget,
                    cfg_fb,
                )
                pub_date, url_year_fb = CitationMiddleware._merge_publication_date_with_url(
                    pub_date, normalized_url, title=clean_title
                )

                # Derive simple site (domain) from the URL itself — MUST use normalized_url
                # for each citation independently; remove www./m. prefixes for cleaner display.
                site = ""
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(normalized_url)
                    netloc = parsed.netloc.lower() if parsed.netloc else ""
                    if netloc:
                        if netloc.startswith("www."):
                            site = netloc[4:]
                        elif netloc.startswith("m."):
                            site = netloc[2:]
                        else:
                            site = netloc
                except Exception:
                    site = ""

                year = CitationMiddleware._year_from_published_or_snippet(
                    pub_date,
                    snippet,
                    url_year_fb,
                    normalized_url,
                    allow_snippet_year=not CitationMiddleware._is_reddit_url(normalized_url),
                )

                raw_ctx = (blob or "").strip()[:80_000]

                citations.append(
                    {
                        "title": title.strip(),
                        "url": normalized_url,
                        "citation_tag": normalized_citation_tag,
                        "snippet": snippet,
                        "tool": getattr(msg, "name", ""),
                        "site": site,
                        "year": year,
                        "author_org": (meta.get("author_org") or "").strip(),
                        "published_date": pub_date,
                        "raw_content": raw_ctx,
                    }
                )

        return citations

    @staticmethod
    def _format_references_ieee(citations: list[dict]) -> str:
        """Format citations into a numbered reference list.

        Each line: ``[n] "Title". https://…`` (period after closing quote, space, URL).
        The quoted title may end with ``. org. YYYY`` when a publish year is known.

        Each citation dict needs at least ``title`` and ``url`` (``tool`` / ``snippet`` optional).
        Returned string does NOT include the ``## 参考文献`` heading.

        Rules:
        - One reference per unique URL; numbering ``[1]``, ``[2]``, …
        - Blank line between entries
        """
        if not citations:
            return ""

        # Deduplicate by normalized URL while preserving order
        seen = set()
        unique: list[dict] = []
        for c in citations:
            url = (c.get("url") or "").strip()
            if not url:
                continue
            try:
                normalized_url = CitationMiddleware._normalize_url(url)
                if normalized_url in seen:
                    continue
                seen.add(normalized_url)
            except Exception:
                # If normalization fails, still add it but skip duplicates by raw URL
                if url in seen:
                    continue
                seen.add(url)
            unique.append(c)

        lines: list[str] = []
        for idx, c in enumerate(unique, start=1):
            title = (c.get("title") or "").strip()
            url = (c.get("url") or "").strip()

            # Normalize URL to ensure consistency
            if url:
                try:
                    url = CitationMiddleware._normalize_url(url)
                except Exception:
                    pass  # Keep original if normalization fails

            # Fallback: derive a short title from snippet if title missing
            if not title:
                snippet = (c.get("snippet") or "").strip()
                if snippet:
                    short = snippet[:80]
                    if len(snippet) > 80:
                        short += "..."
                    title = short
            if not title:
                title = "Unspecified Source"

            row = dict(c)
            row["url"] = url
            CitationMiddleware._enrich_facebook_citation_dates(
                row,
                source_url=url,
                snippet=(c.get("snippet") or ""),
            )
            CitationMiddleware._enrich_zhihu_citation_dates(
                row,
                source_url=url,
                snippet=(c.get("snippet") or ""),
            )
            _ao = CitationMiddleware._facebook_ieee_organization(
                url,
                (row.get("author_org") or "").strip(),
                (row.get("site") or "").strip(),
            )
            _dom = display_domain_for_reference(url, (row.get("site") or "").strip())
            _pub = (row.get("published_date") or "").strip()
            # In the displayed reference list, only the 4-digit year is shown (not full date).
            # _compose_ieee_reference_title stores the full date internally; we pass the year here.
            _year_only = ""
            if _pub:
                _ym = re.search(r"\b(19|20)\d{2}\b", _pub)
                if _ym:
                    _year_only = _ym.group(0)
            composed = CitationMiddleware._compose_ieee_reference_title(
                title.replace('"', "'"),
                _ao,
                _year_only,
                year_fallback=(row.get("year") or "").strip(),
                domain=_dom,
            )
            title = CitationMiddleware._escape_title_for_reference_line(composed)

            lines.append(CitationMiddleware._ieee_web_reference_line(idx, title, url))

        # Join with blank lines between entries for better readability
        return "\n\n".join(lines) + "\n"

    @staticmethod
    def _extract_tool_citations(messages: list) -> list[dict]:
        """Return deduplicated list of citation dicts from ToolMessages (title, url, metadata)."""
        citations_with_snippets = CitationMiddleware._extract_tool_citations_with_snippets(messages)
        return [
            {
                "title": c["title"],
                "url": c["url"],
                "tool": c["tool"],
                "site": c.get("site", ""),
                "year": c.get("year", ""),
                "author_org": c.get("author_org", ""),
                "published_date": c.get("published_date", ""),
                "raw_content": c.get("raw_content", ""),
            }
            for c in citations_with_snippets
        ]

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Check if URL has valid format and common patterns.
        
        Returns False for:
        - Empty/None URLs
        - URLs without http/https scheme
        - URLs without a proper domain (no dot in netloc)
        - URLs matching known OBVIOUS hallucination patterns
        """
        if not url or not isinstance(url, str):
            return False
        
        url = url.strip()
        
        # Must start with http/https
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Check for OBVIOUS hallucination patterns (false positives acceptable to reduce hallucinations)
        obvious_hallucination_patterns = [
            r'fake.*\.com',  # fake-website.com, fake.com, etc.
            r'test.*\.com',   # test-site.com, test.com, etc.
            r'sample.*\.com',
            r'placeholder.*\.com',
            r'\[\d+\]\.\w+$',  # URLs ending with [number].format
            r'\.fake$',        # example.fake (not a real TLD)
            r'\.test$',        # example.test (reserved TLD)
            r'\.localhost$',   # example.localhost
            r'\.invalid$',     # example.invalid (reserved)
        ]
        
        for pattern in obvious_hallucination_patterns:
            if re.search(pattern, url, re.I):
                logger.warning(
                    "CitationMiddleware: URL matches obvious hallucination pattern: %s", url[:80]
                )
                return False
        
        # Must have a domain with at least one dot (but IP addresses are ok)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.netloc:
                return False
            
            # Allow both domain names and IP addresses
            # "192.168.1.1" has dots (valid IP)
            # "localhost" no dots (could be invalid for public web)
            # "example.com" has dots (valid domain)
            if '.' not in parsed.netloc and ':' not in parsed.netloc:
                # No dots and no port means likely invalid (e.g., "http://localhost" without port)
                logger.warning(
                    "CitationMiddleware: URL domain lacks identifier (possibly invalid): %s", url[:80]
                )
                return False
            
            return True
        except Exception as e:
            logger.warning("CitationMiddleware: URL parsing failed for '%s...': %s", url[:60], e)
            return False

    @staticmethod
    def _clean_url_for_reference(url: str) -> str:
        """Clean URL for use in reference list.
        
        Removes any citation tags or nested content from the URL.
        If URL contains nested citations, extracts the actual URL from the nested citation.
        """
        import re
        if not url:
            return ""
        
        # If URL contains citation tags, extract the actual URL
        # Pattern 1: URL with nested citation like "https://www[citation:title](https://actual-url)"
        # Try to find nested citation - match even if the outer URL is incomplete
        # Look for [citation:...](https://...) pattern anywhere in the string
        nested_match = re.search(r'\[citation:[^\]]+\]\((https?://[^\)]+)\)', url)
        if nested_match:
            # Use the nested URL (the actual URL inside the citation tag)
            cleaned = nested_match.group(1)
            # Normalize the cleaned URL
            return CitationMiddleware._normalize_url(cleaned)
        
        # Pattern 2: If no complete nested citation found, but URL contains citation tags,
        # try to extract the URL from the citation tag even if the outer structure is broken
        # This handles cases like: "https://www[citation:title](https://real-url" (missing closing paren)
        if '[citation:' in url:
            # Try to find any URL after [citation:...]( that looks like a complete URL
            # Match: [citation:...](https://... up to end of string or next [citation: or )
            partial_match = re.search(r'\[citation:[^\]]+\]\((https?://[^\)\[\]]+)', url)
            if partial_match:
                potential_url = partial_match.group(1)
                # Only use if it looks like a valid URL (has domain)
                if '.' in potential_url and len(potential_url) > 10:
                    return CitationMiddleware._normalize_url(potential_url)
            
            # Fallback: Extract everything before the first [citation: tag
            match = re.search(r'(https?://[^\[]+)', url)
            if match:
                # Only use if it's a reasonable URL (not just "https://www")
                prefix = match.group(1)
                if len(prefix) > 15 or '.' in prefix.split('/')[-1]:
                    return CitationMiddleware._normalize_url(prefix)
        
        # URL is clean, just normalize it
        return CitationMiddleware._normalize_url(url)
    
    @staticmethod
    def _strip_spaces_in_url_scheme_path(url: str) -> str:
        """Remove literal spaces in scheme/host/path (before ? or #).

        LLMs often break hostnames like ``eet-china`` into ``eet - china``, which
        makes URLs invalid and breaks Markdown ``](...)`` parsing. Query/fragment
        after ``?`` / ``#`` is left unchanged so ``?q=hello world`` can still be
        percent-encoded later.
        """
        import re
        s = url.strip()
        low = s.lower()
        if not (low.startswith("http://") or low.startswith("https://")):
            return s
        q = s.find("?")
        h = s.find("#")
        cut = len(s)
        if q != -1:
            cut = min(cut, q)
        if h != -1:
            cut = min(cut, h)
        # LLM / copy-paste may use NBSP or other Unicode space; ``isspace()`` catches them.
        head = s[:cut]
        # First, normalize patterns like " - " to "-" before removing all spaces
        # This handles LLM-generated URLs like "machine - learning" → "machine-learning"
        head = re.sub(r'\s*-\s+', '-', head)  # " -  " or " - " → "-"
        head = re.sub(r'\s+', '', head)  # Remove any remaining spaces
        head = re.sub(r'-+', '-', head)  # Collapse multiple consecutive hyphens
        return head + s[cut:]

    @staticmethod
    def _sanitize_youtube_video_id(raw: str) -> str:
        """Trim trailing junk after an 11-character YouTube ``v`` id (Markdown/copy often adds ``_``)."""
        s = (raw or "").strip()
        if not s:
            return s
        s = re.sub(r"[_)\]]+$", "", s)
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
            return s
        m = re.match(r"^([A-Za-z0-9_-]{11})", s)
        return m.group(1) if m else s

    @staticmethod
    def _unwrap_facebook_redirect_url(url: str) -> str | None:
        """Resolve Facebook link-shim URLs (``l.facebook.com/l.php?u=…``) to the real target.

        Search indexes often return the wrapper; user-facing citations use ``www.facebook.com``.
        Without unwrapping, tool-grounded URLs never match inline citations.
        """
        try:
            from urllib.parse import parse_qsl, unquote, urlparse

            p = urlparse(url.strip())
            host = (p.hostname or "").lower()
            if host not in ("l.facebook.com", "lm.facebook.com"):
                return None
            inner: str | None = None
            for k, v in parse_qsl(p.query or "", keep_blank_values=True):
                if k != "u" or not v:
                    continue
                cand = (v or "").strip()
                if cand.startswith(("http://", "https://")):
                    inner = cand
                    break
                try:
                    once = unquote(cand).strip()
                except Exception:
                    once = cand
                if once.startswith(("http://", "https://")):
                    inner = once
                    break
            if not inner:
                return None
            # Rare double-encoding from copy-paste chains
            for _ in range(2):
                try:
                    nxt = unquote(inner).strip()
                except Exception:
                    break
                if nxt == inner or not (nxt.startswith("http://") or nxt.startswith("https://")):
                    break
                inner = nxt
            return inner if inner.startswith(("http://", "https://")) else None
        except Exception:
            return None

    @staticmethod
    def _canonicalize_facebook_netloc_for_citations(netloc: str) -> str:
        """Unify apex/mobile Facebook hosts so the same post dedupes across SERP vs browser URLs."""
        h = (netloc or "").lower()
        if h in (
            "business.facebook.com",
            "developers.facebook.com",
            "studio.facebook.com",
            "connect.facebook.com",
            "graph.facebook.com",
            "l.facebook.com",
            "lm.facebook.com",
        ):
            return h
        if h in (
            "facebook.com",
            "m.facebook.com",
            "mobile.facebook.com",
            "web.facebook.com",
            "touch.facebook.com",
            "mbasic.facebook.com",
        ):
            return "www.facebook.com"
        return h

    @staticmethod
    def _is_facebook_citation_host(netloc: str) -> bool:
        h = (netloc or "").lower()
        return h == "facebook.com" or h.endswith(".facebook.com")

    @staticmethod
    def _facebook_domain_only_site_label(site_or_org: str) -> bool:
        """True when *site_or_org* is only a Facebook registrable host, not a real org name."""
        s = (site_or_org or "").strip().lower().removeprefix("www.")
        return s in ("facebook.com", "m.facebook.com", "l.facebook.com", "lm.facebook.com", "fb.com", "facebook")

    @staticmethod
    def _facebook_group_org_from_url(url: str) -> str:
        """Readable organization segment for IEEE ``Title. Org. Year`` (not bare ``facebook.com``)."""
        if "facebook.com" not in (url or "").lower():
            return ""
        try:
            from urllib.parse import unquote, urlparse

            path = unquote(urlparse((url or "").strip()).path or "")
        except Exception:
            return "Facebook"
        m = re.match(r"(?i)^/groups/([^/]+)/posts/\d+", path)
        if m:
            seg = (m.group(1) or "").strip()
            if seg.isdigit():
                return "Facebook Group"
            return f"Facebook Group · {seg}"
        return "Facebook"

    @staticmethod
    def _facebook_ieee_organization(url: str, author_org: str, site: str) -> str:
        """Pick IEEE organization field for facebook.com citations."""
        if "facebook.com" not in (url or "").lower():
            return (author_org or site or "").strip()
        ao = (author_org or "").strip()
        st = (site or "").strip()
        if ao and not CitationMiddleware._facebook_domain_only_site_label(ao):
            if "," in ao:
                left, right = [x.strip() for x in ao.split(",", 1)]
                if CitationMiddleware._facebook_domain_only_site_label(right):
                    derived = CitationMiddleware._facebook_group_org_from_url(url)
                    return f"{left}, {derived}" if left else derived
            return ao
        if st and not CitationMiddleware._facebook_domain_only_site_label(st):
            return st
        return CitationMiddleware._facebook_group_org_from_url(url)

    @staticmethod
    def _enrich_facebook_citation_dates(cit: dict, *, source_url: str, snippet: str) -> None:
        """Fill missing calendar date and/or year from FB JSON in raw_content / snippet."""
        if "facebook.com" not in (source_url or "").lower():
            return
        pub = (cit.get("published_date") or "").strip()
        yr = (cit.get("year") or "").strip()
        if pub and yr:
            return
        if pub and not yr:
            y2 = CitationMiddleware._year_from_published_or_snippet(
                pub,
                snippet,
                "",
                source_url,
                allow_snippet_year=not CitationMiddleware._is_reddit_url(source_url),
            )
            if y2:
                cit["year"] = y2
            return

        raw_blob = ((cit.get("raw_content") or "") + "\n" + (snippet or "")).strip()
        if not raw_blob:
            return
        try:
            from deerflow.utils.publication_date import infer_publication_calendar_date

            got = infer_publication_calendar_date(raw_blob[:1_000_000], source_url=source_url)
        except Exception:
            got = ""
        got = (got or "").strip()
        if not got:
            return
        cit["published_date"] = got
        cit["year"] = CitationMiddleware._year_from_published_or_snippet(
            got,
            snippet,
            "",
            source_url,
            allow_snippet_year=not CitationMiddleware._is_reddit_url(source_url),
        )

    @staticmethod
    def _enrich_zhihu_citation_dates(cit: dict, *, source_url: str, snippet: str) -> None:
        """Fill missing calendar date and/or year from Zhihu JSON/markdown in raw_content / snippet.

        The bibliography builder only invoked Facebook-specific enrichment before; Zhihu rows
        otherwise kept empty ``published_date`` when the inline phase had no full date.
        """
        try:
            from urllib.parse import urlparse

            host = (urlparse((source_url or "").strip()).hostname or "").lower()
        except Exception:
            host = ""
        if host != "zhihu.com" and not host.endswith(".zhihu.com"):
            return
        pub = (cit.get("published_date") or "").strip()
        yr = (cit.get("year") or "").strip()
        if pub and yr:
            return
        if pub and not yr:
            y2 = CitationMiddleware._year_from_published_or_snippet(
                pub,
                snippet,
                "",
                source_url,
                allow_snippet_year=not CitationMiddleware._is_reddit_url(source_url),
            )
            if y2:
                cit["year"] = y2
            return

        raw_blob = ((cit.get("raw_content") or "") + "\n" + (snippet or "")).strip()
        if not raw_blob:
            return
        try:
            from deerflow.utils.publication_date import infer_publication_calendar_date

            got = infer_publication_calendar_date(raw_blob[:1_000_000], source_url=source_url)
        except Exception:
            got = ""
        got = (got or "").strip()
        if not got:
            return
        cit["published_date"] = got
        cit["year"] = CitationMiddleware._year_from_published_or_snippet(
            got,
            snippet,
            "",
            source_url,
            allow_snippet_year=not CitationMiddleware._is_reddit_url(source_url),
        )

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL by encoding spaces and special characters.
        
        This function ensures consistent URL representation for deduplication:
        - Encodes spaces and special characters (but doesn't double-encode)
        - Removes trailing slashes from path (except root)
        - Normalizes scheme and netloc to lowercase
        
        Note: If URL is already encoded, this function will preserve the encoding
        by parsing and reconstructing it properly.
        """
        from urllib.parse import parse_qsl, quote, unquote, urlencode, urlparse, urlunparse
        try:
            url = CitationMiddleware._strip_spaces_in_url_scheme_path(url)
            for _ in range(4):
                fb_inner = CitationMiddleware._unwrap_facebook_redirect_url(url)
                if not fb_inner:
                    break
                url = CitationMiddleware._strip_spaces_in_url_scheme_path(fb_inner)
            parsed = urlparse(url.strip())
            # Normalize scheme and netloc to lowercase for consistent comparison
            normalized_scheme = parsed.scheme.lower() if parsed.scheme else ''
            normalized_netloc = parsed.netloc.lower() if parsed.netloc else ''
            if normalized_netloc and CitationMiddleware._is_facebook_citation_host(normalized_netloc):
                normalized_netloc = CitationMiddleware._canonicalize_facebook_netloc_for_citations(normalized_netloc)
            
            # Remove trailing slash from path (except root path '/')
            normalized_path = parsed.path.rstrip('/') if parsed.path != '/' else '/'
            
            try:
                decoded_path = unquote(normalized_path)
            except Exception:
                decoded_path = normalized_path

            if normalized_netloc.endswith("youtu.be"):
                segs = [s for s in decoded_path.split("/") if s]
                if len(segs) == 1:
                    segs[0] = CitationMiddleware._sanitize_youtube_video_id(segs[0])
                    decoded_path = "/" + segs[0]

            if "prezi.com" in normalized_netloc and decoded_path:
                psegs = [p for p in decoded_path.split("/") if p]
                if psegs and len(psegs[-1]) > 6 and psegs[-1].endswith("_date"):
                    psegs[-1] = psegs[-1][:-5]
                    decoded_path = "/" + "/".join(psegs)

            try:
                normalized_path = quote(decoded_path, safe='/')
            except Exception:
                normalized_path = quote(normalized_path, safe='/')
            
            query_preencoded = False
            try:
                decoded_query = unquote(parsed.query)
            except Exception:
                decoded_query = parsed.query

            yt_hosts = ("youtube.com", "youtube-nocookie.com")
            yt_watch = any(
                normalized_netloc == h or normalized_netloc.endswith("." + h) for h in yt_hosts
            ) and (decoded_path.rstrip("/").endswith("/watch") or decoded_path.rstrip("/") == "/watch")
            if yt_watch and decoded_query:
                pairs = parse_qsl(decoded_query, keep_blank_values=True)
                fixed_pairs = [
                    (k, CitationMiddleware._sanitize_youtube_video_id(v) if k == "v" else v)
                    for k, v in pairs
                ]
                decoded_query = urlencode(fixed_pairs, doseq=True)
                query_preencoded = True

            if query_preencoded:
                normalized_query = decoded_query
            else:
                try:
                    normalized_query = quote(decoded_query, safe='=&')
                except Exception:
                    normalized_query = quote(parsed.query, safe='=&')
            
            try:
                decoded_fragment = unquote(parsed.fragment)
                normalized_fragment = quote(decoded_fragment, safe='')
            except Exception:
                normalized_fragment = quote(parsed.fragment, safe='')
            
            return urlunparse((
                normalized_scheme,
                normalized_netloc,
                normalized_path,
                parsed.params,
                normalized_query,
                normalized_fragment
            ))
        except Exception:
            # Fallback: strip broken spaces in host/path, then encode remaining spaces
            u = CitationMiddleware._strip_spaces_in_url_scheme_path(url)
            return u.strip().replace(" ", "%20")
    
    @staticmethod
    def _fix_nested_citations(text: str) -> str:
        """Fix nested or malformed citation tags.
        
        Strategy: Manually parse citation structures to extract all nested citations,
        then reconstruct them as separate, properly formatted citations.
        
        Fixes cases like:
        [citation:title1](https://www[citation:title2](https://url2).com)
        to:
        [citation:title2](https://url2) [citation:title1](https://www.com)
        """
        import re
        
        def extract_citations_manual(s: str):
            """Manually extract all citation tags, handling nested structures."""
            citations = []
            i = 0
            while i < len(s):
                # Look for [citation:
                if s[i:i+10] == '[citation:':
                    # Find the title (until ])
                    title_start = i + 10
                    title_end = s.find(']', title_start)
                    if title_end == -1:
                        i += 1
                        continue
                    title = s[title_start:title_end]
                    
                    # Find the opening paren for URL
                    url_start = title_end + 1
                    if url_start >= len(s) or s[url_start] != '(':
                        i += 1
                        continue
                    
                    # Find the URL (handle nested citations)
                    url_start += 1
                    if url_start >= len(s):
                        i += 1
                        continue
                    
                    # Check if URL starts with http
                    if s[url_start:url_start+7] != 'http://' and s[url_start:url_start+8] != 'https://':
                        i += 1
                        continue
                    
                    # Find the end of URL (matching closing paren, but handle nested citations)
                    paren_count = 1
                    url_end = url_start
                    found_nested = False
                    
                    for j in range(url_start, len(s)):
                        if s[j] == '(':
                            paren_count += 1
                        elif s[j] == ')':
                            paren_count -= 1
                            if paren_count == 0:
                                url_end = j
                                break
                        # Check for nested citation in URL
                        if s[j:j+10] == '[citation:':
                            found_nested = True
                    
                    if url_end > url_start:
                        url = s[url_start:url_end]
                        # If URL contains nested citations, extract them recursively
                        if found_nested and '[citation:' in url:
                            nested = extract_citations_manual(url)
                            citations.extend(nested)
                        else:
                            # Valid complete citation
                            if '.' in url:
                                citations.append((title, url))
                    
                    i = url_end + 1
                else:
                    i += 1
            
            return citations
        
        # Find all malformed nested citation sections
        nested_pattern = re.compile(r'\[citation:([^\]]+)\]\(https?://[^\)]*?\[citation:')
        
        result = text
        max_iterations = 10
        
        for iteration in range(max_iterations):
            matches = list(nested_pattern.finditer(result))
            if not matches:
                break
            
            # Process from end to start to maintain positions
            for match in reversed(matches):
                start = match.start()
                
                # Find the end of this malformed citation
                paren_count = 0
                found_open = False
                end = start
                for i, char in enumerate(result[start:], start):
                    if char == '(':
                        paren_count += 1
                        found_open = True
                    elif char == ')':
                        paren_count -= 1
                        if found_open and paren_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    malformed_section = result[start:end]
                    
                    # Extract all citations (including nested) using manual parsing
                    all_citations = extract_citations_manual(malformed_section)
                    
                    # Also try to extract the outer citation's title and reconstruct its URL
                    # Look for the first citation in the malformed section (the one that contains nested citations)
                    outer_match = re.search(r'\[citation:([^\]]+)\]\(https?://([^\)\[]*?)\[citation:', malformed_section)
                    if outer_match:
                        outer_title = outer_match.group(1)
                        outer_url_start = outer_match.group(2)  # e.g., "www"
                        
                        # Find the URL suffix after all nested citations
                        # Look for patterns like ".autohome.com.cn/ask/18330204.html" after the last citation
                        # Remove all nested citations to find remaining URL parts
                        temp_section = malformed_section
                        for title, url in all_citations:
                            temp_section = temp_section.replace(f"[citation:{title}]({url})", "")
                        
                        # Extract domain suffix (e.g., ".autohome.com.cn/ask/18330204.html")
                        # Look for pattern: .domain.com/path
                        url_suffix_match = re.search(r'\.([a-z0-9.-]+/[^\)\[]*?)(?:\)|$)', temp_section)
                        if url_suffix_match:
                            url_suffix = url_suffix_match.group(1)
                            # Reconstruct outer URL: https://www + .domain.com/path
                            outer_url = f'https://{outer_url_start}.{url_suffix}'
                            if '.' in outer_url and outer_url.count('.') >= 2:  # At least www.domain.com
                                # Check if this URL is not already in all_citations
                                if not any(url in outer_url or outer_url in url for _, url in all_citations):
                                    all_citations.insert(0, (outer_title, outer_url))
                    
                    if all_citations:
                        # Reconstruct as separate citations
                        valid_citations = []
                        seen_urls = set()
                        for title, url in all_citations:
                            normalized_url = CitationMiddleware._normalize_url(url)
                            # Deduplicate by URL
                            if normalized_url not in seen_urls and '.' in normalized_url:
                                seen_urls.add(normalized_url)
                                valid_citations.append(f"[citation:{title}]({normalized_url})")
                        
                        if valid_citations:
                            replacement = ' '.join(valid_citations)
                            result = result[:start] + replacement + result[end:]
                        else:
                            result = result[:start] + result[end:]
                    else:
                        result = result[:start] + result[end:]
        
        return result
    
    @staticmethod
    def _fix_reference_section_nested_citations(text: str) -> str:
        """Fix nested citations in reference section URLs.
        
        Handles cases like:
        Available: https://www[citation:title](https://url2).com
        to:
        Available: [链接](https://url2) [链接](https://www.com)
        """
        import re
        
        # Pattern to match "Available: https://... [citation:...](...)..."
        # This matches nested citations in reference section URLs
        ref_nested_pattern = re.compile(
            r'(Available:\s*https?://[^\)\[]*?)(\[citation:([^\]]+)\]\(https?://([^\)]+)\))([^\)]*?)(\.\s*Accessed:)'
        )
        
        def fix_ref_nested(match):
            available_prefix = match.group(1)  # "Available: https://www"
            nested_url = match.group(4)
            url_suffix = match.group(5)  # ".com" or ".autohome.com.cn/ask/..."
            accessed_suffix = match.group(6)  # ". Accessed:"
            
            # Extract the nested citation URL (ensure it has https://)
            if not nested_url.startswith('http'):
                nested_url = 'https://' + nested_url
            nested_normalized = CitationMiddleware._normalize_url(nested_url)
            
            # Try to reconstruct outer URL if possible
            outer_url_start = available_prefix.find('https://')
            if outer_url_start != -1:
                outer_url_part = available_prefix[outer_url_start + 8:]  # After "https://"
                if url_suffix and '.' in url_suffix:
                    # Reconstruct: https:// + outer_url_part + url_suffix
                    outer_url = f'https://{outer_url_part}{url_suffix}'
                    outer_normalized = CitationMiddleware._normalize_url(outer_url)
                    if '.' in outer_normalized and outer_normalized.count('.') >= 2:
                        return f"Available: [链接]({nested_normalized}) [链接]({outer_normalized}){accessed_suffix}"
            
            # Fallback: just use the nested citation (ensure it has https://)
            if not nested_normalized.startswith('http'):
                nested_normalized = 'https://' + nested_normalized
            return f"Available: [链接]({nested_normalized}){accessed_suffix}"
        
        # Apply fix iteratively
        max_iterations = 10
        for _ in range(max_iterations):
            new_text = ref_nested_pattern.sub(fix_ref_nested, text)
            if new_text == text:
                break
            text = new_text
        
        return text
    
    @staticmethod
    def _normalize_citation_urls(text: str) -> str:
        """Normalize URLs in all citation tags in the text.
        
        Fixes URLs with spaces like:
        [citation:title](https://github.com/path with spaces/file.md)
        to:
        [citation:title](https://github.com/path%20with%20spaces/file.md)
        
        Also fixes nested citations in both body text and reference section.
        """
        # First, fix nested citations in body text (inline citations)
        text = CitationMiddleware._fix_nested_citations(text)
        
        # Then, fix nested citations in reference section
        text = CitationMiddleware._fix_reference_section_nested_citations(text)

        # Balanced-paren rewrite (regex ``[^)]+`` truncates at first ``)`` in the path).
        text = CitationMiddleware._canonicalize_urls_in_citation_markdown(text)
        return text

    @staticmethod
    def _ensure_section_citation_coverage(text: str, citations_with_snippets: list[dict]) -> str:
        """Ensure each major markdown section has at least one inline citation.

        This reduces the "references at bottom but detached from body content" issue.
        Strategy:
        - Split off existing reference section (if any) and only patch main body.
        - For each `##` / `###` section block, if it has no `[citation:...]`, inject
          citation tag(s) into prose lines (first line; if the section is long, a second line).
        - If a section has only one citation but many prose lines, inject one more tag
          from a different source to increase density.
        """
        if not text or not citations_with_snippets:
            return text

        ref_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if ref_match:
            body = text[: ref_match.start()]
            ref_part = text[ref_match.start() :]
        else:
            body = text
            ref_part = ""

        # Build stable, deduplicated citation pool.
        # We keep both:
        # - `citation_tag`: the exact inline marker we inject
        # - `snippet`: used for similarity scoring to avoid "random citation" injection
        unique_citations: list[dict] = []
        seen_norm_tags = set()
        for c in citations_with_snippets:
            tag = (c.get("citation_tag") or "").strip()
            if not tag:
                continue
            norm_tag = CitationMiddleware._normalize_citation_urls(tag)
            if not norm_tag or norm_tag in seen_norm_tags:
                continue
            seen_norm_tags.add(norm_tag)
            unique_citations.append(
                {
                    "citation_tag": tag,
                    "snippet": (c.get("snippet") or "").strip(),
                }
            )

        tags: list[str] = [c["citation_tag"] for c in unique_citations]
        if not tags:
            return text

        lines = body.splitlines()

        def _is_section_heading_line(ln: str) -> bool:
            s = ln.strip()
            if not s or s.startswith("|") or s.startswith("```"):
                return False
            if re.match(r"^\s*#{2,3}\s+\S+", ln):
                return True
            # Chinese numbered section: 一、xxx / 1. xxx
            if re.match(r"^([一二三四五六七八九十百千零〇两]+)[、\.．]\s*\S", s) or re.match(
                r"^\d+[、\.．]\s*\S", s
            ):
                return True
            # Plain-text title line: "背景介绍：" / "Section:" (no sentence-final punctuation before colon)
            if len(s) <= 100 and re.match(r"^[^。！？.!?]{2,80}[：:]\s*$", s):
                return True
            return False

        heading_idx = [i for i, ln in enumerate(lines) if _is_section_heading_line(ln)]
        no_heading = not heading_idx
        if no_heading:
            # No explicit headings: treat the whole body as one pseudo-section
            # so paragraph-level citation coverage still applies.
            heading_idx = [0]

        def is_prose_line(ln: str) -> bool:
            s = ln.strip()
            if not s:
                return False
            if s.startswith("#") or s.startswith("|") or s.startswith("```"):
                return False
            # Reports often use bullet / numbered lists for the main body; those lines
            # must be eligible for citation injection (previously skipped entirely).
            if re.match(r"^[-*+]\s", s):
                rest = re.sub(r"^[-*+]\s+", "", s).strip()
                return len(rest) >= 8
            if re.match(r"^\d+\.\s", s):
                rest = re.sub(r"^\d+\.\s+", "", s).strip()
                return len(rest) >= 8
            return True

        tag_cursor = 0

        def _inject_at_line(idx: int, tag: str) -> None:
            ln = lines[idx].rstrip()
            # Allow multiple citations in the same line; only skip if this tag
            # already exists.
            if tag in ln:
                return
            if re.search(r"[。！？.!?]\s*$", ln):
                lines[idx] = re.sub(r"\s*([。！？.!?])\s*$", f"{tag}\\1", ln)
            else:
                lines[idx] = f"{ln}{tag}"

        for pos, start in enumerate(heading_idx):
            end = heading_idx[pos + 1] if pos + 1 < len(heading_idx) else len(lines)
            block = "\n".join(lines[start:end])
            prose_start = start if no_heading else start + 1
            prose_indices = [
                i for i in range(prose_start, end) if is_prose_line(lines[i])
            ]
            if not prose_indices:
                continue

            citation_count = block.count("[citation:")
            # Section has no citations: inject 1–3 tags spaced through the section
            if citation_count == 0:
                inject_idx = prose_indices[0]
                tag = tags[tag_cursor % len(tags)]
                tag_cursor += 1
                _inject_at_line(inject_idx, tag)
                # Keep spacing roughly uniform through the section.
                if len(prose_indices) >= 3:
                    tag2 = tags[tag_cursor % len(tags)]
                    tag_cursor += 1
                    second_i = prose_indices[min(2, len(prose_indices) - 1)]
                    if second_i != inject_idx:
                        _inject_at_line(second_i, tag2)
                if len(prose_indices) >= 8:
                    tag3 = tags[tag_cursor % len(tags)]
                    tag_cursor += 1
                    third_i = prose_indices[min(6, len(prose_indices) - 1)]
                    if third_i not in (inject_idx, second_i if len(prose_indices) >= 3 else -1):
                        _inject_at_line(third_i, tag3)
                # Extra density for long sections: add a 4th citation.
                if len(prose_indices) >= 12:
                    tag4 = tags[tag_cursor % len(tags)]
                    tag_cursor += 1
                    fourth_i = prose_indices[min(10, len(prose_indices) - 1)]
                    if fourth_i not in (
                        inject_idx,
                        second_i if len(prose_indices) >= 3 else -1,
                        third_i if len(prose_indices) >= 8 else -1,
                    ):
                        _inject_at_line(fourth_i, tag4)

            # Section has few citations but many prose lines: add more from other sources
            if citation_count == 1 and len(prose_indices) >= 4:
                for i in prose_indices:
                    if "[citation:" not in lines[i]:
                        tag = tags[tag_cursor % len(tags)]
                        tag_cursor += 1
                        _inject_at_line(i, tag)
                        break
                # If the section is long, insert one more citation.
                if len(prose_indices) >= 8:
                    for i in prose_indices:
                        if "[citation:" in lines[i]:
                            continue
                        tag = tags[tag_cursor % len(tags)]
                        tag_cursor += 1
                        _inject_at_line(i, tag)
                        break
            if citation_count == 2 and len(prose_indices) >= 10:
                injected = 0
                for i in prose_indices:
                    if "[citation:" in lines[i]:
                        continue
                    tag = tags[tag_cursor % len(tags)]
                    tag_cursor += 1
                    _inject_at_line(i, tag)
                    injected += 1
                    if injected >= 2:
                        break

        patched_body = "\n".join(lines)
        return patched_body + ref_part
    
    @staticmethod
    def _extract_inline_citations(text: str) -> list[tuple[str, str, int]]:
        """Return list of (title, url, ref_number) found inline in *text*.
        
        Extracts ALL citation tags from the text, preserving order of appearance.
        Supports both old format [citation:title](url) and new format [1] (numeric references).
        
        CRITICAL: Handles nested citations by preferring inner citations (URLs without citation tags).
        Returns:
            List of (title, url, ref_number) tuples. ref_number is -1 for old format citations.
        """
        result = []

        valid_matches_old: list[tuple[str, str]] = []
        for title, url in CitationMiddleware._iter_citation_markdown_links(text):
            if "[citation:" in url:
                try:
                    clean_url = CitationMiddleware._clean_url_for_reference(url)
                    if clean_url and clean_url != url and clean_url.startswith("http"):
                        valid_matches_old.append((title.strip(), clean_url))
                        continue
                except Exception as e:
                    logger.debug(
                        "CitationMiddleware: Failed to clean nested URL '%s...': %s",
                        url[:60],
                        e,
                    )
                continue
            valid_matches_old.append((title.strip(), url))

        # Normalize URLs in old format citations
        for title, url in valid_matches_old:
            try:
                normalized_url = CitationMiddleware._normalize_url(url)
                result.append((title.strip(), normalized_url, -1))  # -1 indicates old format
            except Exception as e:
                logger.warning(f"CitationMiddleware: Failed to normalize URL '{url}': {e}")
                result.append((title.strip(), url, -1))
        
        # Pattern 2: Match new format [1], [2], etc. (numeric references)
        # These will be mapped to URLs when building the reference section
        pattern_new = re.compile(r'\[(\d+)\]')
        numeric_matches = pattern_new.findall(text)
        for ref_num_str in numeric_matches:
            ref_num = int(ref_num_str)
            # We'll map this to URL later when we have the reference list
            result.append(("", "", ref_num))
        
        return result

    @staticmethod
    def _has_reference_section(text: str) -> bool:
        """Return True if the text already contains a reference/bibliography section."""
        return bool(
            re.search(
                CitationMiddleware._reference_heading_regex(),
                text,
                re.IGNORECASE | re.MULTILINE,
            )
        )

    @staticmethod
    def _split_into_sentences(text: str) -> list[tuple[int, int, str]]:
        """Split text into sentences, returning list of (start, end, sentence).
        
        Returns list of (start_pos, end_pos, sentence_text) tuples.
        """
        # Remove reference section if present
        ref_section_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if ref_section_match:
            text = text[:ref_section_match.start()]
        
        sentences = []
        # Split by sentence endings (Chinese and English)
        pattern = r"([.!?。！？\n]+)"
        parts = re.split(pattern, text)
        
        current_pos = 0
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if re.match(pattern, part):
                # This is a delimiter
                if current_sentence.strip():
                    start = current_pos - len(current_sentence)
                    sentences.append((start, current_pos, current_sentence.strip()))
                    current_sentence = ""
                current_pos += len(part)
            else:
                current_sentence += part
                current_pos += len(part)
        
        # Add last sentence if exists
        if current_sentence.strip():
            start = current_pos - len(current_sentence)
            sentences.append((start, current_pos, current_sentence.strip()))
        
        return sentences

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts.

        NOTE: overlap-based similarity (Jaccard / coverage / phrase overlap) has been
        intentionally disabled. This method currently returns a neutral score so
        higher-level citation injection falls back to other heuristics.
        """
        # Disabled: overlap-based similarity (Jaccard / coverage / phrase overlap).
        # Citation insertion now relies on other heuristics:
        # - sentence-level: `keyword_overlap` + `title_match` (see `_insert_inline_citations`)
        # - section-level: deterministic fallback when scoring doesn't help
        _ = (text1, text2)
        return 0.0

    @staticmethod
    def _convert_citations_to_numeric(
        text: str, url_to_ref_map: dict[str, int] | None
    ) -> str:
        """Convert all [citation:title](url) format citations to numeric [1] format.
        
        Args:
            text: Text containing citations in old format
            url_to_ref_map:
                - When provided: mapping from normalized/clean URL to reference number (legacy behavior).
                - When None: assign reference numbers by citation occurrence order (strict per-occurrence mode).
            
        Returns:
            Text with citations converted to numeric format
        """
        spans = list(CitationMiddleware._iter_citation_markdown_link_spans(text))
        if not spans:
            return text

        # Without a url_to_ref_map, per-occurrence [n] would not match any reference list.
        if url_to_ref_map is None:
            out: list[str] = []
            last = 0
            for _title, url, start, end in spans:
                out.append(text[last:start])
                out.append("")
                last = end
            out.append(text[last:])
            return "".join(out)

        out: list[str] = []
        last = 0
        for _title, url, start, end in spans:
            out.append(text[last:start])
            ref_num = -1
            try:
                clean_url = CitationMiddleware._clean_url_for_reference(url)
                normalized_url = CitationMiddleware._normalize_url(clean_url)
                ref_num = url_to_ref_map.get(normalized_url, -1)
            except Exception:
                pass
            if ref_num <= 0:
                try:
                    normalized_url = CitationMiddleware._normalize_url(url)
                    ref_num = url_to_ref_map.get(normalized_url, -1)
                except Exception:
                    pass
            if ref_num <= 0:
                ref_num = url_to_ref_map.get(url, -1)
            if ref_num > 0:
                out.append(f"[{ref_num}]")
            else:
                logger.warning(
                    "CitationMiddleware: Could not find reference number for URL '%s...' in map — stripping tag",
                    url[:60],
                )
                out.append("")
            last = end
        out.append(text[last:])
        return "".join(out)

    @staticmethod
    def _normalize_numeric_citations(text: str) -> str:
        """Normalize numeric citations in body text.

        Fixes two common post-processing artifacts:
        1) Duplicate adjacent refs, e.g. ``[6][6]`` -> ``[6]``
        2) Broken URL tails appended after numeric refs, e.g.
           ``[2][3].finance.yahoo.com/news/...`` -> ``[2][3]``
        """
        if not text:
            return text
        
        # 0) Normalize broken numeric references like `3]` → `[3]` so that
        #    downstream patterns (e.g. `[3].htm)[4]`) can be handled uniformly.
        #    CRITICAL: require the digit run NOT be preceded by another digit,
        #    otherwise `0]` inside valid `[10]` is wrongly turned into `[0]`,
        #    producing garbage like `[1[0]` in the reference list.
        text = re.sub(r"(?<!\[)(?<!\d)(\d+)]", r"[\1]", text)
        
        # 1) Collapse adjacent duplicate numeric references.
        text = re.sub(r"(\[(\d+)\])(?:\s*\[\2\])+", r"\1", text)

        # 1.5) Normalize citation clusters to "tight" form:
        #   - "[1] [2]"  -> "[1][2]"
        #   - "[1], [2]" / "[1]、[2]" -> "[1][2]"
        text = re.sub(r"(\[\d+\])\s+(\[\d+\])", r"\1\2", text)
        text = re.sub(r"(\[\d+\])\s*[，、,;]\s*(\[\d+\])", r"\1\2", text)

        # 2) Remove URL-tail garbage immediately following numeric reference clusters.
        #    Keep the citation cluster and terminal punctuation.
        text = re.sub(
            r"((?:\[\d+\]\s*)+)\.(?:[a-zA-Z0-9-]+\.)+(?:com|net|io|org|cn|edu|gov)/[^\s\)）]+[)）]?([。．\.]?)",
            r"\1\2",
            text,
        )

        # 3) Fix patterns where HTML 扩展名和右括号夹在两个数字引用之间，例如：
        #    `[3].htm)[4]` / `[3].html)[4]` / `[3].htm)[4][5]`
        #    这通常来自「链接 markdown 结束 + 引用编号」被错误打散。
        #    我们在这种情况下，直接移除中间的 `.htm)` / `.html)` 片段，合并为连续引用。
        def _fix_html_suffix_between_refs(match: "re.Match[str]") -> str:
            left_ref, middle, right_refs = match.group(1), match.group(2), match.group(3)
            middle_stripped = middle.strip()
            # 兼容更多变体，例如：
            # `[3].htm)[4]` / `[3].html))[4]` / `[3].shtml])[4]`
            # 这些通常是「链接结尾 + 引用编号」被打散的产物，中间只应包含
            # `.htm` / `.html` / `.shtml` 以及若干右括号/右方括号。
            if re.fullmatch(r"\]?\.(?:s?html?)[\]\)]*", middle_stripped, flags=re.IGNORECASE):
                return f"{left_ref}{right_refs}"
            return match.group(0)

        text = re.sub(r"(\[\d+\])([^\[]+?)((?:\[\d+\])+\s*)", _fix_html_suffix_between_refs, text)

        # 4) 移除在数字引用后的 HTML 扩展名残片，例如：
        #    `[5].shtml)。` / `[5].html)。`
        text = re.sub(
            r"(\[\d+\])\s*\.(?:s?html?)[)）]?[。．\.]?",
            r"\1",
            text,
        )

        # 5) 移除在数字引用后的孤立查询串，例如：
        #    `4]?id=1615779)。` 这类通常是 URL 被清理后残留的 `?id=...` 片段。
        text = re.sub(
            r"(\[\d+\])\]?\s*\?id=\d+[)）]?[。．\.]?",
            r"\1",
            text,
        )
        return text

    @staticmethod
    def _strip_inline_urls_from_body(text: str) -> str:
        """Remove explicit URL展示，仅保留正文内容与数字引用。

        目标：链接只在「参考文献」中出现，不在正文中直接展示长 URL。
        策略：
        - 只处理参考文献标题之前的正文部分；
        - 对于普通 Markdown 链接 `[文本](https://...)`（排除 `[citation:...]`），
          将其替换为纯文本 `文本`；
        - 可选：对裸露的 `https://...`/`http://...` 直接删除（正文只保留叙述和 [n]）。
        """
        if not text:
            return text

        ref_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if ref_match:
            body = text[: ref_match.start()]
            ref_part = text[ref_match.start() :]
        else:
            body = text
            ref_part = ""

        # 0) 模型/工具有时会产出残缺 Markdown：``[标题]()`` 或 ``[citation:…]()``（括号内无 URL）。
        #     主流程里的 ``\(([^)]+)\)`` 要求至少一个字符，匹配不到上述形式，会原样泄漏到 UI。
        def _replace_empty_md_link(match: "re.Match[str]") -> str:
            label = match.group(1)
            if label.startswith("citation:"):
                return ""
            if re.fullmatch(r"\d+", label):
                return f"[{label}]"
            return label

        body = re.sub(r"\[([^\]]+)\]\(\s*\)", _replace_empty_md_link, body)

        # 1) 去掉正文中的 Markdown 链接（无论链接目标是否以 http(s) 开头）
        def _replace_md_link(match: "re.Match[str]") -> str:
            label = match.group(1)
            # 对于 citation 链接和普通链接，都不要在正文里再展示 URL；
            # - citation 链接：已经在前面转成了数字引用 + 参考文献，这里直接去掉整段；
            # - 普通链接：正文只保留可读文本 label。
            if label.startswith("citation:"):
                return ""
            # If numeric citations are accidentally wrapped as markdown links,
            # e.g. "[1](https://...)", restore them as plain "[1]" so that body
            # numbering matches the reference list indices.
            if re.fullmatch(r"\d+", label):
                return f"[{label}]"
            return label

        body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_md_link, body)

        # 2) 清理裸露 URL（避免正文出现一整串长链接）
        body = re.sub(r"https?://[^\s\)）]+", "", body)

        # 3) 兜底：清掉任何残留的 citation 结构（如果上面遗漏）
        #    既包括 `[citation:...](...)` 也包括裸露的 `[citation:...]`
        body = re.sub(r"\[citation:[^\]]+\]\([^\)]*\)", "", body)
        body = re.sub(r"\[citation:[^\]]+\]", "", body)
        
        # 4) 最后的兜底：清理任何残留的 `]()` 碎片（空 markdown 链接）
        #    即便经过上面的处理，仍可能有 `]()`、`]( )`、`](  )` 这样的残留
        body = re.sub(r"\]\(\s*\)", "", body)

        return body + ref_part

    @staticmethod
    def _strip_markdown_from_body(text: str) -> str:
        """Remove common Markdown artifacts from the body.

        We keep the (pure-text) reference section intact.
        """
        if not text:
            return text

        ref_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if ref_match:
            body = text[: ref_match.start()]
            ref_part = text[ref_match.start() :]
        else:
            body = text
            ref_part = ""

        # Remove markdown headings (e.g. "## XXX" -> "XXX")
        body = re.sub(r"(?m)^#{2,6}\s+", "", body)

        # Remove unordered list bullets at line start
        body = re.sub(r"(?m)^[ \t]*[-*+]\s+", "", body)

        # Remove horizontal rules
        body = re.sub(r"(?m)^---+\s*$", "", body)

        # Convert bold to plain text
        body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)

        # Clean up excessive blank lines introduced by stripping
        body = re.sub(r"\n{3,}", "\n\n", body)

        return body + ref_part

    @staticmethod
    def _repair_flattened_markdown_tables(text: str) -> str:
        """Repair table rows that were accidentally concatenated on one line."""
        if not text or "|" not in text:
            return text

        fixed_lines: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line
            # Guardrails: only touch obvious malformed table lines.
            if line.lstrip().startswith("|") and (line.count("| |") >= 2 or "| ----" in line):
                line = re.sub(
                    r"\|\s*\|\s*(?=(?:\d+|[-:]{2,}|[A-Za-z\u4e00-\u9fff#]))",
                    "|\n| ",
                    line,
                )
            fixed_lines.extend(line.split("\n"))
        return "\n".join(fixed_lines)

    @staticmethod
    def _parse_reference_entry_line(line: str) -> int | None:
        """If *line* starts a bibliography row ``[n] …``, return *n*; else None.

        Accepts both ``[1] "Title," url`` and tight ``[1]"Title,"`` so entries are not
        misclassified as "tail" prose (which used to leak uncited rows back into output).
        """
        if not line.strip():
            return None
        m = re.match(r"^\s*(?:[-*+]\s*)?\[(\d+)\](.*)$", line)
        if not m:
            return None
        rest = (m.group(2) or "").strip()
        if not rest:
            return None
        return int(m.group(1))

    @staticmethod
    def _rewrite_ref_line_index_prefix(line: str, new_num: int) -> str:
        """Replace the leading ``[k]`` index on a bibliography row with ``[new_num]``."""
        m = re.match(r"^(\s*(?:[-*+]\s*)?)\[(\d+)\]", line)
        if not m:
            return line
        return f"{m.group(1)}[{new_num}]" + line[m.end() :]

    @staticmethod
    def _prune_reference_section_to_body_citations(text: str) -> str:
        """Keep only reference entries cited in the body and renumber them.

        Rules:
        - Source of truth is numeric citations in body text (before reference heading),
          e.g. [2], [5] → 说明正文只用了 2 号和 5 号。
        - 参考文献区只保留这些被正文实际引用到的条目（且参考文献里确实存在对应行）；
        - 正文里出现在参考文献中找不到的 [n] 会被移除，避免「正文有号、列表无条」；
        - 同时对正文与参考文献进行「紧凑重排」：按正文首次出现顺序
          把引用编号重新映射为 [1]...[N]（例如正文只出现 [2][5]，则
          它们会被重排为 [1][2]，不会出现「[3] 被删除后中间空洞」的问题）。
        - 解析失败的文献行不再通过 tail 拼回正文（此前会把未引用条目原样泄漏到文末）。
        """
        if not text:
            return text

        ref_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if not ref_match:
            return text

        body = text[: ref_match.start()]
        ref_part = text[ref_match.start() :]

        # Log the split for debugging misdetections
        body_len = len(body.strip())
        ref_len = len(ref_part.strip())
        split_pos_ratio = ref_match.start() / len(text) if text else 0
        logger.debug(
            "CitationMiddleware._prune: ref_heading at %.1f%% of text (body=%d chars, ref=%d chars)",
            split_pos_ratio * 100, body_len, ref_len
        )

        # 1) 找出正文中实际用到的数字引用，按首次出现顺序去重。
        #    使用 [1]..[999]，避免把年份式括号 [2024]、页码等四位数字误当作引用编号。
        body_ref_nums: list[int] = []
        seen: set[int] = set()
        for n_str in re.findall(r"\[([1-9]\d{0,2})\]", body):
            n = int(n_str)
            if n not in seen:
                seen.add(n)
                body_ref_nums.append(n)

        if not body_ref_nums:
            # 正文里根本没有数字引用：通常表示模型只在文末贴了参考文献、正文未标号。
            #
            # 防御机制：检查是否可能发生了误判（参考文献标题被错误识别）
            # 如果body太短而ref很长，通常表示分割点选错了位置
            body_len = len(body.strip())
            ref_len = len(ref_part.strip())
            
            # 情形1：body完全为空 → 肯定是误判，保留原文
            if not body.strip():
                logger.warning(
                    "CitationMiddleware._prune: Body is empty after reference split; "
                    "likely misdetected reference heading. Keeping original text."
                )
                return text
            
            # 情形2：body极短而ref很长 → 可能是误判的强烈信号
            if body_len > 0 and ref_len > body_len * 3:
                # 额外验证：检查ref_part是否真的包含有效的参考文献条目
                has_valid_ref_entries = bool(re.search(r"\[\d+\]", ref_part))
                if not has_valid_ref_entries:
                    logger.warning(
                        "CitationMiddleware._prune: Body is very short (%.0f chars) "
                        "but reference part is very long (%.0f chars) with no [n] citations; "
                        "likely reference heading misdetection. Keeping original text.",
                        body_len, ref_len
                    )
                    return text
            
            logger.info(
                "CitationMiddleware._prune: No numeric citations in body; "
                "removing reference section (body=%.0f chars, ref=%.0f chars).",
                body_len, ref_len
            )
            return body.rstrip()

        # 2) Parse bibliography rows: first occurrence of each legacy index wins.
        ref_lines = ref_part.splitlines()
        ref_by_old: dict[int, str] = {}
        first_entry_i: int | None = None
        for i, line in enumerate(ref_lines):
            legacy = CitationMiddleware._parse_reference_entry_line(line)
            if legacy is None:
                continue
            if first_entry_i is None:
                first_entry_i = i
            ref_by_old.setdefault(legacy, line)

        header_lines = ref_lines[:first_entry_i] if first_entry_i is not None else list(ref_lines)

        # 3) Only indices that appear in BOTH body and the parsed bibliography get a slot.
        kept_ordered = [n for n in body_ref_nums if n in ref_by_old]

        if not kept_ordered:
            if not body.strip():
                return text
            body = re.sub(r"\[([1-9]\d{0,2})\]", "", body)
            return body.rstrip()

        # 4) 「旧编号 → 新编号」，例如正文顺序 [2,5] 且两条都在文献表中 → {2:1, 5:2}
        old_to_new: dict[int, int] = {old: i + 1 for i, old in enumerate(kept_ordered)}

        # 5) 在正文中应用该映射；无文献条目的 [n] 直接去掉
        def _renumber_body(match: "re.Match[str]") -> str:
            old = int(match.group(1))
            new = old_to_new.get(old)
            if new is None:
                return ""
            return f"[{new}]"

        body = re.sub(r"\[([1-9]\d{0,2})\]", _renumber_body, body)

        # 6) 参考文献：按正文首次引用顺序输出，编号 1..N；不拼回未解析的 tail（避免孤儿文献行）
        sorted_entry_lines = [
            CitationMiddleware._rewrite_ref_line_index_prefix(ref_by_old[old], old_to_new[old])
            for old in kept_ordered
        ]

        header_part = "\n".join(header_lines).rstrip("\n")
        entries_part = "\n\n".join(sorted_entry_lines)
        new_ref_part_parts: list[str] = []
        if header_part:
            new_ref_part_parts.append(header_part)
        if entries_part:
            new_ref_part_parts.append(entries_part)
        new_ref_part = "\n\n".join(new_ref_part_parts)
        if new_ref_part and not new_ref_part.endswith("\n"):
            new_ref_part += "\n"
        return body + new_ref_part

    @staticmethod
    def _insert_inline_citations(
        text: str, citations_with_snippets: list[dict]
    ) -> str:
        """Insert inline citations into text by matching snippets to sentences.
        
        Strategy:
        1. First, check if LLM has already added citations (preferred)
        2. If citations are missing, use similarity matching to insert them
        3. Ensure all citations from tool results are represented in the text
        4. Convert all citations to numeric format [1], [2], etc.
        
        Args:
            text: The response text
            citations_with_snippets: List of {citation_tag, snippet, ...} dicts
            
        Returns:
            Text with inline citations inserted in numeric format
        """
        # Extract existing inline citations to avoid duplicates
        existing_inline = CitationMiddleware._extract_inline_citations(text)
        # Handle both old format (title, url, -1) and new format (title, url, ref_num)
        existing_urls = {url for _, url, _ in existing_inline if url}
        existing_citation_tags = {tag for tag, _, _ in existing_inline if tag}
        
        # Check if LLM has already added citations for most sources
        # If so, we'll be more conservative about adding more
        citations_from_tools = {cit["url"] for cit in citations_with_snippets}
        citations_in_text = existing_urls & citations_from_tools
        coverage_ratio = len(citations_in_text) / len(citations_from_tools) if citations_from_tools else 0.0
        
        logger.info(
            "CitationMiddleware: Found %d/%d citations already in text (coverage: %.1f%%)",
            len(citations_in_text),
            len(citations_from_tools),
            coverage_ratio * 100,
        )
        
        # Remove reference section before processing (to avoid matching citations in the reference list)
        ref_section_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if ref_section_match:
            text_for_processing = text[:ref_section_match.start()]
        else:
            text_for_processing = text
        
        # Also remove any "Here are the references:" sections
        ref_intro_match = re.search(r"(Here are the references?:|以下是引用|参考文献如下)", text_for_processing, re.IGNORECASE)
        if ref_intro_match:
            text_for_processing = text_for_processing[:ref_intro_match.start()]
        
        # Simple sentence splitting: split by sentence endings.
        # Keep all original segments for lossless reconstruction.
        sentence_endings = r"[。！？.!?\n]+"
        parts = re.split(f"({sentence_endings})", text_for_processing)
        
        # (raw_sentence, ending, normalized_for_matching, is_matchable)
        sentence_segments: list[tuple[str, str, str, bool]] = []
        matchable_indices: list[int] = []
        for i in range(0, len(parts), 2):
            sentence_raw = parts[i] if i < len(parts) else ""
            ending = parts[i + 1] if i + 1 < len(parts) else ""
            sentence_text = sentence_raw.strip()

            # Prefer normal prose; always skip headings, blockquotes, tables, fences.
            # Bullet / numbered list *lines* are common in reports — treat as matchable
            # when there is enough text after the list marker (see is_prose_line).
            is_markdown_structure = bool(
                re.match(r"^(#{1,6}\s|>\s|\|\s*.+\s*\||```)", sentence_text)
            )
            if not is_markdown_structure and re.match(r"^[-*+]\s", sentence_text):
                rest = re.sub(r"^[-*+]\s+", "", sentence_text).strip()
                is_markdown_structure = len(rest) < 8
            if not is_markdown_structure and re.match(r"^\d+\.\s", sentence_text):
                rest = re.sub(r"^\d+\.\s+", "", sentence_text).strip()
                is_markdown_structure = len(rest) < 8
            is_matchable = (
                bool(sentence_text)
                and len(sentence_text) >= 8
                and not is_markdown_structure
                and not re.match(r"^\d+\.\s*\[citation:", sentence_text)
                and "[citation:" not in sentence_text
            )

            seg_idx = len(sentence_segments)
            sentence_segments.append((sentence_raw, ending, sentence_text, is_matchable))
            if is_matchable:
                matchable_indices.append(seg_idx)
        
        # If we filtered out reference section, we need to work with original text positions
        # For now, we'll rebuild from text_for_processing and then append the reference section back
        if ref_section_match:
            # Store reference section to append later
            reference_section = text[ref_section_match.start():]
        else:
            reference_section = None
        
        if not matchable_indices:
            return text
        
        # Build mapping: sentence_index -> list of citation_tags to insert
        citations_to_insert: dict[int, list[str]] = {}
        # Track which sentences already have citations to avoid clustering
        sentences_with_citations: set[int] = set()
        
        # For each citation with snippet, find best matching sentence
        # CRITICAL: Ensure ALL tool results are represented in the text
        uncited_citations = []
        for cit_idx, cit in enumerate(citations_with_snippets):
            citation_tag = cit["citation_tag"]
            url = cit["url"]
            snippet = cit["snippet"]
            
            # Skip if already cited inline (LLM already added it)
            if url in existing_urls or citation_tag in existing_citation_tags:
                logger.debug(
                    "CitationMiddleware: Skipping '%s' - already cited in text",
                    cit.get("title", url)[:40],
                )
                continue
            
            # Track uncited citations to ensure they all get inserted
            uncited_citations.append((cit_idx, cit))
            
            # Even if no snippet, we should still try to insert the citation
            # Use title-based matching as fallback
            has_snippet = snippet and len(snippet) >= 10
            
            # If LLM already cited almost all URLs, be slightly more selective—but keep
            # thresholds high to avoid hallucinated citations.
            #
            # CRITICAL: After fixing hallucination issue, we need much stricter matching:
            # - 0.075 was too low; it allowed 7.5% keyword overlap, causing unrelated citations
            # - New thresholds require 15%+ keyword overlap OR strong title match
            if coverage_ratio > 0.85 and has_snippet:
                min_threshold = 0.25  # Strong semantic connection required
            else:
                min_threshold = 0.18  # Still strict: ~18% keyword overlap + title match
            
            # Find matching sentences with more aggressive matching
            # Strategy: Find multiple potential matches, not just the best one
            candidate_matches: list[tuple[int, float]] = []
            
            for idx in matchable_indices:
                sentence = sentence_segments[idx][2]
                # Skip if sentence already has this specific citation
                if citation_tag in sentence:
                    continue
                
                # Skip very short sentences
                if len(sentence) < 8:
                    continue
                
                # Calculate similarity using enhanced algorithm
                score = CitationMiddleware._calculate_similarity(sentence, snippet)
                
                # Also check if snippet keywords appear in sentence
                snippet_words = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", snippet.lower()))
                sentence_words = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", sentence.lower()))
                if snippet_words:
                    keyword_overlap = len(snippet_words & sentence_words) / len(snippet_words)
                else:
                    keyword_overlap = 0.0
                
                # Check for key terms from snippet title (if available)
                title_match = 0.0
                if "title" in cit:
                    title_words = set(re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", cit["title"].lower()))
                    if title_words:
                        title_match = len(title_words & sentence_words) / len(title_words)
                
                # Combined score with multiple factors
                combined_score = (score * 0.4) + (keyword_overlap * 0.4) + (title_match * 0.2)
                
                if combined_score >= min_threshold:
                    candidate_matches.append((idx, combined_score))
            
            # Sort by score (descending)
            candidate_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Strategy: Insert citation in the best matching sentence
            # If no good match, skip insertion to avoid hallucinations
            best_match_idx = -1
            best_score = 0.0
            
            if candidate_matches:
                # Prefer sentences that don't already have citations
                # This distributes citations across the text
                for idx, score in candidate_matches:
                    if idx not in sentences_with_citations:
                        best_match_idx = idx
                        best_score = score
                        break
                
                # If all candidate sentences already have citations, use the best one anyway
                if best_match_idx == -1:
                    best_match_idx, best_score = candidate_matches[0]
            else:
                # CRITICAL FIX: NO fallback with low thresholds
                # Previous behavior used relaxed_threshold=0.035, allowing citations to attach
                # to sentences with only 3.5% keyword overlap (e.g., both mention "FSD" but 
                # one is about trials and another about performance evaluation).
                #
                # NEW: If primary matching fails (no candidate_matches with min_threshold),
                # we should NOT insert the citation at all, as it likely leads to hallucinated
                # citations where the reference is unrelated to the sentence.
                logger.debug(
                    "CitationMiddleware: No sufficiently related sentence found for '%s' "
                    "(snippet_len=%d, min_threshold=%.3f); skipping insertion to avoid hallucination.",
                    cit.get("title", cit.get("url", ""))[:40],
                    len(snippet),
                    min_threshold,
                )
            
            # Insert citation after matching sentence - store as [citation:title](url) format
            # Will be converted to numeric format later after reference section is built
            if best_match_idx >= 0:
                if best_match_idx not in citations_to_insert:
                    citations_to_insert[best_match_idx] = []
                # Avoid duplicates
                if citation_tag not in citations_to_insert[best_match_idx]:
                    citations_to_insert[best_match_idx].append(citation_tag)
                    sentences_with_citations.add(best_match_idx)
                    logger.info(
                        "CitationMiddleware: Matched citation '%s' to sentence %d (score: %.2f, snippet_len: %d)",
                        cit["title"][:40],
                        best_match_idx,
                        best_score,
                        len(snippet),
                    )
            else:
                logger.warning(
                    "CitationMiddleware: Could not find any insertion point for citation '%s'",
                    cit["title"][:40],
                )
        
        # Rebuild text with citations inserted
        if not citations_to_insert:
            logger.debug("CitationMiddleware: No citations to insert (no matches found)")
            return text
        
        # Rebuild the processed text with citations (lossless: keep all original segments)
        result_parts = []
        inserted_count = 0
        
        for idx, (sentence_raw, ending, _, _) in enumerate(sentence_segments):
            result_parts.append(sentence_raw)
            
            # Insert citations before the ending punctuation
            if idx in citations_to_insert:
                # Keep citations inline to avoid breaking markdown structure.
                citations_str = "".join(citations_to_insert[idx])
                result_parts.append(citations_str)
                inserted_count += len(citations_to_insert[idx])
            
            result_parts.append(ending)
        
        # Reconstruct full text: processed part + reference section
        processed_text = "".join(result_parts)
        if reference_section:
            # Append reference section back
            final_text = processed_text + reference_section
        else:
            final_text = processed_text
        
        logger.info(
            "CitationMiddleware: Inserted %d inline citation(s) into %d sentence(s)",
            inserted_count,
            len(citations_to_insert),
        )
        
        return final_text

    @staticmethod
    def _build_reference_section(
        tool_citations: list[dict],
        inline_urls: set[str],
        inline_citation_pairs: list[tuple[str, str]] = None,
    ) -> tuple[str, dict[str, int]]:
        """Build a Markdown reference section with IEEE format.

        **STRICT RULE**: Only include URLs that (1) appear as inline citations and (2) come from
        *tool_citations* (web_search / web_fetch / etc.). Model-invented plausible URLs are dropped.

        Example row::

            [1] "Report title. Org. 2024". https://example.com/path

        Args:
            tool_citations: List of citations from tool results (for metadata like tool name)
            inline_urls: Set of normalized URLs found in inline citations
            inline_citation_pairs: List of (title, url) tuples from inline citations, in order of appearance
            
        Returns:
            Tuple of (reference_section_text, url_to_ref_map) where url_to_ref_map maps normalized URL to reference number
        """
        # Nothing to render if no inline citations
        # CRITICAL: inline_citation_pairs is the source of truth - if it's empty, no references
        if not inline_citation_pairs:
            return "", {}

        # Blank line between entries so Markdown renders each as its own paragraph (not one merged line).
        ref_header = "\n\n## 参考文献\n\n"
        entry_blocks: list[str] = []
        url_to_ref_map: dict[str, int] = {}

        # Normalize all tool citation URLs for matching (to get tool metadata)
        normalized_tool_citations = {}
        for cit in tool_citations:
            url = cit.get("url", "").strip()
            if not url:
                continue
            normalized_url = CitationMiddleware._normalize_url(url)
            normalized_tool_citations[normalized_url] = {
                "title": cit.get("title", "").strip(),
                "url": normalized_url,
                "tool": cit.get("tool", "").strip(),
                "site": cit.get("site", "").strip(),
                "year": cit.get("year", "").strip(),
                "snippet": (cit.get("snippet") or "")[:1200],
                "author_org": cit.get("author_org", "").strip(),
                "published_date": cit.get("published_date", "").strip(),
                "raw_content": (cit.get("raw_content") or "")[:80_000],
            }

        # Build ordered citations list based on inline_citation_pairs (order of appearance in text).
        #
        # New requirement:
        # - Reference list URL must be unique.
        # - Numeric reference numbers must correspond 1:1 with unique URLs.
        # - If the same URL is cited multiple times in the body, we reuse the same [n] number.
        seen_urls: set[str] = set()
        ordered_citations: list[dict] = []

        for title, url in inline_citation_pairs:
            try:
                # Clean URL first to remove citation tags, then normalize
                clean_url = CitationMiddleware._clean_url_for_reference(url)
                normalized_url = CitationMiddleware._normalize_url(clean_url)
            except Exception as e:
                logger.warning(
                    "CitationMiddleware: Failed to normalize URL '%s': %s", 
                    url[:60] if url else "None", 
                    e
                )
                # Try to clean it even if normalization fails
                try:
                    normalized_url = CitationMiddleware._clean_url_for_reference(url)
                except Exception:
                    normalized_url = url.strip() if url else ""

            if not normalized_url:
                continue

            # Only bibliography entries grounded in tool output (no LLM-invented URLs).
            if normalized_url not in normalized_tool_citations:
                logger.warning(
                    "CitationMiddleware: skipping reference list entry for URL not from tools: %s",
                    normalized_url[:100],
                )
                continue

            # URL uniqueness: keep only first occurrence.
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)

            cit = dict(normalized_tool_citations[normalized_url])
            if title and title.strip() and title.strip() != "Unspecified Source":
                cit["title"] = title.strip()

            # Fuller title when search index truncated with "..."; then author/org + dates; escape quotes
            _raw = (cit.get("title") or "").strip()
            _snip = cit.get("snippet") or ""
            _u = (cit.get("url") or "").strip()
            CitationMiddleware._enrich_facebook_citation_dates(
                cit, source_url=_u, snippet=_snip
            )
            CitationMiddleware._enrich_zhihu_citation_dates(
                cit, source_url=_u, snippet=_snip
            )
            base_polished = strip_trailing_ellipsis(polish_search_hit_title(_raw, _snip, url=_u))
            base_polished = strip_trailing_article_id_from_title(base_polished, _u)
            _ao2 = CitationMiddleware._facebook_ieee_organization(
                _u,
                (cit.get("author_org") or "").strip(),
                (cit.get("site") or "").strip(),
            )
            _dom = display_domain_for_reference(_u, (cit.get("site") or "").strip())
            _pub2 = (cit.get("published_date") or "").strip()
            _year_only2 = ""
            if _pub2:
                _ym2 = re.search(r"\b(19|20)\d{2}\b", _pub2)
                if _ym2:
                    _year_only2 = _ym2.group(0)
            composed = CitationMiddleware._compose_ieee_reference_title(
                base_polished,
                _ao2,
                _year_only2,
                year_fallback=(cit.get("year") or "").strip(),
                domain=_dom,
            )
            cit["title"] = CitationMiddleware._escape_title_for_reference_line(composed)

            ordered_citations.append(cit)

        if not ordered_citations:
            return "", {}

        # Rows that will actually appear in the reference list, with contiguous [1]..[N].
        # CRITICAL: If we skipped an invalid URL in the render loop but still assigned its
        # URL ref number in url_to_ref_map, body [n] and the bibliography would disagree.
        render_rows: list[tuple[dict, str]] = []  # (citation dict, display_text)

        for cit in ordered_citations:
            title = (cit.get("title") or "").strip()
            url = (cit.get("url") or "").strip()

            if not title:
                title = "Unspecified Source"
            else:
                title = strip_trailing_punctuation(title)

            clean_url = CitationMiddleware._clean_url_for_reference(url)
            if not clean_url:
                logger.warning(
                    "CitationMiddleware: skipping reference for '%s' because URL is empty after cleaning",
                    title[:60],
                )
                continue

            if not CitationMiddleware._is_valid_url(clean_url):
                logger.warning(
                    "CitationMiddleware: skipping reference '%s' because URL appears invalid: %s",
                    title[:60],
                    clean_url[:80],
                )
                continue

            display_text = clean_url
            try:
                from urllib.parse import unquote, urlparse

                parsed = urlparse(clean_url)
                scheme = parsed.scheme or "https"
                netloc = parsed.netloc
                path = unquote(parsed.path or "")

                base = f"{scheme}://{netloc}{path}"

                if parsed.query:
                    base = f"{base}?{unquote(parsed.query)}"

                display_text = base
            except Exception:
                display_text = clean_url

            row_cit = dict(cit)
            row_cit["title"] = title
            render_rows.append((row_cit, display_text))

        if not render_rows:
            return "", {}

        for idx, (cit, display_text) in enumerate(render_rows, 1):
            title = (cit.get("title") or "").strip() or "Unspecified Source"
            entry_blocks.append(
                CitationMiddleware._ieee_web_reference_line(idx, title, display_text)
            )

        # URL -> ref id only for rows that appear in the bibliography (contiguous 1..N).
        for idx, (cit, _display_text) in enumerate(render_rows, 1):
            url = (cit.get("url") or "").strip()
            if not url:
                continue
            try:
                clean_url = CitationMiddleware._clean_url_for_reference(url)
                if not clean_url:
                    continue
                normalized_for_map = CitationMiddleware._normalize_url(clean_url)
                url_to_ref_map[normalized_for_map] = idx
                url_to_ref_map[clean_url] = idx
            except Exception:
                continue

        ref_body = "\n\n".join(entry_blocks)
        return f"{ref_header}{ref_body}\n\n", url_to_ref_map

    @staticmethod
    def _validate_ieee_format_compliance(text: str) -> None:
        """Validate that text complies with IEEE numeric citation format.
        
        Logs warnings if:
        - Legacy [citation:...] format is found (should be converted to [n])
        - Reference numbers are not consecutive (1, 2, 3, ...)
        - Reference format doesn't match IEEE spec
        
        This is diagnostic only - doesn't modify text.
        """
        # Check for legacy format remnants
        if "[citation:" in text:
            count = text.count("[citation:")
            logger.warning(
                "CitationMiddleware (FORMAT CHECK): Found %d legacy [citation:...] tags in text. "
                "These should have been converted to [n] format.",
                count
            )
        
        # Extract reference numbers from body
        citation_nums = IEEE_CITATION_FORMAT.get_citation_numbers(text)
        if citation_nums:
            max_num = max(citation_nums)
            expected_nums = set(range(1, max_num + 1))
            if citation_nums != expected_nums:
                missing = expected_nums - citation_nums
                logger.warning(
                    "CitationMiddleware (FORMAT CHECK): Reference numbers are not consecutive. "
                    "Max: %d, Missing: %s. Found: %s",
                    max_num, sorted(missing), sorted(citation_nums)
                )
        
        # Check reference section format
        ref_match = re.search(
            CitationMiddleware._reference_heading_regex(),
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if ref_match:
            ref_section = text[ref_match.start():]
            ref_lines = ref_section.split("\n")[1:]  # Skip heading
            
            ref_count = 0
            for line in ref_lines:
                if re.match(r"^\[?\d+\]", line.strip()):
                    ref_count += 1
            
            if ref_count > 0:
                logger.debug(
                    "CitationMiddleware (FORMAT CHECK): Reference section has %d entries. "
                    "Body has %d citations.",
                    ref_count, len(citation_nums)
                )

    # ------------------------------------------------------------------ core logic

    def _process_state(self, state: CitationMiddlewareState) -> dict | None:
        """Core logic: insert inline citations and append reference section.

        Shared by both sync ``after_model`` and async ``aafter_model``.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last AI message to post-process.
        #
        # LangGraph agent loop: a pure tool-calling turn has AIMessage.tool_calls and
        # (usually) empty content — skip that turn; citation patching runs after the
        # final text response. If a provider emits both text and tool_calls on one
        # message, we still patch that message's content.
        last_ai_message: AIMessage | None = None
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            content_str = (
                msg.content if isinstance(msg.content, str) else str(msg.content or "")
            )
            if tool_calls and not content_str.strip():
                return None
            last_ai_message = msg
            break

        if last_ai_message is None:
            return None

        # ── 1. Extract citations with snippets from all tool results ─────────
        citations_with_snippets = self._extract_tool_citations_with_snippets(messages)
        if not citations_with_snippets:
            logger.debug("CitationMiddleware: no tool citations found, skipping.")
            return None

        response_text = (
            last_ai_message.content
            if isinstance(last_ai_message.content, str)
            else str(last_ai_message.content)
        )
        response_text = CitationMiddleware._strip_legacy_sources_sections(response_text)
        response_text = CitationMiddleware._fix_nested_citations(response_text)
        response_text = CitationMiddleware._canonicalize_urls_in_citation_markdown(response_text)

        allowed_tool_urls: set[str] = set()
        for c in citations_with_snippets:
            u = (c.get("url") or "").strip()
            if not u:
                continue
            try:
                allowed_tool_urls.add(CitationMiddleware._normalize_url(u))
            except Exception:
                continue
        response_text = CitationMiddleware._strip_ungrounded_inline_citations(
            response_text, allowed_tool_urls
        )

        # ── 2. Check if reference section already exists ─────────────────────
        has_ref_section = self._has_reference_section(response_text)
        
        # ── 3. Extract existing inline citations ─────────────────────────────
        # CRITICAL: Extract ALL inline citations from the text - this is the source of truth
        inline_pairs_raw = self._extract_inline_citations(response_text)
        # Convert to (title, url) format for compatibility (filter out numeric-only refs)
        inline_pairs = [(title, url) for title, url, ref_num in inline_pairs_raw if url]
        logger.info(
            "CitationMiddleware: Found %d inline citations in text before processing",
            len(inline_pairs)
        )

        # ── 4. Insert inline citations for uncited sources ──────────────────
        # 无论模型是否已经添加了一部分引用，这里都会尝试为“尚未标注”的来源智能插入行内引用。
        # 仅在检测到文本严重重复（疑似损坏）时跳过。
        words = response_text.split()
        if len(words) > 0:
            # Check for excessive repetition of short phrases
            phrase_counts = {}
            for i in range(len(words) - 2):
                phrase = " ".join(words[i : i + 3])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

            max_repetition = max(phrase_counts.values()) if phrase_counts else 0
            if max_repetition > 10:
                logger.warning(
                    "CitationMiddleware: text appears corrupted (excessive repetition), skipping inline citation insertion."
                )
            else:
                logger.info(
                    "CitationMiddleware: attempting intelligent inline citation insertion for uncited sources."
                )
                try:
                    response_text = self._insert_inline_citations(
                        response_text, citations_with_snippets
                    )
                    # Ensure each major section has at least one inline citation.
                    response_text = self._ensure_section_citation_coverage(
                        response_text, citations_with_snippets
                    )
                    # Re-extract to get updated inline citations after insertion
                    inline_pairs_raw = self._extract_inline_citations(response_text)
                    inline_pairs = [
                        (title, url) for title, url, ref_num in inline_pairs_raw if url
                    ]
                    
                    logger.info(
                        "CitationMiddleware: Found %d inline citations in text after insertion",
                        len(inline_pairs)
                    )
                except Exception as e:
                    logger.error(
                        f"CitationMiddleware: failed to insert inline citations: {e}"
                    )
                    # Continue without inline citations

        # ── 5. Build url_to_ref_map (ALWAYS when body has [citation:](url)) + reference section
        #
        # If the model already wrote "## 参考文献", older code skipped this block entirely,
        # leaving url_to_ref_map empty so step 5.5 never ran — body stayed as raw
        # [citation:Title](URL) instead of [1][2]. We always extract citations from the
        # body only (exclude existing reference heading), build the map, then either
        # append a new section or replace the model's section with our canonical list.
        url_to_ref_map: dict[str, int] = {}
        ref_heading_m = re.search(
            CitationMiddleware._reference_heading_regex(),
            response_text,
            re.IGNORECASE | re.MULTILINE,
        )
        body_for_citation_extract = (
            response_text[: ref_heading_m.start()] if ref_heading_m else response_text
        )
        final_inline_pairs_raw = self._extract_inline_citations(body_for_citation_extract)
        final_inline_pairs = [
            (title, url) for title, url, ref_num in final_inline_pairs_raw if url
        ]
        final_inline_urls = {url.strip() for _, url in final_inline_pairs}

        # If intelligent insertion matched nothing, the body may have no [citation:](url) tags;
        # _build_reference_section would then skip and no 参考文献 appears. Append synthetic
        # tags from *all* tool citations so the reference list + numeric conversion still run.
        if not final_inline_pairs and citations_with_snippets:
            fallback_pairs: list[tuple[str, str]] = []
            for c in citations_with_snippets:
                url = (c.get("url") or "").strip()
                if not url:
                    continue
                raw_title = (c.get("title") or "").strip() or "Source"
                fallback_pairs.append(
                    (CitationMiddleware._sanitize_citation_tag_title(raw_title), url)
                )
            if fallback_pairs:
                logger.info(
                    "CitationMiddleware: no inline [citation:](url) in body; "
                    "appending %d fallback tag(s) from tool results for bibliography",
                    len(fallback_pairs),
                )
                ref_h_fb = re.search(
                    CitationMiddleware._reference_heading_regex(),
                    response_text,
                    re.IGNORECASE | re.MULTILINE,
                )
                if ref_h_fb:
                    body_only_fb = response_text[: ref_h_fb.start()].rstrip()
                    tail_fb = response_text[ref_h_fb.start() :]
                else:
                    body_only_fb = response_text.rstrip()
                    tail_fb = ""
                synthetic = "".join(f"[citation:{t}]({u})" for t, u in fallback_pairs)
                response_text = f"{body_only_fb}\n\n{synthetic}{tail_fb}"
                has_ref_section = self._has_reference_section(response_text)
                ref_heading_m = re.search(
                    CitationMiddleware._reference_heading_regex(),
                    response_text,
                    re.IGNORECASE | re.MULTILINE,
                )
                body_for_citation_extract = (
                    response_text[: ref_heading_m.start()] if ref_heading_m else response_text
                )
                final_inline_pairs_raw = self._extract_inline_citations(body_for_citation_extract)
                final_inline_pairs = [
                    (title, url) for title, url, ref_num in final_inline_pairs_raw if url
                ]
                final_inline_urls = {url.strip() for _, url in final_inline_pairs}

        logger.info(
            "CitationMiddleware: %d [citation:]-style pairs in body (excluding reference heading)",
            len(final_inline_pairs),
        )

        if final_inline_pairs:
            normalized_final_urls = {
                CitationMiddleware._normalize_url(url) for _, url in final_inline_pairs
            }
            filtered_tool_citations = []
            for c in citations_with_snippets:
                url = c.get("url", "").strip()
                if url:
                    normalized_url = CitationMiddleware._normalize_url(url)
                    if normalized_url in normalized_final_urls:
                        filtered_tool_citations.append(
                            {
                                "title": c["title"],
                                "url": c["url"],
                                "tool": c["tool"],
                                "snippet": (c.get("snippet") or "")[:1200],
                                "site": (c.get("site") or ""),
                                "year": (c.get("year") or ""),
                                "author_org": (c.get("author_org") or ""),
                                "published_date": (c.get("published_date") or ""),
                            }
                        )

            logger.info(
                "CitationMiddleware: Filtered tool citations from %d to %d (only those in body)",
                len(citations_with_snippets),
                len(filtered_tool_citations),
            )

            ref_section, url_to_ref_map = self._build_reference_section(
                filtered_tool_citations,
                final_inline_urls,
                inline_citation_pairs=final_inline_pairs,
            )
            if ref_section:
                if not has_ref_section:
                    response_text = response_text + ref_section
                    logger.info(
                        "CitationMiddleware: Appended reference section (%d body citations)",
                        len(final_inline_pairs),
                    )
                else:
                    # Replace model-written reference block so numbering matches post-conversion body.
                    body_only = response_text[: ref_heading_m.start()].rstrip()
                    response_text = body_only + ref_section
                    logger.info(
                        "CitationMiddleware: Replaced model reference section with canonical list (%d citations)",
                        len(final_inline_pairs),
                    )
        
        # ── 5.5. Convert all [citation:title](url) format to numeric [1] format ──
        if url_to_ref_map:
            try:
                response_text = CitationMiddleware._convert_citations_to_numeric(
                    response_text, url_to_ref_map
                )
                logger.info(
                    "CitationMiddleware: Converted citations to numeric format using URL mappings"
                )
            except Exception as e:
                logger.error(f"CitationMiddleware: failed to convert citations to numeric format: {e}")
        # Fallback: if we still have legacy "[citation:...](url)" links in the body,
        # convert them using strict per-occurrence numbering.
        #
        # Without this, `_strip_inline_urls_from_body()` may delete leftover
        # `[citation:...]` tags entirely, which looks like "citations missing"
        # in later sections.
        if "[citation:" in response_text:
            try:
                response_text = CitationMiddleware._convert_citations_to_numeric(
                    response_text, url_to_ref_map=None
                )
                logger.info(
                    "CitationMiddleware: Fallback converted remaining [citation:...] tags to numeric per-occurrence"
                )
            except Exception as e:
                logger.error(
                    "CitationMiddleware: fallback conversion for remaining [citation:...] failed: %s",
                    e,
                )

        # ── 6. Normalize URLs in all citations to fix spaces and special characters ──
        # This also fixes nested citations in both body text and reference section
        try:
            response_text = self._normalize_citation_urls(response_text)
        except Exception as e:
            logger.error(f"CitationMiddleware: failed to normalize citation URLs: {e}")
        
        # ── 7. Cleanup broken URL fragments to avoid "乱码"式尾巴 ─────────────
        try:
            response_text = self._clean_broken_urls(response_text)
        except Exception as e:
            logger.error(f"CitationMiddleware: failed to clean broken URLs: {e}")

        # ── 8. Normalize numeric citation artifacts (e.g. [6][6], [2][3].domain/...) ──
        try:
            response_text = self._normalize_numeric_citations(response_text)
        except Exception as e:
            logger.error(f"CitationMiddleware: failed to normalize numeric citations: {e}")

        # ── 9. Keep only references actually cited in body text ───────────────
        try:
            response_text = self._prune_reference_section_to_body_citations(response_text)
        except Exception as e:
            logger.error(f"CitationMiddleware: failed to prune reference section: {e}")

        # ── 10. Strip inline URLs from body so链接只出现在参考文献中 ────────
        try:
            response_text = self._strip_inline_urls_from_body(response_text)
        except Exception as e:
            logger.error(f"CitationMiddleware: failed to strip inline URLs from body: {e}")

        # ── 10.5. Keep body Markdown (headings, tables, bold) for the chat UI.
        #     Stripping was removing ## / table context and fought Streamdown rendering.

        # ── 10.6. One markdown document: unwrap ```markdown ... ``` on the main body
        #     so 参考文献 is not left outside a code fence (Streamdown).
        try:
            response_text = CitationMiddleware._merge_fenced_body_with_reference_section(
                response_text
            )
        except Exception as e:
            logger.error(
                "CitationMiddleware: failed to merge fenced body with references: %s", e
            )

        # ── 10.7. Repair flattened markdown table rows (e.g. "| ... | | 1 | ... |"). ──
        try:
            response_text = CitationMiddleware._repair_flattened_markdown_tables(response_text)
        except Exception as e:
            logger.error("CitationMiddleware: failed to repair flattened tables: %s", e)

        # ── 10.8. Strip any surviving Sources / Primary Sources blocks (e.g. model H3 after refs).
        try:
            response_text = CitationMiddleware._strip_legacy_sources_sections(response_text)
        except Exception as e:
            logger.error(
                "CitationMiddleware: failed final legacy sources strip: %s", e
            )

        # ── 11. Validate IEEE format compliance ─────────────────────────────────
        self._validate_ieee_format_compliance(response_text)
        
        # ── 12. Final statistics and logging ────────────────────────────────────
        # Calculate final citation statistics
        final_citation_nums = IEEE_CITATION_FORMAT.get_citation_numbers(response_text)
        citation_count = len(final_citation_nums)
        has_legacy_format = "[citation:" in response_text
        ref_section_exists = bool(re.search(
            CitationMiddleware._reference_heading_regex(),
            response_text,
            re.IGNORECASE | re.MULTILINE,
        ))
        
        # Log IEEE format compliance summary
        logger.info(
            "CitationMiddleware (IEEE SUMMARY): "
            "Citations=%d, Format=%s, References=%s, Legacy=[citation:...]=%s",
            citation_count,
            "numeric[n]" if citation_count > 0 else "none",
            "present" if ref_section_exists else "absent",
            "yes" if has_legacy_format else "no",
        )
        
        # ── 13. Return updated message ──────────────────────────────────────
        updated_message = AIMessage(
            id=last_ai_message.id,
            content=response_text,
        )
        
        return {"messages": [updated_message]}

    @classmethod
    def apply_to_markdown_with_allowed_urls(cls, markdown: str, urls: list[str]) -> str:
        """IEEE citation pass without web ``ToolMessage`` history (e.g. three-phase report pipeline).

        Synthesizes a ``web_search``-style tool payload listing ``[citation:…](url)`` for each
        unique HTTP(S) URL so :meth:`_process_state` can ground inline tags and append references.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        text = (markdown or "").strip()
        if not text:
            return markdown

        deduped: list[str] = []
        seen: set[str] = set()
        for u in urls:
            u = (u or "").strip()
            if not u.startswith("http"):
                continue
            try:
                nu = cls._normalize_url(u)
            except Exception:
                nu = u
            if nu in seen:
                continue
            seen.add(nu)
            deduped.append(nu)

        if not deduped:
            return markdown

        lines: list[str] = []
        for u in deduped:
            title = polish_search_hit_title("", "", url=u)
            lines.append(f"[citation:{title}]({u})")
        tool_body = "Evidence URLs:\n" + "\n".join(lines)

        inst = cls()
        state: dict = {
            "messages": [
                ToolMessage(content=tool_body, tool_call_id="evidence", name="web_search"),
                AIMessage(content=text),
            ]
        }
        out = inst._process_state(state)  # type: ignore[arg-type]
        if not out:
            return markdown
        msgs = out.get("messages") or []
        if not msgs:
            return markdown
        new_ai = msgs[-1]
        c = getattr(new_ai, "content", "")
        return c if isinstance(c, str) else str(c)

    # ------------------------------------------------------------------ hooks

    @override
    def after_model(
        self, state: CitationMiddlewareState, runtime: Runtime
    ) -> dict | None:
        """Synchronous hook — called when agent runs via .stream() / .invoke()."""
        return self._process_state(state)

    @override
    async def aafter_model(
        self, state: CitationMiddlewareState, runtime: Runtime
    ) -> dict | None:
        """Async hook — called when agent runs via .astream() / .ainvoke()."""
        return self._process_state(state)