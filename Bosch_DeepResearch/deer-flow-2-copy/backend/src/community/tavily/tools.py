import logging

from langchain.tools import tool
from tavily import TavilyClient

from deerflow.config import get_app_config
from deerflow.utils.reference_titles import polish_fetched_page_title, polish_search_hit_title

logger = logging.getLogger(__name__)


def _get_tavily_client() -> TavilyClient:
    config = get_app_config().get_tool_config("web_search")
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    return TavilyClient(api_key=api_key)


def _extract_published_date_from_content(content: str) -> tuple[str, str, float]:
    """
    Extract published date from HTML content or text with confidence metrics.
    
    Tries multiple methods in order:
    1. Common HTML meta tags (og:published_time, article:published_time, etc.)
    2. HTML5 time tags and semantic elements
    3. Text patterns (ISO dates, month-year, structured text)
    4. Year fallback
    
    Returns: tuple of (date_string, source, confidence_score)
             where source is one of: "meta_tag", "time_tag", "text_pattern", "year_fallback", ""
             confidence_score is 0-1 where 1.0 is highest confidence
    """
    import re
    
    if not content:
        return "", "", 0.0
    
    try:
        # Method 1: Extract from HTML meta tags (highest confidence)
        meta_patterns = [
            (r'<meta\s+property=["\']og:published_time["\'].*?content=["\']([^"\']+)["\']', 0.95),
            (r'<meta\s+property=["\']article:published_time["\'].*?content=["\']([^"\']+)["\']', 0.95),
            (r'<meta\s+property=["\']og:updated_time["\'].*?content=["\']([^"\']+)["\']', 0.90),
            (r'<meta\s+name=["\']publish_date["\'].*?content=["\']([^"\']+)["\']', 0.90),
            (r'<meta\s+name=["\']published["\'].*?content=["\']([^"\']+)["\']', 0.85),
            (r'<meta\s+name=["\']date["\'].*?content=["\']([^"\']+)["\']', 0.85),
            (r'<meta\s+name=["\']article\.published_time["\'].*?content=["\']([^"\']+)["\']', 0.95),
            (r'<meta\s+name=["\']article\.modified_time["\'].*?content=["\']([^"\']+)["\']', 0.90),
            (r'<meta\s+name=["\']twitter:text:published_at["\'].*?content=["\']([^"\']+)["\']', 0.85),
        ]
        
        for pattern, confidence in meta_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                date_str = match.group(1).strip()
                if date_str and _validate_date_format(date_str):
                    return date_str, "meta_tag", confidence
        
        # Method 2: Extract from HTML5 time tags and structured markup
        time_tag_patterns = [
            (r'<time[^>]*datetime=["\']([^"\']+)["\']', 0.90),
            (r'<time[^>]*datetime=([^>\s]+)', 0.85),
            (r'<time[^>]*>([^<]+)</time>', 0.70),
        ]
        
        for pattern, confidence in time_tag_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                date_str = match.group(1).strip()
                if date_str and _validate_date_format(date_str):
                    return date_str, "time_tag", confidence
        
        # Method 3: Extract text patterns with structured markers
        # Look for "Published:", "Updated:", etc. patterns
        structured_text_patterns = [
            (r'(?:Published|Posted|Updated|Modified|Created)[\s:]+(\w+\s+\d{1,2},?\s+\d{4})', 0.80),
            (r'(?:Published|Posted|Updated)[\s:]+(\d{4}-\d{2}-\d{2})', 0.85),
            (r'@datetime\s*=\s*["\']([^"\']+)["\']', 0.75),
            (r'datetime\s*:\s*["\']?([^\s"\']+)["\']?', 0.70),
        ]
        
        for pattern, confidence in structured_text_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                date_str = match.group(1).strip()
                if date_str and _validate_date_format(date_str):
                    return date_str, "text_pattern", confidence
        
        # Method 4: Extract ISO date format from text
        iso_match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
        if iso_match:
            date_str = iso_match.group(1)
            return date_str, "text_pattern", 0.75
        
        # Method 5: Extract common date formats
        date_formats = [
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2},?\s+)(\d{4})', 0.80),
            (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', 0.80),
            (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})', 0.70),
        ]
        
        for pattern, confidence in date_formats:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(0)
                if _validate_date_format(date_str):
                    return date_str, "text_pattern", confidence
        
        # Method 6: Year-only fallback (lowest confidence)
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', content)
        if year_match:
            return year_match.group(1), "year_fallback", 0.40
        
    except Exception as e:
        logger.debug(f"Error extracting published date from content: {e}")
    
    return "", "", 0.0


def _validate_date_format(date_str: str) -> bool:
    """
    Validate if a string looks like a date.
    
    Args:
        date_str: String to validate
        
    Returns:
        True if string appears to be a valid date format
    """
    import re
    from datetime import datetime
    
    if not date_str:
        return False
    
    try:
        # Try ISO format first
        try:
            datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return True
        except:
            pass
        
        # Try common formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%d.%m.%Y']:
            try:
                datetime.strptime(date_str.split('T')[0], fmt)
                return True
            except:
                pass
        
        # Check if it contains date-like patterns
        if re.search(r'\d{4}', date_str):  # Has a year
            if re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})', date_str, re.IGNORECASE):
                return True
        
        return False
    except:
        return False


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """Search the web for information.

    CRITICAL CITATION RULE: After using ANY result from this tool, you MUST insert the
    citation tag (provided as `citation_tag` in each result) IMMEDIATELY after the sentence.
    DO NOT collect citations at the end - insert them inline right after the referenced content.

    Example output format:
    "AI adoption grew 73% in 2025[citation:TechReport 2025](https://techcrunch.com/...).
    The main drivers include cost reduction[citation:Forbes AI](https://forbes.com/...)."

    Args:
        query: The query to search for.
    """
    config = get_app_config().get_tool_config("web_search")
    max_results = 10
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results")

    client = _get_tavily_client()
    try:
        # Use search with include_answer=True to get better metadata
        res = client.search(query, max_results=max_results, include_answer=True)
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower() or "401" in error_msg:
            return (
                f"❌ ERROR: Tavily API key is missing or invalid.\n"
                f"Please configure TAVILY_API_KEY environment variable or set api_key in config.yaml.\n"
                f"Get your API key from: https://tavily.com/\n\n"
                f"Error details: {error_msg}\n\n"
                f"⚠️  Without a valid API key, web_search tool cannot function. "
                f"You MUST inform the user that you cannot search the web without proper configuration."
            )
        else:
            return (
                f"❌ ERROR: Web search failed.\n"
                f"Error: {error_msg}\n\n"
                f"Please try again or check your network connection."
            )
    def normalize_url(url: str) -> str:
        """Normalize URL by encoding spaces and special characters."""
        from urllib.parse import quote, urlparse, urlunparse
        try:
            parsed = urlparse(url)
            # Encode path, query, and fragment components
            normalized_path = quote(parsed.path, safe='/')
            normalized_query = quote(parsed.query, safe='=&')
            normalized_fragment = quote(parsed.fragment, safe='')
            return urlunparse((
                parsed.scheme,
                parsed.netloc,
                normalized_path,
                parsed.params,
                normalized_query,
                normalized_fragment
            ))
        except Exception:
            # Fallback: simple space replacement if parsing fails
            return url.replace(' ', '%20')

    normalized_results = []
    for idx, result in enumerate(res["results"]):
        url = normalize_url(result["url"])
        snippet = result.get("content") or ""
        disp_title = polish_search_hit_title(
            result.get("title") or "",
            snippet,
            url=url,
        )
        
        # Extract published date with multi-level strategy
        pub_raw = result.get("published_date") or result.get("publishedDate") or ""
        published_date = str(pub_raw).strip() if pub_raw else ""
        date_source = "api" if published_date else ""
        date_confidence = 0.95 if published_date else 0.0
        
        # Fallback 1: Try to extract from snippet if no API-provided date
        if not published_date:
            try:
                import re
                # Look for date patterns with priorities
                date_patterns = [
                    (r'(\d{4}-\d{2}-\d{2})', 0.80),  # ISO format - high confidence
                    (r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2},?\s+)?\d{4}', 0.75),  # Month Year
                    (r'(\d{1,2}/\d{1,2}/\d{4})', 0.70),  # MM/DD/YYYY
                ]
                for pattern, confidence in date_patterns:
                    date_match = re.search(pattern, snippet, re.IGNORECASE)
                    if date_match:
                        published_date = date_match.group(0)
                        date_source = "snippet"
                        date_confidence = confidence
                        logger.debug(
                            f"web_search: Extracted date from snippet: '{disp_title[:40]}' → {published_date} (confidence: {confidence})"
                        )
                        break
                
                # Fallback to year only if no structured date found
                if not published_date:
                    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', snippet)
                    if year_match:
                        published_date = year_match.group(1)
                        date_source = "snippet"
                        date_confidence = 0.50
                        logger.debug(
                            f"web_search: Extracted year from snippet: '{disp_title[:40]}' → {published_date}"
                        )
            except Exception:
                pass
        
        # Fallback 2: Try to fetch page for better metadata if no date yet
        # UPDATED: Now fetch for ALL results (removed idx < 3 limitation)
        # The complete date metadata is critical for citation quality
        if not published_date:
            try:
                logger.debug(f"web_search: Fetching page for date extraction: {url[:60]}")
                page_res = client.extract([url], extract_depth="advanced")
                if "results" in page_res and len(page_res["results"]) > 0:
                    page_result = page_res["results"][0]
                    
                    # First try API-provided date from extract
                    page_date = page_result.get("published_date") or page_result.get("publishedDate") or ""
                    if page_date:
                        published_date = str(page_date).strip()
                        date_source = "page_api"
                        date_confidence = 0.90
                        logger.debug(f"web_search: Found date from page fetch (API): {published_date}")
                    else:
                        # Try extracting from raw content
                        raw_content = page_result.get("raw_content", "")
                        extracted_date, ext_source, ext_confidence = _extract_published_date_from_content(raw_content)
                        if extracted_date:
                            published_date = extracted_date
                            date_source = f"page_html_{ext_source}"
                            date_confidence = ext_confidence * 0.85  # Slightly lower confidence for extracted
                            logger.debug(
                                f"web_search: Extracted date from HTML content: {published_date} "
                                f"(source: {ext_source}, confidence: {date_confidence:.2f})"
                            )
            except Exception as e:
                logger.debug(f"web_search: Failed to fetch page for date: {e}")
        
        # Final debug logging for published_date
        if not published_date:
            logger.debug(
                f"web_search: No published_date for '{disp_title[:40]}' after all attempts "
                f"(checked: api_fields={bool(pub_raw)}, snippet, page_fetch)"
            )
        else:
            logger.debug(
                f"web_search: Final published_date for '{disp_title[:40]}' → {published_date} "
                f"(source: {date_source}, confidence: {date_confidence:.2f})"
            )
        
        # Extract source/organization information if available
        source = result.get("source") or ""
        
        normalized_results.append(
            {
                "title": disp_title,
                "url": url,
                "snippet": snippet,
                "citation_tag": f"[citation:{disp_title}]({url})",
                "published_date": published_date,
                "date_source": date_source,  # NEW: Track where date came from
                "date_confidence": date_confidence,  # NEW: Confidence metric
                "source": source,
            }
        )

    # Build format with explicit source attribution for each content block.
    # NOTE: Some unit tests assert fixed substrings in this tool output.
    output_lines = [
        "🚨 CRITICAL CITATION RULES:\n",
        "1. Each result below is from a SPECIFIC SOURCE (see SOURCE: line)\n",
        "2. When you use ANY information from a result, you MUST add its citation_tag IMMEDIATELY after that sentence\n",
        "3. Format: [citation:Title](URL) - Copy EXACTLY as shown in SOURCE line\n",
        "4. Add ALL citation_tags to '## References' section at the END\n",
        "\n[WEB SEARCH RESULTS - Insert citation_tag INLINE]\n",
    ]
    
    for i, r in enumerate(normalized_results, 1):
        citation_tag = r['citation_tag']
        title = r['title']
        snippet = r['snippet']
        
        # Create a pre-formatted example that LLM can directly copy
        example_text = f"根据研究{citation_tag}，{snippet[:100]}..."
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"RESULT {i} - SOURCE: {citation_tag}")
        output_lines.append(f"{'='*60}")
        output_lines.append(f"Title: {title}")
        output_lines.append(f"URL: {r['url']}")
        
        # Extract organization/source if available
        # First try to use explicit 'source' field, fallback to domain name
        source = r.get("source") or ""
        if not source:
            try:
                from urllib.parse import urlparse
                domain = urlparse(r['url']).netloc
                # Remove common prefixes for cleaner display
                source = domain.replace('www.', '').replace('m.', '')
            except Exception:
                source = ""
        
        if source:
            output_lines.append(f"Organization: {source}")
        
        # Include published date if available with confidence indicator
        if r.get("published_date"):
            pub_date = r['published_date']
            confidence = r.get('date_confidence', 0.0)
            date_source = r.get('date_source', 'unknown')
            
            # Add confidence indicator
            if confidence >= 0.90:
                confidence_marker = "✓"
            elif confidence >= 0.70:
                confidence_marker = "~"
            else:
                confidence_marker = "?"
            
            output_lines.append(f"Published: {pub_date} [{confidence_marker} {date_source}]")
        
        # Compatibility with tests: keep this exact phrase.
        output_lines.append(f"Citation tag (insert after sentence): {citation_tag}")
        output_lines.append("⚠️ WHEN USING THIS RESULT, ADD THIS CITATION: " + citation_tag)
        # Compatibility with tests: keep a literal "Content:" label.
        output_lines.append("Content: " + (snippet or ""))
        output_lines.append(f"\n💡 EXAMPLE: {example_text}")
        output_lines.append("")

    return "\n".join(output_lines)


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is an invalid URL.

    CRITICAL CITATION RULE: After using ANY information from this page, you MUST insert the
    citation tag IMMEDIATELY after the sentence. The citation tag is shown at the top of the result.
    DO NOT collect citations at the end - insert them inline.

    Args:
        url: The URL to fetch the contents of.
    """
    client = _get_tavily_client()
    res = client.extract([url])
    if "failed_results" in res and len(res["failed_results"]) > 0:
        return f"Error: {res['failed_results'][0]['error']}"
    elif "results" in res and len(res["results"]) > 0:
        result = res["results"][0]

        # Normalize URL to handle spaces and special characters
        def normalize_url(url: str) -> str:
            """Normalize URL by encoding spaces and special characters."""
            from urllib.parse import quote, urlparse, urlunparse
            try:
                parsed = urlparse(url)
                normalized_path = quote(parsed.path, safe='/')
                normalized_query = quote(parsed.query, safe='=&')
                normalized_fragment = quote(parsed.fragment, safe='')
                return urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    normalized_path,
                    parsed.params,
                    normalized_query,
                    normalized_fragment
                ))
            except Exception:
                return url.replace(' ', '%20')
        
        normalized_url = normalize_url(url)
        content = result.get("raw_content", "") or ""
        raw = content[:8192]
        raw_html = raw if raw.lstrip().startswith("<") else None
        title = polish_fetched_page_title(
            result.get("title") or url,
            raw,
            url=normalized_url,
            raw_html=raw_html,
        )
        content = content[:4096]
        citation_tag = f"[citation:{title}]({normalized_url})"
        # Create example with citation embedded
        content_preview = content[:150] if len(content) > 150 else content
        example_text = f"根据文章{citation_tag}，{content_preview}..."
        author_line = ""
        org_line = ""
        pub_line = ""
        a_raw = result.get("author") or result.get("authors")
        if a_raw:
            author_line = f"Author: {a_raw}\n" if not isinstance(a_raw, (list, tuple)) else f"Author: {', '.join(str(x) for x in a_raw)}\n"
        org_raw = result.get("publisher") or result.get("source") or result.get("organization")
        if org_raw:
            org_line = f"Organization: {org_raw}\n"
        p_raw = result.get("published_date") or result.get("publishedDate")
        if p_raw:
            pub_line = f"Published: {p_raw}\n"
        else:
            # Try extracting date from raw content if not provided by API
            raw_content = result.get("raw_content", "")
            extracted_date, ext_source, ext_confidence = _extract_published_date_from_content(raw_content)
            if extracted_date:
                confidence_marker = "✓" if ext_confidence >= 0.90 else ("~" if ext_confidence >= 0.70 else "?")
                pub_line = f"Published: {extracted_date} [{confidence_marker} {ext_source}]\n"

        return (
            f"{'='*60}\n"
            f"SOURCE: {citation_tag}\n"
            f"{'='*60}\n"
            f"[WEB PAGE - Insert citation tag INLINE]\n"
            f"🚨 CRITICAL: ALL content below is from THIS SOURCE.\n"
            f"When you use ANY information from this page, you MUST add the citation_tag IMMEDIATELY after that sentence.\n"
            f"Citation tag (insert after sentence): {citation_tag}\n"
            f"⚠️ CITATION TAG (COPY THIS EXACTLY): {citation_tag}\n"
            f"Also add to '## References' section at the END.\n"
            f"\n💡 EXAMPLE USAGE (copy this format):\n"
            f"   {example_text}\n"
            f"\n[WEB PAGE CONTENT - All from this source]\n"
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"{author_line}{org_line}{pub_line}"
            f"\n"
            f"--- Page Content ---\n{content}"
        )
    else:
        return "Error: No results found"
