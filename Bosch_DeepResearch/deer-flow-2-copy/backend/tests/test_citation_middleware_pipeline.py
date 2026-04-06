"""Citation middleware helpers used by the three-phase orchestrator."""

import json

import pytest

from deerflow.agents.middlewares.citation_middleware import CitationMiddleware
from deerflow.utils.reference_titles import display_domain_for_reference, polish_search_hit_title


@pytest.fixture
def no_http_publication_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep citation JSON tests offline; default config may fetch Reddit/YouTube HTML."""
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


def test_reference_titles_polish_fallback_host():
    assert polish_search_hit_title("", "", url="https://www.example.com/path") == "example.com"


def test_apply_to_markdown_inserts_numeric_and_references():
    md = "Claim one.\n\n## Section\nMore text from the source."
    urls = ["https://example.org/doc"]
    out = CitationMiddleware.apply_to_markdown_with_allowed_urls(md, urls)
    assert "[1]" in out or "example.org" in out
    assert "References" in out or "参考文献" in out


def test_apply_to_markdown_no_urls_unchanged():
    md = "Just prose without citations."
    assert CitationMiddleware.apply_to_markdown_with_allowed_urls(md, []) == md


def test_strip_legacy_sources_sections_removes_sources_keeps_other_headings():
    md = (
        "## Executive\n\nHello.\n\n## Sources\n\n### Primary Sources\n\n"
        "- [A](https://example.com/a)\n\n## Confidence\n\nHigh."
    )
    out = CitationMiddleware._strip_legacy_sources_sections(md)
    assert "## Sources" not in out
    assert "Primary Sources" not in out
    assert "example.com" not in out
    assert "## Executive" in out
    assert "## Confidence" in out


def test_strip_legacy_h3_primary_sources_without_h2_sources():
    md = (
        "## Intro\n\nBody.\n\n### Primary Sources\n\n"
        "- [x](https://example.org/p)\n\n## Next\n\nMore."
    )
    out = CitationMiddleware._strip_legacy_sources_sections(md)
    assert "### Primary Sources" not in out
    assert "example.org" not in out
    assert "## Intro" in out
    assert "## Next" in out


def test_strip_legacy_h3_after_references_heading():
    md = (
        "## 参考文献\n\n[1] \"t\". https://a.test/x\n\n"
        "### Primary Sources\n\n- [z](https://z.test)\n"
    )
    out = CitationMiddleware._strip_legacy_sources_sections(md)
    assert "### Primary" not in out
    assert "z.test" not in out
    assert "## 参考文献" in out
    assert "a.test" in out


def test_ieee_web_reference_line_format():
    line = CitationMiddleware._ieee_web_reference_line(2, "Title. Org. 2024", "https://ex.com/p")
    assert line == '[2] "Title. Org. 2024". https://ex.com/p'
    assert "[Online]" not in line
    assert "Available:" not in line


def test_citations_from_json_raw_content_infers_published_date():
    payload = json.dumps(
        [
            {
                "title": "Example",
                "url": "https://example.com/article",
                "snippet": "Short summary without a year.",
                "raw_content": '<meta property="article:published_time" content="2021-06-10" />',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2021-06-10"
    assert cites[0].get("year") == "2021"


def test_citations_json_url_path_ymd_overrides_wrong_inferred_date():
    """Explicit ``/YYYY/MM/DD/`` in the URL wins over a conflicting ``raw_content`` guess."""
    payload = json.dumps(
        [
            {
                "title": "Review",
                "url": "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review",
                "snippet": "Mind blowing FSD.",
                "raw_content": '<meta property="article:published_time" content="2016-01-01T00:00:00Z" />',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-12-16"
    assert cites[0].get("year") == "2025"


def test_ieee_compose_shows_full_calendar_date_for_iso_datetime():
    """ISO datetimes normalize to calendar day; bibliography shows ``YYYY-MM-DD`` when known."""
    composed = CitationMiddleware._compose_ieee_reference_title(
        "Electrek review",
        "electrek.co",
        "2025-12-16T14:22:00+00:00",
        year_fallback="",
        domain="electrek.co",
    )
    assert "2025-12-16" in composed
    assert composed.count("electrek.co") == 1


def test_ieee_compose_adds_domain_when_org_is_not_hostname():
    composed = CitationMiddleware._compose_ieee_reference_title(
        "Investigation Report",
        "National Highway Traffic Safety Administration",
        "2024-03-01",
        year_fallback="",
        domain="nhtsa.gov",
    )
    assert "2024-03-01" in composed
    assert "nhtsa.gov" in composed
    assert "National Highway" in composed


def test_display_domain_for_reference_strips_www():
    assert display_domain_for_reference("https://www.electrek.co/path", "") == "electrek.co"
    assert display_domain_for_reference("", "WWW.BBC.co.uk") == "bbc.co.uk"


def test_merge_publication_slug_year_wins_when_title_echoes_year():
    """Editorial slug ``…-best-tech-2026`` overrides a conflicting inferred year when the title cites that year."""
    pub, fb = CitationMiddleware._merge_publication_date_with_url(
        "2023-01-01",
        "https://www.motortrend.com/features/tesla-fsd-driver-assistance-system-best-tech-2026",
        title="Best Tech 2026: Tesla FSD",
    )
    assert pub == ""
    assert fb == "2026"


def test_year_from_snippet_upper_band_prefers_article_year_over_history():
    """Older narrative year + newer publish year: upper band min still picks the recent article year."""
    y = CitationMiddleware._year_from_published_or_snippet(
        "",
        "In 2016 Tesla shipped early AP; by 2025 FSD v12 replaced most hand-written code.",
        "",
        "https://www.fredpope.com/blog/machine-learning/tesla-fsd-12",
    )
    assert y == "2025"


def test_year_from_snippet_upper_band_avoids_future_year_sidebar_spam():
    """Many repeated future/related years (e.g. 2026) must not beat the actual 2023 publish context."""
    sn = (
        "September 2023 — Tesla FSD v12 shifts away from rules-based coding. "
        + "Related: 2026 Roadster. 2026 Cybertruck. 2026 updates. " * 6
    )
    y = CitationMiddleware._year_from_published_or_snippet("", sn, "", "https://www.teslarati.com/x")
    assert y == "2023"


def test_epochtimes_url_path_yy_mm_dd_calendar_hint():
    cal, y, strength = CitationMiddleware._date_hints_from_url_path(
        "https://www.epochtimes.com/b5/25/10/7/n14611472.htm"
    )
    assert strength == "path_ymd"
    assert cal == "2025-10-07"
    assert y == "2025"


def test_normalize_url_strips_trailing_underscore_from_youtube_v():
    u = CitationMiddleware._normalize_url("https://www.youtube.com/watch?v=_W2W1f_Dmc8_")
    assert "_W2W1f_Dmc8_" not in u
    assert "_W2W1f_Dmc8" in u


def test_normalize_url_strips_prezi_slug_date_artifact():
    u = "https://prezi.com/p/auywp8hsa7gw/the-impact-of-pure-vision-systems-on-autonomous-driving_date"
    n = CitationMiddleware._normalize_url(u)
    assert "_date" not in n
    assert n.endswith("autonomous-driving")


def test_normalize_url_unwraps_facebook_l_php_u_param():
    inner = "https://www.facebook.com/groups/609768003478912/posts/1238019363987103"
    wrapped = (
        "https://l.facebook.com/l.php?u="
        + "https%3A%2F%2Fwww.facebook.com%2Fgroups%2F609768003478912%2Fposts%2F1238019363987103"
        + "&h=abc"
    )
    n = CitationMiddleware._normalize_url(wrapped)
    assert "l.facebook.com" not in n
    assert n.startswith("https://www.facebook.com/")
    assert "1238019363987103" in n


def test_normalize_url_facebook_mobile_host_maps_to_www():
    u = "https://m.facebook.com/groups/609768003478912/posts/1238019363987103"
    n = CitationMiddleware._normalize_url(u)
    assert n.startswith("https://www.facebook.com/")
    assert "m.facebook.com" not in n


def test_polish_search_hit_title_strips_trailing_facebook_com_on_fb_url():
    t = polish_search_hit_title(
        "Why does Tesla's FSD slow down in certain facebook.com",
        "",
        url="https://www.facebook.com/groups/x/posts/1",
    )
    assert "facebook.com" not in t.lower()


def test_polish_search_hit_title_facebook_fsd_lane_change_user_reported_shape():
    """Regression: SERP title must not end with ``- Facebook. facebook.com`` when URL is the post."""
    fb = "https://www.facebook.com/groups/teslamodelyownerclub/posts/1559499865078537"
    raw = "How does fsd decide when to change lanes? - Facebook. facebook.com"
    t = polish_search_hit_title(raw, "", url=fb)
    assert "facebook.com" not in t.lower()
    assert not t.lower().endswith("facebook")
    assert "change lanes" in t.lower()


def test_polish_search_hit_title_facebook_strips_dash_facebook_only_suffix():
    t = polish_search_hit_title(
        "Some group discussion - Facebook",
        "",
        url="https://www.facebook.com/groups/foo/posts/1",
    )
    assert t == "Some group discussion"


def test_build_reference_section_facebook_tesla_modely_group_no_duplicate_facebook_labels():
    """Vanity group slug becomes ``Facebook Group · …``; no bare ``. facebook.com`` org tail."""
    fb = "https://www.facebook.com/groups/teslamodelyownerclub/posts/1559499865078537"
    tool_citations = [
        {
            "title": "How does fsd decide when to change lanes? - Facebook. facebook.com",
            "url": CitationMiddleware._normalize_url(fb),
            "tool": "web_search",
            "site": "facebook.com",
            "year": "",
            "snippet": "",
            "author_org": "",
            "published_date": "",
        }
    ]
    # Inline title may echo noisy SERP text from the model.
    inline_pairs = [
        (
            "How does fsd decide when to change lanes? - Facebook. facebook.com",
            fb,
        )
    ]
    section, _ = CitationMiddleware._build_reference_section(
        tool_citations,
        {CitationMiddleware._normalize_url(fb)},
        inline_citation_pairs=inline_pairs,
    )
    assert "1559499865078537" in section
    assert ". facebook.com" not in section
    assert "Facebook. facebook.com" not in section
    assert "facebook.com/groups/teslamodelyownerclub" in section
    assert "Facebook Group · teslamodelyownerclub" in section


def test_build_reference_section_facebook_user_night_post_includes_group_org_and_year():
    """Regression: [8]-style row should show group org + year, not title-only."""
    fb = "https://www.facebook.com/groups/teslamodelyownerclub/posts/1627079344987255"
    nu = CitationMiddleware._normalize_url(fb)
    tool_citations = [
        {
            "title": "How does fsd perform at night",
            "url": nu,
            "tool": "web_search",
            "site": "facebook.com",
            "year": "2026",
            "snippet": "",
            "author_org": "",
            "published_date": "",
            "raw_content": "",
        }
    ]
    section, _ = CitationMiddleware._build_reference_section(
        tool_citations,
        {nu},
        inline_citation_pairs=[("How does fsd perform at night", fb)],
    )
    assert "1627079344987255" in section
    assert "Facebook Group · teslamodelyownerclub" in section
    assert "2026" in section


def test_build_reference_section_facebook_numeric_group_story_publish_time_fills_year():
    """Regression: [9]-style numeric group id + embedded JSON timestamp → year in IEEE title."""
    fb = "https://www.facebook.com/groups/3258771700819965/posts/25635668892703593"
    nu = CitationMiddleware._normalize_url(fb)
    raw = '{"story_publish_time":1696000000}'
    tool_citations = [
        {
            "title": "fsd performance issues in rainy conditions",
            "url": nu,
            "tool": "web_search",
            "site": "facebook.com",
            "year": "",
            "snippet": "",
            "author_org": "",
            "published_date": "",
            "raw_content": raw,
        }
    ]
    section, _ = CitationMiddleware._build_reference_section(
        tool_citations,
        {nu},
        inline_citation_pairs=[("fsd performance issues in rainy conditions", fb)],
    )
    assert "25635668892703593" in section
    assert "Facebook Group" in section
    # Org line is plain ``Facebook Group``, not the numeric group id as a label.
    assert "Facebook Group · 3258771700819965" not in section
    assert "2023" in section


def test_build_reference_section_zhihu_zhuanlan_empty_published_date_filled_from_raw():
    """Bibliography must re-infer Zhihu publish date from ``raw_content`` like Facebook."""
    zh = "https://zhuanlan.zhihu.com/p/572974435"
    nu = CitationMiddleware._normalize_url(zh)
    raw = (
        '{"entities":{"articles":{"572974435":'
        '{"title":"x","publishedTime":1666262400000}}}}'
    )
    tool_citations = [
        {
            "title": "Tesla FSD analysis",
            "url": nu,
            "tool": "web_search",
            "site": "zhuanlan.zhihu.com",
            "year": "",
            "snippet": "",
            "author_org": "",
            "published_date": "",
            "raw_content": raw,
        }
    ]
    section, _ = CitationMiddleware._build_reference_section(
        tool_citations,
        {nu},
        inline_citation_pairs=[("Tesla FSD analysis", zh)],
    )
    assert "2022" in section
    assert "zhuanlan.zhihu.com" in section


def test_build_reference_section_facebook_does_not_append_site_as_org():
    tool_url = CitationMiddleware._normalize_url(
        "https://l.facebook.com/l.php?u="
        + "https%3A%2F%2Fm.facebook.com%2Fgroups%2Fg%2Fposts%2F123"
    )
    tool_citations = [
        {
            "title": "Why does Tesla FSD brake",
            "url": tool_url,
            "tool": "web_search",
            "site": "facebook.com",
            "year": "",
            "snippet": "",
            "author_org": "",
            "published_date": "",
        }
    ]
    inline_pairs = [
        (
            "Why does Tesla FSD brake",
            "https://www.facebook.com/groups/g/posts/123",
        )
    ]
    section, _ = CitationMiddleware._build_reference_section(
        tool_citations,
        {tool_url},
        inline_citation_pairs=inline_pairs,
    )
    assert "https://www.facebook.com/" in section
    assert ". facebook.com" not in section
    assert "Facebook Group · g" in section
    assert '"Why does Tesla FSD brake"' in section or "Why does Tesla FSD brake" in section


def test_canonicalize_urls_in_citation_markdown_collapses_llm_spaces_before_grounding():
    """Spaced URLs in [citation:](...) must rewrite so allowed_tool_urls grounding hits."""
    raw = (
        "[citation:Electrek](https://electrek.co/2025/12/16/tesla-full-self-driving-v14 - review/)."
    )
    fixed = CitationMiddleware._canonicalize_urls_in_citation_markdown(raw)
    assert " " not in fixed.split("](", 1)[1].split(")")[0]
    allowed = {
        CitationMiddleware._normalize_url("https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review")
    }
    kept = CitationMiddleware._strip_ungrounded_inline_citations(fixed, allowed)
    assert "[citation:Electrek]" in kept
    assert "electrek.co" in kept


def test_clean_broken_urls_preserves_citation_links_with_spaces_in_url():
    """LLM often inserts spaces around hyphens in URLs; bare-URL protect must not split them."""
    raw = (
        "[citation:Electrek](https://electrek.co/2025/12/16/tesla-full-self-driving-v14 - review/). "
        "Tail."
    )
    out = CitationMiddleware._clean_broken_urls(raw)
    assert "__PROTECTED_" not in out
    assert "](https://electrek.co/2025/12/16/tesla-full-self-driving-v14 - review/)" in out
    assert " - review/)." in out


def test_normalize_url_strips_unicode_whitespace_in_http_path():
    from urllib.parse import urlparse

    messy = "https://electrek.co/2025/12/16/tesla-full-self-driving-v14\u00a0- review/"
    n = CitationMiddleware._normalize_url(messy)
    assert "\u00a0" not in n
    assert " " not in urlparse(n).path.replace("/", "")
    assert "tesla-full-self-driving-v14-review" in n


def test_citations_json_reddit_snippet_year_not_used_without_created_utc(
    no_http_publication_fetch,
) -> None:
    """Correct output for this sample: no invented date when ``created_utc`` is absent."""
    payload = json.dumps(
        [
            {
                "title": "Thread",
                "url": "https://www.reddit.com/r/TeslaFSD/comments/1rqhwdz/brief_frequent",
                "snippet": "Posted in 2026 — brand new Model Y failures.",
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert cites[0].get("year") == ""
    assert cites[0].get("published_date") == ""


def test_citations_json_reddit_with_created_utc_in_raw():
    payload = json.dumps(
        [
            {
                "title": "Thread",
                "url": "https://www.reddit.com/r/x/comments/abc/title",
                "snippet": "x",
                "raw_content": '{"created_utc": 1700000000}',
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert cites[0].get("published_date") == "2023-11-14"
    assert cites[0].get("year") == "2023"


def test_date_hints_skip_semantic_scholar_hex_paper_slug():
    cal, y, s = CitationMiddleware._date_hints_from_url_path(
        "https://www.semanticscholar.org/paper/Title/4ec5170c3c9735f2dfae8f4ef23bc8d28a017397"
    )
    assert (cal, y, s) == ("", "", "")


def test_citations_json_slug_suffix_year_when_no_publish_date():
    """Fixture: slug carries ``2026``; ``article:published_time`` gives full calendar ground truth."""
    payload = json.dumps(
        [
            {
                "title": "Best Tech",
                "url": "https://www.motortrend.com/features/tesla-fsd-driver-assistance-system-best-tech-2026",
                "snippet": "Tesla FSD is impressive.",
                "raw_content": (
                    '<meta property="article:published_time" content="2026-01-02T12:00:00Z" />'
                ),
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2026-01-02"
    assert cites[0].get("year") == "2026"


# --- Regression: real URLs from user-reported Tesla / FSD citation batches ---


@pytest.mark.parametrize(
    "url,expected_cal,expected_y,expected_strength",
    [
        (
            "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review",
            "2025-12-16",
            "2025",
            "path_ymd",
        ),
        (
            "https://teslanorth.com/2026/03/19/nhtsa-probes-tesla-fsd-performance-in-glare-and-low-visibility",
            "2026-03-19",
            "2026",
            "path_ymd",
        ),
        (
            "https://www.fool.com/investing/2026/03/21/should-tesla-be-worried-about-nhtsa-fsd-yes",
            "2026-03-21",
            "2026",
            "path_ymd",
        ),
        (
            "https://oecd.ai/en/incidents/2025-09-02-b4ea",
            "2025-09-02",
            "2025",
            "path_ymd",
        ),
        (
            "https://insideevs.com/news/738204/tesla-pure-vision-camera-only",
            "",
            "",
            "",
        ),
        (
            "https://www.notateslaapp.com/news/2571/un",
            "",
            "",
            "",
        ),
        (
            "https://www.fredpope.com/blog/machine-learning/tesla-fsd-12",
            "",
            "",
            "",
        ),
        (
            "https://medium.com/techlife/how-tesla-rewrote-the-driving-rules-the-revolutionary-move-that-will-change-everything-35551ba6bbde",
            "",
            "",
            "",
        ),
        (
            "https://teslamotorsclub.com/tmc/threads/fsd-beta-perception-vs-path-planning-driving-logic.246140",
            "",
            "",
            "",
        ),
        (
            "https://www.mattpopovich.com/posts/tesla-fsd-vs-25mph-well-marked-curvy-road",
            "",
            "",
            "",
        ),
        (
            "https://www.youtube.com/watch?v=w43uWn6Zhcg",
            "",
            "",
            "",
        ),
        (
            "https://www.51fusa.com/client/information/informationdetail/id/2045.html",
            "",
            "",
            "",
        ),
    ],
)
def test_date_hints_from_user_reported_citation_urls(
    url: str,
    expected_cal: str,
    expected_y: str,
    expected_strength: str,
) -> None:
    cal, y, strength = CitationMiddleware._date_hints_from_url_path(url)
    assert (cal, y, strength) == (expected_cal, expected_y, expected_strength)


def test_citations_json_user_batch_path_dates_merge_into_published_date() -> None:
    """Search-style JSON rows: URL path dates must land in ``published_date`` / ``year``."""
    payload = json.dumps(
        [
            {
                "title": "Electrek FSD v14",
                "url": "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review",
                "snippet": "Review excerpt.",
            },
            {
                "title": "TeslaNorth NHTSA",
                "url": "https://teslanorth.com/2026/03/19/nhtsa-probes-tesla-fsd-performance-in-glare-and-low-visibility",
                "snippet": "Probe story.",
            },
            {
                "title": "OECD incident",
                "url": "https://oecd.ai/en/incidents/2025-09-02-b4ea",
                "snippet": "Incident record.",
            },
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 3
    assert cites[0]["published_date"] == "2025-12-16"
    assert cites[1]["published_date"] == "2026-03-19"
    assert cites[2]["published_date"] == "2025-09-02"


def test_citations_json_youtube_still_some_issues_tesla_fsd_supervised_ieee_row(
    no_http_publication_fetch,
) -> None:
    """Regression: user batch [6] Still Some Issues… — YouTube. youtube.com. 2024."""
    raw = (
        "<script>var ytInitialPlayerResponse = "
        '{"microformat":{"playerMicroformatRenderer":'
        '{"publishDate":"2025-01-01T08:00:00Z"}}};</script>'
    )
    payload = json.dumps(
        [
            {
                "title": "Still Some Issues on the Highway (Tesla FSD Supervised) - YouTube",
                "url": "https://www.youtube.com/watch?v=w43uWn6Zhcg",
                "snippet": "Highway driving with FSD Supervised.",
                "raw_content": raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == "2025-01-01"
    assert cites[0].get("year") == "2025"
    refs = CitationMiddleware._format_references_ieee(cites)
    assert "Still Some Issues on the Highway (Tesla FSD Supervised) - YouTube" in refs
    assert "youtube.com. 2025" in refs
    assert "w43uWn6Zhcg" in refs


def test_citations_json_youtube_snippet_date_no_raw_content(
    no_http_publication_fetch,
) -> None:
    """YouTube snippet with 'Month D, YYYY —' prefix extracts date without raw_content or HTTP."""
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD vs. Rainy Chaos: The Results Are Insane! - YouTube",
                "url": "https://www.youtube.com/watch?v=a5Fe6HV3PRM",
                "snippet": "Mar 27, 2025 — Tesla FSD navigates rainy conditions perfectly. Watch how the system handles the storm.",
            },
            {
                "title": "I Pushed Tesla FSD V14 to Its Limits in Dangerous Weather - YouTube",
                "url": "https://www.youtube.com/watch?v=6GCaDg--FDg",
                "snippet": "January 2, 2026 — Testing Tesla FSD V14 in extreme dangerous weather conditions.",
            },
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 2
    assert cites[0].get("published_date") == "2025-03-27", f"got {cites[0].get('published_date')!r}"
    assert cites[0].get("year") == "2025"
    assert cites[1].get("published_date") == "2026-01-02", f"got {cites[1].get('published_date')!r}"
    assert cites[1].get("year") == "2026"
    refs = CitationMiddleware._format_references_ieee(cites)
    assert "youtube.com. 2025" in refs
    assert "youtube.com. 2026" in refs


def test_citations_json_youtube_snippet_relative_date(
    no_http_publication_fetch,
) -> None:
    """YouTube snippet with 'N months/weeks/days ago' infers approximate date from today."""
    from datetime import date, timedelta

    today = date.today()
    payload = json.dumps(
        [
            {
                "title": "Tesla FSD Rainy Chaos - YouTube",
                "url": "https://www.youtube.com/watch?v=a5Fe6HV3PRM",
                "snippet": "3 months ago — Tesla FSD navigates rainy conditions perfectly.",
            },
            {
                "title": "Tesla FSD V14 Limits - YouTube",
                "url": "https://www.youtube.com/watch?v=6GCaDg--FDg",
                "snippet": "5 days ago • Testing Tesla FSD V14 in extreme weather.",
            },
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 2
    expected_3mo = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    expected_5d = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    assert cites[0].get("published_date") == expected_3mo, f"got {cites[0].get('published_date')!r}"
    assert cites[1].get("published_date") == expected_5d, f"got {cites[1].get('published_date')!r}"
    # Year must be present in references
    refs = CitationMiddleware._format_references_ieee(cites)
    assert f"youtube.com. {today.year}" in refs or f"youtube.com. {today.year - 1}" in refs



def test_format_references_ieee_user_style_electrek_and_insideevs() -> None:
    """End-to-end IEEE rows shaped like user-visible ``[n] "Title". …`` output."""
    cites = [
        {
            "title": "I tested Tesla's latest FSD v14 | Electrek",
            "url": "https://electrek.co/2025/12/16/tesla-full-self-driving-v14-review",
            "snippet": "",
            "tool": "web_search",
            "site": "electrek.co",
            "year": "2025",
            "author_org": "",
            "published_date": "2025-12-16",
        },
        {
            "title": "Tesla Bet On Pure Vision | InsideEVs",
            "url": "https://insideevs.com/news/738204/tesla-pure-vision-camera-only",
            "snippet": "Camera-only discussion.",
            "tool": "web_search",
            "site": "insideevs.com",
            "year": "2022",
            "author_org": "",
            "published_date": "2022-11-30",
        },
    ]
    refs = CitationMiddleware._format_references_ieee(cites)
    assert "2025-12-16" not in refs
    assert "electrek.co. 2025" in refs
    assert "electrek.co" in refs
    assert "insideevs.com" in refs
    assert "insideevs.com. 2022" in refs
    assert "738204" in refs


def test_normalize_url_prezi_user_exact_slug() -> None:
    """Exact Prezi URL shape from user report (trailing ``_date`` artifact)."""
    raw = (
        "https://prezi.com/p/auywp8hsa7gw/"
        "the-impact-of-pure-vision-systems-on-autonomous-driving_date"
    )
    n = CitationMiddleware._normalize_url(raw)
    assert "autonomous-driving_date" not in n
    assert "the-impact-of-pure-vision-systems-on-autonomous-driving" in n


def test_citations_json_reddit_teslafsd_urls_no_snippet_year(
    no_http_publication_fetch,
) -> None:
    """Correct output: empty calendar/year (snippet prose must not substitute for ``created_utc``)."""
    for path in (
        "1mayn1q/dangerous_fsd_failures",
        "1qse2jz/tesla_fsd_error_running_a_red_light",
    ):
        payload = json.dumps(
            [
                {
                    "title": "Reddit thread",
                    "url": f"https://www.reddit.com/r/TeslaFSD/comments/{path}",
                    "snippet": "Discussion from 2026 about failures.",
                }
            ],
            ensure_ascii=False,
        )
        cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
        assert cites[0].get("year") == ""
        assert cites[0].get("published_date") == ""


def test_ieee_reference_line_includes_year_after_domain():
    """``published_date`` as YYYY-MM-DD yields a 4-digit year after site (domain) in references."""
    cites = [
        {
            "title": "Python 3.13.1",
            "url": "https://www.python.org/downloads/release/python-3131/",
            "snippet": "Maintenance release.",
            "tool": "web_search",
            "site": "python.org",
            "year": "2024",
            "author_org": "",
            "published_date": "2024-12-03",
        }
    ]
    refs = CitationMiddleware._format_references_ieee(cites)
    assert "2024-12-03" not in refs
    assert "python.org. 2024" in refs
    assert "https://www.python.org/downloads/release/python-3131" in refs


@pytest.mark.parametrize(
    "url,synthetic_raw,expected_published",
    [
        (
            "https://www.youtube.com/watch?v=hlb5WRN0SGg",
            'ytInitialPlayerResponse = {"microformat":{"playerMicroformatRenderer":'
            '{"publishDate":"2023-09-01T10:00:00Z"}}}',
            "2023-09-01",
        ),
        (
            "https://www.youtube.com/watch?v=w43uWn6Zhcg",
            (
                "<script>var ytInitialPlayerResponse = "
                '{"microformat":{"playerMicroformatRenderer":'
                '{"publishDate":"2025-01-01T12:00:00Z"}}};</script>'
            ),
            "2025-01-01",
        ),
        (
            "https://www.reddit.com/r/TeslaFSD/comments/1jwxbfd/mark_rober_debunk_heavy_rain",
            '{"createdAt":"2025-02-10T08:00:00.000Z","title":"Heavy Rain Test"}',
            "2025-02-10",
        ),
        (
            "https://www.reddit.com/r/TeslaModelY/comments/1s2vivt/fsd_left_turn",
            '{"created_utc": 1730000000}',
            "2024-10-27",
        ),
        (
            "https://www.reddit.com/r/TeslaFSD/comments/1rqhwdz/fsd_failures",
            '{"created_utc": 1720000000.0}',
            "2024-07-03",
        ),
        (
            "https://www.reddit.com/r/teslamotors/comments/1jakb3w/fsd_china",
            '{"createdAt":"2025-03-15T12:00:00.000Z"}',
            "2025-03-15",
        ),
        (
            "https://www.facebook.com/groups/3258771700819965/posts/25635668892703593",
            '{"story_publish_time":1696000000}',
            "2023-09-29",
        ),
        (
            "https://www.facebook.com/groups/1743863285929939/posts/4242925042690405",
            '{"creation_time":1700000000}',
            "2023-11-14",
        ),
        (
            "https://dianawolftorres.substack.com/p/full-self-driving-gone-wrong-tesla",
            '<script type="application/ld+json">'
            '{"@type":"BlogPosting","datePublished":"2023-12-05T12:00:00.000Z"}'
            "</script>",
            "2023-12-05",
        ),
        (
            "https://www.cybertruckownersclub.com/forum/threads/phantom-braking-and-steering.51388/",
            '<meta property="article:published_time" content="2024-05-20T12:00:00+00:00" />',
            "2024-05-20",
        ),
        (
            "https://www.fredpope.com/blog/machine-learning/tesla-fsd-12",
            '<meta property="article:published_time" content="2018-03-12T00:00:00Z" />',
            "2018-03-12",
        ),
        (
            "https://www.teslaacessories.com/blogs/news/the-fsd-v12-4-paradigm-shift",
            '<meta property="article:published_time" content="2024-06-15T00:00:00Z" />',
            "2024-06-15",
        ),
        (
            "https://www.teslaacessories.com/blogs/news/nhtsa-escalates-fsd-visibility-investigation-"
            "what-it-means-for-3.2-million-us-tesla-owners-and-how-to-stay-safe",
            (
                '<script type="application/ld+json">'
                '{"@type":"WebPage","datePublished":"2020-01-01T00:00:00Z"}'
                "</script>"
                '{"published_at":"2026-03-19T12:00:00-05:00"}'
            ),
            "2026-03-19",
        ),
        (
            "https://teslamotorsclub.com/tmc/threads/fsdb-acceleration-deceleration-rate-hmm.325289",
            '<meta property="article:published_time" content="2024-08-01T00:00:00Z" />',
            "2024-08-01",
        ),
        (
            "https://electrek.co/2026/03/19/nhtsa-upgrades-tesla-fsd-visibility",
            "",
            "2026-03-19",
        ),
        (
            "https://zhuanlan.zhihu.com/p/572974435",
            (
                '{"entities":{"articles":{"572974435":'
                '{"title":"《2022年Tesla AI Day——FSD技术进展分析总结》",'
                '"publishedTime":1666262400000}}}}'
            ),
            "2022-10-20",
        ),
        (
            "https://insideevs.com/news/738204/tesla-pure-vision-camera-only",
            '<meta property="article:published_time" content="2022-11-30T12:00:00Z" />',
            "2022-11-30",
        ),
        (
            "https://www.notateslaapp.com/news/2571/un",
            '<meta property="article:published_time" content="2023-04-10T00:00:00Z" />',
            "2023-04-10",
        ),
        (
            "https://medium.com/techlife/how-tesla-rewrote-the-driving-rules-the-revolutionary-move-that-will-change-everything-35551ba6bbde",
            '{"firstPublishedAt":1704067200000}',
            "2024-01-01",
        ),
        (
            "https://teslamotorsclub.com/tmc/threads/fsd-beta-perception-vs-path-planning-driving-logic.246140",
            '<meta property="article:published_time" content="2021-09-15T00:00:00Z" />',
            "2021-09-15",
        ),
        (
            "https://www.mattpopovich.com/posts/tesla-fsd-vs-25mph-well-marked-curvy-road",
            '<meta property="article:published_time" content="2024-02-01T00:00:00Z" />',
            "2024-02-01",
        ),
        (
            "https://www.51fusa.com/client/information/informationdetail/id/2045.html",
            '<meta property="article:published_time" content="2021-08-20T08:00:00+08:00" />',
            "2021-08-20",
        ),
        (
            "https://www.eeworld.com.cn/qcdz/eic682225.html",
            "发布者：x最新更新时间：2024-10-29 来源: 智驾最前沿",
            "2024-10-29",
        ),
    ],
)
def test_citations_json_social_and_blog_shapes_infer_publish_date(
    url: str,
    synthetic_raw: str,
    expected_published: str,
) -> None:
    """User-reported URLs: fixture ``raw_content`` defines ground-truth ``YYYY-MM-DD``.

    Synthetic HTML/JSON mirrors typical fetch output. Empty ``raw`` still uses URL path
    hints (Electrek-style). Every case must match both ``published_date`` and ``year``.
    """
    payload = json.dumps(
        [
            {
                "title": "Synthetic fixture",
                "url": url,
                "snippet": "Snippet without a year.",
                "raw_content": synthetic_raw,
            }
        ],
        ensure_ascii=False,
    )
    cites = CitationMiddleware._citations_from_plain_json_search_results(payload, "web_search")
    assert len(cites) == 1
    assert cites[0].get("published_date") == expected_published
    assert cites[0].get("year") == (expected_published[:4] if expected_published else "")
