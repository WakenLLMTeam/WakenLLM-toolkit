"""Tavily web_search: Zhihu rows get a Jina Reader date pass when Tavily extract misses."""

import json
from unittest.mock import MagicMock, patch

from deerflow.community.tavily import tools as tavily_tools


def test_web_search_zhihu_row_enriched_via_jina_after_empty_extract() -> None:
    tools = tavily_tools

    web_tool = MagicMock()
    web_tool.model_extra = {
        "api_key": "dummy-key",
        "max_results": 5,
        "include_raw_content": False,
        "enrich_dates_with_extract": True,
        "enrich_dates_max_urls": 8,
        "enrich_zhihu_dates_with_jina": True,
    }
    app_cfg = MagicMock()
    app_cfg.get_tool_config = lambda name: web_tool if name == "web_search" else None

    class FakeTavily:
        def search(self, query, **kw):
            return {
                "results": [
                    {
                        "title": "《2022年Tesla AI Day》",
                        "url": "https://zhuanlan.zhihu.com/p/572974435",
                        "content": "FSD 分析摘要",
                    }
                ]
            }

        def extract(self, urls, **kw):
            return {"results": [{"url": urls[0], "raw_content": ""}]}

    with (
        patch.object(tools, "get_app_config", return_value=app_cfg),
        patch.object(tools, "_get_tavily_client", return_value=FakeTavily()),
        patch.object(
            tools,
            "infer_publication_calendar_via_jina_reader",
            return_value="2022-10-20",
        ) as mock_jina,
    ):
        out = tools.web_search_tool.invoke({"query": "tesla zhihu zhuanlan"})

    mock_jina.assert_called()
    rows = json.loads(out)
    assert len(rows) == 1
    assert rows[0].get("published_date") == "2022-10-20"
