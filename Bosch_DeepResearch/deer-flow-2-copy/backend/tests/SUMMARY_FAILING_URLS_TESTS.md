## 发布日期提取测试用例 - 完成总结

### 📋 任务概览

为三个失败的链接创建发布日期提取的测试用例：

1. **Facebook Group Post** - `https://www.facebook.com/groups/1242309506256283/posts/1995955427558350`
2. **Zhihu Answer** - `https://www.zhihu.com/answer/3246911414`  
3. **Finviz News** - `https://finviz.com/news/264338/tesla-fsd-approaches-7b-miles-with-25b-on-urban-streets`

### ✅ 完成成果

#### 新增测试文件

1. **`test_citation_examples_failing_urls.py`** (离线单元测试)
   - 16个测试用例
   - 测试Facebook、知乎、Finviz的各种JSON/HTML结构变体
   - 模拟真实平台的embed数据

2. **`test_citation_examples_live_failing_urls.py`** (实时集成测试)
   - 10个测试用例
   - 包含实际网络获取、SSRF防护、模拟响应测试
   - 支持 `CITATION_EXAMPLES_LIVE=1` 启用实时测试

3. **`README_FAILING_URLS_TESTS.md`** (文档)
   - 详细的测试说明和背景
   - 提取逻辑说明
   - 爬取限制和注意事项

#### 测试统计

```
总计: 81 tests passed, 7 skipped
├── test_publication_date.py (45 tests) ✅
├── test_citation_examples_offline.py (17 tests) ✅
├── test_citation_examples_failing_urls.py (16 tests) ✅ [新增]
└── test_citation_examples_live_failing_urls.py (10 tests, 7 skipped) ✅ [新增]

执行时间: 0.34s
```

### 🧪 测试覆盖内容

#### Facebook Group Posts (5个测试)
```python
✅ test_facebook_group_post_with_story_id_extraction
✅ test_facebook_group_post_with_legacy_id_format  
✅ test_facebook_group_post_milliseconds_epoch
✅ test_facebook_post_with_permalink_id
✅ test_citations_facebook_group_post_raw_content
```

**关键特性:**
- 从URL提取Post ID，作为JSON中的搜索锚点
- 支持10位(秒)和13位(毫秒)Unix时间戳
- 处理multiple creation_time字段，优先选择接近Post ID的

#### Zhihu Answers (6个测试)
```python
✅ test_zhihu_answer_with_published_time_json
✅ test_zhihu_answer_with_created_time_field
✅ test_zhihu_answer_published_time_as_string
✅ test_zhihu_answer_iso_published_time
✅ test_zhihu_answer_with_question_id_alternative_path
✅ test_citations_zhihu_answer_raw_content
```

**关键特性:**
- 从URL提取Answer/Question ID
- 支持publishedTime和createdTime字段
- 支持Unix时间戳、字符串格式、ISO 8601
- Markdown内容中的「发布于」/「发表于」模式识别

#### Finviz News Articles (5个测试)
```python
✅ test_finviz_news_article_with_article_published_time
✅ test_finviz_news_article_with_og_published_time
✅ test_finviz_news_article_json_ld_datetime
✅ test_finviz_path_date_extraction_from_url
✅ test_citations_finviz_news_raw_content
```

**关键特性:**
- 使用标准HTML meta标签 (`article:published_time`, `og:published_time`)
- 支持JSON-LD NewsArticle schema
- 支持ISO 8601和其他标准datetime格式
- 优先级链: article:published_time > og:published_time > JSON-LD

#### 集成测试 (3个基础通过 + 7个实时)
```python
✅ test_facebook_url_passes_ssrf_guard (Passed)
✅ test_zhihu_url_passes_ssrf_guard (Passed)
✅ test_finviz_url_passes_ssrf_guard (Passed)

⏭️  test_facebook_group_post_live_http_fetch (Skipped - needs network)
⏭️  test_zhihu_answer_live_http_fetch (Skipped - needs network)
⏭️  test_zhihu_answer_via_jina_reader (Skipped - needs network)
⏭️  test_finviz_news_live_http_fetch (Skipped - needs network)
⏭️  test_facebook_extraction_with_mock_response (Skipped - mock only)
⏭️  test_zhihu_extraction_with_mock_response (Skipped - mock only)
⏭️  test_finviz_extraction_with_mock_response (Skipped - mock only)
```

### 🚀 快速开始

**运行所有新增测试:**
```bash
cd backend
PYTHONPATH=. uv run pytest tests/test_citation_examples_failing_urls.py -v
```

**运行完整测试套件:**
```bash
cd backend
make test
```

**启用实时网络测试:**
```bash
cd backend
CITATION_EXAMPLES_LIVE=1 PYTHONPATH=. uv run pytest \
  tests/test_citation_examples_live_failing_urls.py::test_finviz_news_live_http_fetch -v
```

### 📝 相关文件

**新增文件:**
- `backend/tests/test_citation_examples_failing_urls.py` - 16个离线单元测试
- `backend/tests/test_citation_examples_live_failing_urls.py` - 10个实时集成测试  
- `backend/tests/README_FAILING_URLS_TESTS.md` - 详细文档

**核心实现:**
- `backend/packages/harness/deerflow/utils/publication_date.py` - 日期提取逻辑
- `backend/packages/harness/deerflow/utils/publication_date_fallback_fetch.py` - HTTP fallback

### 🔍 关键改进点

1. **范围定位** - 使用URL提取的ID作为锚点，避免误匹配
2. **多格式支持** - Unix时间戳(秒/毫秒)、ISO 8601、中文日期等
3. **平台检测** - 针对特定平台的优化提取逻辑
4. **Fallback策略** - HTML meta tags → JSON-LD → Markdown内容
5. **错误恢复** - SSRF防护、网络超时处理、graceful degradation

### ⚠️ 已知限制

1. **Facebook** - 通常被爬虫反制，需要代理或官方API
2. **知乎** - 有爬虫防护，推荐使用Jina Reader API
3. **Finviz** - 财务网站，可能需要User-Agent或代理

### 🎯 下一步工作

1. 实施HTTP Fallback + Jina Reader自动重试
2. 集成代理配置以绕过反爬虫机制
3. 缓存已提取日期，避免重复请求
4. 添加更多平台支持（LinkedIn, Medium等）
5. 性能优化和错误日志增强
