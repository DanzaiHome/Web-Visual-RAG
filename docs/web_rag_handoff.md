# Visual Web RAG 网页检索与爬取优化交接文档

更新时间：2026-05-05

## 1. 当前项目定位

本项目是一个面向图文问答的 Visual Web RAG 系统。用户输入图片和问题后，系统会：

1. 用视觉语言模型生成搜索 query。
2. 通过博查 Web Search 获取候选网页 URL。
3. 对候选 URL 抓取真实网页正文。
4. 对正文做清洗、去重、质量评估和 chunk 检索。
5. 结合图片相似度、网页质量、来源可信度等信号排序证据。
6. 将最终证据包和原始图片、问题一起交给 VLM 生成答案。

当前优化的核心方向是：把博查从“最终摘要证据源”改造成“候选 URL 发现器”，再由项目自己抓取网页正文并构建更可靠的 RAG evidence context。

## 2. 当前代码结构

主要源码已经迁移到 `src/rag_v1` 包结构：

- `src/rag_v1/services/web_search.py`：博查搜索、候选网页整理、网页抓取接入。
- `src/rag_v1/services/web_page_fetcher.py`：网页抓取、HTML 正文抽取、清洗、图片和元信息提取、缓存。
- `src/rag_v1/services/image_checker.py`：CLIP 图片 embedding 相似度评分。
- `src/rag_v1/retrieval/chunk_extractor.py`：网页文本切 chunk、文本/多模态 chunk 召回。
- `src/rag_v1/pipeline/rag_pipeline.py`：RAG 主流程、网页去重、证据评分、证据 rescue、context 聚合。
- `src/rag_v1/config.py`：API、模型、CLIP server、网页抓取配置。
- `tests/`：当前单元测试目录。

兼容旧入口仍保留：

- `pipeline.py`
- `clip_server.py`
- `config.py`

## 3. 已完成的主要优化

### 3.1 博查搜索从摘要源升级为候选 URL 源

之前：

```text
Bocha Search -> summary/snippet -> chunk -> answer
```

现在：

```text
Bocha Search -> candidate URL -> WebPageFetcher 抓正文 -> 清洗正文 -> chunk -> answer
```

如果网页正文抓取成功且质量足够，会使用 `content_source = web_page`。如果抓取失败、正文过短或质量太低，会回退到 `bocha_summary_fallback`。

### 3.2 新增网页抓取器 `WebPageFetcher`

新增 `src/rag_v1/services/web_page_fetcher.py`，支持：

- URL canonicalization，去掉 `utm_`、`fbclid`、`ref` 等跟踪参数。
- 请求超时、重试、HTTP 错误处理。
- HTML / plain text 页面处理。
- meta charset 识别，改善中文乱码问题。
- 标题、站点名、发布时间、语言、图片 URL 提取。
- 主正文抽取，过滤 `nav`、`footer`、`aside`、广告、推荐、版权、cookie 等噪声。
- listing 页面识别。
- 正文质量评分 `quality_score`。
- 页面缓存，避免重复下载和重复解析。

默认缓存目录：

```text
.cache/web_pages
```

### 3.3 图片相似度从硬过滤改成软评分

原始逻辑更偏向：

```text
图片相似度 < IMAGE_MATCH_THRESHOLD -> 删除网页
```

现在改为软评分和诊断信号：

- `strong_match`
- `weak_match`
- `low_similarity`
- `no_images`
- `failed`
- `unavailable`

图片相似度高会加分，低相似度或无图不一定直接删除。这样可以避免新闻网页因为配图、logo、社交卡片不匹配而误杀高质量文本证据。

### 3.4 新增高质量网页正文 rescue

如果图片过滤后只剩弱证据，系统会从被图片过滤掉的文档中救回高质量网页正文。

典型日志：

```text
Rescued 1 high-quality webpage body doc(s) after image filtering left only weak evidence
Evidence source: webpage body, cache, quality=..., image-filter rescue
```

这对“人物 + 最近新闻/政策”类问题很重要，因为这类网页的页面图片经常不直接对应输入人物图。

### 3.5 新增综合 evidence score

现在 chunk 排序不再只看 embedding 相似度，而是综合多个信号：

```text
semantic_score
content_source adjustment
web_fetch_quality_score
image_similarity_bonus
source_reliability_bonus
freshness_bonus
listing_penalty
```

运行日志中会输出 `score_breakdown`，方便调试为什么某个 chunk 排在前面。

### 3.6 来源可信度加权

`rag_pipeline.py` 中加入了来源可信度评分。官方站点、政府站点、白宫官网等会获得额外加分。低质量 fallback、listing 页面、观点页等会被降权。

这使得类似 White House ballroom 的问题中，`whitehouse.gov` 能稳定排在较前位置。

### 3.7 多来源、多样性和去重

当前已经支持：

- canonical URL 去重。
- 近重复正文去重。
- 每个 URL / 域名限制 chunk 数。
- 抑制低相关 listing chunk。
- context 按文档聚合展示，而不是把所有 chunk 平铺。

在 White House ballroom 测试中，最终 context 能保留 White House、OregonLive、Xinhuanet 等多个来源。

### 3.8 图片 URL 清洗

之前有非图片资源混入 `image_urls`，例如：

```text
.woff2
.js
captcha script
tracking script
```

现在已加规则过滤非图片资源、小图标、字体、脚本、CSS、favicon 等，避免 CLIP server 对非图片 URL 报错。

### 3.9 chunk 切分增强

`chunk_extractor.py` 已增强：

- 英文按词切分。
- 中文无空格文本用重叠字符 chunk。
- 保留段落边界。
- 处理“一行一个词”的网页抽取异常。
- 避免中文正文被错误吞掉。

## 4. 当前验证结果

测试命令：

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

当前结果：

```text
Ran 31 tests in 1.099s
OK
```

覆盖测试包括：

- 网页正文抽取
- URL canonicalization
- meta charset 编码处理
- 页面缓存
- listing 页面识别
- 图片 URL 过滤
- 图片相似度软评分
- evidence rescue
- evidence score breakdown
- 多来源 chunk 选择
- context 格式化
- 中文/英文 chunk 切分

## 5. 已验证的示例效果

### 5.1 Trump recent policy 问题

命令：

```powershell
python -m rag_v1.apps.pipeline --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal
```

观察到的效果：

- 能抓到真实网页正文。
- 缓存生效，`web_fetch_from_cache=True`。
- 对低图片相似度但正文质量较高的网页能进行 rescue。
- 最终能回答 TrumpIRA.gov、退休储蓄计划扩展等政策。

当前不足：

- 有时证据覆盖面偏窄，可能只保留一个主来源。
- 对“recent policies”这种复数问题，还应尽量召回多个不同政策和来源。

### 5.2 White House State Ballroom 问题

图片：

```text
pictures/white_house_state_ballroom_east_colonnade.jpeg
```

命令：

```powershell
python -m rag_v1.apps.pipeline --question "Is the project shown in the image the traditional East Room? If not, what part of the White House does it replace and what is its intended purpose?" --images pictures/white_house_state_ballroom_east_colonnade.jpeg --use-multimodal
```

观察到的效果：

- Query 生成准确：能生成类似 `McCrery Architects White House ballroom rendering East Room replacement`。
- 检索到 `whitehouse.gov` 官方来源。
- 抓取到网页正文。
- 图片匹配正常，无明显非图片资源混入。
- 最终回答能区分 East Room 和 East Wing。
- 能回答新项目是 White House ballroom，涉及 East Wing，用于大型接待/国家访问/外交活动等。

当前不足：

- 有时标题 chunk 会占据 top-k，挤掉更有信息量的正文 chunk。
- 最终回答还没有强制引用 `[Doc 1]`、`[Doc 2]`。

## 6. 常用运行命令

### 6.1 启动 CLIP server

在一个 PowerShell 终端中：

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -m rag_v1.apps.clip_server
```

### 6.2 运行主 pipeline

另一个 PowerShell 终端中：

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -m rag_v1.apps.pipeline --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal
```

### 6.3 White House ballroom 示例

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -m rag_v1.apps.pipeline --question "Is the project shown in the image the traditional East Room? If not, what part of the White House does it replace and what is its intended purpose?" --images pictures/white_house_state_ballroom_east_colonnade.jpeg --use-multimodal
```

### 6.4 运行全部测试

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

### 6.5 单独验证网页抓取

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -c "from rag_v1.services.web_page_fetcher import WebPageFetcher; p=WebPageFetcher().fetch('https://en.wikipedia.org/wiki/Computer_vision'); print('status=',p.fetch_status,'ok=',p.ok,'cache=',p.from_cache,'chars=',len(p.text),'quality=',round(p.quality_score,3)); print('title=',p.title); print(p.text[:1200])"
```

第二次运行同一命令时，如果看到 `cache=True`，说明缓存生效。

### 6.6 验证博查搜索 + 网页正文抓取

需要已设置 `BOCHA_API_KEY`：

```powershell
cd D:\Library\computer_vision\CV_new\CV-project
$env:PYTHONPATH="src"
python -c "from rag_v1.services.web_search import WebSearcher; docs=WebSearcher().search('computer vision recent news', candidate_k=3, fetch_pages=True); [print('%d. %s %s chars=%d q=%s %s' % (i, d.get('content_source'), d.get('web_fetch_status'), len(d.get('full_content') or ''), d.get('web_fetch_quality_score'), d.get('url'))) for i,d in enumerate(docs,1)]"
```

重点观察：

```text
content_source=web_page
```

如果是 `web_page`，说明使用了真实网页正文。如果是 `bocha_summary_fallback`，说明抓取失败或正文质量不够，回退到了博查摘要。

## 7. 配置项

网页抓取相关配置在 `src/rag_v1/config.py`：

```text
WEB_FETCH_ENABLED
WEB_FETCH_TIMEOUT
WEB_FETCH_MAX_RETRIES
WEB_FETCH_CACHE_TTL_SECONDS
WEB_FETCH_MAX_BYTES
WEB_FETCH_MIN_TEXT_CHARS
WEB_FETCH_CACHE_DIR
```

默认：

- 开启网页抓取。
- 抓取缓存位于 `.cache/web_pages`。
- 缓存 TTL 约 7 天。
- 最小正文长度阈值为 300 字符。

## 8. 当前仍建议继续优化的点

### 8.1 降低纯标题 chunk 权重

White House ballroom 测试中，标题 chunk 有时会排进最终 context。后续应降低纯标题、短标题、目录式标题的权重，优先选择能直接回答问题的正文段落。

### 8.2 最终答案增加引用

当前 prompt 还没有强制答案引用来源。建议改成：

```text
请在关键事实后标注 [Doc 1]、[Doc 2]。
如果多个文档支持同一事实，合并引用。
如果证据不足，请明确说明不确定。
```

这样更像 RAG，也更方便课程报告展示。

### 8.3 搜索 query 多样化

现在每个问题通常只生成一个 query。后续可以生成 2-3 个 query 变体：

```text
原始 VLM query
official/source-oriented query
fact sheet / executive order / policy-oriented query
```

再合并搜索结果并去重。这样能提高覆盖面，尤其适合“recent policies”这类开放问题。

### 8.4 增加 Wikipedia 检索分支

目前主要是 Web RAG。对于地标、历史建筑、艺术品、人物背景等稳定知识，Wikipedia 通常比泛 Web Search 更干净。后续可以增加 Wikipedia retriever，并把 Wikipedia evidence 与 Web evidence 一起排序。

### 8.5 建立 50 题评测集和自动评测脚本

建议测试集分布：

```text
15 个纯视觉题
20 个 Wikipedia / 稳定知识题
10 个实时网页检索题
5 个抗干扰 / 拒答题
```

每条数据建议记录：

```json
{
  "image": "...",
  "question": "...",
  "gold_answer": "...",
  "answer_aliases": ["..."],
  "requires_external_knowledge": true,
  "evidence_urls": ["..."],
  "snapshot_time": "2026-05-05"
}
```

## 9. 交接重点

当前项目最重要的变化是：

```text
以前：博查 summary/snippet 直接作为 RAG 文本。
现在：博查只负责发现 URL，系统自己抓网页正文、清洗、打分、缓存、排序。
```

这对 RAG 效果提升很关键，因为最终答案不再完全依赖搜索 API 给出的短摘要，而是可以使用更完整的网页正文证据。

目前整体状态：

- 单元测试通过。
- 网页正文抓取链路已打通。
- 图片过滤已从硬过滤改为软评分。
- 高质量 evidence rescue 已加入。
- 综合证据评分已加入。
- 多来源 context 聚合已加入。
- 仍需继续优化 chunk 选择、引用格式、多 query 检索和正式评测集。
