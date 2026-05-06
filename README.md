# RAG V2

相比 v1，规范了项目结构，简化了运行方式。

一个面向图文问答的 Visual Web RAG 项目。

用户输入一张或多张图片，以及一个问题后，系统会先调用视觉语言模型生成搜索查询，再通过 Web Search 获取候选网页，结合图片相似度过滤和文本/多模态 chunk 检索，最后把聚合后的上下文交给视觉语言模型生成答案。

## 2026.5.6 update xzh

这次更新重点把系统从“搜索摘要 RAG”推进到“网页证据 RAG”：

- Web Search 现在主要用于发现候选 URL，系统会继续抓取真实网页正文、抽取标题/发布时间/页面图片并做缓存。
- 新增网页质量评分、canonical URL 去重、listing 页面识别和低质量内容降权。
- 图片匹配从硬过滤改为软评分：强匹配会加分，弱匹配或无图不会直接误杀高质量文本证据。
- chunk 检索加入综合 evidence score，融合语义相似度、网页质量、图片相似度、来源可信度、freshness 和 listing penalty。
- 增强中英文 chunk 切分，改善中文长文本、英文段落和异常换行网页的召回效果。
- Prompt 更新：搜索 query、freshness 判断和最终回答更强调“最新/最近/当前”问题的时间证据，不在证据不足时强行断言。
- CLIP server 更新：过滤无效图片 URL，减少字体、脚本、favicon 等非图片资源导致的 embedding 请求失败。
- 新增单元测试覆盖网页抓取、图片软评分、chunk 切分、context 聚合和 temporal prompt 行为。

## 项目结构

当前仓库采用 `src` 布局，核心源码都放在 `src/rag_v1/` 下：

```text
RAG_V1/
├─ src/
│  └─ rag_v1/
│     ├─ apps/          # 应用入口
│     ├─ clients/       # 内部/外部服务客户端
│     ├─ pipeline/      # 主流程编排
│     ├─ retrieval/     # 检索与切块
│     ├─ services/      # CLIP / Web / VLM 等服务逻辑
│     ├─ config.py      # 项目配置
│     └─ prompts.py     # Prompt 模板
├─ pictures/            # 示例图片
├─ models/              # 本地模型缓存
├─ logs/                # 日志
├─ pipeline.py          # 兼容入口
├─ clip_server.py       # 兼容入口
├─ config.py            # 兼容配置导出
├─ pyproject.toml
└─ requirements.txt
```

## 主要模块

- `src/rag_v1/pipeline/rag_pipeline.py`：主流程入口，负责串联搜索 query 生成、网页检索、图片过滤、chunk 检索、上下文聚合与最终回答。
- `src/rag_v1/services/vl_router.py`：调用兼容 OpenAI Chat Completions 的视觉语言模型，用于生成搜索 query、判断 freshness 和最终回答。
- `src/rag_v1/services/web_search.py`：调用博查 Web Search API 获取候选 URL，并接入真实网页正文抓取。
- `src/rag_v1/services/web_page_fetcher.py`：抓取网页正文、提取元信息和图片 URL、做质量评分与缓存。
- `src/rag_v1/services/image_checker.py`：对候选页面做图片相似度软评分。
- `src/rag_v1/retrieval/chunk_extractor.py`：对网页文本切块，并进行文本检索或多模态检索。
- `src/rag_v1/services/clip_server.py`：本地 CLIP embedding 服务。
- `src/rag_v1/clients/clip.py`：CLIP server 的 HTTP client。
- `src/rag_v1/config.py`：集中管理 API、模型、超时和阈值配置。
- `src/rag_v1/prompts.py`：集中存放 Prompt 模板。

## 安装依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

如果你希望使用标准包方式运行，也可以执行：

```powershell
pip install -e .
```

## 运行方式

### 推荐方式

先启动 CLIP 服务：

```powershell
$env:PYTHONPATH="src"
python -m rag_v1.apps.clip_server
```

然后在另一个终端启动主流程：

```powershell
$env:PYTHONPATH="src"
python -m rag_v1.apps.pipeline --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal
```

如果已经执行过 `pip install -e .`，也可以直接使用命令行入口：

```powershell
rag-clip-server
rag-pipeline --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal
```

### 兼容旧方式

根目录仍保留了轻量兼容入口，因此以下命令仍可使用：

```powershell
python clip_server.py
python pipeline.py --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal
```

这些脚本会自动转发到 `src/rag_v1/` 下的新实现。

## 整体流程

1. 启动 `CLIP server`，在本地加载 CLIP 模型并监听默认地址 `127.0.0.1:8001`。
2. 主流程接收用户图片和问题。
3. `vl_router.generate_search_query(...)` 根据图片和问题生成 Web Search query。
4. `vl_router.choose_search_freshness(...)` 根据 query 和当前时间选择 freshness，例如 `oneDay`、`oneWeek`、`oneMonth`、`oneYear` 或 `noLimit`。
5. `web_search.WebSearcher.search(...)` 调用搜索接口获取候选网页及其页面图片。
6. `web_page_fetcher.WebPageFetcher` 抓取真实网页正文，并为网页打质量分。
7. `image_checker.filter_docs_by_image_match(...)` 使用 CLIP embedding 进行页面图片软评分。
8. 对通过过滤或被高质量证据救回的页面执行切块与检索。
9. 当 `use_multimodal=False` 时，仅进行文本相似度召回。
10. 当 `use_multimodal=True` 时，会额外计算图片与 chunk 文本的 CLIP 相似度，并与文本分数融合。
11. 聚合 top-k chunk 作为上下文。
12. `vl_router.answer_question(...)` 结合图片、问题和上下文生成最终答案。

## CLIP Server

默认地址：

```text
http://127.0.0.1:8001
```

接口：

- `GET /health`：检查服务是否正常启动。
- `POST /embed/images`：输入图片路径、图片 URL 或 base64 data URL，返回图片 embedding。
- `POST /embed/texts`：输入文本列表，返回文本 embedding。

返回的 embedding 已做 L2 normalize，可直接用于 cosine similarity。

## 配置

主要配置集中在 `src/rag_v1/config.py`，根目录 `config.py` 只是兼容导出层。

可通过环境变量覆盖：

- `BOCHA_API_KEY`：博查搜索 API key
- `BOCHA_API_URL`：博查搜索 API URL
- `BOCHA_TIMEOUT`：博查搜索超时
- `CHAT_API_KEY` 或 `DASHSCOPE_API_KEY`：视觉语言模型 API key
- `CHAT_API_BASE`：兼容 OpenAI Chat Completions 的 API base
- `CHAT_MODEL`：视觉语言模型名称
- `CHAT_API_MODE`：当前默认使用 `chat_completions`
- `CHAT_MAX_RETRIES`：接口最大重试次数
- `CHAT_CONNECT_TIMEOUT`：连接超时
- `CHAT_READ_TIMEOUT`：读取超时
- `IMAGE_MATCH_THRESHOLD`：页面图片过滤阈值，默认 `0.5`
- `CLIP_SERVER_HOST`：CLIP server host，默认 `127.0.0.1`
- `CLIP_SERVER_PORT`：CLIP server port，默认 `8001`
- `CLIP_SERVER_REQUEST_TIMEOUT`：调用 CLIP server 的请求超时
- `CLIP_MODEL_ID`：CLIP 模型 ID
- `CLIP_HF_ENDPOINT`：CLIP 模型下载镜像，默认 `https://hf-mirror.com`
- `CLIP_LOCAL_MODEL_DIR`：本地 CLIP 模型目录
- `CLIP_IMAGE_TIMEOUT`：CLIP server 下载远程图片时的超时

## 导入与开发说明

项目现在统一使用包导入，例如：

```python
from rag_v1.services.web_search import WebSearcher
from rag_v1.retrieval.chunk_extractor import ChunkExtractor
```

不再依赖“Python 文件刚好放在同一目录”这种脆弱方式。这样做的好处是：

- 源码可以安全地移动到 `src/` 下
- 内部模块层级更清晰
- 后续更容易做测试、打包和扩展

## 示例说明

### Trump 政策示例

示例命令会读取：

```text
pictures/trump.jpg
```

并通过 CLI 参数传入问题与图片，例如：

```powershell
rag-pipeline --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal
```

### James / Lakers 最近比赛示例

示例图片文件名是 `pictures/Curry.jpg`，但图中人物实际是 LeBron James。这个例子用于验证系统能先根据图片识别人物和球队，再结合网页证据回答最近正式比赛比分。

```powershell
$env:PYTHONPATH="src"
python -m rag_v1.apps.pipeline --question "图中这个人所在的球队最近一场正式比赛的最终比分是多少？这支球队得了多少分？" --images pictures/Curry.jpg --candidate-k 20 --top-k 10 --top-n-images 5 --use-multimodal
```

一次验证中，系统检索到了湖人对阵雷霆的最近正式比赛比分证据，并回答：

```text
图中人物是勒布朗·詹姆斯（LeBron James），他所在的球队是**洛杉矶湖人队**（Los Angeles Lakers）。

根据提供的网页信息（特别是Doc 4），该球队最近一场正式比赛是北京时间2026年5月6日进行的NBA季后赛，对阵俄克拉荷马城雷霆队。

*   **最终比分**：雷霆 108 - 90 湖人（雷霆胜）。
*   **湖人队得分**：90分。

```

这类问题依赖实时网页证据。若运行时上下文不能证明某场比赛就是“最近一场正式比赛”，最终回答应明确说明不确定，而不是只根据相关旧比分断言。

### White House State Ballroom 示例

```powershell
$env:PYTHONPATH="src"
python -m rag_v1.apps.pipeline --question "Is the project shown in the image the traditional East Room? If not, what part of the White House does it replace and what is its intended purpose?" --images pictures/white_house_state_ballroom_east_colonnade.jpeg --use-multimodal
```

该示例用于验证官方网页正文抓取、图片相似度软评分和多来源证据聚合。期望回答应说明图片展示的不是传统 East Room，而是 White House State Ballroom 项目，涉及 East Wing，并用于大型官方、国事和外交活动。

## 测试

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -v
```

当前测试覆盖网页正文抽取、URL canonicalization、图片 URL 清洗、图片软评分、chunk 切分、context 聚合和 Prompt 时间敏感行为。

## TODO

1. 预先缓存图片 embedding，减少重复计算。
2. 继续优化“最近/最新/当前”问题的事件时间验证，区分页发布时间和真实事件时间。
3. 最终答案增加稳定的 `[Doc]` 引用标注，方便课程报告展示证据来源。
4. 建立小型评测集，覆盖纯视觉题、稳定知识题、实时网页题和抗干扰拒答题。
