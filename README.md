# RAG V2

## 2025.5.24 update mhc - RAG v3
- 增加.env.example，并修改启动服务时读取配置的设置，现在实现从.env中便可直接读取项目各配置，便于调试运行。
- Readme 补充终止服务和检查服务运行的指令。
- 强化最终回答规则：所有依赖检索证据的事实必须带 [Doc n]；不允许编造 Doc id；证据不支持时明确说“无法从证据确定”。
- 把 context 格式改成明确的 Citation Rules + [Doc n] Evidence Block + Evidence text，让模型更容易稳定引用来源。
- 修复补充检索循环中的 Doc 编号重复问题：后续检索会从已有编号继续，例如初始有 [Doc 1]、[Doc 2]，下一轮从 [Doc 3] 开始。
- 增加/更新了测试，覆盖 evidence block、Doc 编号延续、answer prompt 强制引用规则。
- 修复两个原本导致全量测试失败的问题：过期的 chunk 测试 monkeypatch，以及本地 WebPageFetcher 测试被环境代理拦截的问题。

实现 FastAPI 后端 + Vite/React 前端界面，浏览器可以上传图片、输入问题，后端保存临时图片后调用现有 answer_with_rag，再把模型最终回复返回到页面展示。
  - FastAPI 后端：CV-project/src/rag_v1/apps/web_api.py
      - POST /api/ask：接收 question 和 images
      - 临时保存上传图片到 .tmp_uploads/web_ui
      - 调用 answer_with_rag(...)
      - 返回 answer / elapsed_seconds / image_count
      - 请求完成后自动清理临时图片
  - React 前端：frontend/
      - 图片选择与预览
      - 问题输入
      - 可调检索参数：topK、候选网页数、chunk 大小、补充检索轮数、多模态召回
      - loading、错误提示、最终回复展示
  - 依赖配置：
      - CV-project/requirements.txt 增加 fastapi、uvicorn[standard]、python-multipart
      - CV-project/pyproject.toml 增加同样依赖和 rag-web-api 入口
      - frontend/package.json 配置 Vite/React/TypeScript/lucide
  - README 已补充 Web 前端启动说明：CV-project/README.md

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

一个面向图文问答的 Visual Web RAG 项目。

用户输入一张或多张图片，以及一个问题后，系统会先调用视觉语言模型生成搜索查询，再通过 Web Search 获取候选网页，结合图片相似度过滤和文本/多模态 chunk 检索，最后把聚合后的上下文交给视觉语言模型生成答案。

## 2026.5.6 update hxz

- 引入参数 time、debug.
- 引入循环检索和自我验证，以应对单次搜索无法获取全部消息的复杂情况.

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
- `src/rag_v1/services/web_search.py`：调用博查 Web Search API，返回网页候选、摘要和页面图片。
- `src/rag_v1/services/image_checker.py`：对候选页面做图片相似度过滤。
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

建议在项目根目录运行。

先启动 CLIP 服务：

```bash
rag-clip-server > "./logs/clip-server.log" 2>&1 &
```

同时，需要启动 sentence-transformer 服务：
```bash
rag-text-retrieval-server > "./logs/text-retrieval.log" 2>&1 &
```

然后在另一个终端启动主流程：

```powershell
rag-pipeline --question "What new policies did this person announce recently?" --images pictures/trump.jpg --use-multimodal > "./logs/qa.log" 2>&1 &
```

检查当前服务是否运行
```
ps -ef | grep -E 'rag-clip-server|rag-text-retrieval-server' | grep -v grep
```

终止服务
```
  pkill -f '/root/miniconda3/envs/cvpj/bin/rag-clip-server'
  pkill -f '/root/miniconda3/envs/cvpj/bin/rag-text-retrieval-server'
```

确保存在API key
```
  export DASHSCOPE_API_KEY="你的真实 DashScope API Key"
  export CHAT_API_KEY="$DASHSCOPE_API_KEY"
  export BOCHA_API_KEY="你的真实 Bocha API Key"
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
6. `image_checker.filter_docs_by_image_match(...)` 使用 CLIP embedding 进行页面图片过滤。
7. 对通过过滤的页面执行切块与检索。
8. 当 `use_multimodal=False` 时，仅进行文本相似度召回。
9. 当 `use_multimodal=True` 时，会额外计算图片与 chunk 文本的 CLIP 相似度，并与文本分数融合。
10. 聚合 top-k chunk 作为上下文。
11. 模型判断当前 context 是否足够回答问题；如果不够，重新生成 query，回到 `4` 继续检索。如果信息足够或者超出循环限制，进入下一步。
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
- `CLIP_LOCAL_MODEL_DIR`：本地 CLIP 模型目录
- `CLIP_IMAGE_TIMEOUT`：CLIP server 下载远程图片时的超时

## 运行参数

- `--question` 必填，问题
- `--images` 必填，一个或多个本地图片路径
- `--top-n-images` 每个搜索结果保留的页面图片数，默认 `3`
- `--use-multimodal` 开关，开启多模态 chunk 检索
- `--debug` 开关，打印详细调试信息
- `--max-sufficiency-iterations` 充分性检查和补充检索的最大迭代次数，默认 `3`
- `--time` 开关，打印整条 pipeline 的 timing 统计
- `--simple-time` 开关，开启 `--time` 时，简化输出
- `--candidate-k` 搜索阶段拿多少候选网页，默认 `10`
- `--chunks-per-doc` 每个文档最多取多少 chunks，默认 `3`
- `--top-k` 最后保留多少 chunk 进入上下文，默认 `5`
- `--chunk-size` 每个 chunk 大小，默认 `400`

```bash
rag-pipeline \
    --question "What new activities has this man been up to lately?" \
    --images pictures/trump.jpg \
    --max-sufficiency-iterations 3 \
    --candidate-k 10 \
    --chunks-per-doc 3 \
    --top-k 5 \
    --chunk-size 400 \
    --use-multimodal \
    --time \
    --simple-time \
    > "./logs/qa.log" 2>&1 &
```


## Web 前端界面

本项目新增 FastAPI + Vite/React 的轻量前端，用于在浏览器上传图片、输入问题并查看 `answer_with_rag` 的最终回复。

### 1. 准备依赖

Python 后端依赖安装到环境：

```bash
pip install -r requirements.txt
```

Node 前端依赖安装在 `frontend/`：

```bash
cd CV-project/frontend
npm install
```

### 2. 启动基础 RAG 服务

Web UI 仍复用原有 pipeline，因此需要先启动 CLIP 和文本检索服务：

```bash
cd CV-project
rag-clip-server > ./logs/clip-server.log 2>&1 &
rag-text-retrieval-server > ./logs/text-retrieval.log 2>&1 &
```

确保环境变量已经配置：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
export CHAT_API_KEY="$DASHSCOPE_API_KEY"
export BOCHA_API_KEY="你的 Bocha API Key"
```

### 3. 启动 Web 后端

```bash
cd CV-project
python -m uvicorn rag_v1.apps.web_api:app --host 127.0.0.1 --port 8010
```

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:8010/health
```

### 4. 启动前端页面

```bash
cd CV-project/frontend
npm run dev 【本地模式】
// npm run serve:ports 【服务器开发者模式】
```

浏览器打开：

```text
http://127.0.0.1:5173
```

前端开发服务器会把 `/api` 请求代理到 `http://127.0.0.1:8010`。

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

示例命令会读取：

```text
pictures/trump.jpg
```

并通过 CLI 参数传入问题与图片，例如：

```bash
rag-pipeline --question "What new activities have these two people been up to lately?" --images pictures/trump.jpg pictures/Curry.jpg --use-multimodal --debug --max-sufficiency-iterations 3 --time > "./logs/qa.log" 2>&1 &

rag-pipeline --question "Who is this man?" --images pictures/mjq.jpg --use-multimodal --debug --max-sufficiency-iterations 3 --time > "./logs/qa.log" 2>&1 &
```

RAG requirement 测试：
```bash
python -m rag_v1.pipeline.rag_requirement --question "Who is this man?" --images pictures/trump.png --debug
python -m rag_v1.pipeline.rag_requirement --question "What is 1 + 1?" --debug
```