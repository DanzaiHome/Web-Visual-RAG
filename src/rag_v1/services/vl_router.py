import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import requests

from rag_v1.config import CHAT_API_CONFIG
from rag_v1.prompts import prompts


PROMPT_QUERY = prompts.web_prompt_en
PROMPT_ANSWER = prompts.answer_prompt_en
PROMPT_FRESHNESS = prompts.freshness_prompt_en
VALID_FRESHNESS_VALUES = {"oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"}
CURRENT_FACTOID_TERMS = (
    "最近",
    "最新",
    "当前",
    "今日",
    "今天",
    "昨天",
    "最近一场",
    "最终比分",
    "得分",
    "last",
    "latest",
    "most recent",
    "current",
    "today",
    "yesterday",
    "final score",
    "result",
)
SPORT_RESULT_TERMS = (
    "比分",
    "得分",
    "比赛",
    "正式比赛",
    "final score",
    "score",
    "result",
    "game",
    "match",
)
PRICE_TERMS = ("价格", "股价", "报价", "price", "stock", "quote")
RELEASE_TERMS = ("发布", "发行", "release", "released", "launch")

oai_config = {
    "apikey": CHAT_API_CONFIG.api_key,
    "apibase": CHAT_API_CONFIG.api_base,
    "model": CHAT_API_CONFIG.model,
    "api_mode": CHAT_API_CONFIG.api_mode,
    "max_retries": CHAT_API_CONFIG.max_retries,
    "connect_timeout": CHAT_API_CONFIG.connect_timeout,
    "read_timeout": CHAT_API_CONFIG.read_timeout,
}

api_session = requests.Session()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _query_contains_cjk(query: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in query)


def _append_missing_terms(query: str, terms: Sequence[str]) -> str:
    augmented = query.strip()
    lowered = augmented.lower()
    missing_terms = [term for term in terms if term.lower() not in lowered]
    if missing_terms:
        augmented = f"{augmented} {' '.join(missing_terms)}".strip()
    return augmented


def _normalize_query_spacing(query: str) -> str:
    if _query_contains_cjk(query) and not re.search(r"[A-Za-z]", query):
        return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", query)
    return query


def _augment_current_factoid_query(query: str, question: str) -> str:
    stripped_query = query.strip()
    if not stripped_query:
        return stripped_query

    combined_text = f"{question} {stripped_query}"
    if not _contains_any(combined_text, CURRENT_FACTOID_TERMS):
        return stripped_query

    use_chinese_terms = _query_contains_cjk(stripped_query)
    if _contains_any(combined_text, SPORT_RESULT_TERMS):
        terms = (
            ("最近一场", "已结束", "最终比分", "赛程", "结果", "战报")
            if use_chinese_terms
            else ("latest", "completed", "final score", "schedule", "results", "box score")
        )
        return _normalize_query_spacing(_append_missing_terms(stripped_query, terms))

    if _contains_any(combined_text, PRICE_TERMS):
        terms = (
            ("当前", "价格", "官方", "行情", "报价")
            if use_chinese_terms
            else ("current", "price", "quote", "market data", "official")
        )
        return _normalize_query_spacing(_append_missing_terms(stripped_query, terms))

    if _contains_any(combined_text, RELEASE_TERMS):
        terms = (
            ("最新", "发布", "官方", "日期")
            if use_chinese_terms
            else ("latest", "release", "official", "date")
        )
        return _normalize_query_spacing(_append_missing_terms(stripped_query, terms))

    terms = ("最新", "官方") if use_chinese_terms else ("latest", "official")
    return _normalize_query_spacing(_append_missing_terms(stripped_query, terms))


def _encode_image(image_path: Union[str, Path]) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _build_multimodal_content(
    prompt: str,
    img_paths: Sequence[Union[str, Path]],
) -> List[Dict[str, object]]:
    content = [{"type": "text", "text": prompt}]

    for img_path in img_paths:
        image_base64 = _encode_image(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            }
        )

    return content


def _extract_text_from_response(result: dict) -> str:
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError(f"Invalid API response: {json.dumps(result, ensure_ascii=False)}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    raise RuntimeError(f"Unsupported API response format: {json.dumps(result, ensure_ascii=False)}")


def call_api(
    prompt: str,
    img_paths: Sequence[Union[str, Path]] = (),
    temperature: float = 0.2,
) -> str:
    if oai_config["api_mode"] != "chat_completions":
        raise ValueError("Only chat_completions mode is supported in this script.")

    url = f"{oai_config['apibase']}/chat/completions"
    payload = {
        "model": oai_config["model"],
        "messages": [
            {
                "role": "user",
                "content": _build_multimodal_content(prompt, img_paths),
            }
        ],
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {oai_config['apikey']}",
        "Content-Type": "application/json",
    }
    timeout = (
        oai_config["connect_timeout"],
        oai_config["read_timeout"],
    )

    max_attempts = oai_config["max_retries"] + 1
    last_error = None  # type: Optional[Exception]

    for attempt in range(max_attempts):
        try:
            response = api_session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                response.raise_for_status()

            response.raise_for_status()
            return _extract_text_from_response(response.json())
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(min(2 ** attempt, 4))
                continue
            break

    raise RuntimeError(f"API call failed after {max_attempts} attempts: {last_error}")


def generate_search_query(
    img_paths: Sequence[Union[str, Path]],
    question: str,
) -> str:
    prompt = PROMPT_QUERY.format(question=question)
    query = call_api(prompt=prompt, img_paths=img_paths, temperature=0.1)
    return _augment_current_factoid_query(query=query, question=question)


def choose_search_freshness(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    query: str,
    current_time: str,
) -> str:
    prompt = PROMPT_FRESHNESS.format(
        question=question,
        query=query,
        current_time=current_time,
    )
    freshness = call_api(prompt=prompt, img_paths=img_paths, temperature=0.0).strip()

    if freshness in VALID_FRESHNESS_VALUES:
        return freshness

    for value in VALID_FRESHNESS_VALUES:
        if value in freshness:
            return value

    print(f"Invalid freshness from model: {freshness!r}; fallback to noLimit")
    return "noLimit"


def answer_question(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    context: str,
) -> str:
    prompt = PROMPT_ANSWER.format(context=context, question=question)
    print(f"-----------------------------------\nFinal prompt:\n{prompt}")
    return call_api(prompt=prompt, img_paths=img_paths, temperature=0.2)
