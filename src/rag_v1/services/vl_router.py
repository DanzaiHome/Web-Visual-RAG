import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import requests

from rag_v1.config import CHAT_API_CONFIG
from rag_v1.prompts import prompts
from rag_v1.timing import get_active_timing


PROMPT_QUERY = prompts.web_prompt_en
PROMPT_ENTITY_CANDIDATES = prompts.entity_candidates_prompt_en
PROMPT_ENTITY_GUIDED_QUERY = prompts.entity_guided_web_prompt_en
PROMPT_ANSWER = prompts.answer_prompt_en
PROMPT_NO_RAG_ANSWER = prompts.no_rag_prompt_en
PROMPT_FRESHNESS = prompts.freshness_prompt_en
PROMPT_SUFFICIENCY = prompts.sufficiency_prompt_en
VALID_FRESHNESS_VALUES = {"oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"}

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


def _iter_json_objects(response_text: str) -> List[Dict[str, object]]:
    decoder = json.JSONDecoder()
    objects: List[Dict[str, object]] = []
    for match in re.finditer(r"\{", response_text):
        try:
            payload, _ = decoder.raw_decode(response_text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
    return objects


def _normalize_candidate_text(value: object) -> str:
    return str(value or "").strip()


def extract_entity_candidates(response_text: str, max_candidates: int = 3) -> List[Dict[str, object]]:
    for payload in _iter_json_objects(response_text):
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            continue

        candidates: List[Dict[str, object]] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue

            name = _normalize_candidate_text(item.get("name"))
            if not name:
                continue

            aliases: List[str] = []
            raw_aliases = item.get("aliases")
            if isinstance(raw_aliases, list):
                for alias in raw_aliases:
                    normalized_alias = _normalize_candidate_text(alias)
                    if (
                        normalized_alias
                        and normalized_alias.lower() != name.lower()
                        and normalized_alias not in aliases
                    ):
                        aliases.append(normalized_alias)

            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            candidates.append(
                {
                    "name": name,
                    "type": _normalize_candidate_text(item.get("type")),
                    "aliases": aliases,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "reason": _normalize_candidate_text(item.get("reason")),
                    "missing_slot": _normalize_candidate_text(item.get("missing_slot")),
                }
            )

        if candidates:
            candidates.sort(
                key=lambda candidate: (
                    float(candidate.get("confidence") or 0.0),
                    len(str(candidate.get("name") or "")),
                ),
                reverse=True,
            )
            return candidates[:max_candidates]

    return []


def _format_entity_candidates_for_prompt(candidates: Sequence[Dict[str, object]]) -> str:
    if not candidates:
        return "- none"

    lines: List[str] = []
    for index, candidate in enumerate(candidates, start=1):
        name = str(candidate.get("name") or "").strip()
        entity_type = str(candidate.get("type") or "").strip() or "unknown"
        confidence = float(candidate.get("confidence") or 0.0)
        aliases = [
            str(alias).strip()
            for alias in (candidate.get("aliases") or [])
            if str(alias).strip()
        ]
        reason = str(candidate.get("reason") or "").strip()
        missing_slot = str(candidate.get("missing_slot") or "").strip()

        line = f"- Candidate {index}: name={name}; type={entity_type}; confidence={confidence:.2f}"
        if aliases:
            line += f"; aliases={', '.join(aliases[:3])}"
        if reason:
            line += f"; reason={reason}"
        if missing_slot:
            line += f"; missing_slot={missing_slot}"
        lines.append(line)
    return "\n".join(lines)


def _format_previous_queries_for_prompt(previous_queries: Sequence[str]) -> str:
    normalized_queries = [
        " ".join(str(query or "").strip().split())
        for query in previous_queries
        if " ".join(str(query or "").strip().split())
    ]
    if not normalized_queries:
        return "- none"
    return "\n".join(f"- {query}" for query in normalized_queries)


def _dedupe_queries(queries: Sequence[str], max_queries: int = 3) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for query in queries:
        normalized = " ".join(str(query or "").strip().split())
        if not normalized:
            continue
        signature = normalized.lower()
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(normalized)
        if len(deduped) >= max_queries:
            break
    return deduped


def _candidate_variants(candidate: Dict[str, object]) -> List[Dict[str, object]]:
    variants = [candidate]
    aliases = [
        str(alias).strip()
        for alias in (candidate.get("aliases") or [])
        if str(alias).strip()
    ]
    for alias in aliases[:1]:
        variants.append(
            {
                "name": alias,
                "type": candidate.get("type", ""),
                "aliases": [],
                "confidence": candidate.get("confidence", 0.0),
                "reason": candidate.get("reason", ""),
            }
        )
    return variants


def generate_entity_candidates(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    debug: bool = False,
) -> List[Dict[str, object]]:
    if not img_paths:
        return []

    prompt = PROMPT_ENTITY_CANDIDATES.format(question=question)
    response_text = call_api(prompt=prompt, img_paths=img_paths, temperature=0.0)
    candidates = extract_entity_candidates(response_text)
    if debug:
        print(f"-----------------------------------\nEntity candidates raw response:\n{response_text}")
        print(f"Parsed entity candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}")
    return candidates


def generate_search_queries(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    debug: bool = False,
    max_queries: int = 3,
    entity_candidates: Optional[Sequence[Dict[str, object]]] = None,
) -> List[str]:
    candidates = list(entity_candidates) if entity_candidates is not None else generate_entity_candidates(
        img_paths=img_paths,
        question=question,
        debug=debug,
    )

    candidate_queries: List[str] = []
    if candidates:
        for candidate in candidates[:max_queries]:
            focused_candidates = _candidate_variants(candidate)
            focused_prompt = PROMPT_ENTITY_GUIDED_QUERY.format(
                question=question,
                entity_candidates=_format_entity_candidates_for_prompt(focused_candidates),
            )
            candidate_queries.append(
                call_api(prompt=focused_prompt, img_paths=img_paths, temperature=0.1)
            )

    if not candidate_queries:
        candidate_queries.append(
            call_api(
                prompt=PROMPT_QUERY.format(question=question),
                img_paths=img_paths,
                temperature=0.1,
            )
        )

    queries = _dedupe_queries(candidate_queries, max_queries=max_queries)
    if debug:
        print(f"Generated search queries:\n{json.dumps(queries, ensure_ascii=False, indent=2)}")
    return queries


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
    start_time = time.perf_counter()

    try:
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

                if response.status_code >= 400:
                    response_text = response.text.strip()
                    if len(response_text) > 1200:
                        response_text = response_text[:1200] + "..."
                    last_error = RuntimeError(
                        "Chat API request failed: "
                        f"status={response.status_code}, body={response_text}"
                    )
                    break

                return _extract_text_from_response(response.json())
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                break
    finally:
        timing = get_active_timing()
        if timing is not None:
            timing.add("chat_api", time.perf_counter() - start_time)

    raise RuntimeError(f"API call failed after {max_attempts} attempts: {last_error}")


def generate_search_query(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    debug: bool = False,
    entity_candidates: Optional[Sequence[Dict[str, object]]] = None,
) -> str:
    queries = generate_search_queries(
        img_paths=img_paths,
        question=question,
        debug=debug,
        max_queries=1,
        entity_candidates=entity_candidates,
    )
    return queries[0] if queries else ""


def choose_search_freshness(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    query: str,
    current_time: str,
    debug: bool = False,
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

    if debug:
        print(f"Invalid freshness from model: {freshness!r}; fallback to noLimit")
    return "noLimit"


def answer_question(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    context: str,
    debug: bool = False,
) -> str:
    prompt = PROMPT_ANSWER.format(context=context, question=question)
    if debug:
        print(f"-----------------------------------\nFinal prompt:\n{prompt}")
    return call_api(prompt=prompt, img_paths=img_paths, temperature=0.2)


def answer_question_no_rag(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    debug: bool = False,
) -> str:
    prompt = PROMPT_NO_RAG_ANSWER.format(question=question)
    if debug:
        print(f"-----------------------------------\nFinal prompt (no RAG):\n{prompt}")
    return call_api(prompt=prompt, img_paths=img_paths, temperature=0.2)


def extract_sufficiency_json(response_text: str) -> Optional[Dict[str, Optional[str]]]:
    matches = re.findall(r"\{[\s\S]*?\}", response_text)
    if not matches:
        return None

    for match in matches:
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            continue

        raw_judgement = payload.get("judgement", payload.get("judge"))
        if raw_judgement is None:
            continue

        judgement = str(raw_judgement).strip().upper()
        if judgement == "TRUE":
            judgement = "YES"
        elif judgement == "FALSE":
            judgement = "NO"

        if judgement not in {"YES", "NO"}:
            continue

        addition = payload.get("addition")
        if addition is None:
            normalized_addition = None
        else:
            normalized_addition = str(addition).strip() or None

        return {
            "judgement": judgement,
            "addition": normalized_addition,
        }

    return None


def judge_context_sufficiency(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    context: str,
    debug: bool = False,
    entity_candidates: Optional[Sequence[Dict[str, object]]] = None,
    previous_queries: Optional[Sequence[str]] = None,
) -> str:
    prompt = PROMPT_SUFFICIENCY.format(
        question=question,
        entity_candidates=_format_entity_candidates_for_prompt(entity_candidates or []),
        previous_queries=_format_previous_queries_for_prompt(previous_queries or []),
        context=context,
    )
    if debug:
        print(f"-----------------------------------\nSufficiency prompt:\n{prompt}")
    return call_api(prompt=prompt, img_paths=img_paths, temperature=0.0)
