from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from urllib.parse import urlsplit

from rag_v1.config import IMAGE_MATCH_THRESHOLD, WEB_FETCH_CONFIG
from rag_v1.retrieval.chunk_extractor import ChunkExtractor
from rag_v1.services.image_checker import score_docs_by_image_match
from rag_v1.services.vl_router import (
    answer_question,
    choose_search_freshness,
    extract_sufficiency_json,
    generate_search_query,
    judge_context_sufficiency,
)
from rag_v1.services.web_page_fetcher import canonicalize_url
from rag_v1.services.web_search import WebSearcher


def _canonical_url(url: object) -> str:
    return canonicalize_url(url)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _domain_key(url: object) -> str:
    canonical_url = _canonical_url(url)
    parsed = urlsplit(canonical_url or str(url or ""))
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    return host


def _normalized_text_signature(text: object, max_len: int = 800) -> str:
    normalized = "".join(ch.lower() for ch in str(text or "") if ch.isalnum())
    return normalized[:max_len]


def _doc_signature(doc: Dict[str, object]) -> str:
    text = " ".join(
        str(doc.get(key) or "")
        for key in ("name", "full_content", "content", "summary", "snippet")
    )
    return _normalized_text_signature(text)


def _is_near_duplicate(
    signature: str,
    existing_signatures: Sequence[str],
    threshold: float = 0.9,
) -> bool:
    if len(signature) < 80:
        return False

    return any(
        SequenceMatcher(None, signature, existing).ratio() >= threshold
        for existing in existing_signatures
        if existing
    )


def _looks_like_listing_noise(text: object) -> bool:
    content = str(text or "").strip()
    if not content:
        return False

    lowered = content.lower()
    listing_tokens = (
        "advertisement",
        "latest news",
        "more from",
        "most popular",
        "recommended",
        "related articles",
        "share this",
        "subscribe",
        "trending",
        "上一篇",
        "下一篇",
        "广告",
        "更多",
        "热门",
        "相关",
        "推荐",
        "相关阅读",
        "推荐阅读",
    )
    token_hits = sum(1 for token in listing_tokens if token in lowered)
    if token_hits >= 2:
        return True

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) >= 5:
        short_lines = sum(1 for line in lines if len(line.split()) <= 7 and len(line) <= 80)
        if short_lines / len(lines) >= 0.7:
            return True

    return token_hits >= 1 and len(content) < 900


def _is_usable_chunk_candidate(chunk: Dict[str, object]) -> bool:
    content = str(chunk.get("content") or "").strip()
    if not content:
        return False

    semantic_score = _safe_float(chunk.get("semantic_score"), _safe_float(chunk.get("score")))
    text_score = _safe_float(chunk.get("text_score"), semantic_score)
    quality_score = _safe_float(chunk.get("web_fetch_quality_score"))
    probable_listing = bool(chunk.get("web_fetch_is_probable_listing"))
    listing_noise = _looks_like_listing_noise(content)

    if not probable_listing and not listing_noise:
        return True

    if semantic_score >= 0.3 or text_score >= 0.18:
        return True
    if quality_score >= 0.55 and len(content) >= 900 and semantic_score >= 0.22:
        return True

    return False


def deduplicate_web_docs(web_docs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped_docs: List[Dict[str, object]] = []
    seen_urls = set()
    seen_signatures: List[str] = []

    for doc in web_docs:
        canonical_url = _canonical_url(doc.get("url"))
        if canonical_url and canonical_url in seen_urls:
            continue

        signature = _doc_signature(doc)
        if _is_near_duplicate(signature, seen_signatures):
            continue

        doc["canonical_url"] = canonical_url
        deduped_docs.append(doc)
        if canonical_url:
            seen_urls.add(canonical_url)
        if signature:
            seen_signatures.append(signature)

    if len(deduped_docs) != len(web_docs):
        print(f"Deduplicated web docs: kept {len(deduped_docs)}/{len(web_docs)}")

    return deduped_docs


def _select_diverse_chunks(
    ranked_chunks: Sequence[Dict[str, object]],
    top_k: int,
    max_chunks_per_url: int = 2,
    max_chunks_per_domain: int = 3,
) -> List[Dict[str, object]]:
    candidate_pool = [chunk for chunk in ranked_chunks if _is_usable_chunk_candidate(chunk)]
    if candidate_pool and len(candidate_pool) != len(ranked_chunks):
        print(f"Suppressed low-relevance listing chunks: kept {len(candidate_pool)}/{len(ranked_chunks)}")
    if not candidate_pool:
        candidate_pool = list(ranked_chunks)

    selected: List[Dict[str, object]] = []
    per_url_counts: Dict[str, int] = {}
    per_domain_counts: Dict[str, int] = {}
    seen_signatures: List[str] = []

    def try_select(
        chunk: Dict[str, object],
        url_limit: int,
        domain_limit: int,
    ) -> bool:
        canonical_url = _canonical_url(chunk.get("canonical_url") or chunk.get("url"))
        domain = _domain_key(canonical_url or chunk.get("url"))
        if canonical_url and per_url_counts.get(canonical_url, 0) >= url_limit:
            return False
        if domain and per_domain_counts.get(domain, 0) >= domain_limit:
            return False

        signature = _normalized_text_signature(chunk.get("content"))
        if _is_near_duplicate(signature, seen_signatures, threshold=0.92):
            return False

        selected.append(chunk)
        if canonical_url:
            per_url_counts[canonical_url] = per_url_counts.get(canonical_url, 0) + 1
        if domain:
            per_domain_counts[domain] = per_domain_counts.get(domain, 0) + 1
        if signature:
            seen_signatures.append(signature)
        return True

    for chunk in candidate_pool:
        try_select(chunk, url_limit=1, domain_limit=1)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for chunk in candidate_pool:
            if chunk in selected:
                continue
            try_select(
                chunk,
                url_limit=max_chunks_per_url,
                domain_limit=max_chunks_per_domain,
            )
            if len(selected) >= top_k:
                break

    return selected


def _source_reliability_score(doc: Dict[str, object]) -> float:
    url = str(doc.get("canonical_url") or doc.get("url") or "")
    host = _domain_key(url)
    path = urlsplit(url).path.lower()
    site_name = str(doc.get("site_name") or "").lower()
    source_text = f"{host} {path} {site_name}"

    if host.endswith(".gov") or ".gov." in host or host.endswith(".mil"):
        return 0.08
    if host.endswith(".edu") or ".edu." in host or ".ac." in host:
        return 0.04

    trusted_tokens = (
        "apnews",
        "bbc",
        "congress",
        "fact-sheet",
        "fact_sheet",
        "official",
        "reuters",
        "state.gov",
        "whitehouse",
        "who.int",
    )
    if any(token in source_text for token in trusted_tokens):
        return 0.04

    weak_tokens = ("forum", "bbs", "toutiao", "weibo", "zhihu", "wordpress")
    if any(token in source_text for token in weak_tokens):
        return -0.02

    return 0.0


def _freshness_score(doc: Dict[str, object]) -> float:
    raw_date = str(doc.get("date_published") or doc.get("date_last_crawled") or "").strip()
    if not raw_date:
        return 0.0

    try:
        parsed = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
    except ValueError:
        return 0.0

    if parsed.tzinfo is None:
        now = datetime.now()
    else:
        now = datetime.now(parsed.tzinfo)
    age_days = (now - parsed).total_seconds() / 86400
    if age_days < 0:
        return 0.0
    if age_days <= 7:
        return 0.04
    if age_days <= 30:
        return 0.025
    if age_days <= 365:
        return 0.005
    return 0.0


def _image_similarity_score(doc: Dict[str, object]) -> float:
    status = str(doc.get("image_match_status") or "")
    decision = str(doc.get("image_match_decision") or "")
    similarity = _safe_float(doc.get("max_image_similarity"))

    if status != "matched":
        return 0.0
    if decision == "strong_match":
        return min(0.05, max(0.0, (similarity - IMAGE_MATCH_THRESHOLD) * 0.12 + 0.025))
    if decision == "weak_match":
        return 0.01
    if decision == "low_similarity":
        return -0.005
    return 0.0


def _evidence_score_breakdown(chunk_score: float, doc: Dict[str, object]) -> Dict[str, float]:
    quality_score = _safe_float(doc.get("web_fetch_quality_score"))
    content_source = str(doc.get("content_source") or "")
    content_adjustment = 0.0
    listing_penalty = 0.0
    quality_bonus = 0.0

    if doc.get("web_fetch_is_probable_listing"):
        listing_penalty = -0.04
    if content_source == "web_page":
        content_adjustment = 0.04
        quality_bonus = min(0.06, quality_score * 0.06)
    elif "fallback" in content_source:
        content_adjustment = -0.04

    image_bonus = _image_similarity_score(doc)
    source_bonus = _source_reliability_score(doc)
    freshness_bonus = _freshness_score(doc)
    total = (
        chunk_score
        + content_adjustment
        + quality_bonus
        + image_bonus
        + source_bonus
        + freshness_bonus
        + listing_penalty
    )

    return {
        "semantic_score": float(chunk_score),
        "content_adjustment": content_adjustment,
        "web_quality_bonus": quality_bonus,
        "image_similarity_bonus": image_bonus,
        "source_reliability_bonus": source_bonus,
        "freshness_bonus": freshness_bonus,
        "listing_penalty": listing_penalty,
        "total_score": total,
    }


def _doc_text_len(doc: Dict[str, object]) -> int:
    return len(str(doc.get("full_content") or doc.get("content") or "").strip())


def _is_high_quality_web_body(doc: Dict[str, object]) -> bool:
    if str(doc.get("content_source") or "") != "web_page":
        return False
    if str(doc.get("web_fetch_status") or "") not in {"success", "short_content"}:
        return False

    quality_score = _safe_float(doc.get("web_fetch_quality_score"))
    text_len = _doc_text_len(doc)
    if text_len < WEB_FETCH_CONFIG.min_text_chars or quality_score < 0.25:
        return False
    if bool(doc.get("web_fetch_is_probable_listing")):
        return text_len >= 1200 and quality_score >= 0.35
    return True


def _is_usable_summary_fallback(doc: Dict[str, object]) -> bool:
    content_source = str(doc.get("content_source") or "")
    if "fallback" not in content_source:
        return False
    if _doc_text_len(doc) < 300:
        return False
    if str(doc.get("image_match_status") or "") == "matched":
        return True
    return _source_reliability_score(doc) > 0


def select_quality_evidence_docs(
    image_filtered_docs: Sequence[Dict[str, object]],
    pre_image_filter_docs: Sequence[Dict[str, object]],
    max_docs: int = 8,
) -> List[Dict[str, object]]:
    usable_docs = [
        doc
        for doc in image_filtered_docs
        if _is_high_quality_web_body(doc) or _is_usable_summary_fallback(doc)
    ]
    if usable_docs:
        if len(usable_docs) != len(image_filtered_docs):
            print(
                f"Quality evidence filter kept {len(usable_docs)}/"
                f"{len(image_filtered_docs)} image-filtered docs"
            )
        usable_docs = sorted(usable_docs, key=_doc_quality_sort_key, reverse=True)
        return usable_docs[:max_docs]

    rescued_docs = [doc for doc in pre_image_filter_docs if _is_high_quality_web_body(doc)]
    if rescued_docs:
        rescued_docs = sorted(rescued_docs, key=_doc_quality_sort_key, reverse=True)
        for doc in rescued_docs:
            doc["evidence_rescue_status"] = "image_filter_rescue"
        print(
            f"Rescued {len(rescued_docs)} high-quality webpage body doc(s) "
            "after image filtering left only weak evidence"
        )
        return rescued_docs[:max_docs]

    return list(image_filtered_docs)


def _doc_quality_sort_key(doc: Dict[str, object]) -> tuple[float, float, float, int]:
    return (
        _safe_float(doc.get("web_fetch_quality_score")),
        _source_reliability_score(doc),
        _image_similarity_score(doc),
        _doc_text_len(doc),
    )


def retrieve_web_context(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    top_n_images: int = 3,
    chunk_size: int = 400,
    chunks_per_doc: int = 3,
    use_multimodal: bool = False,
    debug: bool = False,
    current_time: Optional[str] = None,
) -> List[Dict[str, object]]:
    web_searcher = WebSearcher()
    if current_time is None:
        current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    freshness = choose_search_freshness(
        img_paths=img_paths,
        question=question,
        query=query,
        current_time=current_time,
        debug=debug,
    )
    if debug:
        print(f"\nSearch freshness: {freshness} (current_time={current_time})\n")

    web_docs = web_searcher.search(
        query=query,
        candidate_k=candidate_k,
        summary=True,
        freshness=freshness,
        top_n_images=top_n_images,
        content_preview_len=4000,
        debug=debug,
    )
    if not web_docs:
        return []

    web_docs = deduplicate_web_docs(web_docs)
    if not web_docs:
        return []

    image_scored_docs = score_docs_by_image_match(
        web_docs=web_docs,
        prompt_image_paths=img_paths,
        threshold=IMAGE_MATCH_THRESHOLD,
    )
    web_docs = select_quality_evidence_docs(
        image_filtered_docs=image_scored_docs,
        pre_image_filter_docs=web_docs,
        max_docs=max(top_k * 2, top_k + 2),
    )
    if not web_docs:
        return []

    chunk_candidates: List[Dict[str, object]] = []
    for doc in web_docs:
        extractor = ChunkExtractor(
            document=doc,
            question=query,
            image_paths=img_paths,
            chunk_size=chunk_size,
        )

        if not use_multimodal:
            chunk_results = extractor.retrieve_text_chunks(top_n=chunks_per_doc)
        else:
            chunk_results = extractor.retrieve_multimodal_chunks(top_n=chunks_per_doc)

        for chunk in chunk_results:
            semantic_score = float(chunk["score"])
            score_breakdown = _evidence_score_breakdown(semantic_score, doc)
            chunk_candidates.append(
                {
                    "url": chunk.get("url", doc.get("url", "")),
                    "canonical_url": chunk.get("canonical_url", doc.get("canonical_url", "")),
                    "score": score_breakdown["total_score"],
                    "semantic_score": semantic_score,
                    "score_breakdown": score_breakdown,
                    "text_score": float(chunk.get("text_score", chunk["score"])),
                    "content": chunk["text"],
                    "image_urls": chunk.get("image_urls", doc.get("image_urls", [])),
                    "max_image_similarity": float(doc.get("max_image_similarity", 0.0)),
                    "chunk_id": chunk["chunk_id"],
                    "chunk_size": chunk["chunk_size"],
                    "name": chunk.get("name", doc.get("name")),
                    "site_name": chunk.get("site_name", doc.get("site_name")),
                    "display_url": chunk.get("display_url", doc.get("display_url")),
                    "date_published": doc.get("date_published"),
                    "date_last_crawled": doc.get("date_last_crawled"),
                    "content_source": doc.get("content_source"),
                    "web_fetch_status": doc.get("web_fetch_status"),
                    "web_fetch_from_cache": doc.get("web_fetch_from_cache"),
                    "web_fetch_quality_score": doc.get("web_fetch_quality_score"),
                    "web_fetch_content_hash": doc.get("web_fetch_content_hash"),
                    "web_fetch_is_probable_listing": doc.get("web_fetch_is_probable_listing"),
                    "evidence_rescue_status": doc.get("evidence_rescue_status"),
                    "image_match_status": doc.get("image_match_status"),
                    "image_match_decision": doc.get("image_match_decision"),
                    "image_match_image_count": doc.get("image_match_image_count"),
                    "image_match_failed_count": doc.get("image_match_failed_count"),
                }
            )

    if not chunk_candidates:
        return []

    ranked = sorted(chunk_candidates, key=lambda item: item["score"], reverse=True)
    if debug:
        print(f"Ranked chunks:\n{ranked}")
    selected = _select_diverse_chunks(ranked, top_k=top_k)
    if debug and len(selected) != min(top_k, len(ranked)):
        print(f"Diversified chunks: kept {len(selected)}/{len(ranked)}")
    return selected


def _format_score(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _format_image_match(item: Dict[str, object]) -> str:
    status = str(item.get("image_match_status") or "").strip()
    if not status:
        return ""

    if status == "matched":
        score = _format_score(item.get("max_image_similarity"))
        image_count = item.get("image_match_image_count")
        decision = str(item.get("image_match_decision") or "").strip()
        if score:
            if decision:
                return f"matched, score={score}, decision={decision}, images={image_count}"
            return f"matched, score={score}, images={image_count}"
        return "matched"

    if status == "no_images":
        return "no page images available"

    if status == "failed":
        failed_count = item.get("image_match_failed_count")
        return f"page image fetch failed, failed_images={failed_count}"

    if status == "unavailable":
        return "CLIP image matching unavailable"

    return status


def _group_retrieved_docs(
    retrieved_docs: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped_docs: List[Dict[str, object]] = []
    by_url: Dict[str, Dict[str, object]] = {}

    for item in retrieved_docs:
        url = str(item.get("url") or "").strip()
        group_key = _canonical_url(item.get("canonical_url") or url) or url
        content = str(item.get("content") or "").strip()
        content_signature = _normalized_text_signature(content)
        chunk = {
            "chunk_id": item.get("chunk_id"),
            "content": content,
            "score": float(item.get("score") or 0.0),
            "semantic_score": float(item.get("semantic_score") or item.get("score") or 0.0),
        }

        group = by_url.get(group_key)
        if group is None:
            group = {
                "url": url,
                "canonical_url": str(item.get("canonical_url") or "").strip(),
                "name": str(item.get("name") or "").strip(),
                "site_name": str(item.get("site_name") or item.get("display_url") or "").strip(),
                "display_url": str(item.get("display_url") or "").strip(),
                "date_published": str(item.get("date_published") or "").strip(),
                "date_last_crawled": str(item.get("date_last_crawled") or "").strip(),
                "score": float(item.get("score") or 0.0),
                "semantic_score": float(item.get("semantic_score") or item.get("score") or 0.0),
                "max_image_similarity": float(item.get("max_image_similarity") or 0.0),
                "content_source": str(item.get("content_source") or "").strip(),
                "web_fetch_status": str(item.get("web_fetch_status") or "").strip(),
                "web_fetch_from_cache": bool(item.get("web_fetch_from_cache")),
                "web_fetch_quality_score": float(item.get("web_fetch_quality_score") or 0.0),
                "web_fetch_is_probable_listing": bool(item.get("web_fetch_is_probable_listing")),
                "evidence_rescue_status": str(item.get("evidence_rescue_status") or "").strip(),
                "image_match_status": str(item.get("image_match_status") or "").strip(),
                "image_match_decision": str(item.get("image_match_decision") or "").strip(),
                "image_match_image_count": item.get("image_match_image_count"),
                "image_match_failed_count": item.get("image_match_failed_count"),
                "chunks": [chunk] if content else [],
                "_content_signatures": [content_signature] if content_signature else [],
            }
            by_url[group_key] = group
            grouped_docs.append(group)
            continue

        if content_signature and _is_near_duplicate(
            content_signature,
            group["_content_signatures"],
            threshold=0.92,
        ):
            continue

        if content:
            group["chunks"].append(chunk)
        if content_signature:
            group["_content_signatures"].append(content_signature)
        group["score"] = max(float(group.get("score") or 0.0), float(item.get("score") or 0.0))
        group["semantic_score"] = max(
            float(group.get("semantic_score") or 0.0),
            float(item.get("semantic_score") or item.get("score") or 0.0),
        )
        group["max_image_similarity"] = max(
            float(group.get("max_image_similarity") or 0.0),
            float(item.get("max_image_similarity") or 0.0),
        )

        if not group.get("name") and item.get("name"):
            group["name"] = str(item.get("name") or "").strip()
        if not group.get("site_name") and item.get("site_name"):
            group["site_name"] = str(item.get("site_name") or "").strip()
        if not group.get("display_url") and item.get("display_url"):
            group["display_url"] = str(item.get("display_url") or "").strip()
        if not group.get("date_published") and item.get("date_published"):
            group["date_published"] = str(item.get("date_published") or "").strip()
        if not group.get("date_last_crawled") and item.get("date_last_crawled"):
            group["date_last_crawled"] = str(item.get("date_last_crawled") or "").strip()
        if not group.get("content_source") and item.get("content_source"):
            group["content_source"] = str(item.get("content_source") or "").strip()
        if not group.get("web_fetch_status") and item.get("web_fetch_status"):
            group["web_fetch_status"] = str(item.get("web_fetch_status") or "").strip()
        if item.get("web_fetch_from_cache"):
            group["web_fetch_from_cache"] = True
        group["web_fetch_quality_score"] = max(
            float(group.get("web_fetch_quality_score") or 0.0),
            float(item.get("web_fetch_quality_score") or 0.0),
        )
        if item.get("web_fetch_is_probable_listing"):
            group["web_fetch_is_probable_listing"] = True
        if not group.get("evidence_rescue_status") and item.get("evidence_rescue_status"):
            group["evidence_rescue_status"] = str(item.get("evidence_rescue_status") or "").strip()
        if not group.get("image_match_status") and item.get("image_match_status"):
            group["image_match_status"] = str(item.get("image_match_status") or "").strip()
        if not group.get("image_match_decision") and item.get("image_match_decision"):
            group["image_match_decision"] = str(item.get("image_match_decision") or "").strip()
        if not group.get("image_match_image_count") and item.get("image_match_image_count"):
            group["image_match_image_count"] = item.get("image_match_image_count")
        if not group.get("image_match_failed_count") and item.get("image_match_failed_count"):
            group["image_match_failed_count"] = item.get("image_match_failed_count")

    return grouped_docs


def _chunk_sort_key(chunk: Dict[str, object]) -> tuple[int, float]:
    try:
        chunk_id = int(chunk.get("chunk_id"))
    except (TypeError, ValueError):
        chunk_id = 10**9

    return (chunk_id, -float(chunk.get("score") or 0.0))


def _format_evidence_source(item: Dict[str, object]) -> str:
    content_source = str(item.get("content_source") or "").strip()
    fetch_status = str(item.get("web_fetch_status") or "").strip()
    from_cache = bool(item.get("web_fetch_from_cache"))
    quality = _format_score(item.get("web_fetch_quality_score"))
    is_listing = bool(item.get("web_fetch_is_probable_listing"))
    rescue_status = str(item.get("evidence_rescue_status") or "").strip()

    if content_source == "web_page":
        details = ["webpage body"]
        if from_cache:
            details.append("cache")
        if quality:
            details.append(f"quality={quality}")
        if rescue_status == "image_filter_rescue":
            details.append("image-filter rescue")
        if is_listing:
            details.append("possible listing page")
        return ", ".join(details)

    if "fallback" in content_source:
        if fetch_status:
            return f"Bocha summary fallback, fetch_status={fetch_status}"
        return "Bocha summary fallback"

    return content_source


def aggregate_context(
    query: str,
    retrieved_docs: Sequence[Dict[str, object]],
    current_time: Optional[str] = None,
) -> str:
    sections: List[str] = []

    if current_time:
        sections.append(f"Pipeline Current Time:\n{current_time}")

    if query.strip():
        sections.append(f"Search Query:\n{query.strip()}")

    if retrieved_docs:
        sections.append(
            "Temporal Note:\n"
            "Document date fields below are page/search metadata. For latest/current questions, "
            "verify event dates and completed/current status from titles or content."
        )
        grouped_docs = _group_retrieved_docs(retrieved_docs)
        doc_blocks = []
        for index, item in enumerate(grouped_docs, start=1):
            title = str(item.get("name") or "").strip()
            site_name = str(item.get("site_name") or item.get("display_url") or "").strip()
            published = str(item.get("date_published") or "").strip()
            last_crawled = str(item.get("date_last_crawled") or "").strip()
            score = _format_score(item.get("score"))
            image_match = _format_image_match(item)
            evidence_source = _format_evidence_source(item)

            lines = [f"[Doc {index}] URL: {item['url']}"]
            if title:
                lines.append(f"Title: {title}")
            if site_name:
                lines.append(f"Source: {site_name}")
            if published:
                lines.append(
                    "Page metadata - Published: "
                    f"{published} (not necessarily the event date)"
                )
            elif last_crawled:
                lines.append(
                    "Page metadata - Last crawled: "
                    f"{last_crawled} (not necessarily the event date)"
                )
            if score:
                lines.append(f"Retrieval score: {score}")
            if evidence_source:
                lines.append(f"Evidence source: {evidence_source}")
            if image_match:
                lines.append(f"Image match: {image_match}")

            chunks = sorted(item.get("chunks", []), key=_chunk_sort_key)
            contents = [chunk["content"] for chunk in chunks if str(chunk.get("content", "")).strip()]
            if contents:
                if len(contents) == 1:
                    lines.append(f"Content: {contents[0]}")
                else:
                    lines.append("Content:")
                    for chunk_index, content in enumerate(contents, start=1):
                        lines.append(f"- Chunk {chunk_index}: {content}")
            doc_blocks.append("\n".join(lines))
        sections.append("Retrieved Context:\n" + "\n\n".join(doc_blocks))

    return "\n\n".join(sections).strip()


def answer_with_rag(
    img_paths: Sequence[Union[str, Path]],
    question: str,
    top_k: int = 5,
    candidate_k: int = 10,
    top_n_images: int = 3,
    chunk_size: int = 400,
    chunks_per_doc: int = 3,
    use_multimodal: bool = False,
    debug: bool = False,
    max_sufficiency_iterations: int = 3,
) -> str:
    print("Step 1/4: Generating initial search query...")
    query = generate_search_query(img_paths=img_paths, question=question)
    if debug:
        print(f"\nquery:\n{query}\n")

    current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    print("Step 2/4: Retrieving initial web context...")
    retrieved_docs = retrieve_web_context(
        img_paths=img_paths,
        question=question,
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        top_n_images=top_n_images,
        chunk_size=chunk_size,
        chunks_per_doc=chunks_per_doc,
        use_multimodal=use_multimodal,
        debug=debug,
        current_time=current_time,
    )
    context = aggregate_context(query=query, retrieved_docs=retrieved_docs, current_time=current_time)
    if debug:
        print(f"\nInitial context:\n{context}\n")

    consecutive_parse_failures = 0
    iteration = 0
    exhausted_sufficiency_iterations = False

    while iteration < max_sufficiency_iterations:
        iteration += 1
        print(f"Step 3/4: Checking context sufficiency (iteration {iteration})...")
        if debug:
            print(f"\n===== Sufficiency Loop Iteration {iteration} =====")
        raw_judgement = judge_context_sufficiency(
            img_paths=img_paths,
            question=question,
            context=context,
            debug=debug,
        )
        if debug:
            print(f"Sufficiency raw response:\n{raw_judgement}\n")

        judgement_payload = extract_sufficiency_json(raw_judgement)
        if judgement_payload is None:
            consecutive_parse_failures += 1
            if debug:
                print(
                    "Failed to parse a valid sufficiency JSON object. "
                    f"consecutive_failures={consecutive_parse_failures}"
                )
            if consecutive_parse_failures >= 3:
                print("Sufficiency check failed three times. Proceeding to final answer.")
                break
            continue

        consecutive_parse_failures = 0
        judgement = judgement_payload["judgement"]
        addition = judgement_payload["addition"]
        if debug:
            print(
                "Parsed sufficiency JSON: "
                f"judgement={judgement}, addition={addition!r}"
            )

        if judgement == "YES":
            print("Context is sufficient.")
            break

        if not addition:
            print("Sufficiency returned no usable additional query. Proceeding to final answer.")
            break

        print("Context is insufficient. Running additional retrieval...")
        if debug:
            print(f"Running additional retrieval with query:\n{addition}\n")
        additional_docs = retrieve_web_context(
            img_paths=img_paths,
            question=question,
            query=addition,
            top_k=top_k,
            candidate_k=candidate_k,
            top_n_images=top_n_images,
            chunk_size=chunk_size,
            chunks_per_doc=chunks_per_doc,
            use_multimodal=use_multimodal,
            debug=debug,
            current_time=current_time,
        )
        additional_context = aggregate_context(
            query=addition,
            retrieved_docs=additional_docs,
            current_time=current_time,
        )
        if debug:
            print(f"Additional context:\n{additional_context}\n")

        if additional_context:
            context = f"{context}\n\n{additional_context}".strip() if context else additional_context
            if debug:
                print(f"Updated context after iteration {iteration}:\n{context}\n")
        else:
            print("Additional retrieval returned empty context. Proceeding to final answer.")
            break
    else:
        exhausted_sufficiency_iterations = max_sufficiency_iterations > 0

    if exhausted_sufficiency_iterations:
        print("Reached maximum sufficiency iterations. Proceeding to final answer.")

    print("Step 4/4: Generating final answer...")
    return answer_question(
        img_paths=img_paths,
        question=question,
        context=context,
        debug=debug,
    )
