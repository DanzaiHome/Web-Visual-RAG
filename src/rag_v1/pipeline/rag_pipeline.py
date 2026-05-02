from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Union

from rag_v1.config import IMAGE_MATCH_THRESHOLD
from rag_v1.retrieval.chunk_extractor import ChunkExtractor
from rag_v1.services.image_checker import filter_docs_by_image_match
from rag_v1.services.vl_router import (
    answer_question,
    choose_search_freshness,
    generate_search_query,
)
from rag_v1.services.web_search import WebSearcher


def retrieve_web_context(
    img_paths: Sequence[Union[str, Path]],
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
    top_n_images: int = 3,
    chunk_size: int = 400,
    chunks_per_doc: int = 3,
    use_multimodal: bool = False,
) -> List[Dict[str, object]]:
    web_searcher = WebSearcher()
    current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    freshness = choose_search_freshness(
        img_paths=img_paths,
        query=query,
        current_time=current_time,
    )
    print(f"\nSearch freshness: {freshness} (current_time={current_time})\n")

    web_docs = web_searcher.search(
        query=query,
        candidate_k=candidate_k,
        summary=True,
        freshness=freshness,
        top_n_images=top_n_images,
        content_preview_len=4000,
    )
    if not web_docs:
        return []

    web_docs = filter_docs_by_image_match(
        web_docs=web_docs,
        prompt_image_paths=img_paths,
        threshold=IMAGE_MATCH_THRESHOLD,
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
            chunk_candidates.append(
                {
                    "url": chunk.get("url", doc.get("url", "")),
                    "score": float(chunk["score"]),
                    "text_score": float(chunk.get("text_score", chunk["score"])),
                    "content": chunk["text"],
                    "image_urls": chunk.get("image_urls", doc.get("image_urls", [])),
                    "max_image_similarity": float(doc.get("max_image_similarity", 0.0)),
                    "chunk_id": chunk["chunk_id"],
                    "chunk_size": chunk["chunk_size"],
                    "name": chunk.get("name", doc.get("name")),
                    "site_name": chunk.get("site_name", doc.get("site_name")),
                }
            )

    if not chunk_candidates:
        return []

    ranked = sorted(chunk_candidates, key=lambda item: item["score"], reverse=True)
    print(f"Ranked chunks:\n{ranked}")
    return ranked[:top_k]


def aggregate_context(
    query: str,
    retrieved_docs: Sequence[Dict[str, object]],
) -> str:
    sections: List[str] = []

    if query.strip():
        sections.append(f"Search Query:\n{query.strip()}")

    if retrieved_docs:
        doc_blocks = []
        for index, item in enumerate(retrieved_docs, start=1):
            doc_blocks.append(
                f"[Doc {index}] URL: {item['url']}\n"
                f"Content: {item['content']}"
            )
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
) -> str:
    query = generate_search_query(img_paths=img_paths, question=question)
    print(f"\nquery:\n{query}\n")
    retrieved_docs = retrieve_web_context(
        img_paths=img_paths,
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        top_n_images=top_n_images,
        chunk_size=chunk_size,
        chunks_per_doc=chunks_per_doc,
        use_multimodal=use_multimodal,
    )
    context = aggregate_context(query=query, retrieved_docs=retrieved_docs)
    return answer_question(img_paths=img_paths, question=question, context=context)
