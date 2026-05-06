from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rag_v1.clients.clip import ClipClient
from rag_v1.services.web_page_fetcher import is_valid_page_image_url


class ImageChecker:
    def __init__(
        self,
        list_a: Sequence[str],
        list_b: Sequence[str],
        clip_client: Optional[ClipClient] = None,
    ) -> None:
        self.list_a = list(list_a)
        self.list_b = list(list_b)
        self.clip_client = clip_client or ClipClient()

    def compute_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        embeddings_a = self.clip_client.embed_images(self.list_a)
        embeddings_b = self.clip_client.embed_images(self.list_b)
        return embeddings_a, embeddings_b

    @staticmethod
    def compute_similarity_matrix(
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> np.ndarray:
        return embeddings_a @ embeddings_b.T

    @staticmethod
    def _hungarian_max(similarity_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        rows = len(similarity_matrix)
        cols = len(similarity_matrix[0]) if rows else 0
        if rows == 0 or cols == 0:
            return []

        size = max(rows, cols)
        max_value = max(max(row) for row in similarity_matrix)

        cost = [[max_value for _ in range(size)] for _ in range(size)]
        for i in range(rows):
            for j in range(cols):
                cost[i][j] = max_value - similarity_matrix[i][j]

        u = [0.0] * (size + 1)
        v = [0.0] * (size + 1)
        p = [0] * (size + 1)
        way = [0] * (size + 1)

        for i in range(1, size + 1):
            p[0] = i
            j0 = 0
            minv = [float("inf")] * (size + 1)
            used = [False] * (size + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float("inf")
                j1 = 0
                for j in range(1, size + 1):
                    if used[j]:
                        continue
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
                for j in range(size + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignments: List[Tuple[int, int]] = []
        for j in range(1, size + 1):
            i = p[j]
            if i == 0:
                continue
            row_index = i - 1
            col_index = j - 1
            if row_index < rows and col_index < cols:
                assignments.append((row_index, col_index))

        assignments.sort(key=lambda pair: pair[0])
        return assignments

    def match(self, similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        assignments = self._hungarian_max(similarity_matrix.tolist())
        return [
            (row_index, col_index, float(similarity_matrix[row_index, col_index]))
            for row_index, col_index in assignments
        ]


def _normalize_image_urls(image_urls: object) -> List[str]:
    if not image_urls:
        return []
    if isinstance(image_urls, str):
        image_url = image_urls.strip()
        return [image_url] if is_valid_page_image_url(image_url) else []

    normalized_urls: List[str] = []
    seen = set()
    for url in image_urls:
        image_url = str(url or "").strip()
        if not image_url or image_url in seen:
            continue
        if not is_valid_page_image_url(image_url):
            continue
        seen.add(image_url)
        normalized_urls.append(image_url)
    return normalized_urls


def _embed_images_lenient(
    clip_client: ClipClient,
    image_addresses: Sequence[str],
) -> Tuple[List[np.ndarray], List[Tuple[str, Exception]]]:
    embeddings: List[np.ndarray] = []
    failures: List[Tuple[str, Exception]] = []

    for image_address in image_addresses:
        try:
            image_embedding = clip_client.embed_images([image_address])
        except Exception as exc:
            failures.append((image_address, exc))
            continue

        if len(image_embedding) > 0:
            embeddings.append(image_embedding[0])

    return embeddings, failures


def compute_page_image_match_score(
    prompt_image_paths: Sequence[Union[str, Path]],
    page_image_urls: Sequence[str],
    clip_client: Optional[ClipClient] = None,
    default_score: float = 1.0,
) -> float:
    if not page_image_urls:
        return default_score

    prompt_images = [str(path) for path in prompt_image_paths]
    page_images = _normalize_image_urls(page_image_urls)
    if not prompt_images or not page_images:
        return default_score

    clip_client = clip_client or ClipClient()
    prompt_embeddings = clip_client.embed_images(prompt_images)
    page_embedding_rows, _ = _embed_images_lenient(clip_client, page_images)
    if not page_embedding_rows:
        return default_score

    page_embeddings = np.vstack(page_embedding_rows)
    similarity_matrix = prompt_embeddings @ page_embeddings.T
    return float(np.max(similarity_matrix))


def _image_match_decision(match_status: str, match_score: float, threshold: float) -> str:
    if match_status != "matched":
        return match_status
    if match_score >= threshold:
        return "strong_match"
    if match_score >= threshold * 0.7:
        return "weak_match"
    return "low_similarity"


def score_docs_by_image_match(
    web_docs: Sequence[Dict[str, object]],
    prompt_image_paths: Sequence[Union[str, Path]],
    threshold: float = 0.5,
    clip_client: Optional[ClipClient] = None,
) -> List[Dict[str, object]]:
    if not prompt_image_paths:
        for doc in web_docs:
            doc["image_match_status"] = "not_requested"
            doc["image_match_decision"] = "not_requested"
            doc["max_image_similarity"] = 0.0
            doc["image_match_image_count"] = 0
            doc["image_match_failed_count"] = 0
        return list(web_docs)

    clip_client = clip_client or ClipClient()
    scored_docs: List[Dict[str, object]] = []
    status_counts: Dict[str, int] = {}
    decision_counts: Dict[str, int] = {}
    prompt_images = [str(path) for path in prompt_image_paths]

    try:
        prompt_embeddings = clip_client.embed_images(prompt_images)
    except Exception as exc:
        print(f"Image match unavailable: {exc}; keeping all pages as soft evidence")
        for doc in web_docs:
            doc["image_match_status"] = "unavailable"
            doc["image_match_decision"] = "unavailable"
            doc["max_image_similarity"] = 0.0
            doc["image_match_image_count"] = 0
            doc["image_match_failed_count"] = 0
        return list(web_docs)

    for doc in web_docs:
        image_urls = _normalize_image_urls(doc.get("image_urls"))
        match_status = "matched"
        failed_count = 0
        matched_image_count = 0

        if not image_urls:
            match_score = 0.0
            match_status = "no_images"
        else:
            page_embedding_rows, failures = _embed_images_lenient(clip_client, image_urls)
            failed_count = len(failures)
            matched_image_count = len(page_embedding_rows)
            if failures and page_embedding_rows:
                print(
                    f"Skipped {len(failures)}/{len(image_urls)} page image(s) "
                    f"for {doc.get('url', '')}"
                )

            if not page_embedding_rows:
                match_score = 0.0
                match_status = "failed"
                if failures:
                    first_error = failures[0][1]
                    print(
                        f"Image match skipped for {doc.get('url', '')}: "
                        f"all {len(failures)} page image(s) failed: {first_error}; "
                        f"fallback score={match_score}"
                    )
            else:
                page_embeddings = np.vstack(page_embedding_rows)
                similarity_matrix = prompt_embeddings @ page_embeddings.T
                match_score = float(np.max(similarity_matrix))

        if not np.isfinite(match_score):
            match_score = 0.0
            match_status = "failed"
            print(
                f"Image match produced non-finite score for {doc.get('url', '')}; "
                f"fallback score={match_score}"
            )

        doc["image_match_status"] = match_status
        doc["max_image_similarity"] = match_score
        doc["image_match_image_count"] = matched_image_count
        doc["image_match_failed_count"] = failed_count
        decision = _image_match_decision(match_status, match_score, threshold)
        doc["image_match_decision"] = decision
        status_counts[match_status] = status_counts.get(match_status, 0) + 1
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        scored_docs.append(doc)

    print(
        f"Image match scored {len(scored_docs)}/{len(web_docs)} pages "
        f"(threshold={threshold}, statuses={status_counts}, decisions={decision_counts})"
    )
    return scored_docs


def filter_docs_by_image_match(
    web_docs: Sequence[Dict[str, object]],
    prompt_image_paths: Sequence[Union[str, Path]],
    threshold: float = 0.5,
    default_score: float = 1.0,
) -> List[Dict[str, object]]:
    scored_docs = score_docs_by_image_match(
        web_docs=web_docs,
        prompt_image_paths=prompt_image_paths,
        threshold=threshold,
    )
    filtered_docs: List[Dict[str, object]] = []

    for doc in scored_docs:
        match_status = str(doc.get("image_match_status") or "")
        match_score = float(doc.get("max_image_similarity") or 0.0)
        if match_status == "matched" and match_score < threshold:
            print(
                f"Filtered page by image similarity: score={match_score:.4f}, "
                f"threshold={threshold:.4f}, url={doc.get('url', '')}"
            )
            continue

        filtered_docs.append(doc)

    print(
        f"Image match filter kept {len(filtered_docs)}/{len(scored_docs)} pages "
        f"(threshold={threshold})"
    )
    return filtered_docs
