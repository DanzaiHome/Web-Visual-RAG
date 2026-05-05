from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rag_v1.clients.clip import ClipClient


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
        return [image_url] if image_url else []

    return [str(url).strip() for url in image_urls if str(url).strip()]


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

    checker = ImageChecker(
        list_a=prompt_images,
        list_b=page_images,
        clip_client=clip_client,
    )
    embeddings_a, embeddings_b = checker.compute_embeddings()
    if embeddings_a.size == 0 or embeddings_b.size == 0:
        return default_score

    similarity_matrix = checker.compute_similarity_matrix(embeddings_a, embeddings_b)
    matches = checker.match(similarity_matrix)

    if not matches:
        return default_score

    return max(score for _, _, score in matches)


def filter_docs_by_image_match(
    web_docs: Sequence[Dict[str, object]],
    prompt_image_paths: Sequence[Union[str, Path]],
    threshold: float = 0.5,
    default_score: float = 1.0,
) -> List[Dict[str, object]]:
    if not prompt_image_paths:
        return list(web_docs)

    clip_client = ClipClient()
    filtered_docs: List[Dict[str, object]] = []

    for doc in web_docs:
        image_urls = _normalize_image_urls(doc.get("image_urls"))
        try:
            match_score = compute_page_image_match_score(
                prompt_image_paths=prompt_image_paths,
                page_image_urls=image_urls,
                clip_client=clip_client,
                default_score=default_score,
            )
        except Exception as exc:
            match_score = 0.0
            print(
                f"Image match failed for {doc.get('url', '')}: {exc}; "
                f"fallback score={match_score}"
            )

        doc["max_image_similarity"] = match_score
        if match_score < threshold:
            print(
                f"Filtered page by image similarity: score={match_score:.4f}, "
                f"threshold={threshold:.4f}, url={doc.get('url', '')}"
            )
            continue

        filtered_docs.append(doc)

    print(
        f"Image match filter kept {len(filtered_docs)}/{len(web_docs)} pages "
        f"(threshold={threshold})"
    )
    return filtered_docs
