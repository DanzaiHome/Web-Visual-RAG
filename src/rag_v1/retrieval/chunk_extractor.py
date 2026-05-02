from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_v1.clients.clip import ClipClient


class ChunkExtractor:
    _similarity_model: Optional[SentenceTransformer] = None
    _clip_client: Optional[ClipClient] = None

    def __init__(
        self,
        document: Union[str, Dict[str, Any]],
        question: str,
        image_paths: Sequence[Union[str, Path]],
        chunk_size: int = 400,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.document = document
        self.question = question
        self.image_paths = [str(path) for path in image_paths]
        self.chunk_size = chunk_size
        self.text = self._extract_text(document)
        self.chunks = self._split_into_chunks(self.text, chunk_size)

    @classmethod
    def _get_similarity_model(cls) -> SentenceTransformer:
        if cls._similarity_model is None:
            cls._similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._similarity_model

    @classmethod
    def _get_clip_client(cls) -> ClipClient:
        if cls._clip_client is None:
            cls._clip_client = ClipClient()
        return cls._clip_client

    @staticmethod
    def _extract_text(document: Union[str, Dict[str, Any]]) -> str:
        if isinstance(document, str):
            return document.strip()

        for key in ("full_content", "content", "summary", "snippet", "text"):
            value = document.get(key)
            if value:
                return str(value).strip()

        return ""

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int) -> List[str]:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []

        return [
            normalized_text[index:index + chunk_size]
            for index in range(0, len(normalized_text), chunk_size)
        ]

    def _build_chunk_result(
        self,
        chunk_id: int,
        text: str,
        score: float,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "text": text,
            "score": score,
            "chunk_size": self.chunk_size,
        }

        if isinstance(self.document, dict):
            for key in ("url", "name", "display_url", "site_name", "image_urls"):
                if key in self.document:
                    result[key] = self.document[key]

        return result

    def _compute_text_scores(self) -> np.ndarray:
        model = self._get_similarity_model()
        query_emb = model.encode([self.question])[0]
        chunk_embs = model.encode(self.chunks)

        return np.dot(chunk_embs, query_emb) / (
            np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )

    def retrieve_text_chunks(self, top_n: int = 3) -> List[Dict[str, Any]]:
        if top_n <= 0 or not self.chunks:
            return []

        scores = self._compute_text_scores()
        result_count = min(top_n, len(self.chunks))
        ranked_indices = np.argsort(scores)[::-1][:result_count]

        return [
            self._build_chunk_result(
                chunk_id=int(index),
                text=self.chunks[int(index)],
                score=float(scores[int(index)]),
            )
            for index in ranked_indices
        ]

    def retrieve_multimodal_chunks(
        self,
        top_n: int = 3,
        a: float = 0.5,
    ) -> List[Dict[str, Any]]:
        if a < 0 or a > 1:
            raise ValueError("a must be between 0 and 1")
        if top_n <= 0 or not self.chunks:
            return []

        text_scores = self._compute_text_scores()

        if not self.image_paths:
            image_avg_scores = np.zeros(len(self.chunks), dtype=np.float32)
        else:
            clip_client = self._get_clip_client()
            clip_texts = [f"{self.question} {chunk}" for chunk in self.chunks]
            text_features = clip_client.embed_texts(clip_texts)
            image_features = clip_client.embed_images(self.image_paths)
            image_scores = text_features @ image_features.T
            print(image_scores)
            image_avg_scores = image_scores.mean(axis=1)

        final_scores = text_scores * a + image_avg_scores * (1 - a)
        result_count = min(top_n, len(self.chunks))
        ranked_indices = np.argsort(final_scores)[::-1][:result_count]

        results: List[Dict[str, Any]] = []
        for index in ranked_indices:
            chunk_index = int(index)
            result = self._build_chunk_result(
                chunk_id=chunk_index,
                text=self.chunks[chunk_index],
                score=float(final_scores[chunk_index]),
            )
            result["text_score"] = float(text_scores[chunk_index])
            result["image_avg_score"] = float(image_avg_scores[chunk_index])
            result["a"] = a
            results.append(result)

        return results
