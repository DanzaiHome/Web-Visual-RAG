import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np

from rag_v1.clients.clip import ClipClient
from rag_v1.timing import get_active_timing

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any


class ChunkExtractor:
    _similarity_model: Optional[SentenceTransformer] = None

    def __init__(
        self,
        document: Union[str, Dict[str, object]],
        question: str,
        image_paths: Sequence[Union[str, Path]],
        chunk_size: int = 400,
        precomputed_chunks: Optional[Sequence[str]] = None,
    ) -> None:
        self.document = document
        self.question = question
        self.image_paths = [str(path) for path in image_paths]
        self.chunk_size = chunk_size
        self.document_text = self._get_document_text(document)
        if precomputed_chunks is None:
            self.chunks = self._split_into_chunks(self.document_text, chunk_size)
        else:
            self.chunks = [str(chunk) for chunk in precomputed_chunks if str(chunk).strip()]

    @staticmethod
    def _get_document_text(document: Union[str, Dict[str, object]]) -> str:
        if isinstance(document, dict):
            return str(document.get("full_content") or document.get("content") or "")
        return str(document)

    @classmethod
    def _get_similarity_model(cls) -> SentenceTransformer:
        if cls._similarity_model is None:
            try:
                from sentence_transformers import SentenceTransformer as _SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for text chunk retrieval. "
                    "Install project requirements or run inside the cv conda environment."
                ) from exc

            cls._similarity_model = _SentenceTransformer("all-MiniLM-L6-v2")
        return cls._similarity_model

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int) -> List[str]:
        paragraphs = ChunkExtractor._normalize_paragraphs(text)
        if not paragraphs:
            return []

        overlap = min(80, max(0, chunk_size // 5))
        chunks: List[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    ChunkExtractor._split_long_segment(
                        paragraph,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                )
                continue

            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate

        if current:
            chunks.append(current)

        return ChunkExtractor._dedupe_chunks(chunks)

    @staticmethod
    def _normalize_paragraphs(text: str) -> List[str]:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        raw_paragraphs = re.split(r"\n\s*\n+", normalized)
        if len(raw_paragraphs) == 1:
            raw_paragraphs = normalized.splitlines()

        paragraphs: List[str] = []
        for paragraph in raw_paragraphs:
            cleaned = " ".join(paragraph.split())
            if cleaned:
                paragraphs.append(cleaned)

        if ChunkExtractor._should_merge_fragmented_paragraphs(paragraphs):
            merged = ChunkExtractor._merge_fragmented_paragraphs(paragraphs)
            if merged:
                return [merged]
        return ChunkExtractor._merge_fragmented_latin_runs(paragraphs)

    @staticmethod
    def _should_merge_fragmented_paragraphs(paragraphs: Sequence[str]) -> bool:
        if len(paragraphs) < 6:
            return False

        joined = " ".join(paragraphs)
        letters = [ch for ch in joined if ch.isalpha()]
        if not letters:
            return False

        ascii_letters = [ch for ch in letters if ch.isascii()]
        if len(ascii_letters) / len(letters) < 0.6:
            return False

        word_counts = [len(paragraph.split()) for paragraph in paragraphs]
        if not word_counts:
            return False

        short_ratio = sum(1 for count in word_counts if count <= 2) / len(word_counts)
        avg_words = sum(word_counts) / len(word_counts)
        max_words = max(word_counts)
        return (avg_words <= 2.5 and short_ratio >= 0.7) or (max_words <= 3 and short_ratio >= 0.85)

    @staticmethod
    def _merge_fragmented_paragraphs(paragraphs: Sequence[str]) -> str:
        joined = " ".join(paragraphs)
        joined = re.sub(r"\s+([,.;:!?%)\]])", r"\1", joined)
        joined = re.sub(r"([(\[])\s+", r"\1", joined)
        joined = re.sub(r"\s{2,}", " ", joined)
        return joined.strip()

    @staticmethod
    def _merge_fragmented_latin_runs(paragraphs: Sequence[str]) -> List[str]:
        merged: List[str] = []
        run: List[str] = []

        def flush_run() -> None:
            if not run:
                return
            if len(run) >= 4:
                merged.append(ChunkExtractor._merge_fragmented_paragraphs(run))
            else:
                merged.extend(run)
            run.clear()

        for paragraph in paragraphs:
            if ChunkExtractor._is_fragmented_latin_piece(paragraph):
                run.append(paragraph)
                continue

            prefix_match = re.match(r"^([A-Za-z][A-Za-z0-9.'-]{0,24})(?=[^\x00-\x7f])", paragraph)
            if run and prefix_match:
                run.append(prefix_match.group(1))
                flush_run()
                remainder = paragraph[prefix_match.end():].strip()
                if remainder:
                    merged.append(remainder)
                continue

            flush_run()
            merged.append(paragraph)

        flush_run()
        return merged

    @staticmethod
    def _is_fragmented_latin_piece(paragraph: str) -> bool:
        if len(paragraph) > 32:
            return False
        if len(paragraph.split()) > 2:
            return False

        letters = [ch for ch in paragraph if ch.isalpha()]
        if not letters:
            return False
        ascii_letters = [ch for ch in letters if ch.isascii()]
        return len(ascii_letters) / len(letters) >= 0.6

    @staticmethod
    def _split_long_segment(segment: str, chunk_size: int, overlap: int) -> List[str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?])|(?<=[.!?])\s+", segment)
            if sentence.strip()
        ]
        if len(sentences) <= 1:
            if " " in segment:
                return ChunkExtractor._split_words(
                    segment,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            step = max(1, chunk_size - overlap)
            return [
                segment[index:index + chunk_size]
                for index in range(0, len(segment), step)
                if segment[index:index + chunk_size].strip()
            ]

        chunks: List[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(
                    ChunkExtractor._split_long_segment(
                        sentence,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                )
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _split_words(segment: str, chunk_size: int, overlap: int) -> List[str]:
        words = segment.split()
        chunks: List[str] = []
        current_words: List[str] = []
        current_len = 0

        for word in words:
            added_len = len(word) + (1 if current_words else 0)
            if current_words and current_len + added_len > chunk_size:
                chunks.append(" ".join(current_words))

                overlap_words: List[str] = []
                overlap_len = 0
                for overlap_word in reversed(current_words):
                    word_len = len(overlap_word) + (1 if overlap_words else 0)
                    if overlap_words and overlap_len + word_len > overlap:
                        break
                    overlap_words.insert(0, overlap_word)
                    overlap_len += word_len

                current_words = overlap_words
                current_len = len(" ".join(current_words))

            current_words.append(word)
            current_len += len(word) + (1 if current_len else 0)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    @staticmethod
    def _dedupe_chunks(chunks: Sequence[str]) -> List[str]:
        deduped: List[str] = []
        seen = set()
        for chunk in chunks:
            cleaned = chunk.strip()
            signature = "".join(ch.lower() for ch in cleaned if ch.isalnum())[:1000]
            if not cleaned or signature in seen:
                continue
            seen.add(signature)
            deduped.append(cleaned)
        return deduped

    def _build_chunk_result(
        self,
        chunk_id: int,
        text: str,
        score: float,
        text_score: float,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "text": text,
            "score": float(score),
            "text_score": float(text_score),
            "chunk_size": len(text),
        }

        if isinstance(self.document, dict):
            for key in (
                "url",
                "canonical_url",
                "name",
                "display_url",
                "site_name",
                "image_urls",
                "date_published",
                "content_source",
                "web_fetch_status",
                "web_fetch_from_cache",
                "web_fetch_quality_score",
                "web_fetch_content_hash",
            ):
                if key in self.document:
                    result[key] = self.document[key]

        return result

    def _compute_text_scores(self) -> np.ndarray:
        start_time = time.perf_counter()
        try:
            try:
                model = self._get_similarity_model()
            except RuntimeError as exc:
                print(f"{exc} Falling back to lexical chunk scoring.")
                return self._compute_lexical_text_scores()

            query_emb = model.encode([self.question])[0]
            chunk_embs = model.encode(self.chunks)

            return (chunk_embs @ query_emb) / (
                np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
            )
        finally:
            timing = get_active_timing()
            if timing is not None:
                timing.add("text_retrieval", time.perf_counter() - start_time)

    def _compute_text_scores_from_embeddings(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
    ) -> np.ndarray:
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        chunk_matrix = np.asarray(chunk_embeddings, dtype=np.float32)
        if query_vector.size == 0 or chunk_matrix.size == 0:
            return np.zeros(len(self.chunks), dtype=np.float32)

        if chunk_matrix.ndim != 2:
            raise ValueError("chunk_embeddings must be a 2D matrix")
        if query_vector.ndim != 1:
            raise ValueError("query_embedding must be a 1D vector")
        if chunk_matrix.shape[0] != len(self.chunks):
            raise ValueError("chunk_embeddings row count must match chunk count")

        return (chunk_matrix @ query_vector) / (
            np.linalg.norm(chunk_matrix, axis=1) * np.linalg.norm(query_vector) + 1e-8
        )

    @staticmethod
    def _lexical_tokens(text: str) -> set[str]:
        lowered = str(text or "").lower()
        latin_tokens = re.findall(r"[a-z0-9]+", lowered)
        cjk_tokens = re.findall(r"[\u4e00-\u9fff]", lowered)
        return {token for token in latin_tokens + cjk_tokens if token}

    def _compute_lexical_text_scores(self) -> np.ndarray:
        query_tokens = self._lexical_tokens(self.question)
        if not query_tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)

        scores = []
        for chunk in self.chunks:
            chunk_tokens = self._lexical_tokens(chunk)
            if not chunk_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & chunk_tokens)
            recall = overlap / max(len(query_tokens), 1)
            precision = overlap / max(len(chunk_tokens), 1)
            scores.append(recall * 0.75 + precision * 0.25)

        return np.array(scores, dtype=np.float32)

    def retrieve_text_chunks(
        self,
        top_n: int = 3,
        query_text_embedding: Optional[np.ndarray] = None,
        chunk_text_embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if top_n <= 0 or not self.chunks:
            return []

        if query_text_embedding is not None and chunk_text_embeddings is not None:
            text_scores = self._compute_text_scores_from_embeddings(
                query_embedding=query_text_embedding,
                chunk_embeddings=chunk_text_embeddings,
            )
        else:
            text_scores = self._compute_text_scores()
        order = np.argsort(-text_scores)[:top_n]
        return [
            self._build_chunk_result(
                chunk_id=int(index),
                text=self.chunks[int(index)],
                score=float(text_scores[int(index)]),
                text_score=float(text_scores[int(index)]),
            )
            for index in order
        ]

    def retrieve_multimodal_chunks(
        self,
        top_n: int = 3,
        a: float = 0.5,
        prompt_image_embeddings: Optional[np.ndarray] = None,
        chunk_clip_text_embeddings: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if a < 0 or a > 1:
            raise ValueError("a must be between 0 and 1")
        if top_n <= 0 or not self.chunks:
            return []

        text_scores = self._compute_text_scores()
        image_avg_scores = np.zeros(len(self.chunks), dtype=np.float32)

        if self.image_paths:
            if chunk_clip_text_embeddings is not None:
                text_features = np.asarray(chunk_clip_text_embeddings, dtype=np.float32)
            else:
                clip_client = ClipClient()
                clip_texts = [f"{self.question}\n\n{chunk}" for chunk in self.chunks]
                text_features = clip_client.embed_texts(clip_texts)

            if prompt_image_embeddings is not None:
                image_features = np.asarray(prompt_image_embeddings, dtype=np.float32)
            else:
                clip_client = ClipClient()
                image_features = clip_client.embed_images(self.image_paths)

            image_scores = text_features @ image_features.T
            image_avg_scores = image_scores.mean(axis=1)

        final_scores = text_scores * a + image_avg_scores * (1 - a)
        order = np.argsort(-final_scores)[:top_n]
        return [
            self._build_chunk_result(
                chunk_id=int(index),
                text=self.chunks[int(index)],
                score=float(final_scores[int(index)]),
                text_score=float(text_scores[int(index)]),
            )
            for index in order
        ]
