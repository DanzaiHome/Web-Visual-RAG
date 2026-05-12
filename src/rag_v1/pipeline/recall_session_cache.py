from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from rag_v1.clients.clip import ClipClient
from rag_v1.clients.text_retrieval import TextRetrievalClient
from rag_v1.config import CLIP_SERVER_CONFIG, PROJECT_ROOT, TEXT_RETRIEVAL_SERVER_CONFIG
from rag_v1.retrieval.chunk_extractor import ChunkExtractor
from rag_v1.services.web_page_fetcher import is_valid_page_image_url
from rag_v1.timing import get_active_timing


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name or "").strip())
    return sanitized.strip("_") or "clip_model"


def _now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")


def _doc_key(doc: Dict[str, object]) -> str:
    return str(doc.get("canonical_url") or doc.get("url") or "").strip()


def _matrix_from_payload(values: object) -> np.ndarray:
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _matrix_to_payload(values: np.ndarray) -> List[List[float]]:
    if values.size == 0:
        return []
    return values.astype(np.float32).tolist()


def _vector_to_payload(values: np.ndarray) -> List[float]:
    if values.size == 0:
        return []
    return values.astype(np.float32).tolist()


def _question_hash(question: str) -> str:
    return sha256(str(question or "").encode("utf-8")).hexdigest()[:16]


def _text_retrieval_model_name() -> str:
    return TEXT_RETRIEVAL_SERVER_CONFIG.model_id


@dataclass
class RecallSessionCache:
    session_path: Path
    payload: Dict[str, object]
    clip_client: ClipClient = field(default_factory=ClipClient)
    text_retrieval_client: TextRetrievalClient = field(default_factory=TextRetrievalClient)
    dirty: bool = False

    @classmethod
    def create(
        cls,
        question: str,
        img_paths: Sequence[Union[str, Path]],
        use_multimodal: bool,
    ) -> "RecallSessionCache":
        timing = get_active_timing()
        if timing is None:
            return cls._create_impl(question=question, img_paths=img_paths, use_multimodal=use_multimodal)
        with timing.scope("session_cache.create", label="RecallSessionCache.create"):
            return cls._create_impl(question=question, img_paths=img_paths, use_multimodal=use_multimodal)

    @classmethod
    def _create_impl(
        cls,
        question: str,
        img_paths: Sequence[Union[str, Path]],
        use_multimodal: bool,
    ) -> "RecallSessionCache":
        cache_dir = (PROJECT_ROOT / "cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{_sanitize_model_name(CLIP_SERVER_CONFIG.model_id)}_{_now_stamp()}.json"
        session_path = cache_dir / filename
        payload: Dict[str, object] = {
            "session_type": "visual_rag_recall",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "clip_model_id": CLIP_SERVER_CONFIG.model_id,
            "text_retrieval_model_id": _text_retrieval_model_name(),
            "question": str(question or ""),
            "question_hash": _question_hash(question),
            "use_multimodal": bool(use_multimodal),
            "prompt_images": [str(path) for path in img_paths if str(path or "").strip()],
            "prompt_image_embeddings": [],
            "question_clip_text_embedding": [],
            "query_text_retrieval_embeddings": {},
            "queries": [],
            "docs": {},
        }
        session = cls(session_path=session_path, payload=payload, dirty=True)
        session.flush(force=True)
        return session

    def mark_dirty(self) -> None:
        self.dirty = True

    def flush(self, force: bool = False) -> None:
        if not force and not self.dirty:
            return
        timing = get_active_timing()
        if timing is None:
            self._flush_impl()
            return
        with timing.scope("session_cache.flush_io", label=f"session_cache.flush_io[{self.session_path.name}]"):
            self._flush_impl()

    def _flush_impl(self) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        with self.session_path.open("w", encoding="utf-8") as file:
            json.dump(self.payload, file, ensure_ascii=False, indent=2)
        self.dirty = False

    def ensure_prompt_image_embeddings(
        self,
        img_paths: Sequence[Union[str, Path]],
    ) -> np.ndarray:
        timing = get_active_timing()
        if timing is None:
            return self._ensure_prompt_image_embeddings_impl(img_paths)
        with timing.scope("session_cache.ensure_prompt_image_embeddings", label="ensure_prompt_image_embeddings"):
            return self._ensure_prompt_image_embeddings_impl(img_paths)

    def _ensure_prompt_image_embeddings_impl(
        self,
        img_paths: Sequence[Union[str, Path]],
    ) -> np.ndarray:
        cached = _matrix_from_payload(self.payload.get("prompt_image_embeddings"))
        if cached.size > 0:
            return cached

        prompt_images = [str(path) for path in img_paths if str(path or "").strip()]
        if not prompt_images:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = self.clip_client.embed_images(prompt_images)
        self.payload["prompt_image_embeddings"] = _matrix_to_payload(embeddings)
        self.mark_dirty()
        return embeddings

    def ensure_question_text_embedding(self, question: str) -> np.ndarray:
        timing = get_active_timing()
        if timing is None:
            return self._ensure_question_text_embedding_impl(question)
        with timing.scope("session_cache.ensure_question_text_embedding", label="ensure_question_text_embedding"):
            return self._ensure_question_text_embedding_impl(question)

    def _ensure_question_text_embedding_impl(self, question: str) -> np.ndarray:
        cached = np.asarray(
            self.payload.get("question_clip_text_embedding") or [],
            dtype=np.float32,
        )
        if cached.size > 0:
            return cached

        if not str(question or "").strip():
            return np.zeros((0,), dtype=np.float32)

        embedding = self.clip_client.embed_texts([str(question)])[0]
        vector = np.asarray(embedding, dtype=np.float32)
        self.payload["question_clip_text_embedding"] = _vector_to_payload(vector)
        self.mark_dirty()
        return vector

    def ensure_query_text_retrieval_embedding(self, query: str) -> np.ndarray:
        timing = get_active_timing()
        if timing is None:
            return self._ensure_query_text_retrieval_embedding_impl(query)
        with timing.scope(
            "session_cache.ensure_query_text_retrieval_embedding",
            label="ensure_query_text_retrieval_embedding",
        ):
            return self._ensure_query_text_retrieval_embedding_impl(query)

    def _ensure_query_text_retrieval_embedding_impl(self, query: str) -> np.ndarray:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return np.zeros((0,), dtype=np.float32)

        query_payload = self.payload.setdefault("query_text_retrieval_embeddings", {})
        if not isinstance(query_payload, dict):
            query_payload = {}
            self.payload["query_text_retrieval_embeddings"] = query_payload

        query_key = _question_hash(normalized_query)
        cached = np.asarray(query_payload.get(query_key) or [], dtype=np.float32)
        if cached.size > 0:
            return cached

        embedding = self.text_retrieval_client.embed_texts([normalized_query])[0]
        embedding = np.asarray(embedding, dtype=np.float32)
        query_payload[query_key] = _vector_to_payload(embedding)
        self.mark_dirty()
        return embedding

    def record_query(self, query: str, freshness: str) -> None:
        queries = self.payload.setdefault("queries", [])
        if not isinstance(queries, list):
            queries = []
            self.payload["queries"] = queries
        queries.append(
            {
                "query": str(query or ""),
                "freshness": str(freshness or ""),
                "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        self.mark_dirty()

    def register_docs(self, docs: Sequence[Dict[str, object]]) -> None:
        timing = get_active_timing()
        if timing is not None:
            with timing.scope("session_cache.register_docs", label=f"register_docs[{len(docs)}]"):
                self._register_docs_impl(docs)
            return
        self._register_docs_impl(docs)

    def _register_docs_impl(self, docs: Sequence[Dict[str, object]]) -> None:
        docs_payload = self.payload.setdefault("docs", {})
        if not isinstance(docs_payload, dict):
            docs_payload = {}
            self.payload["docs"] = docs_payload

        for doc in docs:
            doc_key = _doc_key(doc)
            if not doc_key:
                continue

            existing = docs_payload.get(doc_key)
            if not isinstance(existing, dict):
                existing = {
                    "url": str(doc.get("url") or ""),
                    "canonical_url": str(doc.get("canonical_url") or ""),
                    "name": str(doc.get("name") or ""),
                    "site_name": str(doc.get("site_name") or ""),
                    "display_url": str(doc.get("display_url") or ""),
                    "content_source": str(doc.get("content_source") or ""),
                    "web_fetch_status": str(doc.get("web_fetch_status") or ""),
                    "web_fetch_quality_score": float(doc.get("web_fetch_quality_score") or 0.0),
                    "date_published": str(doc.get("date_published") or ""),
                    "date_last_crawled": str(doc.get("date_last_crawled") or ""),
                    "full_content": str(doc.get("full_content") or doc.get("content") or ""),
                    "image_urls": list(doc.get("image_urls") or []),
                    "page_image_embeddings": [],
                    "page_image_embedding_urls": [],
                    "page_image_failed_urls": [],
                    "page_image_embeddings_ready": False,
                    "chunk_profiles": {},
                }
                docs_payload[doc_key] = existing
            else:
                existing["full_content"] = str(doc.get("full_content") or doc.get("content") or "")
                existing["image_urls"] = list(doc.get("image_urls") or [])
                existing["content_source"] = str(doc.get("content_source") or existing.get("content_source") or "")
                existing["web_fetch_status"] = str(doc.get("web_fetch_status") or existing.get("web_fetch_status") or "")
                existing["web_fetch_quality_score"] = float(
                    doc.get("web_fetch_quality_score") or existing.get("web_fetch_quality_score") or 0.0
                )

        self.mark_dirty()

    def populate_doc_page_image_embeddings(self, doc: Dict[str, object]) -> None:
        timing = get_active_timing()
        doc_label = str(doc.get("url") or doc.get("name") or "doc")[:60]
        if timing is not None:
            with timing.scope(
                "session_cache.populate_doc_page_image_embeddings",
                label=f"populate_doc_page_image_embeddings[{doc_label}]",
            ):
                self._populate_doc_page_image_embeddings_impl(doc)
            return
        self._populate_doc_page_image_embeddings_impl(doc)

    def _populate_doc_page_image_embeddings_impl(self, doc: Dict[str, object]) -> None:
        doc_key = _doc_key(doc)
        if not doc_key:
            doc["clip_page_image_embeddings"] = []
            doc["clip_page_image_embedding_urls"] = []
            doc["clip_page_image_failed_urls"] = []
            return

        docs_payload = self.payload.setdefault("docs", {})
        record = docs_payload.get(doc_key)
        if not isinstance(record, dict):
            self.register_docs([doc])
            record = self.payload["docs"].get(doc_key, {})

        cached_urls = list(record.get("page_image_embedding_urls") or [])
        cached_embeddings = _matrix_from_payload(record.get("page_image_embeddings"))
        cached_failures = list(record.get("page_image_failed_urls") or [])
        ready = bool(record.get("page_image_embeddings_ready"))

        if ready and (
            (cached_urls and cached_embeddings.shape[0] == len(cached_urls))
            or (not cached_urls and cached_embeddings.size == 0)
        ):
            doc["clip_page_image_embeddings"] = _matrix_to_payload(cached_embeddings)
            doc["clip_page_image_embedding_urls"] = cached_urls
            doc["clip_page_image_failed_urls"] = cached_failures
            return

        image_urls = [
            str(url).strip()
            for url in (doc.get("image_urls") or [])
            if str(url or "").strip() and is_valid_page_image_url(url)
        ]
        if not image_urls:
            record["page_image_embeddings"] = []
            record["page_image_embedding_urls"] = []
            record["page_image_failed_urls"] = []
            record["page_image_embeddings_ready"] = True
            doc["clip_page_image_embeddings"] = []
            doc["clip_page_image_embedding_urls"] = []
            doc["clip_page_image_failed_urls"] = []
            self.mark_dirty()
            return

        embedding_rows: List[np.ndarray] = []
        embedded_urls: List[str] = []
        failed_urls: List[str] = []

        for image_url in image_urls:
            try:
                embedding = self.clip_client.embed_images([image_url])
            except Exception:
                failed_urls.append(image_url)
                continue
            if len(embedding) == 0:
                failed_urls.append(image_url)
                continue
            embedding_rows.append(np.asarray(embedding[0], dtype=np.float32))
            embedded_urls.append(image_url)

        matrix = np.vstack(embedding_rows) if embedding_rows else np.zeros((0, 0), dtype=np.float32)
        record["page_image_embeddings"] = _matrix_to_payload(matrix)
        record["page_image_embedding_urls"] = embedded_urls
        record["page_image_failed_urls"] = failed_urls
        record["page_image_embeddings_ready"] = True
        doc["clip_page_image_embeddings"] = record["page_image_embeddings"]
        doc["clip_page_image_embedding_urls"] = embedded_urls
        doc["clip_page_image_failed_urls"] = failed_urls
        self.mark_dirty()

    def get_chunk_profile(
        self,
        doc: Dict[str, object],
        question: str,
        chunk_size: int,
        use_multimodal: bool,
    ) -> Dict[str, object]:
        timing = get_active_timing()
        doc_label = str(doc.get("url") or doc.get("name") or "doc")[:60]
        if timing is not None:
            with timing.scope("session_cache.get_chunk_profile", label=f"get_chunk_profile[{doc_label}]"):
                return self._get_chunk_profile_impl(
                    doc=doc,
                    question=question,
                    chunk_size=chunk_size,
                    use_multimodal=use_multimodal,
                )
        return self._get_chunk_profile_impl(
            doc=doc,
            question=question,
            chunk_size=chunk_size,
            use_multimodal=use_multimodal,
        )

    def _get_chunk_profile_impl(
        self,
        doc: Dict[str, object],
        question: str,
        chunk_size: int,
        use_multimodal: bool,
    ) -> Dict[str, object]:
        doc_key = _doc_key(doc)
        if not doc_key:
            return {"chunks": [], "clip_text_embeddings": []}

        docs_payload = self.payload.setdefault("docs", {})
        record = docs_payload.get(doc_key)
        if not isinstance(record, dict):
            self.register_docs([doc])
            record = self.payload["docs"].get(doc_key, {})

        chunk_profiles = record.setdefault("chunk_profiles", {})
        profile_key = f"q={self.payload.get('question_hash')}|size={int(chunk_size)}"
        existing = chunk_profiles.get(profile_key)
        if isinstance(existing, dict):
            if not existing.get("text_retrieval_chunk_embeddings"):
                existing = None
            elif use_multimodal and not existing.get("clip_text_embeddings"):
                existing = None
            else:
                return existing

        extractor = ChunkExtractor(
            document=doc,
            question=question,
            image_paths=[],
            chunk_size=chunk_size,
        )
        chunks = list(extractor.chunks)

        profile: Dict[str, object] = {
            "question": str(question or ""),
            "chunk_size": int(chunk_size),
            "chunks": [
                {
                    "chunk_id": index,
                    "text": chunk,
                }
                for index, chunk in enumerate(chunks)
            ],
            "clip_text_embeddings": [],
            "text_retrieval_chunk_embeddings": [],
        }

        if chunks:
            text_embeddings = self.text_retrieval_client.embed_texts(chunks)
            profile["text_retrieval_chunk_embeddings"] = _matrix_to_payload(text_embeddings)

        if use_multimodal and chunks:
            clip_texts = [f"{question}\n\n{chunk}" for chunk in chunks]
            embeddings = self.clip_client.embed_texts(clip_texts)
            profile["clip_text_embeddings"] = _matrix_to_payload(embeddings)

        chunk_profiles[profile_key] = profile
        self.mark_dirty()
        return profile
