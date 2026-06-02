from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import requests

from rag_v1.clients.clip import ClipClient
from rag_v1.clients.text_retrieval import TextRetrievalClient
from rag_v1.config import CLIP_SERVER_CONFIG, PROJECT_ROOT, TEXT_RETRIEVAL_SERVER_CONFIG
from rag_v1.retrieval.chunk_extractor import ChunkExtractor
from rag_v1.services.web_page_fetcher import is_valid_page_image_url
from rag_v1.timing import get_active_timing


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name or "").strip())
    return sanitized.strip("_") or "clip_model"


def _sanitize_filename(value: str, default: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return sanitized.strip("_") or default


def _now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")


def _doc_key(doc: Dict[str, object]) -> str:
    return str(doc.get("canonical_url") or doc.get("url") or "").strip()


def _doc_hash(doc_key: str) -> str:
    return sha256(str(doc_key or "").encode("utf-8")).hexdigest()[:16]


def _question_hash(question: str) -> str:
    return sha256(str(question or "").encode("utf-8")).hexdigest()[:16]


def _page_image_hash(image_url: str) -> str:
    return sha256(str(image_url or "").encode("utf-8")).hexdigest()[:16]


def _text_retrieval_model_name() -> str:
    return TEXT_RETRIEVAL_SERVER_CONFIG.model_id


class PageImageDownloader:
    def __init__(self, timeout: int = CLIP_SERVER_CONFIG.image_timeout) -> None:
        self.timeout = timeout

    @staticmethod
    def _download_headers() -> Dict[str, str]:
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36 RAGPageImageDownloader/1.0"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }

    def download(self, image_url: str, destination: Path) -> bool:
        if destination.exists():
            return True

        try:
            response = requests.get(
                image_url,
                headers=self._download_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout:
            return False
        except requests.RequestException:
            return False

        content_type = str(response.headers.get("Content-Type") or "").lower()
        if content_type and "image" not in content_type:
            return False

        content = response.content
        if not content:
            return False

        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as file:
                file.write(content)
            tmp_path.replace(destination)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False

        return True


@dataclass
class RecallSessionCache:
    session_dir: Path
    session_path: Path
    payload: Dict[str, object]
    clip_client: ClipClient = field(default_factory=ClipClient)
    text_retrieval_client: TextRetrievalClient = field(default_factory=TextRetrievalClient)
    page_image_downloader: PageImageDownloader = field(default_factory=PageImageDownloader)
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
        cache_root = (PROJECT_ROOT / "cache").resolve()
        cache_root.mkdir(parents=True, exist_ok=True)

        session_name = f"{_sanitize_model_name(CLIP_SERVER_CONFIG.model_id)}_{_now_stamp()}"
        session_dir = cache_root / session_name
        session_path = session_dir / "manifest.json"
        payload: Dict[str, object] = {
            "session_type": "visual_rag_recall",
            "storage_format": "manifest+npy",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "clip_model_id": CLIP_SERVER_CONFIG.model_id,
            "text_retrieval_model_id": _text_retrieval_model_name(),
            "question": str(question or ""),
            "question_hash": _question_hash(question),
            "use_multimodal": bool(use_multimodal),
            "prompt_images": [str(path) for path in img_paths if str(path or "").strip()],
            "prompt_image_embeddings_file": "",
            "question_clip_text_embedding_file": "",
            "query_text_retrieval_embeddings": {},
            "queries": [],
            "docs": {},
        }
        session = cls(
            session_dir=session_dir,
            session_path=session_path,
            payload=payload,
            dirty=True,
        )
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
        self.session_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.session_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(self.payload, file, ensure_ascii=False, indent=2)
        tmp_path.replace(self.session_path)
        self.dirty = False

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.session_dir).as_posix()

    def _absolute_path(self, relative_path: object) -> Optional[Path]:
        normalized = str(relative_path or "").strip()
        if not normalized:
            return None
        return self.session_dir / Path(normalized)

    def _npy_path(self, *parts: str) -> Path:
        return self.session_dir.joinpath(*parts)

    def _write_array(self, path: Path, values: np.ndarray) -> str:
        array = np.asarray(values, dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as file:
            np.save(file, array, allow_pickle=False)
        tmp_path.replace(path)
        return self._relative_path(path)

    def _load_array(self, relative_path: object) -> np.ndarray:
        path = self._absolute_path(relative_path)
        if path is None or not path.exists():
            return np.zeros((0,), dtype=np.float32)
        with path.open("rb") as file:
            return np.asarray(np.load(file, allow_pickle=False), dtype=np.float32)

    def _clear_array_file(self, relative_path: object) -> str:
        path = self._absolute_path(relative_path)
        if path is not None:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass
        return ""

    def _store_array_or_clear(self, path: Path, values: np.ndarray, existing: object = "") -> str:
        array = np.asarray(values, dtype=np.float32)
        if array.size == 0:
            return self._clear_array_file(existing)
        return self._write_array(path, array)

    def _doc_dir(self, doc_key: str) -> Path:
        return self.session_dir / "docs" / _doc_hash(doc_key)

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
        cached = self._load_array(self.payload.get("prompt_image_embeddings_file"))
        if cached.size > 0:
            return cached

        prompt_images = [str(path) for path in img_paths if str(path or "").strip()]
        if not prompt_images:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings = np.asarray(self.clip_client.embed_images(prompt_images), dtype=np.float32)
        self.payload["prompt_image_embeddings_file"] = self._write_array(
            self._npy_path("prompt_image_embeddings.npy"),
            embeddings,
        )
        self.mark_dirty()
        return embeddings

    def ensure_question_text_embedding(self, question: str) -> np.ndarray:
        timing = get_active_timing()
        if timing is None:
            return self._ensure_question_text_embedding_impl(question)
        with timing.scope("session_cache.ensure_question_text_embedding", label="ensure_question_text_embedding"):
            return self._ensure_question_text_embedding_impl(question)

    def _ensure_question_text_embedding_impl(self, question: str) -> np.ndarray:
        cached = self._load_array(self.payload.get("question_clip_text_embedding_file"))
        if cached.size > 0:
            return cached.reshape(-1)

        if not str(question or "").strip():
            return np.zeros((0,), dtype=np.float32)

        embedding = np.asarray(self.clip_client.embed_texts([str(question)])[0], dtype=np.float32)
        self.payload["question_clip_text_embedding_file"] = self._write_array(
            self._npy_path("question_clip_text_embedding.npy"),
            embedding,
        )
        self.mark_dirty()
        return embedding

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
        cached = self._load_array(query_payload.get(query_key))
        if cached.size > 0:
            return cached.reshape(-1)

        embedding = np.asarray(self.text_retrieval_client.embed_texts([normalized_query])[0], dtype=np.float32)
        query_payload[query_key] = self._write_array(
            self._npy_path("queries", f"{query_key}.npy"),
            embedding,
        )
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
                    "page_image_local_files": {},
                    "page_image_embeddings_file": "",
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
        self.populate_docs_page_image_embeddings([doc])

    def populate_docs_page_image_embeddings(self, docs: Sequence[Dict[str, object]]) -> None:
        timing = get_active_timing()
        if timing is not None:
            with timing.scope(
                "session_cache.populate_doc_page_image_embeddings",
                label=f"populate_doc_page_image_embeddings[{len(docs)} docs]",
            ):
                self._populate_docs_page_image_embeddings_impl(docs)
            return
        self._populate_docs_page_image_embeddings_impl(docs)

    def _populate_doc_page_image_embeddings_impl(self, doc: Dict[str, object]) -> None:
        self._populate_docs_page_image_embeddings_impl([doc])

    def _page_image_local_path(self, doc_key: str, image_url: str) -> Path:
        return self._doc_dir(doc_key) / "page_images" / f"{_page_image_hash(image_url)}.img"

    def _hydrate_page_image_embedding_state(
        self,
        doc: Dict[str, object],
        record: Dict[str, object],
        embeddings: np.ndarray,
        embedded_urls: Sequence[str],
        failed_urls: Sequence[str],
    ) -> None:
        doc["clip_page_image_embeddings"] = embeddings
        doc["clip_page_image_embedding_urls"] = list(embedded_urls)
        doc["clip_page_image_failed_urls"] = list(failed_urls)
        doc["clip_page_image_embeddings_ready"] = bool(record.get("page_image_embeddings_ready"))

    def _populate_docs_page_image_embeddings_impl(self, docs: Sequence[Dict[str, object]]) -> None:
        docs_payload = self.payload.setdefault("docs", {})
        if not isinstance(docs_payload, dict):
            docs_payload = {}
            self.payload["docs"] = docs_payload

        pending_docs: List[Dict[str, object]] = []
        download_targets: Dict[str, Path] = {}

        for doc in docs:
            doc_key = _doc_key(doc)
            if not doc_key:
                doc["clip_page_image_embeddings"] = np.zeros((0, 0), dtype=np.float32)
                doc["clip_page_image_embedding_urls"] = []
                doc["clip_page_image_failed_urls"] = []
                doc["clip_page_image_embeddings_ready"] = True
                continue

            record = docs_payload.get(doc_key)
            if not isinstance(record, dict):
                self.register_docs([doc])
                record = self.payload["docs"].get(doc_key, {})

            cached_urls = list(record.get("page_image_embedding_urls") or [])
            cached_embeddings = self._load_array(record.get("page_image_embeddings_file"))
            cached_failures = list(record.get("page_image_failed_urls") or [])
            ready = bool(record.get("page_image_embeddings_ready"))

            if ready and (
                (cached_urls and cached_embeddings.ndim == 2 and cached_embeddings.shape[0] == len(cached_urls))
                or (not cached_urls and cached_embeddings.size == 0)
            ):
                self._hydrate_page_image_embedding_state(
                    doc=doc,
                    record=record,
                    embeddings=cached_embeddings,
                    embedded_urls=cached_urls,
                    failed_urls=cached_failures,
                )
                continue

            image_urls = [
                str(url).strip()
                for url in (doc.get("image_urls") or [])
                if str(url or "").strip() and is_valid_page_image_url(url)
            ]
            if not image_urls:
                record["page_image_local_files"] = {}
                record["page_image_embeddings_file"] = self._clear_array_file(record.get("page_image_embeddings_file"))
                record["page_image_embedding_urls"] = []
                record["page_image_failed_urls"] = []
                record["page_image_embeddings_ready"] = True
                self._hydrate_page_image_embedding_state(
                    doc=doc,
                    record=record,
                    embeddings=np.zeros((0, 0), dtype=np.float32),
                    embedded_urls=[],
                    failed_urls=[],
                )
                self.mark_dirty()
                continue

            local_files = record.get("page_image_local_files")
            if not isinstance(local_files, dict):
                local_files = {}
                record["page_image_local_files"] = local_files

            doc_pending = {
                "doc": doc,
                "doc_key": doc_key,
                "record": record,
                "image_urls": image_urls,
            }
            pending_docs.append(doc_pending)

            for image_url in image_urls:
                cached_relpath = str(local_files.get(image_url) or "").strip()
                cached_path = self._absolute_path(cached_relpath)
                if cached_path is not None and cached_path.exists():
                    continue
                download_targets.setdefault(image_url, self._page_image_local_path(doc_key, image_url))

        download_results: Dict[str, Optional[Path]] = {}
        if download_targets:
            timing = get_active_timing()
            if timing is not None:
                with timing.scope(
                    "session_cache.page_image_download",
                    label=f"page_image_download[{len(download_targets)} images]",
                ):
                    download_results = self._download_page_images(download_targets)
            else:
                download_results = self._download_page_images(download_targets)

        for pending in pending_docs:
            doc = pending["doc"]
            doc_key = str(pending["doc_key"])
            record = pending["record"]
            image_urls = list(pending["image_urls"])
            local_files = record.get("page_image_local_files")
            if not isinstance(local_files, dict):
                local_files = {}
                record["page_image_local_files"] = local_files

            local_paths: List[Path] = []
            embedded_urls: List[str] = []
            failed_urls: List[str] = []

            for image_url in image_urls:
                cached_relpath = str(local_files.get(image_url) or "").strip()
                cached_path = self._absolute_path(cached_relpath)
                if cached_path is not None and cached_path.exists():
                    local_paths.append(cached_path)
                    embedded_urls.append(image_url)
                    continue

                downloaded_path = download_results.get(image_url)
                if downloaded_path is None or not downloaded_path.exists():
                    failed_urls.append(image_url)
                    continue

                local_files[image_url] = self._relative_path(downloaded_path)
                local_paths.append(downloaded_path)
                embedded_urls.append(image_url)

            embeddings, embedded_urls, embed_failures = self._embed_cached_page_images(local_paths, embedded_urls)
            failed_urls.extend(embed_failures)

            record["page_image_embeddings_file"] = self._store_array_or_clear(
                self._doc_dir(doc_key) / "page_image_embeddings.npy",
                embeddings,
                existing=record.get("page_image_embeddings_file"),
            )
            record["page_image_embedding_urls"] = embedded_urls
            record["page_image_failed_urls"] = failed_urls
            record["page_image_embeddings_ready"] = True
            self._hydrate_page_image_embedding_state(
                doc=doc,
                record=record,
                embeddings=embeddings,
                embedded_urls=embedded_urls,
                failed_urls=failed_urls,
            )
            self.mark_dirty()

    def _download_page_images(self, download_targets: Dict[str, Path]) -> Dict[str, Optional[Path]]:
        if not download_targets:
            return {}

        results: Dict[str, Optional[Path]] = {}
        max_workers = min(12, max(2, len(download_targets)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.page_image_downloader.download, image_url, destination): image_url
                for image_url, destination in download_targets.items()
            }
            for future in as_completed(future_to_url):
                image_url = future_to_url[future]
                destination = download_targets[image_url]
                try:
                    ok = bool(future.result())
                except Exception:
                    ok = False
                results[image_url] = destination if ok and destination.exists() else None

        return results

    def _embed_cached_page_images(
        self,
        local_paths: Sequence[Path],
        image_urls: Sequence[str],
    ) -> tuple[np.ndarray, List[str], List[str]]:
        if not local_paths or not image_urls:
            return np.zeros((0, 0), dtype=np.float32), [], []

        try:
            embeddings = np.asarray(
                self.clip_client.embed_images([str(path) for path in local_paths]),
                dtype=np.float32,
            )
            if embeddings.ndim == 2 and embeddings.shape[0] == len(image_urls):
                return embeddings, list(image_urls), []
        except Exception:
            pass

        embedding_rows: List[np.ndarray] = []
        embedded_urls: List[str] = []
        failed_urls: List[str] = []
        for image_url, local_path in zip(image_urls, local_paths):
            try:
                embedding = self.clip_client.embed_images([str(local_path)])
            except Exception:
                failed_urls.append(image_url)
                continue
            if len(embedding) == 0:
                failed_urls.append(image_url)
                continue
            embedding_rows.append(np.asarray(embedding[0], dtype=np.float32))
            embedded_urls.append(image_url)

        matrix = np.vstack(embedding_rows) if embedding_rows else np.zeros((0, 0), dtype=np.float32)
        return matrix, embedded_urls, failed_urls

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

    def _hydrate_chunk_profile(self, profile: Dict[str, object]) -> Dict[str, object]:
        hydrated = dict(profile)
        hydrated["clip_text_embeddings"] = self._load_array(profile.get("clip_text_embeddings_file"))
        hydrated["text_retrieval_chunk_embeddings"] = self._load_array(
            profile.get("text_retrieval_chunk_embeddings_file")
        )
        return hydrated

    def _get_chunk_profile_impl(
        self,
        doc: Dict[str, object],
        question: str,
        chunk_size: int,
        use_multimodal: bool,
    ) -> Dict[str, object]:
        doc_key = _doc_key(doc)
        if not doc_key:
            return {
                "chunks": [],
                "clip_text_embeddings": np.zeros((0, 0), dtype=np.float32),
                "text_retrieval_chunk_embeddings": np.zeros((0, 0), dtype=np.float32),
            }

        docs_payload = self.payload.setdefault("docs", {})
        record = docs_payload.get(doc_key)
        if not isinstance(record, dict):
            self.register_docs([doc])
            record = self.payload["docs"].get(doc_key, {})

        chunk_profiles = record.setdefault("chunk_profiles", {})
        profile_key = f"q={self.payload.get('question_hash')}|size={int(chunk_size)}"
        existing = chunk_profiles.get(profile_key)
        if isinstance(existing, dict):
            text_embeddings = self._load_array(existing.get("text_retrieval_chunk_embeddings_file"))
            clip_embeddings = self._load_array(existing.get("clip_text_embeddings_file"))
            if text_embeddings.size > 0 and (not use_multimodal or clip_embeddings.size > 0):
                return self._hydrate_chunk_profile(existing)

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
            "clip_text_embeddings_file": "",
            "text_retrieval_chunk_embeddings_file": "",
        }

        profile_dir = self._doc_dir(doc_key) / "chunk_profiles" / _sanitize_filename(profile_key, "profile")

        if chunks:
            text_embeddings = np.asarray(self.text_retrieval_client.embed_texts(chunks), dtype=np.float32)
            profile["text_retrieval_chunk_embeddings_file"] = self._write_array(
                profile_dir / "text_retrieval_chunk_embeddings.npy",
                text_embeddings,
            )

        if use_multimodal and chunks:
            clip_texts = [f"{question}\n\n{chunk}" for chunk in chunks]
            embeddings = np.asarray(self.clip_client.embed_texts(clip_texts), dtype=np.float32)
            profile["clip_text_embeddings_file"] = self._write_array(
                profile_dir / "clip_text_embeddings.npy",
                embeddings,
            )

        chunk_profiles[profile_key] = profile
        self.mark_dirty()
        return self._hydrate_chunk_profile(profile)
