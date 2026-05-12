from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse

import torch

from rag_v1.config import TEXT_RETRIEVAL_SERVER_CONFIG


if TEXT_RETRIEVAL_SERVER_CONFIG.hf_endpoint:
    os.environ["HF_ENDPOINT"] = TEXT_RETRIEVAL_SERVER_CONFIG.hf_endpoint

from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger("text_retrieval_server")


class TextRetrievalEmbeddingService:
    def __init__(
        self,
        model_id: str,
        local_model_dir: Path,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.local_model_dir = local_model_dir
        self.batch_size = max(int(batch_size), 1)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        model_path: str | Path
        if (self.local_model_dir / "modules.json").exists():
            model_path = self.local_model_dir
        else:
            self.local_model_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.model_id

        model = SentenceTransformer(
            model_path,
            cache_folder=str(self.local_model_dir.parent),
            device=self.device,
        )

        if model_path == self.model_id:
            model.save(str(self.local_model_dir))

        return model

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        normalized_texts = [str(text) for text in texts if str(text).strip()]
        if not normalized_texts:
            raise ValueError("texts must not be empty")

        embeddings = self.model.encode(
            normalized_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32").tolist()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}

    raw_body = handler.rfile.read(content_length)
    return json.loads(raw_body.decode("utf-8"))


def create_handler(service: TextRetrievalEmbeddingService) -> type[BaseHTTPRequestHandler]:
    class TextRetrievalRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path != "/health":
                _json_response(self, 404, {"error": "not found"})
                return

            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "model_id": service.model_id,
                    "device": service.device,
                    "batch_size": service.batch_size,
                },
            )

        def do_POST(self) -> None:
            path = urlparse(self.path).path

            try:
                payload = _read_json_body(self)
                if path == "/embed/texts":
                    embeddings = service.embed_texts(payload.get("texts") or [])
                    _json_response(self, 200, {"embeddings": embeddings})
                    return

                _json_response(self, 404, {"error": "not found"})
            except Exception as exc:
                LOGGER.exception("Text retrieval request failed")
                _json_response(self, 500, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

    return TextRetrievalRequestHandler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Using Hugging Face endpoint: %s", os.environ.get("HF_ENDPOINT", "default"))
    service = TextRetrievalEmbeddingService(
        model_id=TEXT_RETRIEVAL_SERVER_CONFIG.model_id,
        local_model_dir=TEXT_RETRIEVAL_SERVER_CONFIG.local_model_dir,
        batch_size=TEXT_RETRIEVAL_SERVER_CONFIG.batch_size,
    )
    server = ThreadingHTTPServer(
        (TEXT_RETRIEVAL_SERVER_CONFIG.host, TEXT_RETRIEVAL_SERVER_CONFIG.port),
        create_handler(service),
    )
    LOGGER.info(
        "Text retrieval server listening on http://%s:%s",
        TEXT_RETRIEVAL_SERVER_CONFIG.host,
        TEXT_RETRIEVAL_SERVER_CONFIG.port,
    )
    server.serve_forever()
