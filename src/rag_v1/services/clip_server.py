from __future__ import annotations

import base64
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse

import requests
import torch
from PIL import Image

from rag_v1.config import CLIP_SERVER_CONFIG


if CLIP_SERVER_CONFIG.hf_endpoint:
    os.environ["HF_ENDPOINT"] = CLIP_SERVER_CONFIG.hf_endpoint

from transformers import CLIPModel, CLIPProcessor


LOGGER = logging.getLogger("clip_server")


class ClipEmbeddingService:
    def __init__(
        self,
        model_id: str,
        local_model_dir: Path,
        image_timeout: int = 20,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.local_model_dir = local_model_dir
        self.image_timeout = image_timeout
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> tuple[CLIPModel, CLIPProcessor]:
        if (self.local_model_dir / "config.json").exists():
            model = CLIPModel.from_pretrained(self.local_model_dir)
            processor = CLIPProcessor.from_pretrained(self.local_model_dir)
            return model, processor

        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = str(self.local_model_dir.parent)
        model = CLIPModel.from_pretrained(self.model_id, cache_dir=cache_dir)
        processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        model.save_pretrained(self.local_model_dir)
        processor.save_pretrained(self.local_model_dir)
        return model, processor

    def load_image(self, image_address: str) -> Image.Image:
        if image_address.startswith(("http://", "https://")):
            response = requests.get(image_address, timeout=self.image_timeout)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")

        if image_address.startswith("data:image/"):
            _, encoded = image_address.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")

        return Image.open(image_address).convert("RGB")

    @staticmethod
    def _is_valid_image(image: Image.Image) -> bool:
        width, height = image.size
        return width > 1 and height > 1

    @staticmethod
    def _maybe_project_features(features: torch.Tensor, projection: Any = None) -> torch.Tensor:
        if projection is None:
            return features

        in_features = getattr(projection, "in_features", None)
        if in_features is not None and features.shape[-1] != in_features:
            return features

        return projection(features)

    @classmethod
    def _coerce_feature_tensor(cls, output: Any, projection: Any = None) -> torch.Tensor:
        if torch.is_tensor(output):
            return output

        if hasattr(output, "image_embeds") and torch.is_tensor(output.image_embeds):
            return output.image_embeds

        if hasattr(output, "text_embeds") and torch.is_tensor(output.text_embeds):
            return output.text_embeds

        if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
            features = output.pooler_output
            return cls._maybe_project_features(features, projection)

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item

        raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")

    @staticmethod
    def _normalize_features(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _is_valid_image(image: Image.Image) -> bool:
        width, height = image.size
        return width > 1 and height > 1

    def embed_images(self, image_addresses: Sequence[str]) -> List[List[float]]:
        if not image_addresses:
            raise ValueError("images must not be empty")

        total_start = time.perf_counter()
        load_start = total_start
        images: List[Image.Image] = []
        for address in image_addresses:
            image = self.load_image(address)
            if not self._is_valid_image(image):
                LOGGER.warning(
                    "Skipping invalid image for CLIP embedding: %s (size=%sx%s)",
                    address,
                    image.size[0],
                    image.size[1],
                )
                continue
            images.append(image)
        load_elapsed = time.perf_counter() - load_start

        if not images:
            LOGGER.warning("All images were invalid for CLIP embedding; returning no embeddings")
            return []

        preprocess_start = time.perf_counter()
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        preprocess_elapsed = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.get_image_features(**inputs)
            features = self._coerce_feature_tensor(
                output,
                projection=getattr(self.model, "visual_projection", None),
            )
        inference_elapsed = time.perf_counter() - inference_start

        postprocess_start = time.perf_counter()
        features = self._normalize_features(features)
        result = features.cpu().tolist()
        postprocess_elapsed = time.perf_counter() - postprocess_start
        total_elapsed = time.perf_counter() - total_start

        LOGGER.info(
            "CLIP image embed timing: total=%.3fs, load=%.3fs, preprocess=%.3fs, "
            "inference=%.3fs, postprocess=%.3fs, device=%s, requested=%d, valid=%d",
            total_elapsed,
            load_elapsed,
            preprocess_elapsed,
            inference_elapsed,
            postprocess_elapsed,
            self.device,
            len(image_addresses),
            len(images),
        )
        return result

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts must not be empty")

        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
            features = self._coerce_feature_tensor(
                output,
                projection=getattr(self.model, "text_projection", None),
            )

        features = self._normalize_features(features)
        return features.cpu().tolist()


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


def create_handler(service: ClipEmbeddingService) -> type[BaseHTTPRequestHandler]:
    class ClipRequestHandler(BaseHTTPRequestHandler):
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
                },
            )

        def do_POST(self) -> None:
            path = urlparse(self.path).path

            try:
                payload = _read_json_body(self)
                if path == "/embed/images":
                    embeddings = service.embed_images(payload.get("images") or [])
                    _json_response(self, 200, {"embeddings": embeddings})
                    return

                if path == "/embed/texts":
                    embeddings = service.embed_texts(payload.get("texts") or [])
                    _json_response(self, 200, {"embeddings": embeddings})
                    return

                _json_response(self, 404, {"error": "not found"})
            except Exception as exc:
                LOGGER.exception("CLIP request failed")
                _json_response(self, 500, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.address_string(), format % args)

    return ClipRequestHandler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Using Hugging Face endpoint: %s", os.environ.get("HF_ENDPOINT", "default"))
    service = ClipEmbeddingService(
        model_id=CLIP_SERVER_CONFIG.model_id,
        local_model_dir=CLIP_SERVER_CONFIG.local_model_dir,
        image_timeout=CLIP_SERVER_CONFIG.image_timeout,
    )
    if torch.cuda.is_available():
        LOGGER.info("CUDA is available: %s", torch.cuda.get_device_name(0))
    else:
        LOGGER.info("CUDA is not available; CLIP server will run on CPU")
    LOGGER.info("CLIP server device selected: %s", service.device)
    server = ThreadingHTTPServer(
        (CLIP_SERVER_CONFIG.host, CLIP_SERVER_CONFIG.port),
        create_handler(service),
    )
    LOGGER.info(
        "CLIP server listening on http://%s:%s",
        CLIP_SERVER_CONFIG.host,
        CLIP_SERVER_CONFIG.port,
    )
    server.serve_forever()
