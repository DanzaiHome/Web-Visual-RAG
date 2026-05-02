from typing import Sequence

import numpy as np
import requests

from rag_v1.config import CLIP_SERVER_CONFIG


class ClipClient:
    def __init__(
        self,
        base_url: str = CLIP_SERVER_CONFIG.base_url,
        timeout: int = CLIP_SERVER_CONFIG.request_timeout,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _post_embeddings(self, endpoint: str, key: str, values: Sequence[str]) -> np.ndarray:
        if not values:
            raise ValueError(f"{key} must not be empty")

        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json={key: list(values)},
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        embeddings = result.get("embeddings")
        if embeddings is None:
            raise RuntimeError(f"CLIP server response missing embeddings: {result}")

        return np.asarray(embeddings, dtype=np.float32)

    def embed_images(self, image_addresses: Sequence[str]) -> np.ndarray:
        return self._post_embeddings("/embed/images", "images", image_addresses)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._post_embeddings("/embed/texts", "texts", texts)
