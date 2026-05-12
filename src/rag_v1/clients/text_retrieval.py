import time
from typing import Sequence

import numpy as np
import requests

from rag_v1.config import TEXT_RETRIEVAL_SERVER_CONFIG
from rag_v1.timing import get_active_timing


class TextRetrievalClient:
    def __init__(
        self,
        base_url: str = TEXT_RETRIEVAL_SERVER_CONFIG.base_url,
        timeout: int = TEXT_RETRIEVAL_SERVER_CONFIG.request_timeout,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")

        start_time = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.base_url}/embed/texts",
                json={"texts": list(texts)},
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            embeddings = result.get("embeddings")
            if embeddings is None:
                raise RuntimeError(
                    f"Text retrieval server response missing embeddings: {result}"
                )

            return np.asarray(embeddings, dtype=np.float32)
        finally:
            timing = get_active_timing()
            if timing is not None:
                timing.add("text_retrieval_server", time.perf_counter() - start_time)
