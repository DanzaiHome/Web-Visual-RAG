import contextlib
import io
import unittest

import numpy as np

from rag_v1.services.image_checker import _normalize_image_urls, score_docs_by_image_match


class FakeClipClient:
    def embed_images(self, image_addresses):
        rows = []
        for image_address in image_addresses:
            text = str(image_address)
            if "prompt" in text or "match" in text:
                rows.append([1.0, 0.0])
            elif "weak" in text:
                rows.append([0.4, 0.916515])
            else:
                rows.append([0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


class ImageCheckerTests(unittest.TestCase):
    def test_normalize_image_urls_filters_non_images_and_dedupes(self) -> None:
        image_urls = _normalize_image_urls(
            [
                "https://example.com/photo.jpg",
                "https://example.com/photo.jpg",
                "https://example.com/static/app.js",
                "https://example.com/font.woff2",
                "https://example.com/favicon.ico",
                "https://cdn.example.com/image?id=42",
            ]
        )

        self.assertEqual(
            image_urls,
            [
                "https://example.com/photo.jpg",
                "https://cdn.example.com/image?id=42",
            ],
        )

    def test_score_docs_by_image_match_keeps_low_similarity_docs(self) -> None:
        docs = [
            {
                "url": "https://example.com/match",
                "image_urls": ["https://example.com/match.jpg"],
            },
            {
                "url": "https://example.com/low",
                "image_urls": ["https://example.com/low.jpg"],
            },
            {
                "url": "https://example.com/no-images",
                "image_urls": [],
            },
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            scored = score_docs_by_image_match(
                web_docs=docs,
                prompt_image_paths=["prompt.jpg"],
                threshold=0.5,
                clip_client=FakeClipClient(),
            )

        self.assertEqual(len(scored), 3)
        self.assertEqual(scored[0]["image_match_decision"], "strong_match")
        self.assertEqual(scored[1]["image_match_decision"], "low_similarity")
        self.assertEqual(scored[2]["image_match_decision"], "no_images")


if __name__ == "__main__":
    unittest.main()
