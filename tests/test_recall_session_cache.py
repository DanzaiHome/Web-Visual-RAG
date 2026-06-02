import json
import shutil
import unittest
from pathlib import Path

import numpy as np

from rag_v1.config import PROJECT_ROOT
from rag_v1.pipeline.recall_session_cache import RecallSessionCache


class FakeClipClient:
    def embed_images(self, images):
        rows = [[float(index + 1), float(index + 2)] for index, _ in enumerate(images)]
        return np.asarray(rows, dtype=np.float32)

    def embed_texts(self, texts):
        rows = [[float(len(text)), float(index + 1)] for index, text in enumerate(texts)]
        return np.asarray(rows, dtype=np.float32)


class FakeTextRetrievalClient:
    def embed_texts(self, texts):
        rows = [[float(index + 10), float(len(text))] for index, text in enumerate(texts)]
        return np.asarray(rows, dtype=np.float32)


class FakePageImageDownloader:
    def __init__(self) -> None:
        self.calls = []

    def download(self, image_url, destination):
        self.calls.append((str(image_url), str(destination)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fake-image-bytes")
        return True


class RecallSessionCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = PROJECT_ROOT / ".test_tmp" / "recall_session_cache_tests"
        self._assert_under_project(self.test_root)
        self.test_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self._assert_under_project(self.test_root)
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

    @staticmethod
    def _assert_under_project(path: Path) -> None:
        resolved_path = path.resolve()
        project_root = PROJECT_ROOT.resolve()
        resolved_path.relative_to(project_root)

    def test_embeddings_are_stored_in_npy_files_referenced_by_manifest(self) -> None:
        cache = RecallSessionCache.create(
            question="Who is this person?",
            img_paths=["img-a.jpg"],
            use_multimodal=True,
        )
        original_session_dir = cache.session_dir
        cache.clip_client = FakeClipClient()
        cache.text_retrieval_client = FakeTextRetrievalClient()
        cache.page_image_downloader = FakePageImageDownloader()

        cache.session_dir = self.test_root / cache.session_dir.name
        cache.session_path = cache.session_dir / "manifest.json"
        if original_session_dir.exists():
            shutil.rmtree(original_session_dir)
        cache.flush(force=True)

        prompt_embeddings = cache.ensure_prompt_image_embeddings(["img-a.jpg"])
        question_embedding = cache.ensure_question_text_embedding("Who is this person?")
        cache.record_query("person identity", "noLimit")
        query_embedding = cache.ensure_query_text_retrieval_embedding("person identity")
        cache.register_docs(
            [
                {
                    "url": "https://example.com/story",
                    "canonical_url": "https://example.com/story",
                    "full_content": "Paragraph one.\n\nParagraph two with more detail.",
                    "image_urls": ["https://example.com/image.jpg"],
                }
            ]
        )
        doc = {
            "url": "https://example.com/story",
            "canonical_url": "https://example.com/story",
            "full_content": "Paragraph one.\n\nParagraph two with more detail.",
            "image_urls": ["https://example.com/image.jpg"],
        }
        cache.populate_doc_page_image_embeddings(doc)
        profile = cache.get_chunk_profile(
            doc=doc,
            question="person identity",
            chunk_size=20,
            use_multimodal=True,
        )
        cache.flush()

        with cache.session_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        self.assertEqual(manifest["storage_format"], "manifest+npy")
        self.assertNotIn("prompt_image_embeddings", manifest)
        self.assertTrue(manifest["prompt_image_embeddings_file"].endswith(".npy"))
        self.assertTrue(manifest["question_clip_text_embedding_file"].endswith(".npy"))
        self.assertEqual(manifest["queries"][0]["query"], "person identity")
        self.assertEqual(len(manifest["query_text_retrieval_embeddings"]), 1)

        prompt_path = cache.session_dir / manifest["prompt_image_embeddings_file"]
        question_path = cache.session_dir / manifest["question_clip_text_embedding_file"]
        query_relpath = next(iter(manifest["query_text_retrieval_embeddings"].values()))
        query_path = cache.session_dir / query_relpath
        doc_manifest = manifest["docs"]["https://example.com/story"]
        self.assertEqual(len(doc_manifest["page_image_local_files"]), 1)
        local_image_relpath = next(iter(doc_manifest["page_image_local_files"].values()))
        local_image_path = cache.session_dir / local_image_relpath
        page_image_path = cache.session_dir / doc_manifest["page_image_embeddings_file"]
        chunk_profiles = doc_manifest["chunk_profiles"]
        chunk_profile = next(iter(chunk_profiles.values()))
        text_chunk_path = cache.session_dir / chunk_profile["text_retrieval_chunk_embeddings_file"]
        clip_chunk_path = cache.session_dir / chunk_profile["clip_text_embeddings_file"]

        for path in (
            prompt_path,
            question_path,
            query_path,
            local_image_path,
            page_image_path,
            text_chunk_path,
            clip_chunk_path,
        ):
            self.assertTrue(path.exists(), str(path))

        np.testing.assert_allclose(np.load(prompt_path), prompt_embeddings)
        np.testing.assert_allclose(np.load(question_path), question_embedding)
        np.testing.assert_allclose(np.load(query_path), query_embedding)
        np.testing.assert_allclose(doc["clip_page_image_embeddings"], np.load(page_image_path))
        np.testing.assert_allclose(profile["text_retrieval_chunk_embeddings"], np.load(text_chunk_path))
        np.testing.assert_allclose(profile["clip_text_embeddings"], np.load(clip_chunk_path))


if __name__ == "__main__":
    unittest.main()
