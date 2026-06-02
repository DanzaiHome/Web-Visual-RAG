import unittest

from rag_v1.retrieval.chunk_extractor import ChunkExtractor


class ChunkExtractorTests(unittest.TestCase):
    def test_split_preserves_paragraph_boundaries(self) -> None:
        text = (
            "Paragraph one contains article evidence and should stay readable.\n\n"
            "Paragraph two adds more context for retrieval and ranking.\n\n"
            "Paragraph three closes the evidence with one more useful detail."
        )

        extractor = ChunkExtractor(
            document=text,
            question="article evidence",
            image_paths=[],
            chunk_size=140,
        )

        self.assertGreaterEqual(len(extractor.chunks), 2)
        self.assertIn("Paragraph one", extractor.chunks[0])
        self.assertTrue(all(chunk.strip() for chunk in extractor.chunks))
        self.assertEqual(len(extractor.chunks), len(set(extractor.chunks)))

    def test_long_english_segment_splits_on_words(self) -> None:
        text = " ".join(f"word{i}" for i in range(80))
        extractor = ChunkExtractor(
            document=text,
            question="word",
            image_paths=[],
            chunk_size=90,
        )

        self.assertGreater(len(extractor.chunks), 1)
        for chunk in extractor.chunks:
            self.assertLessEqual(len(chunk), 90)
            self.assertNotRegex(chunk, r"word\d+word\d+")

    def test_word_per_line_text_merges_into_readable_chunk(self) -> None:
        text = (
            "Fact\n\n"
            "Sheet:\n\n"
            "President\n\n"
            "Donald\n\n"
            "J.\n\n"
            "Trump\n\n"
            "announced\n\n"
            "new\n\n"
            "policies\n\n"
            "for\n\n"
            "retirement\n\n"
            "savings."
        )
        extractor = ChunkExtractor(
            document=text,
            question="new policies",
            image_paths=[],
            chunk_size=120,
        )

        self.assertEqual(len(extractor.chunks), 1)
        self.assertIn(
            "Fact Sheet: President Donald J. Trump announced new policies",
            extractor.chunks[0],
        )
        self.assertNotIn("Fact\n\nSheet", extractor.chunks[0])

    def test_word_per_line_prefix_merges_without_swallowing_cjk_body(self) -> None:
        text = (
            "Trump\n\n"
            "is\n\n"
            "reportedly\n\n"
            "set\n\n"
            "to\n\n"
            "sign\n\n"
            "an\n\n"
            "executive\n\n"
            "order\n\n"
            "to\n\n"
            "expand\n\n"
            "the\n\n"
            "coverage\n\n"
            "of\n\n"
            "retirement\n\n"
            "savings\n\n"
            "plans美国总统特朗普将签署行政命令。"
        )
        extractor = ChunkExtractor(
            document=text,
            question="retirement savings",
            image_paths=[],
            chunk_size=300,
        )
        joined = "\n\n".join(extractor.chunks)

        self.assertIn(
            "Trump is reportedly set to sign an executive order to expand "
            "the coverage of retirement savings plans",
            joined,
        )
        self.assertIn("美国总统", joined)
        self.assertNotIn("Trump\n\nis", joined)

    def test_chinese_text_without_spaces_uses_overlapping_character_chunks(self) -> None:
        text = "".join(
            f"第{index}段中文网页正文用来验证长文本切块不会依赖英文空格。"
            for index in range(12)
        )
        extractor = ChunkExtractor(
            document=text,
            question="中文网页正文",
            image_paths=[],
            chunk_size=70,
        )

        self.assertGreater(len(extractor.chunks), 1)
        self.assertTrue(all(len(chunk) <= 70 for chunk in extractor.chunks))
        self.assertIn("中文网页正文", "".join(extractor.chunks))

    def test_retrieve_text_chunks_falls_back_to_lexical_scores(self) -> None:
        original_get_text_retrieval_client = ChunkExtractor._get_text_retrieval_client

        def raise_missing_dependency(cls):
            raise RuntimeError("missing sentence-transformers")

        ChunkExtractor._get_text_retrieval_client = classmethod(raise_missing_dependency)
        try:
            extractor = ChunkExtractor(
                document=(
                    "Weather summary and unrelated background text.\n\n"
                    "Los Angeles Lakers latest completed game final score result."
                ),
                question="Lakers latest completed game final score",
                image_paths=[],
                chunk_size=90,
            )

            chunks = extractor.retrieve_text_chunks(top_n=1)
        finally:
            ChunkExtractor._get_text_retrieval_client = original_get_text_retrieval_client

        self.assertEqual(len(chunks), 1)
        self.assertIn("Lakers latest completed game final score", chunks[0]["text"])
        self.assertGreater(chunks[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
