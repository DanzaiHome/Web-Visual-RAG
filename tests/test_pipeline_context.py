import contextlib
import io
import unittest

from rag_v1.pipeline.rag_pipeline import (
    _evidence_score_breakdown,
    _select_diverse_chunks,
    aggregate_context,
    deduplicate_web_docs,
    select_quality_evidence_docs,
)


class PipelineContextTests(unittest.TestCase):
    def test_deduplicate_web_docs_uses_canonical_url(self) -> None:
        docs = [
            {
                "url": "https://example.com/story?utm_source=search&a=1",
                "name": "Story",
                "full_content": "Article body one.",
            },
            {
                "url": "https://example.com/story?a=1#section",
                "name": "Story duplicate",
                "full_content": "Article body two.",
            },
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            deduped = deduplicate_web_docs(docs)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["canonical_url"], "https://example.com/story?a=1")

    def test_aggregate_context_labels_webpage_body_evidence(self) -> None:
        context = aggregate_context(
            query="example query",
            retrieved_docs=[
                {
                    "url": "https://example.com/story",
                    "canonical_url": "https://example.com/story",
                    "name": "Fetched Story",
                    "site_name": "Example News",
                    "score": 0.72,
                    "chunk_id": 0,
                    "content": "Fetched body chunk used as RAG evidence.",
                    "content_source": "web_page",
                    "web_fetch_status": "success",
                    "web_fetch_from_cache": True,
                    "web_fetch_quality_score": 0.83,
                    "evidence_rescue_status": "image_filter_rescue",
                    "image_match_status": "matched",
                    "image_match_decision": "low_similarity",
                    "max_image_similarity": 0.22,
                    "image_match_image_count": 1,
                }
            ],
        )

        self.assertIn(
            "Evidence source: webpage body, cache, quality=0.830, image-filter rescue",
            context,
        )
        self.assertIn("Image match: matched, score=0.220, decision=low_similarity, images=1", context)
        self.assertIn("Fetched body chunk used as RAG evidence.", context)

    def test_aggregate_context_labels_summary_fallback(self) -> None:
        context = aggregate_context(
            query="example query",
            retrieved_docs=[
                {
                    "url": "https://example.com/missing",
                    "score": 0.2,
                    "chunk_id": 0,
                    "content": "Summary fallback chunk.",
                    "content_source": "bocha_summary_fallback",
                    "web_fetch_status": "http_error",
                }
            ],
        )

        self.assertIn("Evidence source: Bocha summary fallback, fetch_status=http_error", context)

    def test_aggregate_context_marks_page_dates_as_metadata(self) -> None:
        context = aggregate_context(
            query="latest completed game final score",
            current_time="2026-05-06T18:30:00+08:00",
            retrieved_docs=[
                {
                    "url": "https://sports.example.com/game",
                    "canonical_url": "https://sports.example.com/game",
                    "name": "Team A beats Team B 101-99 in last game",
                    "site_name": "Sports Example",
                    "date_published": "2026-05-05T23:30:00+00:00",
                    "score": 0.6,
                    "chunk_id": 0,
                    "content": "The final score was Team A 101, Team B 99.",
                    "content_source": "web_page",
                    "web_fetch_status": "success",
                }
            ],
        )

        self.assertIn("Temporal Note:", context)
        self.assertIn("Pipeline Current Time:\n2026-05-06T18:30:00+08:00", context)
        self.assertIn(
            "Document date fields below are page/search metadata",
            context,
        )
        self.assertIn(
            "Page metadata - Published: 2026-05-05T23:30:00+00:00 "
            "(not necessarily the event date)",
            context,
        )

    def test_quality_evidence_filter_rescues_web_body_when_only_weak_fallback_survives(self) -> None:
        weak_fallback = {
            "url": "https://example.com/dynamic",
            "content_source": "bocha_summary_fallback",
            "web_fetch_status": "empty_content",
            "image_match_status": "no_images",
            "full_content": "Search summary only.",
        }
        rescued_web_body = {
            "url": "https://example.com/article",
            "content_source": "web_page",
            "web_fetch_status": "success",
            "web_fetch_quality_score": 0.55,
            "web_fetch_is_probable_listing": False,
            "image_match_status": "matched",
            "full_content": "A clean fetched article body. " * 30,
        }

        with contextlib.redirect_stdout(io.StringIO()):
            selected = select_quality_evidence_docs(
                image_filtered_docs=[weak_fallback],
                pre_image_filter_docs=[rescued_web_body, weak_fallback],
            )

        self.assertEqual(selected, [rescued_web_body])
        self.assertEqual(rescued_web_body["evidence_rescue_status"], "image_filter_rescue")

    def test_quality_evidence_filter_keeps_matched_fallback_when_no_web_body_exists(self) -> None:
        matched_fallback = {
            "url": "https://example.com/snippet",
            "content_source": "bocha_summary_fallback",
            "web_fetch_status": "http_error",
            "image_match_status": "matched",
            "full_content": "A useful matched search summary. " * 20,
        }

        selected = select_quality_evidence_docs(
            image_filtered_docs=[matched_fallback],
            pre_image_filter_docs=[matched_fallback],
        )

        self.assertEqual(selected, [matched_fallback])

    def test_quality_evidence_filter_keeps_official_fallback_without_images(self) -> None:
        official_fallback = {
            "url": "https://www.whitehouse.gov/fact-sheets/example",
            "content_source": "bocha_summary_fallback",
            "web_fetch_status": "http_error",
            "image_match_status": "no_images",
            "full_content": "Official fact sheet summary. " * 20,
        }

        selected = select_quality_evidence_docs(
            image_filtered_docs=[official_fallback],
            pre_image_filter_docs=[official_fallback],
        )

        self.assertEqual(selected, [official_fallback])

    def test_quality_evidence_filter_keeps_long_listing_like_article_as_soft_penalty(self) -> None:
        listing_article = {
            "url": "https://news.example.com/article",
            "content_source": "web_page",
            "web_fetch_status": "success",
            "web_fetch_quality_score": 0.45,
            "web_fetch_is_probable_listing": True,
            "image_match_status": "matched",
            "image_match_decision": "weak_match",
            "full_content": "A long article body that may look like a listing. " * 50,
        }

        selected = select_quality_evidence_docs(
            image_filtered_docs=[listing_article],
            pre_image_filter_docs=[listing_article],
        )

        self.assertEqual(selected, [listing_article])

    def test_quality_evidence_filter_keeps_multiple_soft_scored_web_bodies(self) -> None:
        docs = [
            {
                "url": f"https://example{index}.com/article",
                "content_source": "web_page",
                "web_fetch_status": "success",
                "web_fetch_quality_score": 0.5 + index * 0.05,
                "web_fetch_is_probable_listing": False,
                "image_match_status": "matched",
                "image_match_decision": "low_similarity",
                "max_image_similarity": 0.2,
                "full_content": "A clean fetched article body. " * 30,
            }
            for index in range(3)
        ]

        selected = select_quality_evidence_docs(
            image_filtered_docs=docs,
            pre_image_filter_docs=docs,
            max_docs=3,
        )

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0]["url"], "https://example2.com/article")

    def test_evidence_score_breakdown_rewards_quality_source_freshness_and_strong_image(self) -> None:
        doc = {
            "url": "https://www.whitehouse.gov/briefing-room/statements-releases/fact-sheet",
            "canonical_url": "https://www.whitehouse.gov/briefing-room/statements-releases/fact-sheet",
            "content_source": "web_page",
            "web_fetch_quality_score": 0.8,
            "web_fetch_is_probable_listing": False,
            "image_match_status": "matched",
            "image_match_decision": "strong_match",
            "max_image_similarity": 0.7,
            "date_published": "2026-05-03T12:00:00+00:00",
        }

        breakdown = _evidence_score_breakdown(0.2, doc)

        self.assertGreater(breakdown["web_quality_bonus"], 0.0)
        self.assertGreater(breakdown["image_similarity_bonus"], 0.0)
        self.assertGreater(breakdown["source_reliability_bonus"], 0.0)
        self.assertGreater(breakdown["freshness_bonus"], 0.0)
        self.assertGreater(breakdown["total_score"], 0.3)

    def test_select_diverse_chunks_prefers_different_urls_first(self) -> None:
        chunks = [
            {"url": "https://a.example.com/story", "content": "Alpha evidence " * 20, "score": 0.9},
            {"url": "https://a.example.com/story", "content": "Beta evidence " * 20, "score": 0.8},
            {"url": "https://b.example.com/story", "content": "Gamma evidence " * 20, "score": 0.7},
        ]

        selected = _select_diverse_chunks(chunks, top_k=2)

        self.assertEqual([chunk["url"] for chunk in selected], [
            "https://a.example.com/story",
            "https://b.example.com/story",
        ])

    def test_select_diverse_chunks_suppresses_low_relevance_listing_chunks(self) -> None:
        chunks = [
            {
                "url": "https://news.example.com/story",
                "content": "A detailed policy article about new retirement savings rules. " * 8,
                "score": 0.88,
                "semantic_score": 0.88,
                "text_score": 0.86,
                "web_fetch_quality_score": 0.74,
                "web_fetch_is_probable_listing": False,
            },
            {
                "url": "https://listing.example.com/page",
                "content": "Trending\nRelated articles\nRecommended\nMore from this site\nHot topics",
                "score": 0.61,
                "semantic_score": 0.11,
                "text_score": 0.09,
                "web_fetch_quality_score": 0.42,
                "web_fetch_is_probable_listing": True,
            },
            {
                "url": "https://policy.example.org/fact-sheet",
                "content": "Official fact sheet with implementation details and concrete policy terms. " * 8,
                "score": 0.8,
                "semantic_score": 0.8,
                "text_score": 0.79,
                "web_fetch_quality_score": 0.81,
                "web_fetch_is_probable_listing": False,
            },
        ]

        selected = _select_diverse_chunks(chunks, top_k=2)
        urls = [chunk["url"] for chunk in selected]

        self.assertEqual(len(selected), 2)
        self.assertIn("https://news.example.com/story", urls)
        self.assertIn("https://policy.example.org/fact-sheet", urls)
        self.assertNotIn("https://listing.example.com/page", urls)


if __name__ == "__main__":
    unittest.main()
