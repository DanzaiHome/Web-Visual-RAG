import unittest
from pathlib import Path
from unittest.mock import patch

from rag_v1.services import vl_router


class EntityQueryGenerationTests(unittest.TestCase):
    def test_extract_entity_candidates_parses_json_payload(self) -> None:
        response_text = """
        Here is the result:
        {
          "candidates": [
            {
              "name": "Jensen Huang",
              "type": "person",
              "aliases": ["NVIDIA CEO", "Jen-Hsun Huang"],
              "confidence": 0.93,
              "reason": "Black leather jacket and product-launch pose",
              "missing_slot": "company_name"
            },
            {
              "name": "Lisa Su",
              "type": "person",
              "aliases": ["AMD CEO"],
              "confidence": 0.21,
              "reason": "Fallback guess",
              "missing_slot": "company_name"
            }
          ]
        }
        """

        candidates = vl_router.extract_entity_candidates(response_text)

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["name"], "Jensen Huang")
        self.assertEqual(candidates[0]["aliases"], ["NVIDIA CEO", "Jen-Hsun Huang"])
        self.assertAlmostEqual(float(candidates[0]["confidence"]), 0.93)
        self.assertEqual(candidates[0]["missing_slot"], "company_name")

    def test_generate_search_query_uses_single_candidate_guided_prompt_when_candidates_exist(self) -> None:
        prompts_seen: list[str] = []

        def fake_call_api(prompt: str, img_paths=(), temperature: float = 0.2) -> str:
            prompts_seen.append(prompt)
            if '"candidates"' in prompt:
                return """
                {
                  "candidates": [
                    {
                      "name": "Jensen Huang",
                      "type": "person",
                      "aliases": ["NVIDIA CEO"],
                      "confidence": 0.94,
                      "reason": "Product keynote image",
                      "missing_slot": "company_name"
                    }
                  ]
                }
                """
            return "Jensen Huang NVIDIA founder CEO company official"

        with patch.object(vl_router, "call_api", side_effect=fake_call_api):
            query = vl_router.generate_search_query(
                img_paths=[Path("example.jpg")],
                question="The person in the image is the founder and CEO of which company?",
            )

        self.assertEqual(query, "Jensen Huang NVIDIA founder CEO company official")
        self.assertEqual(len(prompts_seen), 2)
        self.assertIn("Entity candidates:", prompts_seen[1])
        self.assertIn("Jensen Huang", prompts_seen[1])

    def test_generate_search_query_falls_back_to_plain_query_prompt_when_candidates_missing(self) -> None:
        prompts_seen: list[str] = []

        def fake_call_api(prompt: str, img_paths=(), temperature: float = 0.2) -> str:
            prompts_seen.append(prompt)
            if '"candidates"' in prompt:
                return '{"candidates":[]}'
            return "silver supercar released by which company official"

        with patch.object(vl_router, "call_api", side_effect=fake_call_api):
            query = vl_router.generate_search_query(
                img_paths=[Path("example.jpg")],
                question="Which company released the supercar in the image?",
            )

        self.assertEqual(query, "silver supercar released by which company official")
        self.assertEqual(len(prompts_seen), 2)
        self.assertNotIn("Entity candidates:", prompts_seen[1])
        self.assertIn("construct the query for web retrieval", prompts_seen[1])

    def test_generate_search_queries_emits_multiple_deduped_queries(self) -> None:
        prompts_seen: list[str] = []

        def fake_call_api(prompt: str, img_paths=(), temperature: float = 0.2) -> str:
            prompts_seen.append(prompt)
            if '"candidates"' in prompt:
                return """
                {
                  "candidates": [
                    {
                      "name": "Jensen Huang",
                      "type": "person",
                      "aliases": ["NVIDIA CEO", "Jen-Hsun Huang"],
                      "confidence": 0.96,
                      "reason": "Keynote speaker",
                      "missing_slot": "company_name"
                    },
                    {
                      "name": "Lisa Su",
                      "type": "person",
                      "aliases": ["AMD CEO"],
                      "confidence": 0.42,
                      "reason": "Possible nearby candidate",
                      "missing_slot": "company_name"
                    }
                  ]
                }
                """
            if "Candidate 1: name=Jensen Huang; type=person; confidence=0.96; aliases=NVIDIA CEO, Jen-Hsun Huang" in prompt:
                return "Jensen Huang NVIDIA keynote official"
            if "Candidate 1: name=Jensen Huang; type=person; confidence=0.96; aliases=Jen-Hsun Huang" in prompt:
                return "Jensen Huang NVIDIA keynote official"
            return "Lisa Su AMD keynote official"

        with patch.object(vl_router, "call_api", side_effect=fake_call_api):
            queries = vl_router.generate_search_queries(
                img_paths=[Path("example.jpg")],
                question="Which company is the person in the image the founder and CEO of?",
            )

        self.assertEqual(
            queries,
            [
                "Jensen Huang NVIDIA keynote official",
                "Lisa Su AMD keynote official",
            ],
        )
        self.assertGreaterEqual(len(prompts_seen), 3)
        self.assertNotIn("Candidate 2: name=Lisa Su", prompts_seen[1])
        self.assertIn("Candidate 1: name=Lisa Su", prompts_seen[2])

    def test_judge_context_sufficiency_includes_entity_candidates_in_prompt(self) -> None:
        prompts_seen: list[str] = []

        def fake_call_api(prompt: str, img_paths=(), temperature: float = 0.2) -> str:
            prompts_seen.append(prompt)
            return '{"judgement":"NO","addition":"targeted query"}'

        with patch.object(vl_router, "call_api", side_effect=fake_call_api):
            response = vl_router.judge_context_sufficiency(
                img_paths=[Path("example.jpg")],
                question="Which award did the person in the image win?",
                context="Retrieved Context:\n[Doc 1] Evidence Block",
                entity_candidates=[
                    {
                        "name": "Victor Wembanyama",
                        "type": "person",
                        "aliases": ["Wemby"],
                        "confidence": 0.93,
                        "reason": "Spurs player in jersey",
                        "missing_slot": "award_name",
                    }
                ],
                previous_queries=[
                    "Victor Wembanyama award winner official",
                    "2026 Spurs award result",
                ],
            )

        self.assertEqual(response, '{"judgement":"NO","addition":"targeted query"}')
        self.assertEqual(len(prompts_seen), 1)
        self.assertIn("Entity candidates:", prompts_seen[0])
        self.assertIn("Victor Wembanyama", prompts_seen[0])
        self.assertIn("Previous search queries already tried:", prompts_seen[0])
        self.assertIn("Victor Wembanyama award winner official", prompts_seen[0])
        self.assertIn("The new query must NOT repeat or trivially paraphrase", prompts_seen[0])
        self.assertIn("identify what is still missing for the most plausible candidate", prompts_seen[0])


if __name__ == "__main__":
    unittest.main()
