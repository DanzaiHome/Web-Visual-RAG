import unittest

from rag_v1.prompts import prompts


class TemporalPromptTests(unittest.TestCase):
    def test_web_prompt_guides_latest_current_factoid_queries(self) -> None:
        prompt = prompts.web_prompt_en

        self.assertIn("current/latest factoid questions", prompt)
        self.assertIn("latest completed game final score schedule results", prompt)
        self.assertIn("structured result page terms", prompt)
        self.assertIn("current stock price quote market data official", prompt)

    def test_freshness_prompt_prefers_fresh_ranges_for_latest_facts(self) -> None:
        prompt = prompts.freshness_prompt_en

        self.assertIn("latest/current/recent factual value", prompt)
        self.assertIn("latest scores/results", prompt)
        self.assertIn("most recent completed games/events", prompt)
        self.assertIn("original user question", prompt)

    def test_answer_prompt_requires_temporal_verification(self) -> None:
        prompt = prompts.answer_prompt_en

        self.assertIn("do not assume the top-ranked document", prompt)
        self.assertIn("Distinguish page metadata from event time", prompt)
        self.assertIn("Do not cite page metadata as proof", prompt)
        self.assertIn("compare event dates", prompt)
        self.assertIn("later scheduled/preview event", prompt)
        self.assertIn("do not fall back to an older completed result", prompt)
        self.assertIn("single old-looking score", prompt)
        self.assertIn("insufficient to confirm it is the latest/current result", prompt)

    def test_answer_prompt_requires_doc_citations_for_retrieved_facts(self) -> None:
        prompt = prompts.answer_prompt_en

        self.assertIn("Every factual claim that depends on retrieved web evidence", prompt)
        self.assertIn("[Doc 1] or [Doc 1][Doc 3]", prompt)
        self.assertIn("无法从证据确定", prompt)
        self.assertIn("Do not invent document ids", prompt)

if __name__ == "__main__":
    unittest.main()
