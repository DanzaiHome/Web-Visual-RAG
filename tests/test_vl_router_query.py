import unittest

from rag_v1.services.vl_router import _augment_current_factoid_query


class QueryAugmentationTests(unittest.TestCase):
    def test_augments_chinese_latest_sports_score_query(self) -> None:
        query = _augment_current_factoid_query(
            query="洛杉矶湖人队最近一场正式比赛最终比分",
            question="图中这个人所在的球队最近一场正式比赛的最终比分是多少？",
        )

        self.assertIn("洛杉矶湖人队", query)
        self.assertIn("已结束", query)
        self.assertIn("赛程", query)
        self.assertIn("结果", query)
        self.assertIn("战报", query)

    def test_augments_english_latest_sports_score_query(self) -> None:
        query = _augment_current_factoid_query(
            query="Los Angeles Lakers latest game score",
            question="What was the final score of this person's team's most recent game?",
        )

        self.assertIn("Los Angeles Lakers", query)
        self.assertIn("completed", query)
        self.assertIn("schedule", query)
        self.assertIn("results", query)
        self.assertIn("box score", query)

    def test_leaves_stable_query_unchanged(self) -> None:
        query = _augment_current_factoid_query(
            query="Huangshan geographical location",
            question="Where was this photo taken?",
        )

        self.assertEqual(query, "Huangshan geographical location")


if __name__ == "__main__":
    unittest.main()
