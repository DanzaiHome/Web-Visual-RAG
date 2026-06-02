import unittest
from unittest.mock import patch

from rag_v1.apps.pipeline import build_parser
from rag_v1.pipeline.rag_pipeline import answer_with_rag


class NoRagModeTests(unittest.TestCase):
    def test_cli_parser_accepts_no_rag_flag(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["--question", "q", "--images", "img.jpg", "--no-RAG"])

        self.assertTrue(args.no_rag)

    def test_no_rag_flag_skips_rag_pipeline_and_calls_vl_once(self) -> None:
        with patch("rag_v1.pipeline.rag_pipeline.should_use_rag") as should_use_rag_mock:
            with patch("rag_v1.pipeline.rag_pipeline.answer_question_no_rag", return_value="direct answer") as answer_mock:
                result = answer_with_rag(
                    img_paths=[],
                    question="Who is this person?",
                    no_rag=True,
                )

        self.assertEqual(result, "direct answer")
        should_use_rag_mock.assert_not_called()
        answer_mock.assert_called_once_with(
            img_paths=[],
            question="Who is this person?",
            debug=False,
        )


if __name__ == "__main__":
    unittest.main()
