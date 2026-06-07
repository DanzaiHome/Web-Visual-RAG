import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from val.eval_baseline import (
    EvalSample,
    build_formal_report,
    build_judge_prompt,
    call_judge_model,
    infer_group,
    parse_judge_verdict,
    run_inference,
    summarize_results,
)


class EvalBaselineTests(unittest.TestCase):
    def test_infer_group_prefers_explicit_or_keyword_then_fallback_order(self) -> None:
        self.assertEqual(
            infer_group(5, "Which company is the person currently the CEO of?", {}),
            "dynamic_web_qa",
        )
        self.assertEqual(
            infer_group(30, "What color is the cat in the image?", {}),
            "static_vqa",
        )
        self.assertEqual(
            infer_group(2, "Which company launched the product in the image?", {}),
            "dynamic_web_qa",
        )
        self.assertEqual(
            infer_group(40, "ignored", {"group": "custom_split"}),
            "custom_split",
        )

    def test_run_inference_uses_no_rag_direct_call(self) -> None:
        sample = EvalSample(
            index=1,
            question="Who is this person?",
            answer="Someone",
            image_path=Path("example.jpg"),
            group="static_vqa",
        )
        args = Namespace(
            mode="no_rag",
            model_id="test-model",
            debug=False,
            top_k=5,
            candidate_k=10,
            top_n_images=3,
            chunk_size=400,
            chunks_per_doc=3,
            use_multimodal=True,
            multimodal_text_weight=0.5,
            trails=1,
            max_sufficiency_iterations=3,
        )

        with patch("val.eval_baseline.answer_question_no_rag", return_value="direct answer") as mock_answer:
            prediction, trace = run_inference(sample=sample, args=args)

        self.assertEqual(prediction, "direct answer")
        self.assertEqual(trace["query_rounds"], [])
        self.assertEqual(trace["final_context_urls"], [])
        mock_answer.assert_called_once()

    def test_run_inference_forced_rag_modes_patch_router(self) -> None:
        sample = EvalSample(
            index=1,
            question="Which award did she win?",
            answer="Award",
            image_path=Path("example.jpg"),
            group="dynamic_web_qa",
        )
        args = Namespace(
            mode="text_only_rag",
            model_id="test-model",
            debug=False,
            top_k=5,
            candidate_k=10,
            top_n_images=3,
            chunk_size=400,
            chunks_per_doc=3,
            use_multimodal=True,
            multimodal_text_weight=0.35,
            trails=1,
            max_sufficiency_iterations=3,
        )

        with patch("val.eval_baseline.rag_pipeline.answer_with_rag", return_value="rag answer") as mock_answer:
            with patch(
                "val.eval_baseline.rag_pipeline.get_last_rag_trace",
                return_value={
                    "query_rounds": [{"stage": "initial", "queries": ["q1", "q2"]}],
                    "final_context_urls": ["https://example.com/story"],
                },
            ):
                prediction, trace = run_inference(sample=sample, args=args)

        self.assertEqual(prediction, "rag answer")
        self.assertEqual(trace["query_rounds"][0]["queries"], ["q1", "q2"])
        self.assertEqual(trace["final_context_urls"], ["https://example.com/story"])
        self.assertFalse(mock_answer.call_args.kwargs["use_multimodal"])
        self.assertEqual(mock_answer.call_args.kwargs["multimodal_text_weight"], 0.35)

    def test_summarize_results_aggregates_groups(self) -> None:
        summary = summarize_results(
            [
                {
                    "group": "dynamic_web_qa",
                    "success_rate": 1.0,
                    "avg_latency_seconds": 1.5,
                },
                {
                    "group": "static_vqa",
                    "success_rate": 0.0,
                    "avg_latency_seconds": 0.5,
                },
            ]
        )

        self.assertEqual(summary["overall"]["count"], 2)
        self.assertAlmostEqual(summary["overall"]["judge_accuracy"], 0.5)
        self.assertAlmostEqual(summary["static_vqa"]["avg_latency_seconds"], 0.5)

    def test_build_formal_report_contains_summary_table(self) -> None:
        summary = {
            "overall": {
                "count": 2,
                "judge_accuracy": 1.0,
                "avg_latency_seconds": 1.25,
            }
        }
        args = Namespace(
            model_id="test-model",
            top_k=5,
            candidate_k=10,
            top_n_images=3,
            chunk_size=400,
            chunks_per_doc=3,
            max_sufficiency_iterations=3,
            use_multimodal=True,
            multimodal_text_weight=0.7,
            trails=3,
            offset=0,
            limit=0,
        )

        report = build_formal_report(
            mode="full_rag",
            model_id="test-model",
            dataset_path=Path("val/rag_vqa_question_answer_image.json"),
            output_path=Path("val/results/report.json"),
            summary=summary,
            args=args,
            sample_count=2,
            eval_started_at="2026-06-06T13:30:00+08:00",
        )

        self.assertIn("Baseline Evaluation Report", report)
        self.assertIn("Model ID: test-model", report)
        self.assertIn("Summary Metrics", report)
        self.assertIn("overall", report)
        self.assertIn("100.00%", report)
        self.assertIn("1.250s", report)
        self.assertIn("multimodal_text_weight=0.7", report)
        self.assertIn("trails=3", report)
        self.assertIn("Judge Model: qwen3.6-flash", report)

    def test_parse_judge_verdict_accepts_json_and_fallback(self) -> None:
        self.assertTrue(parse_judge_verdict('{"verdict":"CORRECT"}'))
        self.assertFalse(parse_judge_verdict('{"verdict":"INCORRECT"}'))
        self.assertTrue(parse_judge_verdict("CORRECT"))

    def test_build_judge_prompt_contains_required_fields(self) -> None:
        prompt = build_judge_prompt(
            question="Which company is the person currently the CEO of?",
            gold_answer="Microsoft",
            prediction="He is the CEO of Microsoft.",
        )

        self.assertIn("Reference answer:", prompt)
        self.assertIn("Predicted answer:", prompt)
        self.assertIn('"verdict":"CORRECT"', prompt)

    def test_call_judge_model_temporarily_switches_model(self) -> None:
        with patch("val.eval_baseline.call_api", return_value='{"verdict":"CORRECT"}') as mock_call_api:
            with patch.dict("val.eval_baseline.oai_config", {"model": "qwen3-vl-8b-instruct"}, clear=False):
                result = call_judge_model("judge prompt")

        self.assertEqual(result, '{"verdict":"CORRECT"}')
        mock_call_api.assert_called_once_with(prompt="judge prompt", img_paths=(), temperature=0.0)
        self.assertEqual(__import__("val.eval_baseline", fromlist=["oai_config"]).oai_config["model"], "qwen3-vl-8b-instruct")


if __name__ == "__main__":
    unittest.main()
