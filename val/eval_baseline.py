import argparse
import json
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_v1.pipeline import rag_pipeline
from rag_v1.pipeline.rag_requirement import RagRequirementDecision
from rag_v1.services import web_search as web_search_service
from rag_v1.services.vl_router import answer_question_no_rag, call_api, oai_config


DEFAULT_DATASET = Path(__file__).resolve().with_name("rag_vqa_question_answer_image.json")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name("results")
DYNAMIC_QUESTION_TERMS = (
    "award",
    "won",
    "win",
    "season",
    "festival",
    "released",
    "release",
    "launched",
    "launch",
    "currently",
    "current",
    "ceo",
    "partner",
    "mvp",
    "finals",
    "conference finals",
    "champions league",
    "computex",
    "eurovision",
    "cannes",
    "bafta",
    "pulitzer",
    "2025",
    "2026",
)
MODE_CHOICES = ("no_rag", "text_only_rag", "full_rag", "route_router")
WEB_SEARCH_PROVIDER_CHOICES = ("bocha", "serpapi")
JUDGE_CORRECT = "CORRECT"
JUDGE_INCORRECT = "INCORRECT"
JUDGE_MODEL = "qwen3.6-flash"


@dataclass
class EvalSample:
    index: int
    question: str
    answer: str
    image_path: Path
    group: str


def build_judge_prompt(question: str, gold_answer: str, prediction: str) -> str:
    return f"""You are evaluating whether a predicted answer should be counted as correct for a visual question answering benchmark.

Question:
{question}

Reference answer:
{gold_answer}

Predicted answer:
{prediction}

Evaluation policy:
1. Return CORRECT only if the predicted answer gives the same final fact as the reference answer.
2. Accept harmless surface-form differences such as casing, punctuation, articles, short paraphrases, and equivalent entity naming.
3. Accept concise supersets if they clearly contain the correct final answer without changing it.
4. Return INCORRECT if the prediction states a different fact, is too vague to confirm the answer, refuses to answer, or mixes in contradictory claims.
5. For explanation questions, return CORRECT only when the central reason matches the reference answer.
6. Be strict: if the prediction is not clearly correct, return INCORRECT.

Output format:
Return a JSON object only, exactly like:
{{"verdict":"CORRECT"}}
or
{{"verdict":"INCORRECT"}}"""


def parse_judge_verdict(response_text: str) -> bool:
    text = str(response_text or "").strip()
    try:
        payload = json.loads(text)
        verdict = str(payload.get("verdict") or payload.get("label") or "").strip().upper()
        if verdict == JUDGE_CORRECT:
            return True
        if verdict == JUDGE_INCORRECT:
            return False
    except json.JSONDecodeError:
        pass

    upper = text.upper()
    if JUDGE_CORRECT in upper and JUDGE_INCORRECT not in upper:
        return True
    if JUDGE_INCORRECT in upper:
        return False
    raise RuntimeError(f"Unable to parse judge verdict: {response_text!r}")


def call_judge_model(prompt: str) -> str:
    original_model = oai_config.get("model")
    oai_config["model"] = JUDGE_MODEL
    try:
        return call_api(prompt=prompt, img_paths=(), temperature=0.0)
    finally:
        oai_config["model"] = original_model


def judge_prediction(sample: EvalSample, prediction: str) -> Dict[str, Any]:
    prompt = build_judge_prompt(
        question=sample.question,
        gold_answer=sample.answer,
        prediction=prediction,
    )
    raw_response = call_judge_model(prompt)
    return {
        "judge_model": JUDGE_MODEL,
        "judge_raw_response": raw_response,
        "judge_correct": parse_judge_verdict(raw_response),
    }


def infer_group(index: int, question: str, payload: Dict[str, Any]) -> str:
    explicit_group = str(payload.get("group") or payload.get("split") or "").strip()
    if explicit_group:
        return explicit_group

    lowered = question.lower()
    if any(term in lowered for term in DYNAMIC_QUESTION_TERMS):
        return "dynamic_web_qa"

    # This validation set is currently ordered with dynamic/web-evidence samples first
    # and static visual-recognition samples later. Keep that fallback explicit so the
    # split remains stable even when a dynamic question lacks a keyword.
    if index <= 22:
        return "dynamic_web_qa"
    return "static_vqa"


def load_dataset(dataset_path: Path) -> List[EvalSample]:
    raw_items = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_dir = dataset_path.parent
    samples: List[EvalSample] = []

    for index, item in enumerate(raw_items, start=1):
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        image_value = str(item.get("image") or "").strip()
        if not question or not answer or not image_value:
            raise ValueError(f"Invalid sample at index {index}: missing question/answer/image")

        image_path = (dataset_dir / image_value).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for sample {index}: {image_path}")

        samples.append(
            EvalSample(
                index=index,
                question=question,
                answer=answer,
                image_path=image_path,
                group=infer_group(index=index, question=question, payload=item),
            )
        )

    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_baseline",
        description="Run baseline evaluation on the validation set.",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to dataset JSON.")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        required=True,
        help="Baseline mode to evaluate.",
    )
    parser.add_argument(
        "--model-id",
        default=str(oai_config.get("model") or ""),
        help="Model id used for the evaluated answering pipeline.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of final chunks to keep.")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="Number of web search candidates to fetch before reranking.",
    )
    parser.add_argument(
        "--top-n-images",
        type=int,
        default=3,
        help="Number of page images to keep per result.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Chunk size used during document splitting.",
    )
    parser.add_argument(
        "--chunks-per-doc",
        type=int,
        default=3,
        help="Maximum number of retrieved chunks per document.",
    )
    parser.add_argument(
        "--max-sufficiency-iterations",
        type=int,
        default=3,
        help="Maximum number of sufficiency-check iterations.",
    )
    parser.add_argument(
        "--multimodal-text-weight",
        type=float,
        default=0.5,
        help="Weight for text score in multimodal retrieval. Image weight = 1 - value.",
    )
    parser.add_argument(
        "--use-multimodal",
        action="store_true",
        default=True,
        help="For route_router mode, enable multimodal chunk retrieval.",
    )
    parser.add_argument(
        "--no-use-multimodal",
        dest="use_multimodal",
        action="store_false",
        help="For route_router mode, disable multimodal chunk retrieval.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N samples.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N samples.")
    parser.add_argument(
        "--trails",
        type=int,
        default=1,
        help="Number of repeated trials per sample. Final score uses average success rate.",
    )
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated result files when --output is not set.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug output from the pipeline.")
    parser.add_argument(
        "--web-search-provider",
        choices=WEB_SEARCH_PROVIDER_CHOICES,
        default=web_search_service.WEB_SEARCH_PROVIDER,
        help="Select the web search backend used during evaluation.",
    )
    return parser


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    filename = f"eval_baseline_{args.mode}_{timestamp}.json"
    return output_dir / filename


def run_inference(sample: EvalSample, args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    image_paths = [sample.image_path]

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                web_search_service,
                "WEB_SEARCH_PROVIDER",
                args.web_search_provider,
            )
        )

        if args.mode == "no_rag":
            prediction = answer_question_no_rag(
                img_paths=image_paths,
                question=sample.question,
                debug=args.debug,
            )
            return prediction, {
                "mode": "no_rag",
                "query_rounds": [],
                "final_context_urls": [],
            }

        if args.mode == "route_router":
            prediction = rag_pipeline.answer_with_rag(
                img_paths=image_paths,
                question=sample.question,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                top_n_images=args.top_n_images,
                chunk_size=args.chunk_size,
                chunks_per_doc=args.chunks_per_doc,
                use_multimodal=args.use_multimodal,
                multimodal_text_weight=args.multimodal_text_weight,
                no_rag=False,
                debug=args.debug,
                max_sufficiency_iterations=args.max_sufficiency_iterations,
            )
            return prediction, rag_pipeline.get_last_rag_trace()

        forced_use_multimodal = args.mode == "full_rag"
        stack.enter_context(
            patch.object(
                rag_pipeline,
                "should_use_rag",
                return_value=RagRequirementDecision(use_rag=True, reason=f"forced:{args.mode}"),
            )
        )
        prediction = rag_pipeline.answer_with_rag(
            img_paths=image_paths,
            question=sample.question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            top_n_images=args.top_n_images,
            chunk_size=args.chunk_size,
            chunks_per_doc=args.chunks_per_doc,
            use_multimodal=forced_use_multimodal,
            multimodal_text_weight=args.multimodal_text_weight,
            no_rag=False,
            debug=args.debug,
            max_sufficiency_iterations=args.max_sufficiency_iterations,
        )
        return prediction, rag_pipeline.get_last_rag_trace()


def summarize_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    grouped["overall"] = list(results)
    for item in results:
        grouped.setdefault(str(item["group"]), []).append(item)

    summary: Dict[str, Any] = {}
    for group_name, group_items in grouped.items():
        count = len(group_items)
        if count == 0:
            continue
        summary[group_name] = {
            "count": count,
            "judge_accuracy": sum(float(item["success_rate"]) for item in group_items) / count,
            "avg_latency_seconds": sum(float(item["avg_latency_seconds"]) for item in group_items) / count,
        }
    return summary


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_formal_report(
    mode: str,
    model_id: str,
    dataset_path: Path,
    output_path: Path,
    summary: Dict[str, Any],
    args: argparse.Namespace,
    sample_count: int,
    eval_started_at: str,
) -> str:
    report_lines = [
        "=" * 72,
        "Baseline Evaluation Report",
        "=" * 72,
        f"Mode: {mode}",
        f"Model ID: {model_id}",
        f"Dataset: {dataset_path}",
        f"Evaluation Time: {eval_started_at}",
        f"Evaluated Samples: {sample_count}",
        f"Output JSON: {output_path}",
        f"Output Text Report: {output_path.with_suffix('.txt')}",
        f"Judge Model: {JUDGE_MODEL}",
        "",
        "Evaluation Configuration",
        "-" * 72,
        f"top_k={args.top_k}",
        f"candidate_k={args.candidate_k}",
        f"top_n_images={args.top_n_images}",
        f"chunk_size={args.chunk_size}",
        f"chunks_per_doc={args.chunks_per_doc}",
        f"max_sufficiency_iterations={args.max_sufficiency_iterations}",
        f"web_search_provider={args.web_search_provider}",
        f"use_multimodal={args.use_multimodal}",
        f"multimodal_text_weight={args.multimodal_text_weight}",
        f"trails={args.trails}",
        f"offset={args.offset}",
        f"limit={args.limit if args.limit else 'all'}",
        "",
        "Summary Metrics",
        "-" * 72,
        f"{'Group':<18} {'Count':>7} {'Judge Acc':>12} {'Avg Latency':>14}",
        "-" * 72,
    ]

    for group_name in ("overall", "dynamic_web_qa", "static_vqa"):
        group_summary = summary.get(group_name)
        if not group_summary:
            continue
        report_lines.append(
            f"{group_name:<18} "
            f"{group_summary['count']:>7} "
            f"{_format_percent(group_summary['judge_accuracy']):>12} "
            f"{group_summary['avg_latency_seconds']:>11.3f}s"
        )

    report_lines.extend(
        [
            "-" * 72,
            "Metric Notes",
            "-" * 72,
            "Judge Acc: binary accuracy from a model-based correctness judge.",
            "The judge model is fixed to qwen3.6-flash and returns only CORRECT or INCORRECT.",
            "=" * 72,
        ]
    )
    return "\n".join(report_lines)


def print_summary(report_text: str) -> None:
    print(report_text)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        parser.error(f"Dataset not found: {dataset_path}")
    if args.limit < 0 or args.offset < 0:
        parser.error("--limit and --offset must be >= 0")
    if args.trails <= 0:
        parser.error("--trails must be >= 1")
    if args.max_sufficiency_iterations < 0:
        parser.error("--max-sufficiency-iterations must be >= 0")
    if args.multimodal_text_weight < 0.0 or args.multimodal_text_weight > 1.0:
        parser.error("--multimodal-text-weight must be between 0 and 1")

    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not str(args.model_id or "").strip():
        parser.error("--model-id must be non-empty")

    samples = load_dataset(dataset_path)
    if args.offset:
        samples = samples[args.offset:]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        parser.error("No samples selected after applying --offset/--limit")

    results: List[Dict[str, Any]] = []
    eval_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    original_model = oai_config.get("model")
    oai_config["model"] = args.model_id
    try:
        for sample in samples:
            print(f"[{sample.index}] Evaluating ({sample.group}): {sample.question}")
            attempts: List[Dict[str, Any]] = []
            success_count = 0
            total_latency = 0.0

            for trail_index in range(1, args.trails + 1):
                print(f"  Trial {trail_index}/{args.trails}")
                start_time = time.perf_counter()
                prediction, inference_trace = run_inference(sample=sample, args=args)
                latency = time.perf_counter() - start_time
                judge_result = judge_prediction(sample=sample, prediction=prediction)
                total_latency += latency
                success_count += int(judge_result["judge_correct"])
                attempts.append(
                    {
                        "trial_index": trail_index,
                        "prediction": prediction,
                        "model_id": args.model_id,
                        "judge_model": judge_result["judge_model"],
                        "judge_correct": judge_result["judge_correct"],
                        "judge_raw_response": judge_result["judge_raw_response"],
                        "latency_seconds": latency,
                        "inference_trace": inference_trace,
                    }
                )
                print(
                    f"    Judge={int(judge_result['judge_correct'])} "
                    f"Latency={latency:.3f}s"
                )

            success_rate = success_count / args.trails
            avg_latency = total_latency / args.trails

            result = {
                "index": sample.index,
                "group": sample.group,
                "question": sample.question,
                "gold_answer": sample.answer,
                "image_path": str(sample.image_path),
                "mode": args.mode,
                "model_id": args.model_id,
                "judge_model": JUDGE_MODEL,
                "trials": args.trails,
                "success_count": success_count,
                "success_rate": success_rate,
                "avg_latency_seconds": avg_latency,
                "attempts": attempts,
                "prediction": attempts[-1]["prediction"],
                "judge_correct": attempts[-1]["judge_correct"],
                "judge_raw_response": attempts[-1]["judge_raw_response"],
                "query_rounds": attempts[-1]["inference_trace"].get("query_rounds", []),
            }
            results.append(result)

            print(
                f"  SuccessRate={success_rate:.3f} "
                f"({success_count}/{args.trails}) "
                f"AvgLatency={avg_latency:.3f}s"
            )
    finally:
        oai_config["model"] = original_model

    summary = summarize_results(results)
    report_text = build_formal_report(
        mode=args.mode,
        model_id=args.model_id,
        dataset_path=dataset_path,
        output_path=output_path,
        summary=summary,
        args=args,
        sample_count=len(results),
        eval_started_at=eval_started_at,
    )
    payload = {
        "mode": args.mode,
        "model_id": args.model_id,
        "dataset": str(dataset_path),
        "evaluated_at": eval_started_at,
        "judge_model": JUDGE_MODEL,
        "args": vars(args),
        "summary": summary,
        "report_text": report_text,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.with_suffix(".txt").write_text(report_text, encoding="utf-8")
    print_summary(report_text)


if __name__ == "__main__":
    main()
