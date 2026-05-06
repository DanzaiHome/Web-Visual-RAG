import argparse
import time
from pathlib import Path
from typing import Sequence

from rag_v1.config import PROJECT_ROOT
from rag_v1.pipeline.rag_pipeline import answer_with_rag
from rag_v1.timing import TimingStats, set_active_timing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-pipeline",
        description="Run the visual web RAG pipeline with a question and input images.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to answer with multimodal RAG.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more local image paths.",
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
        help="Number of page images to keep per search result.",
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
        "--use-multimodal",
        action="store_true",
        help="Enable multimodal chunk retrieval instead of text-only retrieval.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug output.",
    )
    parser.add_argument(
        "--max-sufficiency-iterations",
        type=int,
        default=3,
        help="Maximum number of sufficiency-check / additional-retrieval iterations.",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Print timing statistics for the full run, API calls, and CLIP server calls.",
    )
    return parser


def _resolve_image_path(image: str) -> Path:
    image_path = Path(image).expanduser()
    candidates = [image_path]
    if not image_path.is_absolute():
        candidates.append(PROJECT_ROOT / image_path)

    for candidate in candidates:
        resolved_path = candidate.resolve()
        if resolved_path.exists():
            return resolved_path

    return image_path.resolve()


def _resolve_image_paths(images: Sequence[str]) -> list[Path]:
    resolved_paths = [_resolve_image_path(image) for image in images]
    missing_paths = [str(path) for path in resolved_paths if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(missing_paths)
        raise FileNotFoundError(f"Image file(s) not found: {missing_display}")
    return resolved_paths


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        image_paths = _resolve_image_paths(args.images)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if args.max_sufficiency_iterations < 0:
        parser.error("--max-sufficiency-iterations must be >= 0")
    timing = TimingStats() if args.time else None
    set_active_timing(timing)
    total_start_time = time.perf_counter()
    try:
        response = answer_with_rag(
            img_paths=image_paths,
            question=args.question,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            top_n_images=args.top_n_images,
            chunk_size=args.chunk_size,
            chunks_per_doc=args.chunks_per_doc,
            use_multimodal=args.use_multimodal,
            debug=args.debug,
            max_sufficiency_iterations=args.max_sufficiency_iterations,
        )
    finally:
        total_elapsed = time.perf_counter() - total_start_time
        set_active_timing(None)
    print(f"------------------------\nFinal response:\n{response}")
    if timing is not None:
        chat_api_time = timing.get_duration("chat_api")
        web_search_api_time = timing.get_duration("web_search_api")
        api_time = chat_api_time + web_search_api_time
        clip_server_time = timing.get_duration("clip_server")
        print("------------------------")
        print("Timing:")
        print(f"Total: {total_elapsed:.3f}s")
        print(f"API: {api_time:.3f}s")
        print(
            "API breakdown: "
            f"chat={chat_api_time:.3f}s ({timing.get_count('chat_api')} calls), "
            f"web_search={web_search_api_time:.3f}s ({timing.get_count('web_search_api')} calls)"
        )
        print(
            f"CLIP server: {clip_server_time:.3f}s "
            f"({timing.get_count('clip_server')} calls)"
        )


if __name__ == "__main__":
    main()
