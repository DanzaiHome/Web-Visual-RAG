import argparse
from pathlib import Path
from typing import Sequence

from rag_v1.config import PROJECT_ROOT
from rag_v1.pipeline.rag_pipeline import answer_with_rag


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
    response = answer_with_rag(
        img_paths=image_paths,
        question=args.question,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        top_n_images=args.top_n_images,
        chunk_size=args.chunk_size,
        chunks_per_doc=args.chunks_per_doc,
        use_multimodal=args.use_multimodal,
    )
    print(f"------------------------\nFinal response:\n{response}")


if __name__ == "__main__":
    main()
