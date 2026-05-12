import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the time needed to read one embedding from a cache JSON file.",
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Path to a cache JSON file.",
    )
    parser.add_argument(
        "--key",
        default="",
        help=(
            "Optional dot-separated path to a specific embedding field, "
            'for example: "prompt_image_embeddings" or '
            '"docs.<doc_key>.chunk_profiles.<profile_key>.text_retrieval_chunk_embeddings"'
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeated measurements. Default: 1.",
    )
    return parser


def _resolve_path(payload: Any, key_path: str) -> Any:
    current = payload
    for part in key_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise KeyError(f"Key path not found: {key_path}")
    return current


def _looks_like_embedding_matrix(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not isinstance(value[0], list) or not value[0]:
        return False
    return all(isinstance(item, (int, float)) for item in value[0])


def _looks_like_embedding_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, (int, float)) for item in value)
    )


def _iter_embedding_candidates(payload: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from _iter_embedding_candidates(value, next_prefix)
        return

    if _looks_like_embedding_matrix(payload) or _looks_like_embedding_vector(payload):
        yield prefix, payload


def _find_first_embedding(payload: Any) -> Tuple[str, Any]:
    preferred_suffixes = (
        "prompt_image_embeddings",
        "question_clip_text_embedding",
        "text_retrieval_chunk_embeddings",
        "clip_text_embeddings",
        "page_image_embeddings",
    )

    candidates = list(_iter_embedding_candidates(payload))
    for suffix in preferred_suffixes:
        for path, value in candidates:
            if path.endswith(suffix):
                return path, value

    if not candidates:
        raise ValueError("No embedding-like field was found in the JSON file.")
    return candidates[0]


def _embedding_shape(value: Any) -> str:
    if _looks_like_embedding_matrix(value):
        return f"{len(value)} x {len(value[0])}"
    if _looks_like_embedding_vector(value):
        return f"{len(value)}"
    return "unknown"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    json_path = Path(args.json).expanduser().resolve()
    if not json_path.exists():
        parser.error(f"JSON file not found: {json_path}")
    if args.repeat <= 0:
        parser.error("--repeat must be > 0")

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if args.key:
        key_path = args.key
        embedding_value = _resolve_path(payload, key_path)
    else:
        key_path, embedding_value = _find_first_embedding(payload)

    elapsed_values = []
    loaded_value = None

    for _ in range(args.repeat):
        start_time = time.perf_counter()
        with json_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        loaded_value = _resolve_path(payload, key_path)
        elapsed_values.append(time.perf_counter() - start_time)

    average_ms = sum(elapsed_values) / len(elapsed_values) * 1000.0
    min_ms = min(elapsed_values) * 1000.0
    max_ms = max(elapsed_values) * 1000.0

    print("Cache Embedding Read Benchmark")
    print(f"file: {json_path}")
    print(f"key_path: {key_path}")
    print(f"shape: {_embedding_shape(loaded_value)}")
    print(f"repeat: {args.repeat}")
    print(f"avg_ms: {average_ms:.3f}")
    print(f"min_ms: {min_ms:.3f}")
    print(f"max_ms: {max_ms:.3f}")


if __name__ == "__main__":
    main()

# python measure_cache_embedding_read.py --json cache/openai_clip-vit-base-patch32_20260511T155719042206.json --repeat 20
