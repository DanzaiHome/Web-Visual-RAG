"""Top-level package for the RAG V1 project."""

import os
from pathlib import Path


def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if char == "#" and not in_single_quote and not in_double_quote:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()

    return value.strip()


def _parse_env_value(raw_value: str) -> str:
    value = _strip_inline_comment(raw_value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    if value.startswith("\ufeff"):
        value = value.lstrip("\ufeff")
    return value


def _load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = _parse_env_value(raw_value)
        if value == "":
            continue
        os.environ.setdefault(key, value)


_load_project_env()

_hf_endpoint = (
    os.getenv("CLIP_HF_ENDPOINT") or os.getenv("HF_ENDPOINT") or "https://hf-mirror.com"
)
os.environ["HF_ENDPOINT"] = _hf_endpoint


def answer_with_rag(*args, **kwargs):
    from rag_v1.pipeline.rag_pipeline import answer_with_rag as _answer_with_rag

    return _answer_with_rag(*args, **kwargs)


__all__ = ["answer_with_rag"]
