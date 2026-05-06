"""Top-level package for the RAG V1 project."""

import os


_hf_endpoint = (
    os.getenv("CLIP_HF_ENDPOINT") or os.getenv("HF_ENDPOINT") or "https://hf-mirror.com"
)
os.environ["HF_ENDPOINT"] = _hf_endpoint

def answer_with_rag(*args, **kwargs):
    from rag_v1.pipeline.rag_pipeline import answer_with_rag as _answer_with_rag

    return _answer_with_rag(*args, **kwargs)

__all__ = ["answer_with_rag"]
