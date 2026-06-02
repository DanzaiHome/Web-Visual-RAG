from __future__ import annotations

import shutil
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_v1.config import PROJECT_ROOT
from rag_v1.pipeline.rag_pipeline import answer_with_rag


UPLOAD_ROOT = PROJECT_ROOT / ".tmp_uploads" / "web_ui"
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGES = 4
MAX_IMAGE_SIDE = 1600


class AskResponse(BaseModel):
    answer: str
    elapsed_seconds: float
    image_count: int


class HealthResponse(BaseModel):
    status: str
    upload_root: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="Visual Web RAG API",
        version="0.1.0",
        description="Upload images and ask questions through the existing rag_v1 pipeline.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", upload_root=str(UPLOAD_ROOT))

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(
        question: str = Form(...),
        images: List[UploadFile] = File(...),
        top_k: int = Form(5),
        candidate_k: int = Form(10),
        top_n_images: int = Form(3),
        chunk_size: int = Form(400),
        chunks_per_doc: int = Form(3),
        max_sufficiency_iterations: int = Form(3),
        use_multimodal: bool = Form(True),
        debug: bool = Form(False),
    ) -> AskResponse:
        clean_question = question.strip()
        if not clean_question:
            raise HTTPException(status_code=400, detail="Question must not be empty.")
        if not images:
            raise HTTPException(status_code=400, detail="At least one image is required.")
        if len(images) > MAX_IMAGES:
            raise HTTPException(status_code=400, detail=f"At most {MAX_IMAGES} images are supported.")
        if top_k <= 0 or candidate_k <= 0 or top_n_images <= 0 or chunk_size <= 0 or chunks_per_doc <= 0:
            raise HTTPException(status_code=400, detail="Retrieval parameters must be positive.")
        if max_sufficiency_iterations < 0:
            raise HTTPException(status_code=400, detail="max_sufficiency_iterations must be >= 0.")

        request_dir = UPLOAD_ROOT / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=False)
        saved_paths: list[Path] = []

        try:
            for index, image in enumerate(images, start=1):
                content_type = (image.content_type or "").lower()
                suffix = ALLOWED_IMAGE_TYPES.get(content_type)
                if suffix is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported image type for {image.filename or f'image {index}'}: {content_type or 'unknown'}.",
                    )

                data = await image.read()
                if not data:
                    raise HTTPException(status_code=400, detail=f"Image {index} is empty.")
                if len(data) > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image {index} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
                    )

                try:
                    with Image.open(BytesIO(data)) as opened_image:
                        normalized_image = opened_image.convert("RGB")
                        normalized_image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
                        image_path = request_dir / f"image_{index}.jpg"
                        normalized_image.save(image_path, format="JPEG", quality=92, optimize=True)
                except UnidentifiedImageError as exc:
                    raise HTTPException(status_code=400, detail=f"Image {index} is not a valid image file.") from exc
                saved_paths.append(image_path)

            start = time.perf_counter()
            try:
                answer = await run_in_threadpool(
                    answer_with_rag,
                    img_paths=saved_paths,
                    question=clean_question,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    top_n_images=top_n_images,
                    chunk_size=chunk_size,
                    chunks_per_doc=chunks_per_doc,
                    use_multimodal=use_multimodal,
                    debug=debug,
                    max_sufficiency_iterations=max_sufficiency_iterations,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            elapsed = time.perf_counter() - start
            return AskResponse(answer=answer, elapsed_seconds=elapsed, image_count=len(saved_paths))
        finally:
            for image in images:
                await image.close()
            shutil.rmtree(request_dir, ignore_errors=True)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "rag_v1.apps.web_api:app",
        host="127.0.0.1",
        port=8010,
        reload=False,
    )


if __name__ == "__main__":
    main()
