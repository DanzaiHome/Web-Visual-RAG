from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
import torch
from PIL import Image
import json
import os
from tqdm import tqdm

import time

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_BASE_DIR = BASE_DIR.parent

DATASET_JSON = BASE_DIR / "datasets" / "coco-captions_processed.json"
MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None
_processor = None
_dataset_embeddings = None


def load_clip(model_name):
    global _model, _tokenizer, _processor

    if _model is None:
        _model = CLIPModel.from_pretrained(
            model_name,
            cache_dir= BASE_DIR / "models" / MODEL_NAME
        ).to(DEVICE)

        _tokenizer = CLIPTokenizer.from_pretrained(model_name,
                                                   cache_dir=BASE_DIR / "models" / MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(model_name,
                                                   cache_dir=BASE_DIR / "models" / MODEL_NAME)

        _model.eval()

    return _model, _tokenizer, _processor


def load_dataset_embeddings():
    global _dataset_embeddings

    if _dataset_embeddings is None:
        with open(DATASET_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids = []
        texts = []
        embeddings = []

        for item in tqdm(data, desc="Loading dataset embeddings"):
            ids.append(item["id"])
            texts.append(item["text"])
            embeddings.append(item["embedding"])

        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        _dataset_embeddings = (ids, texts, embeddings)

    return _dataset_embeddings


def compute_image_embedding(model, processor, img_path):
    image = Image.open(img_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features


def images_to_hints(img_paths: str) -> list:
    model, tokenizer, processor = load_clip(MODEL_NAME)
    ids, texts, dataset_embeddings = load_dataset_embeddings()

    best_texts = []
    # t1 = time.time()
    for img_path in img_paths:
        image_embedding = compute_image_embedding(model, processor, img_path)  # [1, D]

        similarities = torch.matmul(image_embedding, dataset_embeddings.T)  # [1, N]

        best_idx = torch.argmax(similarities, dim=-1).item()

        best_text = texts[best_idx]
        best_id = ids[best_idx]
        best_score = similarities[0, best_idx].item()

        # print(f"[MATCH] id={best_id}, score={best_score:.4f}")

        # t2 = time.time()
        # print(t2 - t1)

        best_texts.append(best_text)
        
    return best_texts


# ====== 测试 ======
if __name__ == "__main__":
    result = images_to_hints(
        [PROJECT_BASE_DIR / "RAG_identifier" / "data" / "inference_query" / "thames-river.jpg"]
    )
    print(f"Best matched text: {result}")