import os
import json
import base64
import requests
import re
from tqdm import tqdm

from pathlib import Path

IDENTIFIER_DIR = Path(__file__).resolve().parent

# ==============================
# Config
# ==============================

API_URL = "http://127.0.0.1:18001/v1/chat/completions"
MODEL_NAME = "Qwen3-VL"

TRAIN_JSON = IDENTIFIER_DIR / "data "/ "router_dataset" / "router_train.json"
VAL_JSON = IDENTIFIER_DIR / "data" / "router_dataset" / "router_val.json"

IMAGE_ROOT = IDENTIFIER_DIR / "data " / "router_dataset"

HEADERS = {
    "Content-Type": "application/json"
}

# ==============================
# Prompt
# ==============================

SYSTEM_PROMPT = """
You are a strict classifier to determine whether a question requires RAG (external knowledge retrieval).

Rules:
1. If answering requires external knowledge (e.g., people names, locations, historical facts, world knowledge), output LABEL: 1.
2. Even if you already know the answer, if it belongs to knowledge/memorization, still output LABEL: 1.
3. If the question can be answered from the image using reasoning, counting, or visual understanding with some basic world knowledge, output LABEL: 0.

Output format (STRICT):
LABEL: <0 or 1>

Do NOT output anything else.
"""

# ==============================
# Utils
# ==============================

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def resolve_image_path(image_rel_path):

    filename = os.path.basename(image_rel_path)          # COCO_train2014_xxx.jpg
    subdir = os.path.dirname(image_rel_path)             # train2014

    coco_root = IDENTIFIER_DIR / "data" / "coco"

    return os.path.join(coco_root, subdir, filename)

def call_vl_model(image_path, question):
    image_base64 = encode_image(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    result = response.json()
    text = result["choices"][0]["message"]["content"]

    return text


def extract_label(text):
    match = re.search(r"LABEL:\s*([01])", text)
    if match:
        return int(match.group(1))
    else:
        print("⚠️ Failed to parse:", text)
        return None


# ==============================
# Core processing
# ==============================

def process_file(input_json_path):
    output_json_path = input_json_path.replace(".json", "_modified.json")

    with open(input_json_path, "r") as f:
        data = json.load(f)

    with open(output_json_path, "w") as f:
        json.dump([], f)

    results = []

    for item in tqdm(data):
        image_rel_path = item["image"]
        question = item["question"]

        # image_path = os.path.join(IMAGE_ROOT, image_rel_path)
        image_path = resolve_image_path(image_rel_path)
        
        try:
            response_text = call_vl_model(image_path, question)
            label = extract_label(response_text)

            if label is None:
                continue

            new_item = {
                "image": image_rel_path,
                "question": question,
                "label": label
            }

            results.append(new_item)

            with open(output_json_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            print(f"Error processing {image_rel_path}: {e}")
            continue

def main():
    print("Processing train set...")
    process_file(TRAIN_JSON)

    print("Processing val set...")
    process_file(VAL_JSON)


if __name__ == "__main__":
    main()