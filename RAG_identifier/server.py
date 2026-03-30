import torch
from transformers import AutoProcessor
from model import CLIPRouter
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import io

from pathlib import Path
IDENTIFIER_DIR = Path(__file__).resolve().parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="CLIP Router API",
    description="Binary decision model for RAG routing",
    version="1.0"
)

def load_model(checkpoint_path):
    print(f"Using device: {DEVICE}")

    model = CLIPRouter().to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model

model = load_model("checkpoints/router_best.pt")

# --------------------------------------------------
# processor (global singleton)
# --------------------------------------------------

processor = AutoProcessor.from_pretrained(
    "google/siglip-so400m-patch14-384",
    cache_dir = IDENTIFIER_DIR / "model",
)

# processor = AutoProcessor.from_pretrained(
#     IDENTIFIER_DIR / "model",
#     local_files_only=True
# )

# --------------------------------------------------
# core predict
# --------------------------------------------------

def predict(image: Image.Image, question: str):

    image_tensor = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    with torch.no_grad():
        logit = model(image_tensor, [question])
        prob = torch.sigmoid(logit)

    return prob.item()

# --------------------------------------------------
# request schema (for JSON mode)
# --------------------------------------------------

class PredictRequest(BaseModel):
    question: str

# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.post("/predict")
async def predict_api(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    """
    Multipart form:
    - file: image
    - question: text
    """

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prob = predict(image, question)

    return {
        "probability": prob
    }


@app.get("/")
def health_check():
    return {"status": "ok"}



# uvicorn server:app --host 127.0.0.1 --port 18000 > "/remote-home1/xzhe/projects/CV_project/RAG_identifier/logs/RAG_server.log" 2>&1 &