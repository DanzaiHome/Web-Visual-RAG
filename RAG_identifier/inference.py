import torch
from transformers import AutoProcessor
from model import CLIPRouter
from PIL import Image

from pathlib import Path
IDENTIFIER_DIR = Path(__file__).resolve().parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path):

    print(f"Using device: \"{DEVICE}\"")
    
    model = CLIPRouter().to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])

    model.eval()

    return model


# SigLIP image processor
processor = AutoProcessor.from_pretrained(
    "google/siglip-so400m-patch14-384",
    cache_dir = IDENTIFIER_DIR / "model"
)



def predict(model, image_path, question):

    image = Image.open(image_path).convert("RGB")

    image = processor(
        images=image,
        return_tensors="pt"
    )["pixel_values"].to(DEVICE)

    with torch.no_grad():

        logit = model(image, [question])

        prob = torch.sigmoid(logit)

    return prob.item()


# --------------------------------------------------
# main
# --------------------------------------------------

if __name__ == "__main__":

    model = load_model("checkpoints/router_best.pt")

    prob = predict(
        model,
        IDENTIFIER_DIR / "data" / "inference_query" / "Sacred_Heart_Cathedral_of_Guangzhou_2025.06_02.jpg",
        "Where is this building?"
    )

    print(f"Need RAG probability: {prob}")