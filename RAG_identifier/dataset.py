import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip

class RAGRouterDataset(Dataset):

    def __init__(self, annotation_file, image_root, preprocess):
        self.data = json.load(open(annotation_file))
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        image_path = os.path.join(self.image_root, sample["image"])
        question = sample["question"]
        label = sample["label"]

        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)

        return {
            "image": image,
            "question": question,
            "label": torch.tensor(label).float()
        }