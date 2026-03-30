import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from dataset import RAGRouterDataset
from model import CLIPRouter

from transformers import AutoProcessor


# Argument parser

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="router_dataset")
    parser.add_argument("--coco_dir", type=str, default="coco")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


# accuracy

def compute_accuracy(logits, labels):

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    correct = (preds == labels).float().sum()

    return correct / len(labels)


# validation

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0
    total_acc = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():

        for batch in tqdm(loader, desc="Val", leave=False):

            images = batch["image"].to(device)
            questions = batch["question"]
            labels = batch["label"].to(device)

            logits = model(images, questions)

            loss = criterion(logits, labels)

            acc = compute_accuracy(logits, labels)

            total_loss += loss.item()
            total_acc += acc.item()

    return total_loss / len(loader), total_acc / len(loader)


# train

def train():

    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    # paths

    train_json = os.path.join(
        args.data_root,
        args.dataset_dir,
        "router_train.json",
    )

    val_json = os.path.join(
        args.data_root,
        args.dataset_dir,
        "router_val.json",
    )

    coco_root = os.path.join(
        args.data_root,
        args.coco_dir,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # SigLIP preprocess

    processor = AutoProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    # datasets

    train_dataset = RAGRouterDataset(
        annotation_file=train_json,
        image_root=coco_root,
        preprocess=preprocess,
    )

    val_dataset = RAGRouterDataset(
        annotation_file=val_json,
        image_root=coco_root,
        preprocess=preprocess,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # model

    model = CLIPRouter().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_acc = 0

    # resume

    if args.resume is not None:

        print("Loading checkpoint:", args.resume)

        ckpt = torch.load(args.resume)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0)

    # training loop

    for epoch in range(start_epoch, args.epochs):

        model.train()

        total_loss = 0
        total_acc = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:

            images = batch["image"].to(device)
            questions = batch["question"]
            labels = batch["label"].to(device)

            logits = model(images, questions)

            loss = criterion(logits, labels)

            acc = compute_accuracy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

            pbar.set_postfix(
                loss=total_loss / (pbar.n + 1),
                acc=total_acc / (pbar.n + 1),
            )

        # validation

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"\nEpoch {epoch} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        # save checkpoint

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }

        torch.save(
            checkpoint,
            os.path.join(args.save_dir, f"router_epoch_{epoch}.pt"),
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                checkpoint,
                os.path.join(args.save_dir, "router_best.pt"),
            )

            print("Saved best model")

def main():
    train()
    
if __name__ == "__main__":
    main()
    
# python train.py --data_root /remote-home1/xzhe/projects/CV_project/RAG_identifier/data