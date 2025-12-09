import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import FloorplanDataset
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

from model_unet_small import UNetSmall
 # same as backend
# or from model import UNetSmall if you merged


IMG_DIR = "../data/resized_images"
MASK_DIR = "../data/resized_masks"
OUT_MODEL = "../models/trained_unet.pth"
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    dataset = FloorplanDataset(IMG_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNetSmall(num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(DEVICE)
            # masks: [B,1,H,W] 0 or 1 -> convert to long labels
            labels = masks.squeeze(1).long().to(DEVICE)  # [B,H,W]

            optimizer.zero_grad()
            logits = model(imgs)  # [B,2,H,W]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, loss={avg_loss:.4f}")

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    torch.save(model.state_dict(), OUT_MODEL)
    print("Model saved to", OUT_MODEL)


if __name__ == "__main__":
    train()
