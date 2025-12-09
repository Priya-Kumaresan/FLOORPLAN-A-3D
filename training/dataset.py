import os
import cv2
import torch
from torch.utils.data import Dataset


class FloorplanDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.fnames = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = (img[:, :, 0] * 0)  # fallback zeros

        # normalize
        img = img.astype("float32") / 255.0
        # binary mask 0/1
        mask = (mask > 0).astype("float32")

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]

        return img_t, mask_t
