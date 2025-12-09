import os
import cv2
import numpy as np

IMG_SIZE = 512

RAW_DIR = "../data/input_plans"
GEN_MASK_DIR = "../data/generated_masks"   # after generate_masks.py
RESIZED_IMG_DIR = "../data/resized_images"
RESIZED_MASK_DIR = "../data/resized_masks"


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def resize_all():
    ensure_dir(RESIZED_IMG_DIR)
    ensure_dir(RESIZED_MASK_DIR)

    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(RAW_DIR, fname)
        mask_path = os.path.join(GEN_MASK_DIR, fname)  # same name

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(RESIZED_IMG_DIR, fname), img_resized)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # if mask missing, create simple edge mask
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            mask = edges

        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(mask_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(RESIZED_MASK_DIR, fname), mask_bin)

    print("Resizing done.")


if __name__ == "__main__":
    resize_all()
