import os
import numpy as np
import torch
import cv2

from utils import bytes_to_pil, resize_pil, pil_to_numpy, IMG_SIZE
from model_unet_small import UNetSmall


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "trained_unet.pth")

def _load_model():
    if not os.path.exists(MODEL_PATH):
        print("[floorplan_ai] No trained_unet.pth found. Using OpenCV fallback.")
        return None

    try:
        model = UNetSmall(num_classes=2)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        print("[floorplan_ai] Loaded model:", MODEL_PATH)
        return model

    except Exception as e:
        print("[floorplan_ai] ERROR loading model:", e)
        print("[floorplan_ai] Falling back to OpenCV mask.")
        return None


MODEL = _load_model()


def predict_wall_mask(image_bytes: bytes) -> np.ndarray:
    """
    Returns a binary mask (H,W) of likely wall pixels (1=wall, 0=background).
    Size: 512×512
    """
    pil_img = bytes_to_pil(image_bytes)
    pil_resized = resize_pil(pil_img, IMG_SIZE)
    img_np = pil_to_numpy(pil_resized)  # (H,W,3), uint8

    if MODEL is None:
        # Fallback: Canny edges + dilation
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = (mask > 0).astype(np.uint8)
        return mask

    # Use UNet model
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(img_tensor)
        probs = torch.softmax(logits, dim=1)
        wall_prob = probs[:, 1, :, :]  # channel 1 = wall

    mask = (wall_prob[0].cpu().numpy() > 0.5).astype(np.uint8)
    return mask
