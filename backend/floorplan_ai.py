import os
import numpy as np
import torch
import cv2
import sys

from utils import bytes_to_pil, resize_pil, pil_to_numpy, IMG_SIZE

# Add models directory to path to import model_1427
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.model_1427 import model_1427


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the pretrained model_1427.pth
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_1427.pth")

# Wall class index in the 51-class segmentation model
# Common floorplan segmentation: class 0=background, class 1=walls, others=doors/windows/rooms/etc.
# If class 1 doesn't work well, try: 0 (background), 2-50 (other classes)
WALL_CLASS_IDX = 1


def _load_model():
    if not os.path.exists(MODEL_PATH):
        print("[floorplan_ai] No model_1427.pth found. Using OpenCV fallback.")
        return None

    try:
        # Load the pretrained model architecture
        model = model_1427
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Handle different state dict formats
        if isinstance(state, dict):
            if 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            elif 'model' in state:
                model.load_state_dict(state['model'])
            else:
                model.load_state_dict(state)
        else:
            # Direct state dict
            model.load_state_dict(state)
        
        model.to(DEVICE)
        model.eval()
        print("[floorplan_ai] ✅ Loaded pretrained model_1427.pth successfully!")
        print(f"[floorplan_ai] Using device: {DEVICE}")
        return model

    except Exception as e:
        print(f"[floorplan_ai] ERROR loading model: {e}")
        print("[floorplan_ai] Falling back to OpenCV mask.")
        import traceback
        traceback.print_exc()
        return None


MODEL = _load_model()


def predict_wall_mask(image_bytes: bytes) -> np.ndarray:
    """
    Returns a binary mask (H,W) of likely wall pixels (1=wall, 0=background).
    Size: 512×512
    Uses the pretrained model_1427 for high-quality AI segmentation.
    """
    pil_img = bytes_to_pil(image_bytes)
    pil_resized = resize_pil(pil_img, IMG_SIZE)
    img_np = pil_to_numpy(pil_resized)  # (H,W,3), uint8

    if MODEL is None:
        # Fallback: Canny edges + dilation (noisy, but works)
        print("[floorplan_ai] Using OpenCV fallback (noisy)")
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(edges, kernel, iterations=2)
        mask = (mask > 0).astype(np.uint8)
        return mask

    # Use pretrained model_1427 for high-quality segmentation
    # Normalize image: (H,W,3) uint8 -> (1,3,H,W) float32 [0,1]
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Model outputs: (batch, 51, H_out, W_out)
        logits = MODEL(img_tensor)
        print(f"[floorplan_ai] Model output shape: {logits.shape}")
        
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)
        
        # Debug: Check which classes have high activation
        # Get mean probability per class across the image
        class_mean_probs = probs[0].mean(dim=(1, 2)).cpu().numpy()  # Mean prob per class
        top_classes = np.argsort(class_mean_probs)[-5:][::-1]  # Top 5 classes
        print(f"[floorplan_ai] Top 5 class indices by mean probability: {top_classes}")
        print(f"[floorplan_ai] Their mean probabilities: {class_mean_probs[top_classes]}")
        
        # Extract wall class probability (class 1 = walls)
        wall_prob = probs[:, WALL_CLASS_IDX, :, :]
        print(f"[floorplan_ai] Class {WALL_CLASS_IDX} (wall) stats: min={wall_prob.min():.3f}, max={wall_prob.max():.3f}, mean={wall_prob.mean():.3f}")
        
        # Get binary mask: threshold at 0.5
        mask = wall_prob[0].cpu().numpy()
        print(f"[floorplan_ai] Wall mask shape before resize: {mask.shape}, min={mask.min():.3f}, max={mask.max():.3f}")
        
        # Handle output size mismatch: resize to 512x512 if needed
        if mask.shape != (IMG_SIZE, IMG_SIZE):
            print(f"[floorplan_ai] Resizing mask from {mask.shape} to ({IMG_SIZE}, {IMG_SIZE})")
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Binarize - try adaptive threshold if mask is mostly empty
        binary_mask = (mask > 0.5).astype(np.uint8)
        wall_pixel_count = binary_mask.sum()
        
        # If mask is mostly empty, try lower threshold or use top class
        if wall_pixel_count < 100:  # Less than 100 pixels detected
            print(f"[floorplan_ai] ⚠️  Low wall pixel count ({wall_pixel_count}), trying lower threshold (0.3)...")
            binary_mask = (mask > 0.3).astype(np.uint8)
            wall_pixel_count = binary_mask.sum()
            print(f"[floorplan_ai] After lower threshold: {wall_pixel_count} wall pixels")
            
            # If still empty, try using the class with highest activation
            if wall_pixel_count < 100:
                print(f"[floorplan_ai] ⚠️  Still low, trying top class instead of class {WALL_CLASS_IDX}")
                top_class_idx = top_classes[0]
                if top_class_idx != WALL_CLASS_IDX:
                    print(f"[floorplan_ai] Using class {top_class_idx} instead")
                    top_class_prob = probs[:, top_class_idx, :, :]
                    top_mask = top_class_prob[0].cpu().numpy()
                    if top_mask.shape != (IMG_SIZE, IMG_SIZE):
                        top_mask = cv2.resize(top_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    binary_mask = (top_mask > 0.5).astype(np.uint8)
                    wall_pixel_count = binary_mask.sum()
                    print(f"[floorplan_ai] Using class {top_class_idx}: {wall_pixel_count} pixels")
        
        print(f"[floorplan_ai] Final mask shape: {binary_mask.shape}, wall pixels: {wall_pixel_count}")
        mask = binary_mask
    
    return mask
