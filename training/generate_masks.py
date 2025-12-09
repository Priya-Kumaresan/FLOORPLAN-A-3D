import os
import cv2

RAW_DIR = "data/input_plans"
OUT_DIR = "data/generated_masks"

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def generate_masks():
    ensure_dir(OUT_DIR)

    for fname in os.listdir(RAW_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(RAW_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("Cannot open:", path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(edges, kernel, iterations=2)

        out_path = os.path.join(OUT_DIR, fname)
        cv2.imwrite(out_path, mask)
        print("Generated mask →", out_path)

    print("All masks generated.")

if __name__ == "__main__":
    generate_masks()
