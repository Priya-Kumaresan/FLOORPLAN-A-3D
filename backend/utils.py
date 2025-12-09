import io
import numpy as np
from PIL import Image


IMG_SIZE = 512  # fixed training / inference size


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


def resize_pil(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)
