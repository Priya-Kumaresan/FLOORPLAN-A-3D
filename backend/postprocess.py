import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops


def extract_wall_lines(mask: np.ndarray, min_length: int = 20):
    """
    mask: (H,W) binary uint8 (0/1)
    returns list of line segments: [((x1,y1), (x2,y2)), ...] in pixel coordinates
    """
    # 1. skeletonize to 1px-wide lines
    skel = skeletonize(mask > 0).astype(np.uint8)

    # 2. use Probabilistic Hough to find long straight segments
    lines = cv2.HoughLinesP(
        skel * 255,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_length,
        maxLineGap=5,
    )

    segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(((float(x1), float(y1)), (float(x2), float(y2))))

    return segments
