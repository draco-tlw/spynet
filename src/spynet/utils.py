import cv2
import numpy as np


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Converts a (H, W, 2) flow tensor into a (H, W, 3) BGR image for OpenCV."""
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]

    # Calculate Angle (Hue) and Magnitude (Value)
    ang = np.arctan2(fy, fx) + np.pi
    mag, _ = cv2.cartToPolar(fx, fy)

    # Build HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)  # Hue mapped to 0-179 for OpenCV
    hsv[..., 1] = 255  # Max Saturation

    # Pre-allocate the destination array to satisfy strict type checkers
    norm_mag = np.zeros_like(mag)
    cv2.normalize(mag, norm_mag, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = norm_mag.astype(np.uint8)  # Assign to Value channel

    # Convert to BGR for OpenCV
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
