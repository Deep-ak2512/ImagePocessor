import cv2
import numpy as np
import torch.nn as nn
import torch
from bandsegment import *

# simulated image
def simulate_image(height: int, width: int, bands: int) -> np.ndarray:
    return np.vstack([np.full((height, width), 50 * (i+1), dtype=np.uint8) + np.random.randint(0, 10, (height, width)).astype(np.uint8) for i in range(bands)])

# Test demo
if __name__ == '__main__':
    mock_img = simulate_image(100, 200, 5)
    band_bounds =  band_segment(mock_img)
