import cv2
import numpy as np
import torch.nn as nn
import torch

# Demo Image Creation for Band Segmentation
def create_mock_fused_image(height: int, width: int, bands: int) -> np.ndarray:
    return np.vstack([np.full((height, width), 50 * (i+1), dtype=np.uint8) + np.random.randint(0, 10, (height, width)).astype(np.uint8) for i in range(bands)])

# Test demo
if __name__ == '__main__':
    mock_img = create_mock_fused_image(100, 200, 5)
    band_bounds = detect_band_boundaries_sobel(mock_img)
    bands = segment_bands(mock_img, band_bounds)

    model = SimpleDenoiser()
    denoised_band = denoise_with_model(bands[0], model)
    quality = compute_srf_quality(denoised_band)

    plt.figure(figsize=(10, 2))
    for i, b in enumerate(bands):
        plt.subplot(1, 5, i+1)
        plt.imshow(b, cmap='gray')
        plt.title(f'Band {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("Quality of denoised Band 1:", quality)
