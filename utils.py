import torch
import numpy as np
import tifffile
import cv2

def preprocess_image(path):
    img = tifffile.imread(path)  # (H, W, 12)
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Normalize
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 12, H, W)
    return img

def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay binary mask on image with blue color
    """
    if len(mask.shape) == 3:
        mask = mask.squeeze()

    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = [0, 0, 255]  # Blue

    overlayed = cv2.addWeighted(image, 1, color_mask, alpha, 0)
    return overlayed
