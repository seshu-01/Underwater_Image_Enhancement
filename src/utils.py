"""
Utility functions for image loading, preprocessing, and saving.
Used by the FUnIE-GAN inference pipeline.
"""
import os
import fnmatch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# Standard transform pipeline for FUnIE-GAN (256x256, normalized to [-1, 1])
def get_transform(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_image_paths(data_dir):
    """Recursively collect all image file paths from a directory."""
    exts = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.JPEG', '*.bmp']
    image_paths = []
    for pattern in exts:
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    image_paths.append(os.path.join(root, filename))
    return sorted(image_paths)


def tensor_to_numpy(tensor):
    """Convert a [-1, 1] normalized tensor to a [0, 255] uint8 numpy array."""
    # Remove batch dim if present -> (C, H, W)
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    # (C, H, W) -> (H, W, C)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    # [-1, 1] -> [0, 255]
    img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return img



def save_image(np_img, save_path):
    """Save a numpy uint8 image array to disk."""
    Image.fromarray(np_img).save(save_path)
