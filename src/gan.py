"""
FUnIE-GAN Inference Script
- Defines the generator architecture (5-layer UNet)
- Loads pretrained weights
- Runs inference on all images in data/raw-890/
- Saves enhanced outputs to data/gan/

Usage:
    python src/gan.py
    python src/gan.py --input data/raw-890 --output data/gan --weights models/funie_gan/funie_generator.pth
"""
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
from ntpath import basename

import torch
import torch.nn as nn
from torch.autograd import Variable

# Add project root to path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils import get_transform, get_image_paths, tensor_to_numpy, save_image


# ---------------------------------------------------------------------------
#  FUnIE-GAN Generator Architecture (5-layer UNet)
#  Paper: arxiv.org/pdf/1903.09766.pdf
# ---------------------------------------------------------------------------
class UNetDown(nn.Module):
    """Downsampling block: Conv2d -> (BatchNorm) -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Upsampling block: ConvTranspose2d -> BatchNorm -> ReLU + skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), dim=1)


class GeneratorFunieGAN(nn.Module):
    """5-layer UNet-based generator for underwater image enhancement."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.down1 = UNetDown(in_channels, 32, use_bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, use_bn=False)
        # Decoder
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final(u4)


# ---------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------
def load_generator(weights_path, device):
    """Load the FUnIE-GAN generator with pretrained weights."""
    assert os.path.exists(weights_path), f"Weights not found: {weights_path}"

    model = GeneratorFunieGAN()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[✓] Generator loaded from {weights_path}")
    print(f"[✓] Running on: {device}")
    return model


def run_inference(model, input_dir, output_dir, device):
    """Run the generator on every image in input_dir and save results to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    transform = get_transform(img_size=256)

    image_paths = get_image_paths(input_dir)
    total = len(image_paths)
    print(f"[✓] Found {total} images in {input_dir}")

    if total == 0:
        print("[!] No images found. Exiting.")
        return

    times = []
    for idx, img_path in enumerate(image_paths, 1):
        # Load and preprocess
        img = Image.open(img_path).convert('RGB')
        orig_size = img.size  # (width, height)
        inp = transform(img).unsqueeze(0).to(device)  # (1, 3, 256, 256)

        # Inference
        start = time.time()
        with torch.no_grad():
            enhanced = model(inp)
        elapsed = time.time() - start
        times.append(elapsed)

        # Convert and save
        out_np = tensor_to_numpy(enhanced)

        # 1. Resize back to original dimensions
        out_pil = Image.fromarray(out_np).resize(orig_size, Image.BICUBIC)
        out_np = np.array(out_pil)

        out_name = basename(img_path)
        save_image(out_np, os.path.join(output_dir, out_name))

        if idx % 50 == 0 or idx == total:
            print(f"  Processed {idx}/{total} images...")

    # Summary
    avg_time = np.mean(times[1:]) if len(times) > 1 else times[0]
    total_time = np.sum(times)
    print(f"\n{'='*50}")
    print(f"  Total images : {total}")
    print(f"  Total time   : {total_time:.2f} sec")
    print(f"  Avg FPS      : {1.0 / avg_time:.2f}")
    print(f"  Output saved : {output_dir}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FUnIE-GAN Underwater Image Enhancement")
    parser.add_argument("--input",   type=str, default="data/raw-890",
                        help="Path to input directory containing raw underwater images")
    parser.add_argument("--output",  type=str, default="data/gan",
                        help="Path to output directory for enhanced images")
    parser.add_argument("--weights", type=str, default="models/funie_gan/funie_generator.pth",
                        help="Path to pretrained generator weights (.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = load_generator(args.weights, device)
    run_inference(generator, args.input, args.output, device)
