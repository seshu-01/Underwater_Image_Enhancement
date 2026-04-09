# generate a side-by-side comparison of RAW, enhanced (CLAHE and GAN), and GT images for each file in the dataset. 
# The output will be saved in separate folders for CLAHE and GAN comparisons. Each combined image will have the format: 
# [RAW | Enhanced | GT]. The script also handles missing files gracefully and provides progress updates.

import os
import numpy as np
from PIL import Image

# ====== PATHS ======
raw_dir = "data/raw_subset"
clahe_dir = "data/clahe_output"
gan_dir = "data/Gan_output"
gt_dir = "data/reference_subset"

output_clahe = "Results/RAW_CLAHE_GT"
output_gan = "Results/RAW_GAN_GT"

os.makedirs(output_clahe, exist_ok=True)
os.makedirs(output_gan, exist_ok=True)

# ====== FILE LIST ======
files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Processing {len(files)} images...")

# ====== LOOP ======
for idx, filename in enumerate(files, 1):
    raw_path = os.path.join(raw_dir, filename)
    clahe_path = os.path.join(clahe_dir, filename)
    gan_path = os.path.join(gan_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    # Skip if RAW or GT missing (mandatory for comparison)
    if not os.path.exists(raw_path):
        print(f"[!] Missing RAW: {filename}")
        continue
    if not os.path.exists(gt_path):
        print(f"[!] Missing GT: {filename}")
        continue

    # Load RAW + GT
    try:
        raw_img = Image.open(raw_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
    except Exception as e:
        print(f"[!] Error loading {filename}: {str(e)}")
        continue

    # Resize GT if needed
    if raw_img.size != gt_img.size:
        gt_img = gt_img.resize(raw_img.size, Image.BICUBIC)

    # ===== CLAHE =====
    if os.path.exists(clahe_path):
        try:
            clahe_img = Image.open(clahe_path).convert("RGB")

            if raw_img.size != clahe_img.size:
                clahe_img = clahe_img.resize(raw_img.size, Image.BICUBIC)

            combined = np.hstack((
                np.array(raw_img),
                np.array(clahe_img),
                np.array(gt_img)
            ))

            Image.fromarray(combined).save(os.path.join(output_clahe, filename))
        except Exception as e:
            print(f"[!] Error processing CLAHE for {filename}: {str(e)}")
    else:
        print(f"[!] Missing CLAHE: {filename}")

    # ===== GAN =====
    if os.path.exists(gan_path):
        try:
            gan_img = Image.open(gan_path).convert("RGB")

            if raw_img.size != gan_img.size:
                gan_img = gan_img.resize(raw_img.size, Image.BICUBIC)

            combined = np.hstack((
                np.array(raw_img),
                np.array(gan_img),
                np.array(gt_img)
            ))

            Image.fromarray(combined).save(os.path.join(output_gan, filename))
        except Exception as e:
            print(f"[!] Error processing GAN for {filename}: {str(e)}")
    else:
        print(f"[!] Missing GAN: {filename}")

    if idx % 10 == 0 or idx == len(files):
        print(f"Processed {idx}/{len(files)}")

print("\nDone.")
