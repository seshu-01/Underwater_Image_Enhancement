import os
import numpy as np
from PIL import Image

base_dir = r"c:\1Projects\FUnIE GAN\FUnIE GAN inference\underwater-image-enhancement-comparison-FUnIE-GAN"
raw_dir = os.path.join(base_dir, "data", "raw_subset")
gan_dir = os.path.join(base_dir, "data", "Gan_output")
output_dir = os.path.join(base_dir, "Results", "images_FUnIE GAN")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Find all images in the raw subset
files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Generating side-by-side comparisons for {len(files)} images...")

for idx, filename in enumerate(files, 1):
    raw_path = os.path.join(raw_dir, filename)
    gan_path = os.path.join(gan_dir, filename)
    
    if os.path.exists(gan_path):
        # Open both images
        raw_img = Image.open(raw_path).convert('RGB')
        gan_img = Image.open(gan_path).convert('RGB')
        
        # Ensure sizes match perfectly before merging 
        # (Though they should already match from our previous gan.py script)
        if raw_img.size != gan_img.size:
            gan_img = gan_img.resize(raw_img.size, Image.BICUBIC)
            
        # Horizontally stack them (Raw on left, GAN on right)
        combined_np = np.hstack((np.array(raw_img), np.array(gan_img)))
        combined_img = Image.fromarray(combined_np)
        
        # Save output
        out_path = os.path.join(output_dir, filename)
        combined_img.save(out_path)
        
        if idx % 10 == 0 or idx == len(files):
            print(f"  Processed {idx}/{len(files)} comparisons...")
    else:
        print(f"  [!] Missing GAN output for {filename}")

print(f"\n[✓] Done! Check the '{output_dir}' folder.")
