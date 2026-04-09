import os
import shutil

# Target directory paths
base_dir = r"c:\1Projects\FUnIE GAN\FUnIE GAN inference\underwater-image-enhancement-comparison-FUnIE-GAN"
source_raw_dir = os.path.join(base_dir, "data", "raw-890")
source_gan_dir = os.path.join(base_dir, "data", "gan")
source_ref_dir = os.path.join(base_dir, "data", "reference-890")
target_raw_dir = os.path.join(base_dir, "data", "raw_subset")
target_gan_dir = os.path.join(base_dir, "data", "Gan_output")
target_ref_dir = os.path.join(base_dir, "data", "reference_subset")

# Create directories if they don't exist
os.makedirs(target_raw_dir, exist_ok=True)
os.makedirs(target_gan_dir, exist_ok=True)
os.makedirs(target_ref_dir, exist_ok=True)

# List of files provided by the user
files_to_copy = [
    "2_img_.png", "3_img_.png", "4.png", "4_img_.png", "6_img_.png", "7_img_.png", "8_img_.png", 
    "9_img_.png", "10_img_.png", "11_img_.png", "13_img_.png", "14_img_.png", "15_img_.png", 
    "16_img_.png", "17_img_.png", "18_img_.png", "19_img_.png", "20.png", "22_img_.png", 
    "23_img_.png", "24_img_.png", "25_img_.png", "26_img_.png", "27_img_.png", "28_img_.png", 
    "29_img_.png", "30_img_.png", "31.png", "32_img_.png", "35.png", "37_img_.png", 
    "39_img_.png", "41_img_.png", "42_img_.png", "43_img_.png", "44_img_.png", "45_img_.png", 
    "46_img_.png", "48_img_.png", "49_img_.png", "50_img_.png", "51_img_.png", "52_img_.png", 
    "53_img_.png", "55_img_.png", "56_img_.png", "57_img_.png", "58_img_.png", "59_img_.png", 
    "60_img_.png", "61_img_.png", "62_img_.png", "63_img_.png", "64_img_.png", "65_img_.png", 
    "67_img_.png", "70_img_.png", "72_img_.png", "73_img_.png", "74_img_.png", "76_img_.png", 
    "77_img_.png", "78_img_.png", "79_img_.png", "80_img_.png", "82_img_.png", "86_img_.png", 
    "87_img_.png", "89_img_.png", "90_img_.png", "91_img_.png", "92_img_.png", "94_img_.png", 
    "95_img_.png", "96_img_.png", "97_img_.png", "98_img_.png", "99_img_.png", "100_img_.png", 
    "101_img_.png", "102_img_.png", "104_img_.png", "105_img_.png", "106_img_.png", "107_img_.png", 
    "108_img_.png", "109_img_.png", "111_img_.png", "112_img_.png", "113_img_.png", "114_img_.png", 
    "115_img_.png", "116_img_.png", "117_img_.png", "118_img_.png", "120_img_.png", "121_img_.png", 
    "122_img_.png", "123_img_.png", "124_img_.png", "125_img_.png", "129_img_.png", "130_img_.png", 
    "131_img_.png", "133_img_.png", "135_img_.png", "136_img_.png", "137_img_.png"
]

copied_count_raw = 0
not_found_raw = []

for filename in files_to_copy:
    src_path = os.path.join(source_raw_dir, filename)
    dst_path = os.path.join(target_raw_dir, filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_count_raw += 1
    else:
        not_found_raw.append(filename)

print(f"Copied {copied_count_raw} raw files to {target_raw_dir}")

copied_count_gan = 0
not_found_gan = []

for filename in files_to_copy:
    src_path = os.path.join(source_gan_dir, filename)
    dst_path = os.path.join(target_gan_dir, filename)
    
    if os.path.exists(src_path):
        # We use copy2 (instead of move) to prevent accidental data loss 
        # from the original folder, acting essentially as a moving operation
        shutil.copy2(src_path, dst_path)
        copied_count_gan += 1
    else:
        not_found_gan.append(filename)

print(f"Copied {copied_count_gan} gan files to {target_gan_dir}")

copied_count_ref = 0
not_found_ref = []

for filename in files_to_copy:
    src_path = os.path.join(source_ref_dir, filename)
    dst_path = os.path.join(target_ref_dir, filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        copied_count_ref += 1
    else:
        not_found_ref.append(filename)

print(f"Copied {copied_count_ref} reference files to {target_ref_dir}")
