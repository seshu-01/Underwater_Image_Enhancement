"""
metrics.py
Evaluates PSNR, SSIM, and UIQM for Enhanced GAN images against Ground Truth tracking images.
"""
import os
import argparse
import numpy as np
import math
from PIL import Image
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- UIQM Implementation ---
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    T_a_L = math.ceil(alpha_L*K)
    T_a_R = math.floor(alpha_R*K)
    if (K-T_a_L-T_a_R) <= 0: return 0.0
    weight = (1/(K-T_a_L-T_a_R))
    s = int(T_a_L+1)
    e = int(K-T_a_R)
    val = sum(x[s:e])
    val = weight*val
    return val

def s_a(x, mu):
    val = sum([math.pow((pixel-mu), 2) for pixel in x])
    return val/len(x)

def _uicm(x):
    R = x[:,:,0].flatten()
    G = x[:,:,1].flatten()
    B = x[:,:,2].flatten()
    RG = R-G
    YB = ((R+G)/2)-B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt((math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)))
    r = math.sqrt(s_a_RG+s_a_YB)
    return (-0.0268*l)+(0.1586*r)

def sobel(x):
    dx = ndimage.sobel(x,0)
    dy = ndimage.sobel(x,1)
    mag = np.hypot(dx, dy)
    if np.max(mag) > 0:
        mag *= 255.0 / np.max(mag) 
    return mag

def eme(x, window_size):
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    if k1 == 0 or k2 == 0: return 0.0
    w = 2./(k1*k2)
    x = x[:window_size*k2, :window_size*k1]
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
            max_ = np.max(block)
            min_ = np.min(block)
            if min_ > 0.0 and max_ > 0.0:
                val += math.log(max_/min_)
    return w*val

def _uism(x):
    R, G, B = x[:,:,0], x[:,:,1], x[:,:,2]
    Rs, Gs, Bs = sobel(R), sobel(G), sobel(B)
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)
    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)
    return (0.299*r_eme) + (0.587*g_eme) + (0.144*b_eme)

def _uiconm(x, window_size):
    k1 = int(x.shape[1]/window_size)
    k2 = int(x.shape[0]/window_size)
    if k1 == 0 or k2 == 0: return 0.0
    w = -1./(k1*k2)
    x = x[:window_size*k2, :window_size*k1]
    alpha = 1
    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
            max_ = np.max(block)
            min_ = np.min(block)
            top = max_-min_
            bot = max_+min_
            if not (math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0):
                val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
    return w*val

def getUIQM(x):
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x, 10)
    return (c1*uicm) + (c2*uism) + (c3*uiconm)

# --- PSNR/SSIM Wrappers ---
def getPSNR(r_img, g_img):
    # Using L channel for PSNR to match SOTA norm
    r_img_l = np.array(Image.fromarray(r_img).convert('L'))
    g_img_l = np.array(Image.fromarray(g_img).convert('L'))
    return psnr(r_img_l, g_img_l, data_range=255)

def getSSIM(r_img, g_img):
    # Calculate SSIM across RGB channels
    return ssim(r_img, g_img, channel_axis=-1, data_range=255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan_dir", type=str, default="data/Gan_output")
    parser.add_argument("--clahe_dir", type=str, default="data/clahe_output")
    parser.add_argument("--ref_dir", type=str, default="data/reference_subset")
    args = parser.parse_args()
    
    # Project root (same pattern as before)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gan_dir = os.path.join(BASE_DIR, args.gan_dir)
    clahe_dir = os.path.join(BASE_DIR, args.clahe_dir)
    ref_dir = os.path.join(BASE_DIR, args.ref_dir)

    files = [f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Evaluating GAN + CLAHE on {len(files)} images...\n")

    # ===== METRIC STORAGE =====
    gan_psnr, gan_ssim, gan_uiqm = [], [], []
    clahe_psnr, clahe_ssim, clahe_uiqm = [], [], []

    for idx, filename in enumerate(files, 1):
        ref_path = os.path.join(ref_dir, filename)
        gan_path = os.path.join(gan_dir, filename)
        clahe_path = os.path.join(clahe_dir, filename)

        if not os.path.exists(ref_path):
            continue

        r_im = Image.open(ref_path).convert('RGB')
        r_arr = np.array(r_im)

        # ===== GAN =====
        if os.path.exists(gan_path):
            g_im = Image.open(gan_path).convert('RGB')
            if g_im.size != r_im.size:
                g_im = g_im.resize(r_im.size, Image.BICUBIC)

            g_arr = np.array(g_im)

            gan_psnr.append(getPSNR(r_arr, g_arr))
            gan_ssim.append(getSSIM(r_arr, g_arr))
            gan_uiqm.append(getUIQM(g_arr))

        # ===== CLAHE =====
        if os.path.exists(clahe_path):
            c_im = Image.open(clahe_path).convert('RGB')
            if c_im.size != r_im.size:
                c_im = c_im.resize(r_im.size, Image.BICUBIC)

            c_arr = np.array(c_im)

            clahe_psnr.append(getPSNR(r_arr, c_arr))
            clahe_ssim.append(getSSIM(r_arr, c_arr))
            clahe_uiqm.append(getUIQM(c_arr))

        if idx % 10 == 0 or idx == len(files):
            print(f"\n[{idx}/{len(files)}]")

            if len(gan_psnr) > 0:
                print(f"  GAN   -> PSNR: {gan_psnr[-1]:.2f} | SSIM: {gan_ssim[-1]:.4f} | UIQM: {gan_uiqm[-1]:.2f}")

            if len(clahe_psnr) > 0:
                print(f"  CLAHE -> PSNR: {clahe_psnr[-1]:.2f} | SSIM: {clahe_ssim[-1]:.4f} | UIQM: {clahe_uiqm[-1]:.2f}")


    # ===== FINAL RESULTS =====
    print("\n" + "="*60)
    print("FINAL METRICS (Mean ± Std)\n")

    print("GAN RESULTS:")
    print(f"  PSNR : {np.mean(gan_psnr):.2f} ± {np.std(gan_psnr):.2f}")
    print(f"  SSIM : {np.mean(gan_ssim):.4f} ± {np.std(gan_ssim):.4f}")
    print(f"  UIQM : {np.mean(gan_uiqm):.2f} ± {np.std(gan_uiqm):.2f}\n")

    print("CLAHE RESULTS:")
    print(f"  PSNR : {np.mean(clahe_psnr):.2f} ± {np.std(clahe_psnr):.2f}")
    print(f"  SSIM : {np.mean(clahe_ssim):.4f} ± {np.std(clahe_ssim):.4f}")
    print(f"  UIQM : {np.mean(clahe_uiqm):.2f} ± {np.std(clahe_uiqm):.2f}")

    print("="*60)