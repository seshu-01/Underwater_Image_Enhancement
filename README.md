# Underwater Image Enhancement Comparison

## Overview
This project presents a comparative analysis of three approaches for underwater image enhancement:

1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
2. Generative Adversarial Network (GAN) - FUnIE-GAN
3. Diffusion-based methods

The goal is to evaluate performance using visual results and quantitative metrics.

---

## Problem Statement
Underwater images suffer from:
- Low contrast
- Color distortion (blue/green dominance)
- Scattering and noise

This project explores how different enhancement techniques address these issues.

---

## Methods

### 1. CLAHE
A classical image processing technique that improves local contrast.

### 2. GAN (FUnIE-GAN)
A deep learning model trained to restore underwater images.

### 3. Diffusion Models
State-of-the-art generative models that iteratively refine images.

---

## Project Structure

data/           -> Input and output images  
src/            -> Core scripts   

---

## Setup

1. Clone the repository:
   git clone <repo_link>
   cd underwater-image-enhancement-comparison

2. Install dependencies:
   pip install -r requirements.txt


## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- UIQM (Underwater Image Quality Measure)

---

## Results

Our empirical testing on a custom out-of-distribution dataset subset (n=108) yielded the following metrics for the strictly-paired FUnIE-GAN model:

| Metric | Score | Analytical Finding |
|---|---|---|
| **PSNR** | 18.56 ± 3.49 | Highlights extreme susceptibility to **Domain Shift**. The model heavily applied its EUVP training-set color palette, failing to strictly match the custom ground truth colors. |
| **SSIM** | 0.5910 ± 0.12 | Exposes the 256x256 **Architectural Bottleneck**. Heavy structural detail loss occurred when crushing high-definition images to the network's strict dimensions before upscaling. |
| **UIQM** | 2.64 ± 0.53 | Proves high perceptual capability. Despite mathematical structural flaws, the GAN successfully removed underwater haze, substantially improving contrast and color vibrancy to the human eye. |

Generated evaluation materials are stored in:
- `Results/images_FUnIE GAN/` (Visual Side-by-Side comparisons)
- `data/Gan_output/` (Direct isolated GAN image results)

---

## Conclusion

Our final findings across the evaluated models:
- **CLAHE:** Improves baseline contrast computationally, but may distort image colors heavily without intelligent correction.
- **GAN (FUnIE-GAN):** Exceptional at removing murky underwater haze and rendering vibrant colors in real-time (proven by a strong UIQM score). However, its lightweight architecture severely restricts resolution fidelity (low SSIM) and struggles to generalize to wild environments outside its exact training distribution (low PSNR).
- **Diffusion Models:** Produce highly robust, high-resolution enhancements, but are severely computationally expensive compared to lightweight counterparts.

---

## Contributors

- Member 1: CLAHE + Dataset
- Member 2: GAN Implementation
- Member 3: Metrics + Analysis

---

## Future Work

- Real-time deployment
- Integration with object detection
- Training custom GAN models
