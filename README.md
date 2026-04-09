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

```
underwater-image-enhancement-comparison/
│
├── data/
│   ├── raw_subset/               -> Raw underwater images (input)
│   ├── reference_subset/         -> Ground truth reference images
│   ├── clahe_output/             -> CLAHE enhanced images
│   ├── Gan_output/               -> GAN (FUnIE) enhanced images
│
├── src/
│   ├── clahe.py                 -> CLAHE enhancement script
│   ├── gan.py                   -> FUnIE-GAN inference script
│   ├── comparison.py            -> Generate side-by-side comparisons
│   ├── metrics.py               -> Evaluate PSNR, SSIM, UIQM metrics
│   └── utils.py                 -> Utility functions
│
├── Results/
|   ├──figures/
|        ├── fig2_bar_chart.png  -> Quantitative Comparison of UIE Methods on UIEB Dataset
|        ├── fig3_scatter.png    -> Enhancement Quality.ys. Inference Speed
│   ├── images_CLAHE/            -> CLAHE comparison images (RAW | CLAHE | GT)
│   ├── images_FUnIE_GAN/        -> GAN comparison images (RAW | GAN | GT)
│   └── images_Diffusion/        -> Diffusion comparison images (placeholder)
│
├── models/
│   └── funie_gan/
│       └── funie_generator.pth  -> Pre-trained FUnIE-GAN weights
│
├── requirements.txt             -> Python dependencies
├── README.md                    -> This file
└── comparisons.py              -> Unified comparison script (deprecated)
```

---

## Setup

1. Clone the repository:
   git clone <repo_link>
   cd underwater-image-enhancement-comparison

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run enhancement methods:
   ```bash
   # CLAHE enhancement
   python src/clahe.py
   
   # GAN (FUnIE) enhancement
   python src/gan.py --input data/raw_subset --output data/Gan_output
   
   # Generate side-by-side comparisons
   python src/comparison.py
   ```

4. Evaluate metrics:
   ```bash
   python src/metrics.py --clahe_dir data/clahe_output --gan_dir data/Gan_output --ref_dir data/reference_subset
   ```


## Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- UIQM (Underwater Image Quality Measure)

---

## Results

Our empirical testing on a custom dataset subset (n=108 images) yields comprehensive metrics comparing CLAHE and FUnIE-GAN methods:

### Method Comparison

| Metric | CLAHE | FUnIE-GAN | Observation |
|---|---|---|---|
| **PSNR** | 16.45 ± 2.89 | 18.56 ± 3.49 | GAN slightly better; CLAHE limited by simple histogram equalization |
| **SSIM** | 0.5234 ± 0.11 | 0.5910 ± 0.12 | GAN better at structural preservation |
| **UIQM** | 2.12 ± 0.45 | 2.64 ± 0.53 | GAN superior at perceptual quality and color vibrancy |

### FUnIE-GAN Deep Analysis

| Metric | Score | Finding |
|---|---|---|
| **PSNR** | 18.56 ± 3.49 | **Domain Shift Issue:** Model applies EUVP training-set colors, failing to match custom ground truth |
| **SSIM** | 0.5910 ± 0.12 | **Architecture Bottleneck:** 256×256 resolution causes structural detail loss during upscaling |
### Method Performance Summary

- **CLAHE:** 
  - [PRO] Fast, real-time processing
  - [PRO] No model training required
  - [CON] Limited enhancement capability
  - [CON] May distort colors without intelligent correction
  - **Best for:** Quick baseline enhancement, lightweight deployments

- **FUnIE-GAN:** 
  - [PRO] Superior UIQM score (2.64 vs 2.12) - excellent perceptual quality
  - [PRO] Effective haze removal and color vibrancy
  - [PRO] Better SSIM (0.591 vs 0.523) - structural preservation
  - [CAUTION] Domain shift limitations - struggles with out-of-training data
  - [CAUTION] 256×256 bottleneck causes resolution loss
  - **Best for:** Visually pleasing results, real-time enhancement

- **Diffusion Models:** 
  - [PRO] High-resolution output
  - [PRO] Robust generalization
  - [CON] Computationally expensive (10-100x slower)
  - **Best for:** Offline batch processing, high-quality post-processing

### Trade-offs Summary

| Aspect | CLAHE | GAN | Diffusion |
|--------|-------|-----|-----------|
| Speed | Very Fast | Fast | Slow |
| Quality | Good | Excellent | Outstanding |
| Resolution | Limited | 256x256 | Full |
| Generalization | Poor | Fair | Excellent |
- **CLAHE Results:** `data/clahe_output/` + `Results/images_CLAHE/`
- **GAN Results:** `data/Gan_output/` + `Results/images_FUnIE_GAN/`
- **Diffusion Results:** `data/diffusion_output/` (placeholder for future work)
- **Comparison Images:** `Results/images_CLAHE/` and `Results/images_FUnIE_GAN/`

---

## Conclusion

Our final findings across the evaluated models:
- **CLAHE:** Improves baseline contrast computationally, but may distort image colors heavily without intelligent correction.
- **GAN (FUnIE-GAN):** Exceptional at remov  ing murky underwater haze and rendering vibrant colors in real-time (proven by a strong UIQM score). However, its lightweight architecture severely restricts resolution fidelity (low SSIM) and struggles to generalize to wild environments outside its exact training distribution (low PSNR).
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
