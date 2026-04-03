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

Results are stored in:
- results/images/ (visual comparisons)
- results/graphs/ (plots)
- results/tables/ (metric values)

---

## Conclusion (Expected)

- CLAHE improves contrast but may distort colors
- GAN provides balanced enhancement
- Diffusion models produce high-quality results but are computationally expensive

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
