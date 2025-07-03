# 🗂️ PMRF – Restored-image Outputs

This directory stores **blind-face-restoration results** produced by the official
Posterior-Mean Rectified Flow (PMRF) implementation.  
For every benchmark test-set we keep two sub-folders:

| Sub-folder | Contents | Notes |
|------------|----------|-------|
| `restored_images/` | Final PMRF outputs $\hat X_0$ (photo-realistic, low-distortion) | Obtained after **rectified-flow refinement** with the default setting `K = 25` steps. |
| `restored_images_posterior_mean/` | Stage-1 estimates $\hat X^{\*}$ (posterior mean / MMSE) | Distortion-optimal but typically over-smooth – serves as the baseline “posterior mean” in the paper. |

The five dataset-specific folders are:

| Folder | Benchmark split (paper § 5.1) | Native resolution |
|--------|--------------------------------|-------------------|
| `celeba512_pmrf/`   | **CelebA-Test** (512 × 512 HQ faces)       | 512 × 512 |
| `lfw_pmrf/`         | **LFW-Test** (real-world LQ faces)        | 512 × 512 |
| `wider_pmrf/`       | **WIDER-Test** (real-world LQ faces)      | 512 × 512 |
| `webphoto_pmrf/`    | **WebPhoto-Test** (real-world LQ faces)   | 512 × 512 |
| `celeb_adult_pmrf/` | **CelebAdult-Test** (real-world LQ faces) | 512 × 512 |

