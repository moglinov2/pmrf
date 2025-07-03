# 📊 Controlled Experiments – Restored-image Results

This directory contains the outputs of the four controlled restoration tasks  
(colorization, denoising, inpainting, super-resolution) for various numbers of flow steps \(K\).  


---

## 1️⃣ Top-level: `num_flow_steps_{K}`

Each folder corresponds to the total number of rectified-flow inference steps \(K\in\{5,10,20,100\}\) used by all methods in that run.  

---

## 2️⃣ Per-degradation subfolders

Inside each `num_flow_steps_{K}` you’ll find one folder per degradation type:

| Folder Name                          | Task             | Degradation Description                                      |
|--------------------------------------|------------------|--------------------------------------------------------------|
| `colorization_gaussian_noise_025/`   | Colorization     | Grayscale input + Gaussian noise (std=0.25)                  |
| `gaussian_noise_035/`                | Denoising        | Color input + Gaussian noise (std=0.35)                     |
| `random_inpainting_gaussian_noise_01/` | Inpainting     | 90% pixels masked + Gaussian noise (std=0.10)                |
| `sr_bicubic_x8_gaussian_noise_005/`  | Super-resolution | Downsampled ×8 (bicubic) + Gaussian noise (std=0.05)         |

---

## 3️⃣ Method-specific outputs

Within *each* degradation folder you’ll find five method subfolders:

| Folder Name                      | Method                                       |
|----------------------------------|----------------------------------------------|
| `mmse/`                          | Posterior mean (Stage-1 predictor; low-distortion) |
| `naive_flow/`                    | “Flow from Y” (naïve posterior sampling)     |
| `pmrf/`                          | PMRF output \(\hat X_0\)                     |
| `posterior_conditioned_on_mmse/` | Flow conditioned on \(\hat X^*\) (X\* = MMSE) |
| `posterior_conditioned_on_y/`    | Flow conditioned on \(Y\) (degraded input)   |

Each of these contains a directory of PNG files named exactly as the input IDs (so you can align them with your degraded and ground-truth images).  

---

## 4️⃣ Quick Preview

To display a random batch of PMRF outputs for the colorization task at \(K=10\), for example:

```bash
python - <<'PY'
import glob, random, matplotlib.pyplot as plt
from PIL import Image

# pick 9 random files
imgs = glob.glob('num_flow_steps_10/colorization_gaussian_noise_025/pmrf/*.png')
for f in random.sample(imgs, 9):
    plt.imshow(Image.open(f)); plt.axis('off')
plt.tight_layout(); plt.show()
PY
