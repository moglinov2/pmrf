<div align="center">

# Photo-Realistic Image Restoration via Posterior-Mean Rectified Flow

Code for my Bachelor's thesis at the Institute of Machine Learning, Johannes Kepler University Linz (2026).  
It reproduces and extends *“Posterior-Mean Rectified Flow: Towards Minimum MSE Photo-Realistic Image Restoration”* (Ohayon, Michaeli & Elad, ICLR 2025).

[[Thesis]](LINK_TO_THESIS_PDF) • [[PMRF Paper]](https://arxiv.org/abs/2410.00418) • [[Project Page]](https://pmrf-ml.github.io/) • [[Official Code]](https://github.com/ohayonguy/PMRF) • [[Demo]](https://huggingface.co/spaces/ohayonguy/PMRF)

</div>

---

> [!IMPORTANT]
> **Which notebook do I need?**
> - **`pmrf_imagenet_and_zappos_data.ipynb`** is the main notebook of the thesis (**Experiment III**): data preprocessing, training of all five methods from scratch on ImageNet-mini and UT Zappos50K, inference, evaluation, and the thesis figures.
> - **`pmrf_papers_data.ipynb`** comes from my Practical Work and is still needed for **Experiments I and II** (released PMRF checkpoints on the face benchmarks).

## ✨ What’s in this repo?

This fork of the official PMRF code started as my Practical Work, which reproduced Sections 5.1 and 5.2 of the paper. For the thesis, I extended it with a third experiment in which all methods are trained from scratch on two new image domains.

| Thesis | Experiment | Data | Models | Notebook |
|--------|------------|------|--------|----------|
| **I** | Blind face restoration (reproduces § 5.1 of the paper) | CelebA-Test, LFW-Test, WIDER-Test, WebPhoto-Test, CelebAdult-Test (FFHQ-512 as reference set) | released checkpoint | `pmrf_papers_data.ipynb` |
| **II** | Controlled face restoration (reproduces § 5.2): colorization, denoising, inpainting, 8× super-resolution at K = {5, 10, 20, 100} flow steps | CelebA-Test at 256×256 (FFHQ-256 as reference set) | released checkpoints | `pmrf_papers_data.ipynb` |
| **III** | Newly trained controlled restoration: colorization, denoising, super-resolution (plus inpainting in the initial configuration) | ImageNet-mini, UT Zappos50K at 128×128 | trained from scratch | `pmrf_imagenet_and_zappos_data.ipynb` |

### Compared methods

The thesis compares five methods. For Experiment III, each one has its own training script:

| Method (thesis name) | What it does | Training script |
|----------------------|--------------|-----------------|
| Posterior mean $\hat{X}^{\ast}$ | MMSE predictor (stage 1 of PMRF) | `train_mmse.sh` |
| PMRF | Rectified flow from the noised posterior mean to the clean image (stage 2) | `train_pmrf.sh` |
| Naive Flow | Rectified flow directly from the degraded input $Y$ | `train_naive_flow.sh` |
| Flow cond. on $Y$ | Flow from Gaussian noise, conditioned on $Y$ | `train_posterior_conditioned_on_y.sh` |
| Flow cond. on $\hat{X}^{\ast}$ | Flow from Gaussian noise, conditioned on the posterior mean | `train_posterior_conditioned_on_mmse_model.sh` |

### Experiment III: initial and refined configuration

Experiment III was run in two configurations. The refined one follows the ImageNet setup of Ohayon et al.

|  | Initial configuration | Refined configuration |
|--|-----------------------|-----------------------|
| Posterior-mean predictor | SwinIR-M | HDiT-XL2 |
| Flow model (velocity field) | HDiT-XL2 | HDiT-XL2 |
| Colorization | `colorization_gaussian_noise_025` | `colorization_gaussian_noise_005` |
| Denoising | `gaussian_noise_035` | `gaussian_noise_02` |
| Super-resolution | `sr_bicubic_x8_gaussian_noise_005` (8×) | `sr_bicubic_x4_gaussian_noise_005` (4×) |
| Inpainting | `random_inpainting_gaussian_noise_01` | not retained |
| Source noise σ<sub>s</sub> (PMRF, Naive Flow) | 0.1 | 0.025 |
| Notebook sections | *IMAGENET DATA*, *ZAPPOS DATA* | *ImageNet / Zappos with optimized architecture and hyperparameters* |

Shared settings:
- 128×128 images
- AdamW with learning rate 5·10⁻⁴, weight decay 10⁻² and EMA decay 0.9999
- bfloat16 mixed precision
- 100 epochs at batch size 64 on one NVIDIA A100 (Google Colab)
- K = 100 Euler steps at inference

One exception: the refined UT Zappos50K posterior-mean predictor used learning rate 10⁻⁴, because at 5·10⁻⁴ it collapsed to near-constant outputs.

---

## ⚙️ Setup

I worked entirely in Google Colab. To avoid re-cloning the code after every runtime reset, I synced the whole repo to Google Drive (`/content/drive/MyDrive/PMRF`).

Both notebooks now use the same pinned environment and the same **Setup** cells at the top. The Practical Work originally used a condacolab environment with PyTorch 2.3.1 and CUDA 11.8; the notebooks in this repo now use the pinned environment below.

Run the setup cells in this order in every new runtime:

1. **Installation.** Installs the pinned environment (see below) and applies the BasicSR patches. The cell stops early if the runtime is not Python 3.13.
2. **Runtime → Restart session.**
3. **Post-restart check.** Prints all package versions and asserts the exact pins. It then runs quick CUDA, NATTEN and dctorch tests.
4. **Mount Google Drive:**
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
5. **PMRF compatibility setup.** Checks and patches the repository files listed below and sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.
6. **Smoke test.** Imports the PMRF modules and builds all degradations except `difface`. It then runs a forward and backward pass through PMRF's NATTEN attention block.

<details>
<summary><b>Pinned environment</b></summary>

| Package | Version |
|---------|---------|
| Python | 3.13 |
| PyTorch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 (`+cu128`, CUDA 12.8) |
| NATTEN | 0.21.6 (`+torch2110cu128`) |
| Lightning / PyTorch Lightning | 2.3.3 |
| NumPy / SciPy | 2.1.3 / 1.15.3 |
| timm / einops / torch-ema | 1.0.8 / 0.8.0 / 0.3 |
| transformers / tokenizers / huggingface-hub | 4.46.3 / 0.20.3 / 0.24.5 |
| lpips / piq | 0.1.4 / 0.8.0 |
| opencv-python | 4.10.0.84 |
| wandb | 0.30.0 |
| dctorch | 0.1.2 (installed with `--no-deps`, since its metadata requires NumPy < 2) |
| torch-fidelity | pinned commit `5e211a9` from GitHub |
| BasicSR | 1.4.2, built from source (see patches) |

</details>

### Patches

| File | Patch | Applied by |
|------|-------|------------|
| BasicSR `setup.py` | `get_version()` reads the version into an explicit namespace. Python 3.13 (PEP 667) makes `locals()` a snapshot, so the original installer fails. | Installation cell (before building BasicSR) |
| BasicSR `basicsr/data/degradations.py` | Imports `rgb_to_grayscale` from `torchvision.transforms.functional`, because `functional_tensor` no longer exists in torchvision 0.26. | Installation cell |
| `utils/create_degradation.py` | Wraps the BasicSR imports in `try/except`, so BasicSR is only required for the `difface` degradation. | Compatibility cell, which saves the original as `create_degradation.py.before_optional_basicsr.bak` the first time |
| `arch/hourglass/image_transformer_v2.py` | NATTEN ≥ 0.21 compatibility: uses `natten.functional.na2d(...)`, with a `has_fused_na` guard. | Already part of this repo. The compatibility cell only checks for it and stops if the file is the unpatched original. |

In addition, `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` restores the older `torch.load` behaviour, so that the trusted Lightning checkpoints (released and self-trained) load under PyTorch 2.11.

---

## 📦 Data and checkpoints

| What | Where to download | Where to place |
|------|-------------------|----------------|
| **Pre-trained PMRF** for blind face restoration | [https://drive.google.com/drive/folders/1dfjZATcQ451uhvFH42tKnfMNHRkL6N_A] | keep structure shown below `checkpoints/…` |
| **Controlled-experiment checkpoints** | [https://drive.google.com/drive/folders/1dfjZATcQ451uhvFH42tKnfMNHRkL6N_A] | keep structure shown below `checkpoints/controlled_experiments/…` |
| **Evaluation metric models** (`resnet18_110.pth`, `alignment_WFLW_4HG.pth`) | [https://drive.google.com/drive/folders/1k3RCSliF6PsujCMIdCD1hNM63EozlDIZ] | `evaluation/metrics_ckpt/` |
| **All face test sets** (`data/`) | [https://drive.google.com/drive/folders/10ivacNpoFqq3K9xPeF1IHmrKVU7ZNRIG?usp=sharing] | keep structure shown below `data/…` |

```bash
checkpoints/
├── blind_face_restoration_pmrf.ckpt    # Checkpoint of the blind face image restoration model
├── swinir_restoration512_L1.pth        # Checkpoint of the SwinIR model trained by DifFace
├── controlled_experiments/             # Checkpoints for the controlled experiments
│   ├── colorization_gaussian_noise_025/
│   │   ├── pmrf/
│   │   │   └── epoch=999-step=273000.ckpt
│   │   ├── mmse/
│   │   │   └── epoch=999-step=273000.ckpt
.   .   .
.   .   .
.   .   .
```

```bash
data/
├── CelebA-Test/                # 512×512 HQ faces  (3 000)
├── CelebA-Test-LQ/             # same images, degraded (3 000)
├── LFW-Test/                   # 512×512 (1 711)
├── WIDER-Test/                 # 512×512 (970)
├── WebPhoto-Test/              # 512×512 (407)
├── CelebChild-Test/            # includes Adult and Child images
  ├── Child                     # 512×512 (180)
  └── Adult/                    # 512×512 (180)
└── cropped_faces/              # FFHQ reference sets
  ├── ffhq512/                  # 512×512 (10 000)
  └── ffhq256/                  # 256×256, down-sampled for the controlled experiments (10 000)
├── ImageNet-mini/              # center-cropped to 128×128
  ├── train/                    # 128×128 (34 745)
  ├── val/                      # 128×128 (3 923)
  └── test/                     # 128×128 (5 000), flat folder
├── UT Zappos50k/               # center-cropped to 128×128
  ├── train/                    # 128×128 (40 066)
  ├── val/                      # 128×128 (5 000)
  └── test/                     # 128×128 (5 000), flat folder
```

The datasets used in Experiment III can also be obtained from Kaggle:
[ImageNet 1000 (mini)](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
and [Large Shoe Dataset (UT Zappos50k)](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k).

The cell **Resizing and Cropping Data** preprocesses all images: it rescales the shorter side to 128 px (Lanczos) and center-crops to 128×128. The test splits are flat folders of 5,000 images in `data/imagenet-mini-centercropped/test` and `data/zap50k_128/test`.

> [!WARNING]
> The preprocessing cell works in place: it flattens the folder and deletes the original subfolders. Run it on a copy of the downloaded data.

---

## 🚀 How to run

### Experiment III: `pmrf_imagenet_and_zappos_data.ipynb`

The notebook has four blocks: ImageNet-mini and UT Zappos50K, each in the initial and the refined configuration. Every block follows the same pattern of **Training → Inference → Evaluation**, with one cell per task and method.

**1. Training.** For each task, train the posterior-mean predictor first (*Stage 1*). PMRF (*Stage 2*) and Flow cond. on $\hat{X}^{\ast}$ both use its checkpoint. Naive Flow and Flow cond. on $Y$ are trained independently.

```bash
%cd /content/drive/MyDrive/PMRF
!bash train_mmse.sh
```

- The same five scripts are reused for every task and dataset, so set the degradation, data paths and run name inside the script before each run.
- Checkpoints are written to `PMRF/<run_name>/checkpoints/last.ckpt`. The run names follow one pattern per block, e.g. `colorization_mmse` (ImageNet, initial), `zappos_data/zap_colorization_pmrf` (Zappos, initial), `new_colorization_pmrf` (ImageNet, refined) and `new_zap_colorization_pmrf` (Zappos, refined).
- Training runs are logged to Weights & Biases.

**2. Inference.** Set the checkpoint and output paths once, in the first cell of each *Inference* section. Then run one cell per method, for example:

```bash
!python test.py \
  --precision "32" \
  --degradation "colorization_gaussian_noise_025" \
  --test_data_root "$TEST_ROOT" \
  --num_gpus 1 \
  --batch_size 64 \
  --num_workers 5 \
  --img_size 128 \
  --ckpt_path "$PMRF_CKPT" \
  --results_path "$OUT_PMRF" \
  --num_flow_steps 100
```

For PMRF, the reconstructions end up in `<results_path>/<degradation>/pmrf/num_flow_steps=100/xhat`.

**3. Evaluation.**

```bash
%cd /content/drive/MyDrive/PMRF/evaluation
!python compute_metrics_imagenet.py \
  --gt_path  "$TEST_ROOT" \
  --rec_path "<results_path>/colorization_gaussian_noise_025/pmrf/num_flow_steps=100/xhat"
```

This computes:
- MSE, PSNR, SSIM and LPIPS on the paired test images;
- FID, KID, precision and recall against the clean test split, using Inception features.

The initial UT Zappos50K block uses `compute_metrics_zappos.py`; all other blocks use `compute_metrics_imagenet.py`.

**4. Figures.**
- The *Visualizations* section at the end of the notebook builds the qualitative grids used in the thesis.
- The *Checks* section counts the images in the output folders.

### Experiment I: blind face restoration (`pmrf_papers_data.ipynb`)

Inference (CelebA-Test example):
```bash
python inference.py \
  --ckpt_path ./checkpoints/blind_face_restoration_pmrf.ckpt \
  --lq_data_path ./data/celeba_512_validation_lq \
  --output_dir ./outputs/celeba512_pmrf \
  --batch_size 64 \
  --num_flow_steps 25
```
Change the paths to run it on the other benchmarks.

Evaluation:
```bash
python compute_metrics_blind.py \
  --parent_ffhq_512_path ../data/cropped_faces \
  --rec_path              ../outputs/celeba512_pmrf/restored_images \
  --gt_path               ../data/celeba_512_validation
```

### Experiment II: controlled face restoration (`pmrf_papers_data.ipynb`)

The notebook first creates the 256×256 test set `data/celeba_256_test`, by bicubic downscaling of `data/celeba_512_validation`.

Inference:
```bash
bash test.sh
```
In `test.sh`, adjust these arguments:
- `--test_data_root`: the path of the CelebA-Test 256×256 images;
- `--degradation` and `--ckpt_path`: the degradation you want to assess and the matching checkpoint.

Evaluation:
```bash
python compute_metrics_controlled_experiments.py \
  --parent_ffhq_256_path ../data/cropped_faces \
  --rec_path              ../controlled_experiments_results/num_flow_steps_5/colorization_gaussian_noise_025/mmse/xhat \
  --gt_path               ../data/celeba_256_test
```
Adjust `--rec_path` to evaluate the other methods, K values or degradations.

The raw outputs of my Practical Work evaluation runs for Experiments I and II are in `Paper Reconstruction Results.docx`.

---

## 📊 Main findings

- **Face domain (Experiments I and II, released models):** PMRF achieved a favourable balance between reconstruction fidelity and perceptual quality, and it reached most of its perceptual improvement within few integration steps.
- **New domains (Experiment III, trained from scratch):** this behaviour did not transfer uniformly across tasks.
  - PMRF performed favourably on colorization.
  - It gave mixed results on denoising.
  - For super-resolution in the refined configuration, it did not improve on its own posterior-mean estimate in reconstruction error or perceptual quality.
- **Limitations:** training and degradation settings were changed together between the two configurations, so the experiments do not isolate which factor drives these differences. All results are point estimates without an analysis across independent runs.

All numbers and the full discussion are in the thesis.

---

## 📚 Citation

If you use this code, please cite the original PMRF paper:
```bibtex
@inproceedings{
    ohayon2025posteriormean,
    title={Posterior-Mean Rectified Flow: Towards Minimum {MSE} Photo-Realistic Image Restoration},
    author={Guy Ohayon and Tomer Michaeli and Michael Elad},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=hPOt3yUXii}
}
```

To refer to the thesis itself:
```bibtex
@thesis{morgunov2026pmrf,
    author = {Daniel Morgunov},
    title  = {Photo-Realistic Image Restoration via Posterior-Mean Rectified Flow},
    type   = {Bachelor's thesis},
    school = {Johannes Kepler University Linz},
    year   = {2026}
}
```

## License and acknowledgements

This repository builds on the [official PMRF implementation](https://github.com/ohayonguy/PMRF) by Guy Ohayon. The code is MIT-licensed and re-uses components from BasicSR, SwinIR, VQFR, DifFace and k-diffusion (see the original licenses in each folder).
