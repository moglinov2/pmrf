<div align="center">

# Posterior-Mean Rectified Flow (PMRF)  
Reproducing the results from the ICLR 2025 paper  
*“Posterior-Mean Rectified Flow: Towards Minimum-MSE Photo-Realistic Image Restoration”*

[[Paper]](https://arxiv.org/abs/2410.00418) • [[Project Page]](https://pmrf-ml.github.io/) • [[Demo]](https://huggingface.co/spaces/ohayonguy/PMRF)

</div>

---

## ✨ What’s in this repo?

This fork **only contains the code, checkpoints and evaluation scripts** needed to reproduce the two experimental sections I covered in my Practical Work project:

| Section in Paper | What we reproduced | Where to look |
|------------------|--------------------|---------------|
| § 5.1 (from the paper) Blind face image restoration | Inference + full metric suite on **CelebA-Test, LFW-Test, WIDER-Test, WebPhoto-Test, CelebAdult-Test, FFHQ-512** | `inference.py` · `evaluation/compute_metrics_blind.py` |
| § 5.2 (from the paper) Controlled experiments | Colorization, Denoising, Inpainting, 8× Super-resolution at K = {5, 10, 20, 100} flow steps | `scripts/test.sh` · `evaluation/compute_metrics_controlled_experiments.py` |

---

## ⚙️ Quick Colab install

I worked inside Google Colab and that is how I installed all necessary dependencies.

```bash
!pip install -q condacolab
import condacolab, os
condacolab.install()        # Colab will restart itself once this finishes

!conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
!conda install -y lightning==2.3.3 -c conda-forge

!pip install opencv-python==4.10.0.84 timm==1.0.8 wandb==0.17.5 lovely-tensors==0.1.16 torch-fidelity==0.3.0 einops==0.8.0 dctorch==0.1.2 torch-ema==0.3
!pip install natten==0.17.1+torch230cu118 -f https://shi-labs.com/natten/wheels
!pip install nvidia-cuda-nvcc-cu11
!pip install basicsr==1.4.2
!pip install git+https://github.com/toshas/torch-fidelity.git
!pip install lpips==0.1.4
!pip install piq==0.8.0
!pip install huggingface_hub==0.24.5
```

### Patching files
```bash
# Find Basicsr’s install location → patch the bad import
FILE=$(python - <<'PY'
import importlib.metadata, pathlib, sys
dist = importlib.metadata.distribution("basicsr")
print(pathlib.Path(dist.locate_file("basicsr/data/degradations.py")))
PY
)
echo "Patching $FILE ..."
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$FILE"
# show the patched line for confirmation
grep -n "rgb_to_grayscale" -A0 -B1 "$FILE" | head
```

## Copying repository into Google Drive
In Google Colab, the runtime resets every time you restart the kernel, which means you normally have to re-clone or reload your repository on each session. To avoid that extra step, I synced my entire repo to Google Drive—and now I can pull in my code with a single line:
```
from google.colab import drive
drive.mount("/content/drive")
```

##
| What                                                                        | Where to download                                                                                                          | Where to place                         |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **Pre-trained PMRF** for blind face restoration                             | [https://drive.google.com/drive/folders/1dfjZATcQ451uhvFH42tKnfMNHRkL6N_A]                                                 | any path, or leave the default         |
| **Controlled-experiment checkpoints** (PMRF + 4 baselines × 4 degradations) | [https://drive.google.com/drive/folders/1dfjZATcQ451uhvFH42tKnfMNHRkL6N_A]                                                 | `checkpoints/controlled_experiments/…` |
| **Evaluation metric models** (`resnet18_110.pth`, `alignment_WFLW_4HG.pth`) | [https://drive.google.com/drive/folders/1k3RCSliF6PsujCMIdCD1hNM63EozlDIZ]                                                 | `evaluation/metrics_ckpt/`             |
| **All test data sets** (`data/`)                                            | [https://drive.google.com/drive/folders/10ivacNpoFqq3K9xPeF1IHmrKVU7ZNRIG?usp=sharing]                                     | keep structure shown below             |

```bash
data/
├── CelebA-Test/                # 512×512 HQ faces  (3 000)
├── CelebA-Test-LQ/             # same images, degraded (3 000)
├── LFW-Test/                   # 512×512 (1 711)
├── WIDER-Test/                 # 512×512 (970)
├── WebPhoto-Test/              # 512×512 (407)
├── CelebChild-Test/            # inludes Adult and Child images
  ├── Child                     # 512×512 (180)
  └── Adult/                    # 512×512 (180)
└── cropped_faces/              # includes FFHQ images
  ├── ffhq512/                  # 512×512 (10 000)
  └── ffhq256/                  # 256×256 down-sampled for controlled experiments (10 000)
```

## How to run
### 1. Blind face restoration (CelebA-Test example)
Inference:
```bash
python inference.py \
  --ckpt_path ./checkpoints/blind_face_restoration_pmrf.ckpt \
  --lq_data_path ./data/celeba_512_validation_lq \
  --output_dir ./outputs/celeba512_pmrf \
  --batch_size 64 \
  --num_flow_steps 25
```
Change the paths if you want to run it on different data.

Evaluation:
```bash
python compute_metrics_blind.py \
  --parent_ffhq_512_path ../data/cropped_faces \
  --rec_path              ../outputs/celeba512_pmrf/restored_images \
  --gt_path               ../data/celeba_512_validation
```

### 2. Controlled experiments
Inference:
```bash
bash test.sh
```
Adjust --test_data_root in test.sh to the path of the CelebA-Test 256x256 images, and adjust --degradation and --ckpt_path to the type of degradation you wish to assess and the corresponding model checkpoint.

Evaluation:
```bash
python compute_metrics_controlled_experiments.py \
  --parent_ffhq_256_path ../data/cropped_faces \
  --rec_path              ../controlled_experiments_results/num_flow_steps_5/colorization_gaussian_noise_025/mmse/xhat \
  --gt_path               ../data/celeba_256_test
```
Adjust --rec_path if you want to run it on different algorithms. *Repeat for other K values or degradation folders as needed.*

## Citation
Paper citation:
```
@inproceedings{
    ohayon2025posteriormean,
    title={Posterior-Mean Rectified Flow: Towards Minimum {MSE} Photo-Realistic Image Restoration},
    author={Guy Ohayon and Tomer Michaeli and Michael Elad},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=hPOt3yUXii}
}
```

Code is MIT-licensed and re-uses components from BasicSR, SwinIR, VQFR, DifFace and k-diffusion (see original licenses in each folder).
