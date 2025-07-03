<div align="center">

# Posterior-Mean Rectified Flow (PMRF)  
Reproducing the ICLR 2025 paper  
*“Posterior-Mean Rectified Flow: Towards Minimum-MSE Photo-Realistic Image Restoration”*

[[Paper]](https://arxiv.org/abs/2410.00418) • [[Project Page]](https://pmrf-ml.github.io/) • [[Demo]](https://huggingface.co/spaces/ohayonguy/PMRF)

</div>

---

## ✨ What’s in this repo?

This fork **only contains the code, checkpoints and evaluation scripts** needed to reproduce the two experimental sections we covered in our Bachelor project:

| Section in Paper | What we reproduced | Where to look |
|------------------|--------------------|---------------|
| § 5.1 Blind face image restoration | Inference + full metric suite on **CelebA-Test, LFW-Test, WIDER-Test, WebPhoto-Test, CelebAdult-Test, FFHQ-512** | `inference.py` · `evaluation/compute_metrics_blind.py` |
| § 5.2 Controlled experiments | Colorization, Denoising, Inpainting, 8× Super-resolution at K = {5, 10, 20, 100} flow steps | `scripts/test.sh` · `evaluation/compute_metrics_controlled_experiments.py` |

*All numbers, figures and tables cited in our report were obtained with these exact scripts and checkpoints.*

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
Google Colab has the problem that one has to load your repository everytime one restarts the kernel. That is why i copied my whole repository into Google Drive and with the following simple line I could load my code very easily:
```
from google.colab import drive
drive.mount("/content/drive")
```
