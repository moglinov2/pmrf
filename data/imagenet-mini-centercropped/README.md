## 📂 Dataset directory structure

Place the preprocessed **ImageNet-mini** splits inside **this folder** before running any Experiment III scripts:

| Folder   | Resolution | Images | Notes |
|----------|------------|--------|-------|
| `train/` | 128 × 128  | 34,745 | Training split |
| `val/`   | 128 × 128  | 3,923  | Validation during training |
| `test/`  | 128 × 128  | 5,000  | Inference and evaluation; also the clean reference set for FID, KID, precision and recall |

**Source:** [ImageNet 1000 (mini) on Kaggle](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).

**Preprocessing:** the cell *Resizing and Cropping Data* in `pmrf_imagenet_and_zappos_data.ipynb` rescales the shorter side to 128 px (Lanczos) and center-crops every image to 128 × 128. The cell works in place, so run it on a copy of the downloaded data.

If you use ImageNet, please cite Deng et al. (2009) and Russakovsky et al. (2015).