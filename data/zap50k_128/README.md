## 📂 Dataset directory structure

Place the preprocessed **UT Zappos50K** splits inside **this folder** before running any Experiment III scripts:

| Folder   | Resolution | Images | Notes |
|----------|------------|--------|-------|
| `train/` | 128 × 128  | 40,066 | Training split |
| `val/`   | 128 × 128  | 5,000  | Validation during training |
| `test/`  | 128 × 128  | 5,000  | Inference and evaluation; also the clean reference set for FID, KID, precision and recall |

**Source:** [Large Shoe Dataset (UT Zappos50k) on Kaggle](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k).

**Preprocessing:** the cell *Resizing and Cropping Data* in `pmrf_imagenet_and_zappos_data.ipynb` rescales the shorter side to 128 px (Lanczos) and center-crops every image to 128 × 128. The cell works in place, so run it on a copy of the downloaded data.

UT Zappos50K is for academic, non-commercial use only. If you use it, please cite Yu & Grauman (2014, 2017).