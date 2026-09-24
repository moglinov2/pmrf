# 🧠 Experiment III – Trained checkpoints

This folder contains the checkpoints of all models trained from scratch in Experiment III, on ImageNet-mini and UT Zappos50K. Download them from [Google Drive](https://drive.google.com/drive/folders/19zhGx2uKuI-mrdFWtRyN5OZ2FzyCyULa?usp=sharing) and place them here.

- Folders starting with `new_` are runs of the **refined** configuration. All other folders are runs of the **initial** configuration.
- The ImageNet-mini runs are directly in this folder.
- The initial UT Zappos50K runs are in `zappos_data/` (named `zap_*`), and the refined ones start with `new_zap_`.
- Each run is named `<task>_<method>`, with method `mmse`, `pmrf`, `naive_flow`, `post_con_on_y` or `post_con_on_mmse`.
- The weights are in `<run>/checkpoints/`; the notebook loads `last.ckpt`.