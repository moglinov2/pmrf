# 🗂️ PMRF – Restored-image Outputs

This directory stores the restored images of **Experiment I** (blind face restoration) and **Experiment III** (models trained from scratch on ImageNet-mini and UT Zappos50K). The outputs of Experiment II are in [`../controlled_experiments_results/`](../controlled_experiments_results/).

```bash
outputs/
├── papers_data/celeba512_pmrf/       # Experiment I
├── papers_data/lfw_pmrf/             # Experiment I
├── papers_data/wider_pmrf/           # Experiment I
├── papers_data/webphoto_pmrf/        # Experiment I
├── papers_data/celeb_adult_pmrf/     # Experiment I
├── imagenet_data/        # Experiment III, ImageNet-mini
└── zappos_data/          # Experiment III, UT Zappos50K
```

---

## Experiment I – Blind face restoration

These outputs were produced with the released PMRF checkpoint, using `pmrf_papers_data.ipynb`. Every benchmark folder has two sub-folders:

| Sub-folder | Contents | Notes |
|------------|----------|-------|
| `restored_images/` | Final PMRF outputs $\hat{X}_0$ | Rectified-flow refinement with K = 25 steps |
| `restored_images_posterior_mean/` | Stage-1 estimates $\hat{X}^{\ast}$ (posterior mean) | Low distortion but typically over-smooth; the posterior-mean baseline in the thesis |

| Folder | Benchmark | Resolution |
|--------|-----------|------------|
| `celeba512_pmrf/`   | CelebA-Test (synthetic degradations, paired ground truth) | 512 × 512 |
| `lfw_pmrf/`         | LFW-Test (real-world)        | 512 × 512 |
| `wider_pmrf/`       | WIDER-Test (real-world)      | 512 × 512 |
| `webphoto_pmrf/`    | WebPhoto-Test (real-world)   | 512 × 512 |
| `celeb_adult_pmrf/` | CelebAdult-Test (real-world) | 512 × 512 |

---

## Experiment III – Newly trained models

These outputs were produced with `pmrf_imagenet_and_zappos_data.ipynb`, on the 5,000-image test splits at 128 × 128. All flow-based methods used K = 100 steps.

Each trained model has one folder, named `<task>_<method>_test_K100/`:
- Folders starting with `new_` belong to the **refined** configuration.
- All other folders belong to the **initial** configuration.

| Task | `imagenet_data/` initial | `imagenet_data/` refined | `zappos_data/` initial | `zappos_data/` refined |
|------|--------------------------|--------------------------|------------------------|------------------------|
| Colorization     | `colorization_*`     | `new_colorization_*`     | `colorization_*` | `new_zap_colorization_*`     |
| Denoising        | `gaussian_noise_*`   | `new_gaussian_noise_*`   | `gaussion_*`     | `new_zap_gaussian_noise_*`   |
| Super-resolution | `super_resolution_*` | `new_super_resolution_*` | `sr_*`           | `new_zap_super_resolution_*` |
| Inpainting       | `inpainting_*`       | –                        | `inpainting_*`   | –                            |

In the table, `*` stands for the method (`mmse`, `pmrf`, `naive_flow`, `post_con_on_y` or `post_con_on_mmse`), followed by `_test_K100`. The initial UT Zappos50K denoising runs are named `gaussion_*`, spelled exactly like this.

Inside each folder, the restored images are stored in:

```bash
<task>_<method>_test_K100/<degradation>/<method folder>/num_flow_steps=100/xhat/   # flow-based methods
<task>_<method>_test_K100/<degradation>/mmse/xhat/                                # posterior mean
```

`<degradation>` is the identifier of the task in that configuration, e.g. `colorization_gaussian_noise_025` (initial) or `colorization_gaussian_noise_005` (refined). The main README lists all identifiers. The method folders correspond to the thesis names as follows:

| Method folder | Method in the thesis |
|---------------|----------------------|
| `mmse/` | Posterior mean $\hat{X}^{\ast}$ |
| `pmrf/` | PMRF |
| `naive_flow/` | Naive Flow |
| `posterior_conditioned_on_y/` | Flow cond. on $Y$ |
| `posterior_conditioned_on_mmse/` | Flow cond. on $\hat{X}^{\ast}$ |