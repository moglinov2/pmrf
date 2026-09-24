# /content/drive/MyDrive/PMRF/evaluation/compute_metrics_imagenet.py
import os, argparse, torch, piq
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_fidelity import calculate_metrics
from torch_fidelity.datasets import ImagesPathDataset

torch.set_grad_enabled(False)

def compute_metrics_given_folder(rec_dir, gt_dir):
    results = {}

    # Pairwise distortion metrics (MSE/PSNR/SSIM/LPIPS)
    lpips_met = piq.LPIPS(reduction='mean').cuda()
    rec_files = sorted([os.path.join(rec_dir, f) for f in os.listdir(rec_dir)])
    gt_files  = sorted([os.path.join(gt_dir,  f) for f in os.listdir(gt_dir)])
    assert len(gt_files) == len(rec_files), f"{len(gt_files)} vs {len(rec_files)}"
    for i in range(len(gt_files)):
        assert os.path.basename(gt_files[i]) == os.path.basename(rec_files[i]), \
            f"name mismatch: {os.path.basename(gt_files[i])} vs {os.path.basename(rec_files[i])}"

    gt_ds  = ImagesPathDataset(gt_files)
    rec_ds = ImagesPathDataset(rec_files)
    gt_dl  = DataLoader(gt_ds,  batch_size=128, shuffle=False, drop_last=False, num_workers=10)
    rec_dl = DataLoader(rec_ds, batch_size=128, shuffle=False, drop_last=False, num_workers=10)

    mse = psnr = ssim = lpips = 0.0
    for gt, rec in tqdm(zip(gt_dl, rec_dl), total=len(gt_dl)):
        gt  = gt.cuda().float()
        rec = rec.cuda().float()
        mse   += ((gt - rec) ** 2).mean() * gt.shape[0]
        lpips += lpips_met(gt / 255., rec / 255.) * gt.shape[0]
        psnr  += piq.psnr(gt / 255.,  rec / 255., data_range=1., reduction='sum')
        ssim  += piq.ssim(gt / 255.,  rec / 255., data_range=1., reduction='sum')

    n = len(gt_ds)
    results.update({
        'mse':   (mse / n).item(),
        'psnr':  (psnr / n).item(),
        'ssim':  (ssim / n).item(),
        'lpips': (lpips / n).item(),
    })

    # Distribution metrics (FID/IS/KID/PRC) using GT folder as reference distribution
    fidelity_results = calculate_metrics(
        input1=gt_dir,              # <--- use GT path directly
        input2=rec_dir,             # <--- use REC path directly
        cuda=True,
        batch_size=256,
        isc=True,
        fid=True,
        kid=True,
        prc=True,
        kid_subset_size=min(1000, len(rec_files)),
        verbose=True,
        cache=True,
    )
    results.update(fidelity_results)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec_path', type=str, required=True, help='Folder with reconstructions')
    ap.add_argument('--gt_path',  type=str, required=True, help='Folder with ground-truth images (same filenames)')
    args = ap.parse_args()
    out = compute_metrics_given_folder(args.rec_path, args.gt_path)
    print(out)
