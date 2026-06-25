"""
Evaluate a trained checkpoint on an unseen identity set.

The Reddit advice: test identity retention on people the model never saw
during training.  This script runs on a separate held-out directory and
reports PSNR, SSIM, and mean ArcFace cosine similarity.

Usage:
    python evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --data-dir   /data/vggface2_test_aligned \\
        --output-dir eval_results \\
        --samples    8
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FFHQDataset, VGGFace2Dataset
from models.autoencoder import FaceAutoencoder
from utils.metrics import compute_all


def evaluate(
    checkpoint_path: str,
    data_dir:        str,
    output_dir:      str = 'eval_results',
    num_samples:     int = 8,
    use_vggface2:    bool = False,
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt['cfg']

    model = FaceAutoencoder(
        base_ch    = cfg['model']['base_channels'],
        num_res    = cfg['model']['num_residual_blocks'],
        latent_dim = cfg['model']['latent_dim'],
        style_dim  = cfg['model']['style_dim'],
        image_size = cfg['data']['image_size'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    size = cfg['data']['image_size']
    ds   = (VGGFace2Dataset(data_dir, size, augment=False) if use_vggface2
            else FFHQDataset(data_dir, size, augment=False))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    totals: dict = {}
    n_batches = 0
    saved_grid = False

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc='Evaluating'):
            imgs  = imgs.to(device)
            recon, _, _ = model(imgs)

            for k, v in compute_all(recon, imgs).items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            if not saved_grid:
                k        = min(num_samples, imgs.shape[0])
                combined = torch.cat([imgs[:k], recon[:k]], dim=0)
                grid     = vutils.make_grid(combined, nrow=k, normalize=True, value_range=(-1, 1))
                vutils.save_image(grid, out_dir / 'samples.png')
                saved_grid = True

    print(f"\nEvaluation on {len(ds)} images:")
    for k, v in totals.items():
        print(f"  {k:20s} = {v / n_batches:.4f}")
    print(f"\nSample grid saved → {out_dir}/samples.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--data-dir',    required=True)
    parser.add_argument('--output-dir',  default='eval_results')
    parser.add_argument('--samples',     type=int, default=8)
    parser.add_argument('--vggface2',    action='store_true',
                        help='Use VGGFace2Dataset (identity sub-dirs) instead of flat FFHQ layout')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir, args.samples, args.vggface2)
