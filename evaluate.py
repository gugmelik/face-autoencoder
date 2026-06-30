"""
Evaluate a trained checkpoint on an unseen-identity set.

Reports reconstruction quality (PSNR, SSIM, LPIPS, FID) and identity retention.
Identity is reported two ways:
  - identity_sim       : the training teacher (IR-SE50) — for reference only
  - identity_sim_indep : an INDEPENDENT verifier (InsightFace buffalo_l) — the
    number to put in the paper, since it isn't the network you trained against.

Usage:
    python evaluate.py \\
        --checkpoint checkpoints/best_model.pt \\
        --data-dir   /data/ffhq_test \\
        --output-dir eval_results \\
        --samples    8 [--vggface2] [--wandb]
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


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → uint8 [0,255] (B,3,H,W) for FID."""
    return ((x.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)


def evaluate(
    checkpoint_path: str,
    data_dir:        str,
    output_dir:      str = 'eval_results',
    num_samples:     int = 8,
    use_vggface2:    bool = False,
    use_wandb:       bool = False,
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

    # Independent identity verifier (different net than the training teacher).
    try:
        from utils.eval_identity import IndependentVerifier
        verifier = IndependentVerifier(str(device))
    except Exception as e:
        print(f"WARNING: independent verifier unavailable ({e}); skipping identity_sim_indep.")
        verifier = None

    # FID accumulator (real = originals, fake = reconstructions).
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    except Exception as e:
        print(f"WARNING: FID unavailable ({e}); skipping FID.  pip install torchmetrics torch-fidelity")
        fid = None

    totals: dict = {}
    n_batches = 0
    id_indep_sum, id_indep_n = 0.0, 0
    saved_grid = False

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc='Evaluating'):
            imgs  = imgs.to(device)
            recon, _, _ = model(imgs)

            for k, v in compute_all(recon, imgs).items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

            if verifier is not None:
                cos = verifier.cosine(recon, imgs)
                id_indep_sum += cos.sum().item()
                id_indep_n   += cos.shape[0]

            if fid is not None:
                fid.update(_to_uint8(imgs),  real=True)
                fid.update(_to_uint8(recon), real=False)

            if not saved_grid:
                k        = min(num_samples, imgs.shape[0])
                combined = torch.cat([imgs[:k], recon[:k]], dim=0)
                grid     = vutils.make_grid(combined, nrow=k, normalize=True, value_range=(-1, 1))
                vutils.save_image(grid, out_dir / 'samples.png')
                saved_grid = True

    results = {k: v / n_batches for k, v in totals.items()}
    if id_indep_n:
        results['identity_sim_indep'] = id_indep_sum / id_indep_n
    if fid is not None:
        results['fid'] = fid.compute().item()

    print(f"\nEvaluation on {len(ds)} images:")
    for k, v in results.items():
        print(f"  {k:20s} = {v:.4f}")
    print(f"\nSample grid saved → {out_dir}/samples.png")

    if use_wandb:
        import wandb
        run = wandb.init(project=cfg.get('wandb', {}).get('project', 'face-autoencoder'),
                         job_type='eval', config={'checkpoint': checkpoint_path, 'data_dir': data_dir})
        run.log({**{f"eval/{k}": v for k, v in results.items()},
                 'eval/samples': wandb.Image(str(out_dir / 'samples.png'))})
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--data-dir',    required=True)
    parser.add_argument('--output-dir',  default='eval_results')
    parser.add_argument('--samples',     type=int, default=8)
    parser.add_argument('--vggface2',    action='store_true',
                        help='Use VGGFace2Dataset (identity sub-dirs) instead of flat FFHQ layout')
    parser.add_argument('--wandb',       action='store_true', help='Log results to wandb')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.output_dir,
             args.samples, args.vggface2, args.wandb)
