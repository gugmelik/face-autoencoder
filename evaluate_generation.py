"""
Identity-conditioned generation evaluation — the core result for the
"one reference image → identity-preserving generation" use case.

Given a reference face, we take its identity code z_id (= its ArcFace embedding,
by construction) and sample the style code z_style ~ N(0, I) to synthesise new
images of the *same* person under varied pose/lighting/expression:

    generate = decode([ z_style ~ N(0, I) , z_id(reference) ])

We report:
  - identity_preservation : mean cosine(generation, reference), INDEPENDENT
    verifier (InsightFace buffalo_l). Higher = identity kept across samples.
  - identity_consistency  : per-reference std of that cosine (lower = stable).
  - diversity_lpips        : mean pairwise LPIPS among a reference's samples
    (higher = more varied generations; needs `lpips`).
  - prior match           : per-dim mean/std of z_style over real faces
    (targets 0 and 1) — does the AAE actually match N(0, I)?
  - generation_fid (opt.) : FID of prior-sampled generations vs real faces.

Usage:
    python evaluate_generation.py --checkpoint checkpoints/best_model.pt \
        --data-dir ./datasets/ffhq_test --samples-per-ref 8 --num-refs 100 \
        --grid-refs 6 --output-dir eval_gen [--vggface2] [--fid] [--wandb]
"""
import argparse
import itertools
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FFHQDataset, VGGFace2Dataset
from models.autoencoder import FaceAutoencoder
from utils.metrics import lpips_distance


def _to_uint8(x):
    return ((x.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)


@torch.no_grad()
def prior_diagnostic(model, loader, device, out_path, max_batches=20):
    """Per-dim mean/std of z_style over real faces; save a histogram vs N(0,1)."""
    styles = []
    for i, (imgs, _) in enumerate(tqdm(loader, desc='prior stats', total=min(max_batches, len(loader)))):
        if i >= max_batches:
            break
        z = model.encode(imgs.to(device))
        styles.append(model.split(z)[0].cpu())
    z_style = torch.cat(styles, 0)                       # (N, style_dim)
    per_dim_mean = z_style.mean(0)
    per_dim_std  = z_style.std(0)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(z_style.flatten().numpy(), bins=120, density=True, alpha=0.7, label='z_style')
        xs = np.linspace(-4, 4, 200)
        ax[0].plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), 'r', label='N(0,1)')
        ax[0].set_title('z_style marginal vs N(0,1)'); ax[0].legend()
        ax[1].hist(per_dim_mean.numpy(), bins=40, alpha=0.7, label='per-dim mean (→0)')
        ax[1].hist(per_dim_std.numpy(),  bins=40, alpha=0.7, label='per-dim std (→1)')
        ax[1].set_title('per-dimension statistics'); ax[1].legend()
        fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    except Exception as e:
        print(f"  (histogram skipped: {e})")

    return {
        'style_mean_abs': per_dim_mean.abs().mean().item(),
        'style_std':      per_dim_std.mean().item(),
    }


@torch.no_grad()
def generation_eval(model, ds, verifier, num_refs, k, device, rng):
    """Sample k styles per reference; measure identity preservation/consistency/diversity."""
    ref_idx = torch.randperm(len(ds), generator=rng)[:num_refs].tolist()
    id_means, id_stds, div = [], [], []

    for idx in tqdm(ref_idx, desc='generation eval'):
        ref = ds[idx][0].unsqueeze(0).to(device)
        z_id = model.split(model.encode(ref))[1]
        z_style = torch.randn(k, model.style_dim, device=device,
                              generator=torch.Generator(device=device).manual_seed(idx))
        gen = model.decode(torch.cat([z_style, z_id.repeat(k, 1)], dim=1))

        cos = verifier.cosine(gen, ref.repeat(k, 1, 1, 1))   # (k,)
        id_means.append(cos.mean().item())
        id_stds.append(cos.std().item())

        # diversity: mean pairwise LPIPS among this reference's k generations
        pairs = list(itertools.combinations(range(k), 2))
        if pairs:
            a = torch.cat([gen[[i]] for i, _ in pairs], 0)
            b = torch.cat([gen[[j]] for _, j in pairs], 0)
            dl = lpips_distance(a, b)
            if dl is not None:
                div.append(dl)

    out = {
        'identity_preservation': sum(id_means) / len(id_means),
        'identity_consistency':  sum(id_stds)  / len(id_stds),
    }
    if div:
        out['diversity_lpips'] = sum(div) / len(div)
    return out


@torch.no_grad()
def generation_figure(model, ds, num_refs, k, device, rng, out_path):
    ref_idx = torch.randperm(len(ds), generator=rng)[:num_refs].tolist()
    rows = []
    for idx in ref_idx:
        ref = ds[idx][0].unsqueeze(0).to(device)
        z_id = model.split(model.encode(ref))[1]
        z_style = torch.randn(k, model.style_dim, device=device)
        gen = model.decode(torch.cat([z_style, z_id.repeat(k, 1)], dim=1))
        rows.append(torch.cat([ref, gen], 0))            # reference + k samples
    grid = vutils.make_grid(torch.cat(rows, 0).cpu(), nrow=k + 1,
                            normalize=True, value_range=(-1, 1), padding=2)
    vutils.save_image(grid, out_path)


@torch.no_grad()
def generation_fid(model, ds, device, n_images, batch, rng):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    # real
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    seen = 0
    for imgs, _ in tqdm(loader, desc='fid: real'):
        fid.update(_to_uint8(imgs.to(device)), real=True)
        seen += imgs.shape[0]
        if seen >= n_images:
            break
    # fake: random reference id + prior style
    done = 0
    while done < n_images:
        b = min(batch, n_images - done)
        idx = torch.randint(0, len(ds), (b,), generator=rng).tolist()
        refs = torch.stack([ds[i][0] for i in idx]).to(device)
        z_id = model.split(model.encode(refs))[1]
        z_style = torch.randn(b, model.style_dim, device=device)
        gen = model.decode(torch.cat([z_style, z_id], dim=1))
        fid.update(_to_uint8(gen), real=False)
        done += b
    return fid.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-dir',   required=True, help='UNSEEN-identity references')
    ap.add_argument('--vggface2',   action='store_true')
    ap.add_argument('--samples-per-ref', type=int, default=8)
    ap.add_argument('--num-refs',   type=int, default=100)
    ap.add_argument('--grid-refs',  type=int, default=6)
    ap.add_argument('--fid',        action='store_true')
    ap.add_argument('--fid-images', type=int, default=5000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--output-dir', default='eval_gen')
    ap.add_argument('--wandb',      action='store_true')
    ap.add_argument('--seed',       type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = torch.Generator().manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device)
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
    ds = (VGGFace2Dataset(args.data_dir, size, augment=False) if args.vggface2
          else FFHQDataset(args.data_dir, size, augment=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    from utils.eval_identity import IndependentVerifier
    verifier = IndependentVerifier(str(device))

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    results = {}
    results.update(prior_diagnostic(model, loader, device, out / 'prior_hist.png'))
    results.update(generation_eval(model, ds, verifier, args.num_refs,
                                   args.samples_per_ref, device, rng))
    if args.fid:
        results['generation_fid'] = generation_fid(model, ds, device,
                                                    args.fid_images, args.batch_size, rng)
    generation_figure(model, ds, args.grid_refs, args.samples_per_ref, device, rng,
                      out / 'generations.png')

    print("\nIdentity-conditioned generation (independent verifier):")
    for k, v in results.items():
        print(f"  {k:24s} = {v:.4f}")
    print(f"\nFigures → {out}/generations.png , {out}/prior_hist.png")
    print("Read: identity_preservation high (identity kept), prior std ≈ 1 &")
    print("mean_abs ≈ 0 (AAE matched the prior), diversity_lpips clearly > 0 (varied).")

    if args.wandb:
        import wandb
        run = wandb.init(project=cfg.get('wandb', {}).get('project', 'face-autoencoder'),
                         job_type='eval-generation',
                         config={'checkpoint': args.checkpoint, 'data_dir': args.data_dir})
        run.log({**{f"gen/{k}": v for k, v in results.items()},
                 'gen/generations': wandb.Image(str(out / 'generations.png')),
                 'gen/prior_hist':  wandb.Image(str(out / 'prior_hist.png'))})
        run.finish()


if __name__ == '__main__':
    main()
