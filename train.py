"""
Training script for the face autoencoder (adversarial autoencoder / AAE).

The latent is split into a style half (regularised toward N(0, I) by an
adversarial discriminator) and an identity half (distilled from frozen ArcFace
embeddings).  Each step runs two phases per the AAE recipe:
  1. reconstruction + generator phase  → updates encoder + decoder
  2. regularisation phase              → updates the discriminator

Usage:
    python train.py                          # uses configs/default.yaml
    python train.py --config configs/my.yaml
    python train.py --resume checkpoints/checkpoint_epoch10.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.utils as vutils
import yaml
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.dataset import (
    CombinedDataset,
    FFHQDataset,
    VGGFace2Dataset,
    build_transforms,
)
from models.autoencoder import AAEDiscriminator, FaceAutoencoder
from models.losses import (
    FaceLoss,
    LatentIdentityLoss,
    aae_discriminator_loss,
    aae_generator_loss,
)
from utils.metrics import compute_all


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, augment: bool = False):
    ffhq_root = cfg['data'].get('ffhq_root')
    vgg_root  = cfg['data'].get('vggface2_root')
    size      = cfg['data']['image_size']

    if not ffhq_root and not vgg_root:
        sys.exit("ERROR: set at least one of data.ffhq_root / data.vggface2_root in config.")

    if ffhq_root and vgg_root:
        ffhq_ds = FFHQDataset(ffhq_root, size, augment)
        vgg_ds  = VGGFace2Dataset(vgg_root, size, augment)
        return CombinedDataset(ffhq_ds, vgg_ds, cfg['data'].get('ffhq_ratio', 0.3))

    if vgg_root:
        return VGGFace2Dataset(vgg_root, size, augment)
    return FFHQDataset(ffhq_root, size, augment)


def log_losses(prefix: str, losses: dict, n: int) -> None:
    parts = '  |  '.join(f"{k}={v/n:.4f}" for k, v in losses.items())
    print(f"  {prefix}:  {parts}")


_IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp'}


def load_holdout(cfg: dict, device, max_n: int = 8):
    """
    Load a small, fixed batch of *unseen-identity* faces for qualitative
    validation plots.  Point cfg.data.holdout_dir at a folder containing a few
    images of ~2 people that are NOT in your training roots.  Returns a tensor
    (N, 3, H, W) in [-1, 1], or None if holdout_dir is unset/empty.
    """
    holdout_dir = cfg['data'].get('holdout_dir')
    if not holdout_dir:
        return None
    paths = sorted(p for p in Path(holdout_dir).rglob('*') if p.suffix.lower() in _IMG_EXT)[:max_n]
    if not paths:
        print(f"WARNING: holdout_dir '{holdout_dir}' has no images — skipping unseen-identity plots.")
        return None
    tf = build_transforms(cfg['data']['image_size'], augment=False)
    imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths])
    print(f"Loaded {len(paths)} unseen-identity holdout images for validation plots.")
    return imgs.to(device)


def recon_grid(originals: torch.Tensor, recons: torch.Tensor):
    """Grid with originals on the top row and reconstructions directly below."""
    n = originals.shape[0]
    combined = torch.cat([originals, recons], dim=0)
    return vutils.make_grid(combined, nrow=n, normalize=True, value_range=(-1, 1))


def init_wandb(cfg: dict, resume: str | None):
    """Return a wandb run if cfg.wandb.enabled, else None (training works either way)."""
    wcfg = cfg.get('wandb', {})
    if not wcfg.get('enabled', False):
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb requested but not installed (pip install wandb) — skipping logging.")
        return None
    return wandb.init(
        project = wcfg.get('project', 'face-autoencoder'),
        entity  = wcfg.get('entity'),
        name    = wcfg.get('run_name'),
        mode    = wcfg.get('mode', 'online'),
        config  = cfg,
        resume  = 'allow' if resume else None,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def train(cfg_path: str, resume: str | None = None) -> None:
    cfg    = load_config(cfg_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    run = init_wandb(cfg, resume)

    # Model + adversarial discriminator over the style latent
    model = FaceAutoencoder(
        base_ch    = cfg['model']['base_channels'],
        num_res    = cfg['model']['num_residual_blocks'],
        latent_dim = cfg['model']['latent_dim'],
        style_dim  = cfg['model']['style_dim'],
        image_size = cfg['data']['image_size'],
    ).to(device)
    disc = AAEDiscriminator(style_dim=cfg['model']['style_dim']).to(device)

    # Reconstruction-space loss (L1 + perceptual + ArcFace) and the latent-space
    # identity distillation, which reuses the same frozen ArcFace backbone.
    criterion   = FaceLoss(
        l1_weight             = cfg['loss']['l1_weight'],
        perceptual_weight     = cfg['loss']['perceptual_weight'],
        identity_weight       = cfg['loss']['identity_weight'],
        identity_weights_path = cfg['loss']['identity_weights_path'],
    ).to(device)
    latent_id   = LatentIdentityLoss(criterion.identity).to(device)
    lat_id_w    = cfg['loss']['latent_identity_weight']
    adv_w       = cfg['loss']['adversarial_weight']

    # Two optimisers: one for the autoencoder, one for the discriminator.
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=1e-4)
    d_optimizer = optim.Adam(disc.parameters(),
                             lr=cfg['training'].get('discriminator_lr', 2e-4),
                             betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['num_epochs'])

    start_epoch   = 0
    best_val_loss = float('inf')
    global_step   = 0

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        disc.load_state_dict(ckpt['disc'])
        optimizer.load_state_dict(ckpt['optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        global_step   = ckpt.get('step', 0)
        print(f"Resumed from epoch {ckpt['epoch']} / iter {global_step} (val_loss={best_val_loss:.4f})")

    # Data
    full_ds  = build_dataset(cfg, augment=True)
    val_n    = max(1, int(len(full_ds) * 0.05))
    train_n  = len(full_ds) - val_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))

    loader_kw = dict(num_workers=cfg['training'].get('num_workers', 4), pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'],
                              shuffle=False, **loader_kw)

    out_dir = Path(cfg['training']['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed batch of unseen identities for qualitative validation plots.
    holdout = load_holdout(cfg, device)

    num_epochs  = cfg['training']['num_epochs']
    save_every  = cfg['training'].get('save_every', 10)    # checkpoint every N epochs
    val_every   = cfg['training'].get('val_every', 200)    # validate every N iterations
    log_every   = cfg['training'].get('log_every', 20)     # log train loss every N iterations

    def save_ckpt(path: Path, epoch: int, step: int) -> None:
        torch.save({
            'epoch':         epoch,
            'step':          step,
            'model':         model.state_dict(),
            'disc':          disc.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'd_optimizer':   d_optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'cfg':           cfg,
        }, path)

    def do_validation(step: int) -> float:
        """Full validation pass; print + log to wandb at `step`; return mean val total loss."""
        model.eval()
        disc.eval()
        v_losses: dict  = {}
        v_metrics: dict = {}
        val_vis = None
        with torch.no_grad():
            for imgs, _ in tqdm(val_loader, desc=f"val @ iter {step}", leave=False):
                imgs = imgs.to(device)
                recon, _, (z_style, z_id) = model(imgs)
                _, breakdown = criterion(recon, imgs)
                breakdown['latent_id'] = latent_id(z_id, imgs).item()
                metrics = compute_all(recon, imgs)
                for k, v in breakdown.items():
                    v_losses[k]  = v_losses.get(k, 0.0) + v
                for k, v in metrics.items():
                    v_metrics[k] = v_metrics.get(k, 0.0) + v
                if val_vis is None:  # keep first batch for the recon grid
                    k = min(8, imgs.shape[0])
                    val_vis = (imgs[:k].cpu(), recon[:k].cpu())

        n_va = max(len(val_loader), 1)
        print(f"\n[iter {step}]")
        log_losses("val ", v_losses, n_va)
        log_losses("metr", v_metrics, n_va)

        if run is not None:
            import wandb
            payload = {f"val/{k}": v / n_va for k, v in v_losses.items()}
            payload.update({f"metric/{k}": v / n_va for k, v in v_metrics.items()})
            if val_vis is not None:
                payload['val/reconstructions'] = wandb.Image(
                    recon_grid(*val_vis), caption="top: original — bottom: reconstruction")
            if holdout is not None:
                with torch.no_grad():
                    recon_h, _, _ = model(holdout)
                payload['holdout/unseen_identities'] = wandb.Image(
                    recon_grid(holdout.cpu(), recon_h.cpu()),
                    caption="unseen identities — top: original, bottom: reconstruction")
            run.log(payload, step=step)

        model.train()
        disc.train()
        return v_losses.get('total', 0.0) / n_va

    model.train()
    disc.train()
    for epoch in range(start_epoch, num_epochs):
        t_losses: dict = {}
        for imgs, _ in tqdm(train_loader, desc=f"[{epoch+1}/{num_epochs}] train", leave=False):
            imgs = imgs.to(device)

            # ── phase 1: reconstruction + generator (autoencoder) ──
            recon, _, (z_style, z_id) = model(imgs)
            rec_total, breakdown = criterion(recon, imgs)
            lat_id = latent_id(z_id, imgs)
            g_loss = aae_generator_loss(disc, z_style)
            ae_loss = rec_total + lat_id_w * lat_id + adv_w * g_loss

            optimizer.zero_grad(set_to_none=True)
            ae_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # ── phase 2: regularisation (discriminator) ──
            d_loss = aae_discriminator_loss(disc, z_style.detach())
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            global_step += 1

            breakdown['latent_id'] = lat_id.item()
            breakdown['adv_g']     = g_loss.item()
            breakdown['adv_d']     = d_loss.item()
            breakdown['ae_loss']   = ae_loss.item()
            for k, v in breakdown.items():
                t_losses[k] = t_losses.get(k, 0.0) + v

            # ── per-iteration train loss curves ──
            if run is not None and global_step % log_every == 0:
                run.log({**{f"train/{k}": v for k, v in breakdown.items()},
                         'lr': scheduler.get_last_lr()[0]}, step=global_step)

            # ── iteration-based validation ──
            if global_step % val_every == 0:
                val_total = do_validation(global_step)
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    save_ckpt(out_dir / 'best_model.pt', epoch, global_step)
                    print(f"  ✓ best model saved (val_total={best_val_loss:.4f})")

        # ── epoch console summary ──
        print(f"\nEpoch {epoch+1}/{num_epochs} done  (iter {global_step})")
        log_losses("train(avg)", t_losses, len(train_loader))

        # ── periodic checkpoint (epoch-based) ──
        if (epoch + 1) % save_every == 0:
            save_ckpt(out_dir / f'checkpoint_epoch{epoch+1:04d}.pt', epoch, global_step)

        scheduler.step()

    if run is not None:
        run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(args.config, args.resume)
