"""
Disentanglement evaluation: identity/style swapping and identity interpolation.

This is the test that decides what your paper can claim. The latent is
z = [z_style | z_id]. A swap rebuilds an image from the STYLE of face A and the
IDENTITY of face B:

    z_swap = [ z_style(A) , z_id(B) ]   →   decode(z_swap)

If the split is disentangled, the result should carry B's identity and A's
attributes (pose / expression / lighting). We measure this with an INDEPENDENT
recognition network (see utils/eval_identity.py), never the training teacher.

Outputs:
  - metrics: identity-to-B (want high), identity-to-A (want low), transfer rate,
    reconstruction identity (sanity), and the different-identity floor.
  - swap_matrix.png : rows = attribute source A, cols = identity source B.
  - interpolation.png : fix style, interpolate z_id from one identity to another.

Usage:
    python evaluate_swap.py --checkpoint checkpoints/best_model.pt \
        --data-dir /data/vggface2_test_aligned --vggface2 \
        --num-pairs 1000 --grid 5 --output-dir eval_swap
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from data.dataset import FFHQDataset, VGGFace2Dataset
from models.autoencoder import FaceAutoencoder
from utils.eval_identity import IndependentVerifier


# ── dataset helpers ───────────────────────────────────────────────────────────

def get_labels(ds) -> list[int]:
    """Per-sample identity label. FFHQ has unique people → each index is its own id."""
    if isinstance(ds, VGGFace2Dataset):
        return [lbl for _, lbl in ds.samples]
    return list(range(len(ds)))  # FFHQ: every image is a distinct identity


def sample_distinct(labels: list[int], k: int, rng: torch.Generator) -> list[int]:
    """k dataset indices with pairwise-distinct identity labels."""
    order = torch.randperm(len(labels), generator=rng).tolist()
    seen, picks = set(), []
    for idx in order:
        if labels[idx] not in seen:
            seen.add(labels[idx])
            picks.append(idx)
        if len(picks) == k:
            break
    return picks


def sample_pairs(labels: list[int], n: int, rng: torch.Generator):
    """n (idxA, idxB) pairs with different identities."""
    N = len(labels)
    pairs = []
    while len(pairs) < n:
        a, b = torch.randint(0, N, (2,), generator=rng).tolist()
        if labels[a] != labels[b]:
            pairs.append((a, b))
    return pairs


def stack(ds, idxs, device):
    return torch.stack([ds[i][0] for i in idxs]).to(device)


# ── core ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def swap_decode(model: FaceAutoencoder, img_a: torch.Tensor, img_b: torch.Tensor):
    """decode([style(A), id(B)]) — style from A, identity from B."""
    z_a = model.encode(img_a)
    z_b = model.encode(img_b)
    style_a, _ = model.split(z_a)
    _, id_b    = model.split(z_b)
    return model.decode(torch.cat([style_a, id_b], dim=1))


@torch.no_grad()
def swap_metrics(model, ds, labels, verifier, num_pairs, batch_size, device, rng):
    pairs = sample_pairs(labels, num_pairs, rng)
    sums = {'id_to_B': 0.0, 'id_to_A': 0.0, 'transfer': 0.0,
            'recon_id': 0.0, 'diff_floor': 0.0}
    n = 0
    for s in tqdm(range(0, len(pairs), batch_size), desc='swap metrics'):
        chunk = pairs[s:s + batch_size]
        img_a = stack(ds, [a for a, _ in chunk], device)
        img_b = stack(ds, [b for _, b in chunk], device)

        swapped = swap_decode(model, img_a, img_b)
        recon_a = model.decode(model.encode(img_a))

        cos_sb = verifier.cosine(swapped, img_b)   # identity should come from B
        cos_sa = verifier.cosine(swapped, img_a)   # should NOT match A
        sums['id_to_B']   += cos_sb.sum().item()
        sums['id_to_A']   += cos_sa.sum().item()
        sums['transfer']  += (cos_sb > cos_sa).float().sum().item()
        sums['recon_id']  += verifier.cosine(recon_a, img_a).sum().item()
        sums['diff_floor']+= verifier.cosine(img_a, img_b).sum().item()
        n += len(chunk)
    return {k: v / n for k, v in sums.items()}, n


@torch.no_grad()
def swap_matrix_figure(model, ds, labels, k, device, rng, out_path):
    """Grid: row i = attribute source A_i, col j = identity source B_j."""
    idxs = sample_distinct(labels, k, rng)
    imgs = stack(ds, idxs, device)               # (k, 3, H, W)
    z = model.encode(imgs)
    style, ident = model.split(z)

    H = imgs.shape[-1]
    blank = torch.zeros(1, 3, H, H, device=device)
    cells = [blank, *[imgs[j:j + 1] for j in range(k)]]   # header row: identity sources
    for i in range(k):
        cells.append(imgs[i:i + 1])                       # header col: attribute source
        for j in range(k):
            z_swap = torch.cat([style[i:i + 1], ident[j:j + 1]], dim=1)
            cells.append(model.decode(z_swap))
    grid = vutils.make_grid(torch.cat(cells, 0).cpu(),
                            nrow=k + 1, normalize=True, value_range=(-1, 1), padding=2)
    vutils.save_image(grid, out_path)


@torch.no_grad()
def interpolation_figure(model, ds, labels, steps, device, rng, out_path):
    """Fix style of A, interpolate z_id from A to B → identity morph."""
    a, b = sample_distinct(labels, 2, rng)
    img = stack(ds, [a, b], device)
    z = model.encode(img)
    style, ident = model.split(z)
    rows = []
    for t in torch.linspace(0, 1, steps, device=device):
        z_id_t = (1 - t) * ident[0:1] + t * ident[1:2]
        rows.append(model.decode(torch.cat([style[0:1], z_id_t], dim=1)))
    strip = vutils.make_grid(torch.cat(rows, 0).cpu(),
                             nrow=steps, normalize=True, value_range=(-1, 1), padding=2)
    vutils.save_image(strip, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data-dir',   required=True, help='UNSEEN-identity test set')
    ap.add_argument('--vggface2',   action='store_true', help='identity sub-dir layout')
    ap.add_argument('--num-pairs',  type=int, default=1000)
    ap.add_argument('--grid',       type=int, default=5, help='swap-matrix side length')
    ap.add_argument('--interp-steps', type=int, default=7)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--output-dir', default='eval_swap')
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
    labels = get_labels(ds)
    verifier = IndependentVerifier(str(device))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics, n = swap_metrics(model, ds, labels, verifier,
                              args.num_pairs, args.batch_size, device, rng)

    print(f"\nIdentity/style swap — {n} pairs, independent verifier (InceptionResnetV1/VGGFace2)")
    print(f"  identity→B (transfer target, ↑) : {metrics['id_to_B']:.4f}")
    print(f"  identity→A (leakage, ↓)         : {metrics['id_to_A']:.4f}")
    print(f"  transfer rate (cos_B > cos_A)   : {metrics['transfer']:.4f}")
    print(f"  recon identity (sanity, ↑)      : {metrics['recon_id']:.4f}")
    print(f"  different-identity floor (ref)  : {metrics['diff_floor']:.4f}")
    print("\nRead: good disentanglement → identity→B high & near recon identity,")
    print("identity→A near the different-identity floor, transfer rate near 1.0.")

    swap_matrix_figure(model, ds, labels, args.grid, device, rng, out / 'swap_matrix.png')
    interpolation_figure(model, ds, labels, args.interp_steps, device, rng, out / 'interpolation.png')
    print(f"\nFigures → {out}/swap_matrix.png , {out}/interpolation.png")


if __name__ == '__main__':
    main()
