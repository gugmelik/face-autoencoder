"""
Evaluation metrics for face reconstruction quality and identity retention.

  - PSNR / SSIM : pixel-level reconstruction quality
  - Identity similarity : cosine similarity of ArcFace embeddings between
    the reconstruction and the original — the primary identity-retention signal.
    Test this on identities the model never saw during training.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] (model output) to [0, 1]."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB).  Higher is better."""
    mse = F.mse_loss(_to_01(pred), _to_01(target)).item()
    if mse == 0:
        return float('inf')
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Structural Similarity Index (0–1).  Higher is better.
    Uses pytorch_msssim; falls back to a simple luminance-based approximation
    if the package is not installed.
    """
    pred_01   = _to_01(pred)
    target_01 = _to_01(target)
    try:
        from pytorch_msssim import ssim as _ssim
        return _ssim(pred_01, target_01, data_range=1.0, size_average=True).item()
    except ImportError:
        # Lightweight fallback (less accurate but dependency-free)
        mu1     = F.avg_pool2d(pred_01,   11, stride=1, padding=5)
        mu2     = F.avg_pool2d(target_01, 11, stride=1, padding=5)
        sigma1  = F.avg_pool2d(pred_01   ** 2, 11, 1, 5) - mu1 ** 2
        sigma2  = F.avg_pool2d(target_01 ** 2, 11, 1, 5) - mu2 ** 2
        sigma12 = F.avg_pool2d(pred_01 * target_01, 11, 1, 5) - mu1 * mu2
        c1, c2  = 0.01 ** 2, 0.03 ** 2
        num     = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        den     = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)
        return (num / den).mean().item()


_face_model_cache: dict = {}


def _face_model(device: str):
    if device not in _face_model_cache:
        import os, sys
        repo = os.environ.get(
            'INSIGHTFACE_PYTORCH_PATH',
            os.path.join(os.path.dirname(__file__), '../../InsightFace_Pytorch'),
        )
        if repo not in sys.path:
            sys.path.insert(0, os.path.abspath(repo))
        weights = os.environ.get('INSIGHTFACE_WEIGHTS_PATH', '')
        if not weights:
            raise RuntimeError(
                'Set INSIGHTFACE_WEIGHTS_PATH to the IR-SE50 .pth file '
                'from https://github.com/TreB1eN/InsightFace_Pytorch'
            )
        from model import Backbone
        net = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').eval().to(device)
        net.load_state_dict(torch.load(weights, map_location='cpu'))
        for p in net.parameters():
            p.requires_grad_(False)
        _face_model_cache[device] = net
    return _face_model_cache[device]


def identity_similarity(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean cosine similarity between ArcFace embeddings (0–1).
    Higher is better; 1.0 = identical identity.
    """
    device = pred.device
    model  = _face_model(str(device))

    pred_112   = F.interpolate(pred,   size=(112, 112), mode='bilinear', align_corners=False)
    target_112 = F.interpolate(target, size=(112, 112), mode='bilinear', align_corners=False)

    with torch.no_grad():
        pred_emb   = F.normalize(model(pred_112),   dim=1)
        target_emb = F.normalize(model(target_112), dim=1)

    return (pred_emb * target_emb).sum(dim=1).mean().item()


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Convenience wrapper returning all metrics in one dict."""
    return {
        'psnr':               psnr(pred, target),
        'ssim':               ssim(pred, target),
        'identity_sim':       identity_similarity(pred, target),
    }
