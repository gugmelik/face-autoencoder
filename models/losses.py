import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class PerceptualLoss(nn.Module):
    """
    VGG19 feature matching loss at relu1_2, relu2_2, relu3_3.
    Expects input in [-1, 1]; internally renormalizes to ImageNet stats.
    """

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg)[:4]).eval()   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg)[4:9]).eval()  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg)[9:18]).eval() # relu3_3
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0  # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = self._normalize(pred)
        target = self._normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        for slc in (self.slice1, self.slice2, self.slice3):
            pred   = slc(pred)
            target = slc(target)
            loss   = loss + F.l1_loss(pred, target)
        return loss


class IdentityLoss(nn.Module):
    """
    ArcFace identity loss using InsightFace_Pytorch's IR-SE50 backbone.
    Gradients flow back to the autoencoder through this loss (pure PyTorch).

    Setup:
      1. Clone https://github.com/TreB1eN/InsightFace_Pytorch
      2. Download the IR-SE50 .pth weights from that repo's README
      3. Set INSIGHTFACE_PYTORCH_PATH env var, or clone it at ../InsightFace_Pytorch

    Input: (B, 3, H, W) RGB in [-1, 1] — matches the backbone's expected
    normalisation (ToTensor + Normalize(0.5, 0.5, 0.5)) without any extra step.
    """

    def __init__(self, weights_path: str):
        super().__init__()
        import os, sys
        repo = os.environ.get(
            'INSIGHTFACE_PYTORCH_PATH',
            os.path.join(os.path.dirname(__file__), '../../InsightFace_Pytorch'),
        )
        if repo not in sys.path:
            sys.path.insert(0, os.path.abspath(repo))

        from model import Backbone
        self.net = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').eval()
        self.net.load_state_dict(torch.load(weights_path, map_location='cpu'))
        for p in self.net.parameters():
            p.requires_grad_(False)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised ArcFace embedding (B, 512) of faces in [-1, 1]."""
        x_112 = F.interpolate(x, (112, 112), mode='bilinear', align_corners=False)
        return F.normalize(self.net(x_112), dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        e_pred   = self.embed(pred)
        e_target = self.embed(target)
        return 1.0 - (e_pred * e_target).sum(dim=1).mean()  # 0 = identical, 2 = opposite


class LatentIdentityLoss(nn.Module):
    """
    Supervises the identity half of the latent (z_id) to match the frozen
    ArcFace embedding of the input face, by cosine distance.

    This is the "trained with ArcFace loss" half of the split latent: instead
    of classifying a fixed set of training identities (which would not transfer
    to unseen people), we distil the ArcFace embedding directly into z_id.
    Since ArcFace embeddings generalise to identities never seen in training,
    z_id inherits that generalisation.

    Shares the IR-SE50 backbone with the recon-space IdentityLoss to avoid
    loading the weights twice.  Gradients flow into z_id only — the ArcFace
    target is detached.

    Note: z_id must have the same dimensionality as the ArcFace embedding (512).
    """

    def __init__(self, embedder: IdentityLoss):
        super().__init__()
        self.embedder = embedder

    def forward(self, z_id: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_emb = self.embedder.embed(target)
        z_id_n = F.normalize(z_id, dim=1)
        return 1.0 - (z_id_n * target_emb).sum(dim=1).mean()


# ── AAE adversarial losses on the style latent ────────────────────────────────
# The discriminator learns to separate a true N(0, I) sample from the encoder's
# z_style; the encoder (generator) is trained to fool it.  At equilibrium the
# aggregated posterior of z_style matches N(0, I) — see Makhzani et al., 2015
# (https://arxiv.org/pdf/1511.05644), the adversarial replacement for VAE's KL.

_bce = nn.BCEWithLogitsLoss()


def aae_discriminator_loss(disc: nn.Module, z_style: torch.Tensor) -> torch.Tensor:
    """Discriminator step: real = N(0, I), fake = z_style (pass it detached)."""
    z_real      = torch.randn_like(z_style)
    real_logits = disc(z_real)
    fake_logits = disc(z_style)
    return (_bce(real_logits, torch.ones_like(real_logits))
            + _bce(fake_logits, torch.zeros_like(fake_logits)))


def aae_generator_loss(disc: nn.Module, z_style: torch.Tensor) -> torch.Tensor:
    """Encoder step: push z_style toward the prior by fooling the discriminator."""
    logits = disc(z_style)
    return _bce(logits, torch.ones_like(logits))


class FaceLoss(nn.Module):
    """
    Combined loss: L1 + perceptual (VGG) + identity (ArcFace).
    Weights follow the Reddit advice: start with L1 dominant, perceptual
    at 0.1x, identity at 0.5x and tune from there.
    """

    def __init__(
        self,
        l1_weight:             float = 1.0,
        perceptual_weight:     float = 0.1,
        identity_weight:       float = 0.5,
        identity_weights_path: str   = '',
    ):
        super().__init__()
        self.l1_w   = l1_weight
        self.perc_w = perceptual_weight
        self.id_w   = identity_weight

        self.perceptual = PerceptualLoss()
        self.identity   = IdentityLoss(identity_weights_path)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1   = F.l1_loss(pred, target)
        perc = self.perceptual(pred, target)
        iden = self.identity(pred, target)

        total = self.l1_w * l1 + self.perc_w * perc + self.id_w * iden
        breakdown = {
            'l1':          l1.item(),
            'perceptual':  perc.item(),
            'identity':    iden.item(),
            'total':       total.item(),
        }
        return total, breakdown
