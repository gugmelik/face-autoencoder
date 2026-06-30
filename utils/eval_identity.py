"""
Independent identity verifier — EVALUATION ONLY.

Critical for an honest identity-preservation number: this must be a *different*
network than the one the model was trained against. Training uses IR-SE50
(InsightFace_Pytorch, trained on MS1M). For evaluation we use facenet-pytorch's
InceptionResnetV1 trained on VGGFace2 — a different architecture *and* a
different training set — so identity scores are not measured with the model's
own teacher (which would inflate them).

    pip install facenet-pytorch

State this cross-network protocol explicitly in the paper; it's the same
practice as Nitzan et al. 2020 (train with one net, verify with another).
"""
import torch
import torch.nn.functional as F


class IndependentVerifier:
    """InceptionResnetV1 / VGGFace2 face embedder for identity verification."""

    def __init__(self, device: str = 'cuda'):
        from facenet_pytorch import InceptionResnetV1
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.net = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised embedding (B, 512) for faces in [-1, 1]."""
        x = x.to(self.device)
        x = F.interpolate(x, (160, 160), mode='bilinear', align_corners=False)
        return F.normalize(self.net(x), dim=1)

    @torch.no_grad()
    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-sample cosine similarity (B,) between two image batches in [-1, 1]."""
        return (self.embed(a) * self.embed(b)).sum(dim=1)
