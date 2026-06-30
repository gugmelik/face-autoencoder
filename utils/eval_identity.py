"""
Independent identity verifier — EVALUATION ONLY.

Critical for an honest identity-preservation number: this must be a *different*
network than the one the model was trained against. Training uses IR-SE50
(InsightFace_Pytorch, trained on MS1M). For evaluation we use InsightFace's
**buffalo_l / w600k_r50** recognition model — a different ArcFace backbone
trained on a different dataset (WebFace600K) via ONNX — so identity scores are
not measured with the model's own teacher (which would inflate them).

    pip install insightface onnxruntime         # or onnxruntime-gpu for CUDA

The buffalo_l pack auto-downloads to ~/.insightface/models on first use. State
this cross-network protocol explicitly in the paper.
"""
import numpy as np
import torch
import torch.nn.functional as F


class IndependentVerifier:
    """InsightFace buffalo_l (w600k_r50, ONNX) face embedder for verification."""

    def __init__(self, device: str = 'cuda'):
        from insightface.app import FaceAnalysis
        use_cuda = device.startswith('cuda') and torch.cuda.is_available()
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if use_cuda else ['CPUExecutionProvider'])
        app = FaceAnalysis(name='buffalo_l',
                           allowed_modules=['recognition'],
                           providers=providers)
        app.prepare(ctx_id=0 if use_cuda else -1)
        self.rec = app.models['recognition']

    def _to_bgr_uint8(self, x: torch.Tensor) -> np.ndarray:
        """(B,3,H,W) RGB in [-1,1] → list-ready (B,H,W,3) BGR uint8 for get_feat."""
        arr = ((x.detach().clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().byte()
        arr = arr.permute(0, 2, 3, 1).cpu().numpy()        # RGB HWC
        return np.ascontiguousarray(arr[..., ::-1])         # → BGR (get_feat swaps back)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalised embedding (B, 512) for faces in [-1, 1]."""
        feats = self.rec.get_feat(list(self._to_bgr_uint8(x)))   # (B, 512), unnormalised
        return F.normalize(torch.from_numpy(np.asarray(feats)).float(), dim=1)

    @torch.no_grad()
    def cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-sample cosine similarity (B,) between two image batches in [-1, 1]."""
        return (self.embed(a) * self.embed(b)).sum(dim=1)
