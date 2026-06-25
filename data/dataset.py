from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


_IMG_EXT = {'.jpg', '.jpeg', '.png', '.webp'}


def build_transforms(image_size: int = 256, augment: bool = False) -> T.Compose:
    ops = []
    if augment:
        ops += [
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        ]
    ops += [
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return T.Compose(ops)


class FFHQDataset(Dataset):
    """
    FFHQ / CelebA-HQ: flat directory of high-res face images.
    One image per person — excellent for reconstruction quality.
    Identity label is always -1 (unknown).
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = False):
        self.files = sorted(
            p for p in Path(root).rglob('*') if p.suffix.lower() in _IMG_EXT
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), -1


class VGGFace2Dataset(Dataset):
    """
    VGGFace2: root/<identity_id>/<image>.jpg
    Multiple images per identity — what ArcFace loss actually needs
    to see real pose/lighting variation per person.
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = False):
        root = Path(root)
        identity_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        if not identity_dirs:
            raise FileNotFoundError(f"No identity sub-directories found in {root}")

        self.identity_to_idx = {d.name: i for i, d in enumerate(identity_dirs)}
        self.samples = []
        for d in identity_dirs:
            label = self.identity_to_idx[d.name]
            for p in d.iterdir():
                if p.suffix.lower() in _IMG_EXT:
                    self.samples.append((p, label))

        if not self.samples:
            raise FileNotFoundError(f"No images found under identity dirs in {root}")

        self.transform = build_transforms(image_size, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

    @property
    def num_identities(self) -> int:
        return len(self.identity_to_idx)


class CombinedDataset(Dataset):
    """
    Blends FFHQ (clean, high-res) with VGGFace2 (identity diversity).
    The Reddit advice: train recon quality on FFHQ, let the identity loss
    see real variation through VGGFace2.  ffhq_ratio controls the mix.
    """

    def __init__(
        self,
        ffhq: FFHQDataset,
        vggface2: VGGFace2Dataset,
        ffhq_ratio: float = 0.3,
    ):
        self.ffhq     = ffhq
        self.vggface2 = vggface2

        n_vgg    = len(vggface2)
        n_ffhq   = int(n_vgg * ffhq_ratio / max(1.0 - ffhq_ratio, 1e-6))
        n_ffhq   = min(n_ffhq, len(ffhq))
        perm     = torch.randperm(len(ffhq))[:n_ffhq]
        self.ffhq_indices = perm.tolist()

    def __len__(self) -> int:
        return len(self.vggface2) + len(self.ffhq_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx < len(self.vggface2):
            return self.vggface2[idx]
        return self.ffhq[self.ffhq_indices[idx - len(self.vggface2)]]
