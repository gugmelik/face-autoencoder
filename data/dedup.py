"""
Embedding-based deduplication.

The Reddit advice: de-dup by embedding, not filename.  Duplicates sneak
across train/test splits when you rely on filenames, and they also quietly
inflate your ArcFace identity loss if the same person appears in two
different identity folders.

Two use cases:
  1. find_cross_split_leakage — given a train dir and a test dir, report
     any test images whose embedding is too close to a training image.
  2. find_within_dir_dupes — report near-duplicate pairs inside one dir
     (catches re-named or re-encoded copies of the same image, and the
     "one folder, two people" label-noise problem).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


_TRANSFORM = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def _load_model(device: str, weights_path: str = ''):
    import os, sys
    repo = os.environ.get(
        'INSIGHTFACE_PYTORCH_PATH',
        os.path.join(os.path.dirname(__file__), '../../InsightFace_Pytorch'),
    )
    if repo not in sys.path:
        sys.path.insert(0, os.path.abspath(repo))
    path = weights_path or os.environ.get('INSIGHTFACE_WEIGHTS_PATH', '')
    if not path:
        raise RuntimeError(
            'Set INSIGHTFACE_WEIGHTS_PATH to the IR-SE50 .pth file '
            'from https://github.com/TreB1eN/InsightFace_Pytorch'
        )
    from model import Backbone
    net = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').eval().to(device)
    net.load_state_dict(torch.load(path, map_location='cpu'))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def embed_directory(
    image_dir: str,
    device: str = 'cuda',
    batch_size: int = 64,
    weights_path: str = '',
) -> Tuple[List[Path], np.ndarray]:
    """
    Compute L2-normalised face embeddings for every image in image_dir.

    Returns:
        paths:      list of Path objects in the same order as rows in `embs`.
        embs:       float32 ndarray (N, 512), each row L2-normalised.
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    model   = _load_model(device, weights_path)
    paths   = sorted(p for p in Path(image_dir).rglob('*') if p.suffix.lower() in _EXTS)

    all_embs: List[np.ndarray] = []
    batch_imgs: List[torch.Tensor] = []
    batch_paths_out: List[Path] = []

    def flush():
        with torch.no_grad():
            batch = torch.stack(batch_imgs).to(device)
            embs  = F.normalize(model(batch), dim=1).cpu().numpy()
        all_embs.append(embs)
        batch_imgs.clear()

    valid_paths: List[Path] = []
    for p in tqdm(paths, desc='Embedding'):
        try:
            img = Image.open(p).convert('RGB')
            batch_imgs.append(_TRANSFORM(img))
            valid_paths.append(p)
        except Exception:
            continue
        if len(batch_imgs) == batch_size:
            flush()

    if batch_imgs:
        flush()

    embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 512), dtype=np.float32)
    return valid_paths, embs


def _cosine_pairs_above(embs_a: np.ndarray, embs_b: np.ndarray, threshold: float, chunk: int = 2000):
    """Yield (i, j) pairs where cosine_sim(embs_a[i], embs_b[j]) >= threshold."""
    for start in range(0, len(embs_a), chunk):
        block = embs_a[start:start + chunk]          # (chunk, D)
        sims  = block @ embs_b.T                     # (chunk, len_b)
        hits  = np.argwhere(sims >= threshold)
        for local_i, j in hits:
            i = start + local_i
            yield int(i), int(j)


def find_cross_split_leakage(
    train_dir:    str,
    test_dir:     str,
    threshold:    float = 0.90,
    device:       str   = 'cuda',
    output:       str   = 'leakage.txt',
    weights_path: str   = '',
) -> List[Tuple[Path, Path]]:
    """
    Find test images that are near-duplicates of training images.
    High similarity across splits inflates evaluation metrics.

    threshold=0.90 is conservative; lower it to 0.80 for a stricter clean.
    """
    print("Embedding train …")
    train_paths, train_embs = embed_directory(train_dir, device, weights_path=weights_path)
    print("Embedding test  …")
    test_paths,  test_embs  = embed_directory(test_dir,  device, weights_path=weights_path)

    leaks: List[Tuple[Path, Path]] = []
    for ti, vi in _cosine_pairs_above(test_embs, train_embs, threshold):
        leaks.append((test_paths[ti], train_paths[vi]))

    with open(output, 'w') as f:
        for tp, trp in leaks:
            f.write(f"{tp}\t{trp}\n")

    print(f"Found {len(leaks)} test↔train leakage pairs  →  {output}")
    return leaks


def find_within_dir_dupes(
    image_dir:    str,
    threshold:    float = 0.95,
    device:       str   = 'cuda',
    output:       str   = 'duplicates.txt',
    weights_path: str   = '',
) -> List[Tuple[Path, Path]]:
    """
    Find near-duplicate pairs inside a single directory.
    Useful for checking that one identity folder doesn't contain two people,
    and for de-duping before train/test split.
    """
    paths, embs = embed_directory(image_dir, device, weights_path=weights_path)

    dupes: List[Tuple[Path, Path]] = []
    for i, j in _cosine_pairs_above(embs, embs, threshold):
        if i < j:   # avoid (a,b) and (b,a) both appearing
            dupes.append((paths[i], paths[j]))

    with open(output, 'w') as f:
        for a, b in dupes:
            f.write(f"{a}\t{b}\n")

    print(f"Found {len(dupes)} duplicate pairs  →  {output}")
    return dupes
