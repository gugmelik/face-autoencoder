"""
Consistent face alignment using MTCNN.

The Reddit advice: use the SAME alignment on ALL data.  If alignment is
inconsistent across images the model learns alignment as part of identity,
which silently poisons the ArcFace loss.

MTCNN handles detection + landmark extraction + rotation + crop internally,
so we get a uniform 5-point landmark alignment for every image.
"""
from pathlib import Path
from typing import Optional

from PIL import Image
import torch


def _get_mtcnn():
    import os, sys
    repo = os.environ.get(
        'INSIGHTFACE_PYTORCH_PATH',
        os.path.join(os.path.dirname(__file__), '../../InsightFace_Pytorch'),
    )
    if repo not in sys.path:
        sys.path.insert(0, os.path.abspath(repo))
    from mtcnn import MTCNN
    return MTCNN()


def align_single(img: Image.Image, mtcnn, output_size: int = 256) -> Optional[Image.Image]:
    """
    Align a PIL image. Returns a new PIL image or None if no face detected.
    Uses MTCNN's built-in 5-point landmark alignment (eyes, nose, mouth corners)
    so the result is identical across datasets as long as you use the same call.
    """
    try:
        aligned = mtcnn.align(img)  # returns 112×112 PIL Image, or None if no face
    except Exception:
        return None
    if aligned is None:
        return None
    return aligned.resize((output_size, output_size), Image.LANCZOS)


def align_dataset(
    input_dir:   str,
    output_dir:  str,
    output_size: int  = 256,
    device:      str  = 'cuda',
    skip_existing: bool = True,
) -> None:
    """
    Walk input_dir recursively, align every face image, and save the result
    to output_dir preserving the original folder structure.

    Args:
        input_dir:     Root of unaligned dataset (e.g. raw VGGFace2 or FFHQ).
        output_dir:    Root of aligned output.
        output_size:   Square output resolution in pixels.
        device:        'cuda' or 'cpu'.
        skip_existing: Skip images that already have an aligned counterpart.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mtcnn = _get_mtcnn()  # InsightFace_Pytorch MTCNN auto-selects GPU/CPU

    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    paths = [p for p in input_dir.rglob('*') if p.suffix.lower() in exts]

    ok = failed = skipped = 0
    for img_path in paths:
        rel     = img_path.relative_to(input_dir)
        out     = (output_dir / rel).with_suffix('.jpg')
        out.parent.mkdir(parents=True, exist_ok=True)

        if skip_existing and out.exists():
            skipped += 1
            continue

        try:
            img     = Image.open(img_path).convert('RGB')
            aligned = align_single(img, mtcnn, output_size)
            if aligned is not None:
                aligned.save(out, quality=95)
                ok += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    print(f"align_dataset done — ok={ok}  no_face={failed}  skipped={skipped}")
