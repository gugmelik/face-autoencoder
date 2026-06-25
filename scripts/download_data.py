"""
Download FFHQ (256×256) face images from a Hugging Face mirror and extract them
into a flat directory that FFHQDataset can read.

FFHQ is the easy, fully-automatable path and is enough to start training.
VGGFace2 (identity diversity, multiple images per person) is large and gated —
see the README for the manual steps; it is optional.

Usage:
    python scripts/download_data.py --out ./datasets/ffhq256
    python scripts/download_data.py --out /workspace/datasets/ffhq256   # RunPod volume

If the default mirror/file is unavailable, pass your own:
    python scripts/download_data.py --repo LIAGM/FFHQ_datasets --file FFHQ_256.zip --out ...
See the README for alternative mirrors.
"""
import argparse
import sys
import zipfile
from pathlib import Path

# Default: a mirror that ships FFHQ at 256 as a single zip of image files.
DEFAULT_REPO = "LIAGM/FFHQ_datasets"
DEFAULT_FILE = "FFHQ_256.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',  required=True, help='Output directory for extracted images')
    ap.add_argument('--repo', default=DEFAULT_REPO, help='HF dataset repo id')
    ap.add_argument('--file', default=DEFAULT_FILE, help='Zip filename inside the repo')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Install huggingface_hub first:  pip install huggingface_hub")

    print(f"Downloading {args.file} from hf.co/datasets/{args.repo} (this is several GB) …")
    try:
        zip_path = hf_hub_download(repo_id=args.repo, filename=args.file, repo_type='dataset')
    except Exception as e:
        sys.exit(
            f"Download failed: {e}\n\n"
            "Pick an alternative mirror (see README), e.g.:\n"
            "  --repo bitmind/ffhq-256   (HF datasets / parquet — needs the `datasets` lib)\n"
            "  --repo merkol/ffhq-256\n"
            "or download FFHQ manually and point the config at it."
        )

    print(f"Extracting → {out}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)

    n = sum(1 for p in out.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    print(f"✓ {n} images under {out}")
    print(f"\nSet in your config:  data.ffhq_root: {out.resolve()}")
    print("Tip: hold out a few images of 2 people into a separate folder and set "
          "data.holdout_dir so validation can plot reconstructions on unseen identities.")


if __name__ == '__main__':
    main()
