"""
Set up the frozen ArcFace identity model used by the loss / metrics.

This does two things:
  1. Clones TreB1eN/InsightFace_Pytorch (provides the IR-SE50 `Backbone` and
     `MTCNN` classes that losses.py / metrics.py / align.py import).
  2. Downloads the pretrained IR-SE50 weights (model_ir_se50.pth) from a
     Hugging Face mirror.

After running, export the two env vars it prints (or pass the weights path via
the config's loss.identity_weights_path).

Usage:
    python scripts/download_identity_model.py                 # ./InsightFace_Pytorch + ./weights
    python scripts/download_identity_model.py --dest /workspace
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL  = "https://github.com/TreB1eN/InsightFace_Pytorch"
HF_REPO   = "lithiumice/insightface"
HF_FILE   = "InsightFace_Pytorch+model_ir_se50.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='.', help='Directory to place the repo clone and weights/ in')
    args = ap.parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    repo_dir = dest / 'InsightFace_Pytorch'
    if repo_dir.exists():
        print(f"✓ repo already present: {repo_dir}")
    else:
        print(f"Cloning {REPO_URL} → {repo_dir}")
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(repo_dir)], check=True)

    weights_dir = dest / 'weights'
    weights_dir.mkdir(exist_ok=True)
    weights_path = weights_dir / 'model_ir_se50.pth'

    if weights_path.exists():
        print(f"✓ weights already present: {weights_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit("Install huggingface_hub first:  pip install huggingface_hub")
        print(f"Downloading IR-SE50 weights from hf.co/{HF_REPO} …")
        cached = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        # Copy out of the HF cache to a stable path.
        import shutil
        shutil.copy(cached, weights_path)
        print(f"✓ weights → {weights_path}")

    print("\nAdd these to your shell / RunPod env (and set them before training):")
    print(f"  export INSIGHTFACE_PYTORCH_PATH={repo_dir}")
    print(f"  export INSIGHTFACE_WEIGHTS_PATH={weights_path}")
    print(f"\nAnd in your config set:  loss.identity_weights_path: {weights_path}")


if __name__ == '__main__':
    main()
