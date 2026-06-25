# Face Autoencoder (Adversarial Autoencoder)

A convolutional **adversarial autoencoder (AAE)** for faces, designed to
generalise to **unseen identities**. The flat 1024-d latent is split in two:

- **`z_style` (512 dims)** — regularised toward `N(0, I)` by an adversarial
  discriminator (no KL / no noise injection → sharper than a VAE). Captures
  pose, lighting, expression — everything that is *not* identity.
- **`z_id` (512 dims)** — distilled from a frozen **ArcFace (IR-SE50)** embedding
  of the input. Because ArcFace embeddings generalise to people never seen in
  training, so does this half.

The decoder consumes the concatenated latent. See [models/autoencoder.py](models/autoencoder.py)
and [models/losses.py](models/losses.py). Design follows the AAE paper
([Makhzani et al., 2015](https://arxiv.org/pdf/1511.05644)).

---

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Identity model (required)

The identity loss and the `identity_sim` metric use the IR-SE50 ArcFace
backbone from [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch),
which is **not** pip-installable. This one script clones the repo and downloads
the pretrained weights:

```bash
python scripts/download_identity_model.py --dest .
```

Then export what it prints (point these at the clone and the `.pth`):

```bash
export INSIGHTFACE_PYTORCH_PATH=$PWD/InsightFace_Pytorch
export INSIGHTFACE_WEIGHTS_PATH=$PWD/weights/model_ir_se50.pth
```

Also set `loss.identity_weights_path` in your config to that `.pth` path.

> Weights mirror: [`lithiumice/insightface`](https://huggingface.co/lithiumice/insightface/blob/main/InsightFace_Pytorch%2Bmodel_ir_se50.pth)
> (the original repo's README links a Dropbox copy as well).

## 3. Datasets

### FFHQ (automated — enough to start)

```bash
python scripts/download_data.py --out ./datasets/ffhq256
```

Downloads FFHQ at 256×256 from a Hugging Face mirror and extracts a flat folder
of images. Then set `data.ffhq_root: ./datasets/ffhq256` in the config.

Mirrors (pass `--repo` / `--file` if the default is unavailable):
- [`LIAGM/FFHQ_datasets`](https://huggingface.co/datasets/LIAGM/FFHQ_datasets) — `FFHQ_256.zip` (default)
- [`bitmind/ffhq-256`](https://huggingface.co/datasets/bitmind/ffhq-256), [`merkol/ffhq-256`](https://huggingface.co/datasets/merkol/ffhq-256) (HF `datasets` / parquet)

### VGGFace2 (optional, manual — adds identity diversity)

VGGFace2 gives multiple images per person, which the identity losses benefit
from. It is large (~40 GB+) and gated, so it is **not** scripted. Download from
a mirror and arrange as `root/<identity_id>/<image>.jpg`:

- [`ProgramComputer/VGGFace2`](https://huggingface.co/datasets/ProgramComputer/VGGFace2) (HF)
- [SourceForge mirror](https://sourceforge.net/projects/vggface2.mirror/)

Then set `data.vggface2_root` and (optionally) tune `data.ffhq_ratio`.

### Alignment (recommended)

Use the **same** alignment on all data — inconsistent alignment leaks into the
identity signal. Align with MTCNN before training:

```bash
python preprocess.py align --input <raw_dir> --output <aligned_dir>
```

### Held-out unseen identities (for validation plots)

Create a small folder with a few images of **~2 people that are NOT in your
training roots**, and set `data.holdout_dir` to it. Every validation epoch their
originals + reconstructions are plotted to wandb — your direct read on
generalisation to unseen faces.

## 4. Weights & Biases

Enable in the config:

```yaml
wandb:
  enabled: true
  project: face-autoencoder
  entity:  <your-wandb-username>   # or null
```

```bash
wandb login
```

Logged per epoch: all train/val loss components (`l1`, `perceptual`, `identity`,
`latent_id`, `adv_g`, `adv_d`), metrics (`psnr`, `ssim`, `identity_sim`), `lr`,
a validation reconstruction grid, and the **unseen-identity** grid from
`holdout_dir`. Training runs fine with `enabled: false`.

## 5. Train / evaluate

```bash
python train.py --config configs/default.yaml
python train.py --resume checkpoints/checkpoint_epoch0010.pt   # resume

python evaluate.py --checkpoint checkpoints/best_model.pt \
                   --data-dir ./datasets/vggface2_test --vggface2
```

---

## 6. Deploying on RunPod

**Recommended pod: 1× RTX 4090 (24 GB)** — Community Cloud, ~$0.3–0.5/hr. The
autoencoder (~113M params) plus the frozen VGG19 + IR-SE50 graphs fit
comfortably at `batch_size: 32`, 256×256. Best price/performance for this model.

| Need | Pod | Notes |
|------|-----|-------|
| Sweet spot | **1× RTX 4090 (24 GB)** | batch 32 fine; cheapest sensible option |
| Bigger batches / faster | 1× A40 (48 GB) or A100 (80 GB) | raise `batch_size` to 64–128 |
| Tight budget | 1× RTX A5000 (24 GB) | similar to 4090, often cheaper |

Avoid <16 GB cards (e.g. 3080) at this resolution — drop `batch_size` to 8–16 if
you must.

### Setup steps

1. **Deploy** a GPU pod with the official **RunPod PyTorch** template
   (PyTorch 2.x / CUDA 12.x).
2. **Attach a Network Volume** (~100 GB) mounted at `/workspace` so datasets and
   checkpoints survive pod restarts. FFHQ-256 ≈ 7 GB; add headroom for VGGFace2.
3. In the pod's web terminal:

   ```bash
   cd /workspace
   git clone <your-repo-url> face-autoencoder && cd face-autoencoder
   pip install -r requirements.txt

   python scripts/download_identity_model.py --dest /workspace
   python scripts/download_data.py --out /workspace/datasets/ffhq256

   export INSIGHTFACE_PYTORCH_PATH=/workspace/InsightFace_Pytorch
   export INSIGHTFACE_WEIGHTS_PATH=/workspace/weights/model_ir_se50.pth
   wandb login
   ```
4. Edit `configs/default.yaml`: set `data.ffhq_root`, `data.holdout_dir`,
   `loss.identity_weights_path`, `wandb.enabled: true`,
   `training.output_dir: /workspace/checkpoints`.
5. Train inside `tmux`/`nohup` so it survives disconnects:

   ```bash
   tmux new -s train
   python train.py --config configs/default.yaml
   ```

Monitor live curves and the unseen-identity reconstruction grids in your wandb
project.
