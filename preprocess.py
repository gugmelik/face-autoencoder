"""
Dataset preprocessing: alignment and deduplication.

Run these BEFORE training.  The Reddit advice in order:
  1. Align with the same pipeline on every dataset (FFHQ + VGGFace2)
  2. De-dup by embedding within each dataset
  3. Check for cross-split leakage between train and test

Examples:
  # 1. Align FFHQ
  python preprocess.py align --input /data/ffhq_raw --output /data/ffhq_aligned

  # 2. Align VGGFace2
  python preprocess.py align --input /data/vggface2_raw --output /data/vggface2_aligned

  # 3. Find duplicate images inside the training set
  python preprocess.py dedup --input /data/vggface2_aligned --output dupes_train.txt

  # 4. Find leakage between train and test
  python preprocess.py leakage \\
      --train /data/vggface2_aligned/train \\
      --test  /data/vggface2_aligned/test \\
      --output leakage.txt
"""
import argparse


def cmd_align(args):
    from data.align import align_dataset
    align_dataset(
        input_dir    = args.input,
        output_dir   = args.output,
        output_size  = args.size,
        device       = args.device,
        skip_existing= not args.no_skip,
    )


def cmd_dedup(args):
    from data.dedup import find_within_dir_dupes
    find_within_dir_dupes(
        image_dir = args.input,
        threshold = args.threshold,
        device    = args.device,
        output    = args.output,
    )


def cmd_leakage(args):
    from data.dedup import find_cross_split_leakage
    find_cross_split_leakage(
        train_dir = args.train,
        test_dir  = args.test,
        threshold = args.threshold,
        device    = args.device,
        output    = args.output,
    )


def main():
    parser = argparse.ArgumentParser(description='Face dataset preprocessing')
    sub    = parser.add_subparsers(dest='command', required=True)

    # align
    p_align = sub.add_parser('align', help='Align faces using MTCNN')
    p_align.add_argument('--input',    required=True, help='Raw image directory')
    p_align.add_argument('--output',   required=True, help='Aligned output directory')
    p_align.add_argument('--size',     type=int, default=256, help='Output image size (default 256)')
    p_align.add_argument('--device',   default='cuda')
    p_align.add_argument('--no-skip',  action='store_true', help='Re-process already aligned images')

    # dedup
    p_dedup = sub.add_parser('dedup', help='Find near-duplicate images by face embedding')
    p_dedup.add_argument('--input',     required=True, help='Directory to check')
    p_dedup.add_argument('--output',    default='duplicates.txt')
    p_dedup.add_argument('--threshold', type=float, default=0.95,
                         help='Cosine similarity threshold (default 0.95)')
    p_dedup.add_argument('--device',    default='cuda')

    # leakage
    p_leak = sub.add_parser('leakage', help='Find train/test leakage by face embedding')
    p_leak.add_argument('--train',     required=True, help='Training set directory')
    p_leak.add_argument('--test',      required=True, help='Test set directory')
    p_leak.add_argument('--output',    default='leakage.txt')
    p_leak.add_argument('--threshold', type=float, default=0.90,
                        help='Cosine similarity threshold (default 0.90)')
    p_leak.add_argument('--device',    default='cuda')

    args = parser.parse_args()
    {'align': cmd_align, 'dedup': cmd_dedup, 'leakage': cmd_leakage}[args.command](args)


if __name__ == '__main__':
    main()
