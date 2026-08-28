"""L2 Forward Diagnosis: PLQ_basic score distribution.

Forward-only (no backward, no param update, no checkpoint save).
Loads model + target images, computes PLQ_basic on a few batches,
and prints per-class / per-group statistics plus histogram buckets.

Usage:
    python experiments/pl_refiner/scripts/l2_plq_forward_diagnosis.py \
        --config configs/ablations/omnisam_A_plus4train_plus5train.yaml \
        --ckpt data_pd_exp_final_2000/A_Dtrain_Etrain/checkpoints/best.pth \
        --num_batches 4 \
        [--pd_root data_pd_exp_final_2000/A_Dtrain_Etrain/pseudo_labels]
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import utils.sam2_bootstrap  # noqa: F401

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from adaptation.pseudo_label.plq import (
    compute_confidence,
    compute_entropy_reliability,
    compute_local_agreement,
    compute_plq_basic,
)

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'car',
]
NUM_CLASSES = 13
IGNORE_INDEX = 255

CLASS_GROUPS = {
    'common': [0, 1, 2, 8, 9, 10, 12],
    'rare_hard': [3, 4, 5, 11],
    'extreme_rare': [6, 7],
}


def parse_args():
    p = argparse.ArgumentParser(description="L2 PLQ_basic forward diagnosis")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    p.add_argument("--pd_root", type=str, default=None,
                   help="Pseudo-label directory (PNG). If None, infer from config.")
    p.add_argument("--num_batches", type=int, default=4, help="Batches to diagnose")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--kernel_size", type=int, default=5, help="Local agreement kernel")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional: save results to JSON file")
    return p.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model_for_diag(config, ckpt_path, device):
    """Build OmniSAM_adapt model, load checkpoint, set eval mode."""
    from utils.build import build_model
    from utils.checkpoint_adapt import smart_load_checkpoint

    class Args:
        pass

    args = Args()
    args.backbone = config.get('backbone', 'sam2_s')
    args.num_maskmem = config.get('num_maskmem', 9)
    args.text_fusion = config.get('text_fusion', {'enabled': False})
    args.depth_adapter = config.get('depth_adapter', {'enabled': False})
    args.plugins = config.get('plugins', {})

    for k, v in config.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    model = build_model(args, num_classes=NUM_CLASSES,
                        num_maskmem=args.num_maskmem, device=device)
    smart_load_checkpoint(model, ckpt_path)
    model.eval()
    model.to(device)
    return model


def load_pseudo_labels_from_dir(pd_root, filenames):
    """Load pseudo label PNGs as numpy arrays."""
    from PIL import Image
    labels = {}
    for fn in filenames:
        base = os.path.splitext(fn)[0] + '.png'
        pl_path = os.path.join(pd_root, base)
        if os.path.exists(pl_path):
            lbl = np.array(Image.open(pl_path), dtype=np.int64)
            labels[fn] = lbl
    return labels


def build_target_dataloader(config, pd_root, batch_size):
    """Build DensePASS target dataloader for diagnosis."""
    from dataset.DensePASS import DensePASS

    data_cfg = config.get('dataset', {})
    data_root = data_cfg.get('target_root', 'data/DensePASS')
    split = 'train'

    dataset = DensePASS(
        root=data_root,
        split=split,
        num_classes=NUM_CLASSES,
        pseudo_label_dir=pd_root,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    return loader


def run_diagnosis(args):
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"[L2 PLQ Diag] config: {args.config}")
    print(f"[L2 PLQ Diag] ckpt:   {args.ckpt}")
    print(f"[L2 PLQ Diag] device: {device}")
    print(f"[L2 PLQ Diag] num_batches: {args.num_batches}, kernel: {args.kernel_size}")
    print()

    # Build model
    model = build_model_for_diag(config, args.ckpt, device)

    # Resolve pd_root
    pd_root = args.pd_root
    if pd_root is None:
        pd_root = config.get('pd_root', None)
        if pd_root is None:
            pd_root = os.path.join(os.path.dirname(args.ckpt), '..', 'pseudo_labels')
    print(f"[L2 PLQ Diag] pd_root: {pd_root}")

    # Build dataloader
    loader = build_target_dataloader(config, pd_root, args.batch_size)

    # --- Forward loop (no backward) ---
    per_class_plq = defaultdict(list)
    per_class_conf = defaultdict(list)
    per_class_ent = defaultdict(list)
    per_class_agr = defaultdict(list)
    all_plq_values = []
    total_pixels = 0
    valid_pixels_total = 0

    model.reset_state()  # clear memory bank state

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            img = batch['image'].to(device)
            frame_idx = batch_idx

            pred_t, feat_t, feat_t_de = model(
                frame_idx, img, mem_type='target'
            )

            probs = torch.softmax(pred_t, dim=1)
            conf_t, pseudo_t = probs.max(dim=1)

            plq = compute_plq_basic(
                confidence=conf_t,
                student_logits=pred_t,
                pseudo_label=pseudo_t,
                num_classes=NUM_CLASSES,
                kernel_size=args.kernel_size,
                ignore_index=IGNORE_INDEX,
            )
            ent_rel = compute_entropy_reliability(pred_t, num_classes=NUM_CLASSES)
            local_agr = compute_local_agreement(
                pseudo_t, kernel_size=args.kernel_size, ignore_index=IGNORE_INDEX
            )

            valid = (pseudo_t != IGNORE_INDEX)
            B, H, W = pseudo_t.shape
            total_pixels += B * H * W
            valid_pixels_total += int(valid.sum().item())

            for c in range(NUM_CLASSES):
                mask_c = (pseudo_t == c) & valid
                if mask_c.sum() == 0:
                    continue
                per_class_plq[c].append(plq[mask_c].cpu().numpy())
                per_class_conf[c].append(conf_t[mask_c].cpu().numpy())
                per_class_ent[c].append(ent_rel[mask_c].cpu().numpy())
                per_class_agr[c].append(local_agr[mask_c].cpu().numpy())

            all_plq_values.append(plq[valid].cpu().numpy())

            print(f"  batch {batch_idx}: plq_mean={plq[valid].mean().item():.4f}  "
                  f"conf={conf_t[valid].mean().item():.4f}  "
                  f"ent_rel={ent_rel[valid].mean().item():.4f}  "
                  f"local_agr={local_agr[valid].mean().item():.4f}")

    # --- Aggregate and print results ---
    print("\n" + "=" * 70)
    print("  L2 PLQ_basic Forward Diagnosis — Results")
    print("=" * 70)

    all_plq_cat = np.concatenate(all_plq_values) if all_plq_values else np.array([])

    print(f"\n[Overall] total_pixels={total_pixels}  valid_pixels={valid_pixels_total}  "
          f"valid_ratio={valid_pixels_total / max(1, total_pixels):.4f}")
    if all_plq_cat.size > 0:
        print(f"[Overall] PLQ_basic: mean={all_plq_cat.mean():.4f}  "
              f"std={all_plq_cat.std():.4f}  "
              f"min={all_plq_cat.min():.4f}  max={all_plq_cat.max():.4f}  "
              f"median={np.median(all_plq_cat):.4f}")

    # Histogram buckets
    print(f"\n{'Bucket':<12} {'Count':>10} {'Ratio':>8}")
    print("-" * 32)
    hist_data = {}
    for lo in range(10):
        lo_v = lo * 0.1
        hi_v = (lo + 1) * 0.1
        if lo == 9:
            count = int(((all_plq_cat >= lo_v) & (all_plq_cat <= hi_v)).sum())
        else:
            count = int(((all_plq_cat >= lo_v) & (all_plq_cat < hi_v)).sum())
        ratio = count / max(1, len(all_plq_cat))
        bucket_name = f"[{lo_v:.1f}-{hi_v:.1f}]"
        print(f"{bucket_name:<12} {count:>10} {ratio:>8.4f}")
        hist_data[bucket_name] = {'count': count, 'ratio': ratio}

    # Per-class table
    print(f"\n{'Class':<15} {'Pixels':>9} {'PLQ_mean':>9} {'PLQ_std':>8} "
          f"{'Conf':>7} {'EntRel':>7} {'LocAgr':>7}")
    print("-" * 70)
    per_class_results = {}
    for c in range(NUM_CLASSES):
        if c not in per_class_plq or len(per_class_plq[c]) == 0:
            print(f"{CLASS_NAMES[c]:<15} {'N/A':>9}")
            per_class_results[CLASS_NAMES[c]] = {'pixels': 0}
            continue
        plq_c = np.concatenate(per_class_plq[c])
        conf_c = np.concatenate(per_class_conf[c])
        ent_c = np.concatenate(per_class_ent[c])
        agr_c = np.concatenate(per_class_agr[c])
        print(f"{CLASS_NAMES[c]:<15} {len(plq_c):>9} {plq_c.mean():>9.4f} "
              f"{plq_c.std():>8.4f} {conf_c.mean():>7.4f} "
              f"{ent_c.mean():>7.4f} {agr_c.mean():>7.4f}")
        per_class_results[CLASS_NAMES[c]] = {
            'pixels': int(len(plq_c)),
            'plq_mean': float(plq_c.mean()),
            'plq_std': float(plq_c.std()),
            'conf_mean': float(conf_c.mean()),
            'ent_rel_mean': float(ent_c.mean()),
            'local_agr_mean': float(agr_c.mean()),
        }

    # Per-group table
    print(f"\n{'Group':<15} {'PLQ_mean':>9} {'Conf':>7} {'EntRel':>7} {'LocAgr':>7}")
    print("-" * 50)
    per_group_results = {}
    for gname, gclasses in CLASS_GROUPS.items():
        g_plq = [per_class_plq[c] for c in gclasses if c in per_class_plq and per_class_plq[c]]
        if not g_plq:
            print(f"{gname:<15} {'N/A':>9}")
            per_group_results[gname] = {}
            continue
        g_plq_cat = np.concatenate([np.concatenate(v) for v in g_plq])
        g_conf = np.concatenate([np.concatenate(per_class_conf[c]) for c in gclasses
                                 if c in per_class_conf and per_class_conf[c]])
        g_ent = np.concatenate([np.concatenate(per_class_ent[c]) for c in gclasses
                                if c in per_class_ent and per_class_ent[c]])
        g_agr = np.concatenate([np.concatenate(per_class_agr[c]) for c in gclasses
                                if c in per_class_agr and per_class_agr[c]])
        print(f"{gname:<15} {g_plq_cat.mean():>9.4f} {g_conf.mean():>7.4f} "
              f"{g_ent.mean():>7.4f} {g_agr.mean():>7.4f}")
        per_group_results[gname] = {
            'plq_mean': float(g_plq_cat.mean()),
            'conf_mean': float(g_conf.mean()),
            'ent_rel_mean': float(g_ent.mean()),
            'local_agr_mean': float(g_agr.mean()),
        }

    print("\n" + "=" * 70)
    print("[L2 PLQ Diag] Done. No backward, no param update, no checkpoint save.")

    # Optional JSON output
    if args.output_json:
        results = {
            'config': args.config,
            'ckpt': args.ckpt,
            'num_batches': args.num_batches,
            'kernel_size': args.kernel_size,
            'overall': {
                'total_pixels': total_pixels,
                'valid_pixels': valid_pixels_total,
                'valid_ratio': valid_pixels_total / max(1, total_pixels),
                'plq_mean': float(all_plq_cat.mean()) if all_plq_cat.size > 0 else None,
                'plq_std': float(all_plq_cat.std()) if all_plq_cat.size > 0 else None,
                'plq_min': float(all_plq_cat.min()) if all_plq_cat.size > 0 else None,
                'plq_max': float(all_plq_cat.max()) if all_plq_cat.size > 0 else None,
                'plq_median': float(np.median(all_plq_cat)) if all_plq_cat.size > 0 else None,
            },
            'histogram': hist_data,
            'per_class': per_class_results,
            'per_group': per_group_results,
        }
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[L2 PLQ Diag] Results saved to {args.output_json}")


if __name__ == '__main__':
    run_diagnosis(parse_args())

