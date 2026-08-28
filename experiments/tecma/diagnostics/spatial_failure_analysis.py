#!/usr/bin/env python
"""Spatial distribution diagnostic for baseline vs A_Dtrain_Etrain."""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, "/root/autodl-tmp/OmniSAM-main")

from utils.build import build_model
from utils.metrics import fast_hist, per_class_iu
from dataset.DensePASS import DensePASS

def load_model_and_ckpt(ckpt_path):
    """Load OmniSAM model and checkpoint."""
    # Minimal config for inference
    config = {
        "model": "omnisam_adapt",
        "backbone": "sam2_s",
        "num_classes": 13,
        "lora": {"rank": 4, "enabled": True},
        "efficient_tam_memory": {"enabled": False},
        "efficient_sam2_sparse": {"enabled": False},
    }
    
    # Detect config from path
    if "A_Dtrain_Etrain" in ckpt_path:
        config["efficient_tam_memory"]["enabled"] = True
        config["efficient_sam2_sparse"]["enabled"] = True
    
    model = build_model(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model = model.cuda().eval()
    return model

def infer_densepass_val(model, dataset_root="data/DensePASS", crop_size=(2048, 400)):
    """Run inference on DensePASS val set, return dict of {image_name: pred_argmax_np}."""
    # Load val image list
    val_list = os.path.join(dataset_root, "densepass_list", "val.txt")
    with open(val_list) as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    predictions = {}
    
    for idx, name in enumerate(image_names):
        img_path = os.path.join(dataset_root, "leftImg8bit", "val", name + ".png")
        img = Image.open(img_path).convert("RGB")
        
        # Resize to crop_size
        img = img.resize(crop_size, Image.BILINEAR)
        img_np = np.array(img)
        
        # Normalize and to tensor
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_np = (img_np - mean) / std
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().cuda()
        
        # Sliding window inference (simplified: single forward for 2048x400)
        with torch.no_grad():
            out = model(img_t)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
        
        predictions[name] = pred
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(image_names)}")
    
    return predictions

def spatial_bucket_miou(predictions, dataset_root, bucket_scheme="edges_vs_center"):
    """Compute per-bucket mIoU."""
    results = {}
    
    for name, pred in predictions.items():
        gt_path = os.path.join(dataset_root, "gtFine", "val", name + "_gtFine_labelTrainIds.png")
        gt = np.array(Image.open(gt_path))
        
        H, W = pred.shape
        
        if bucket_scheme == "edges_vs_center":
            # 3 buckets: left 10%, center 80%, right 10%
            left_mask = np.zeros((H, W), dtype=bool)
            left_mask[:, :int(0.1*W)] = True
            center_mask = np.zeros((H, W), dtype=bool)
            center_mask[:, int(0.1*W):int(0.9*W)] = True
            right_mask = np.zeros((H, W), dtype=bool)
            right_mask[:, int(0.9*W):] = True
            
            buckets = {"left_10%": left_mask, "center_80%": center_mask, "right_10%": right_mask}
        else:
            # 5 equal-width buckets
            buckets = {}
            for i in range(5):
                mask = np.zeros((H, W), dtype=bool)
                mask[:, int(i*W/5):int((i+1)*W/5)] = True
                buckets[f"bucket_{i}"] = mask
        
        for bucket_name, mask in buckets.items():
            if bucket_name not in results:
                results[bucket_name] = {"hist": np.zeros((13, 13), dtype=np.int64)}
            
            # Apply mask
            pred_masked = pred[mask]
            gt_masked = gt[mask]
            
            # Filter valid pixels
            valid = (gt_masked < 13) & (gt_masked != 255)
            if valid.sum() == 0:
                continue
            
            pred_masked = pred_masked[valid]
            gt_masked = gt_masked[valid]
            
            # Accumulate histogram
            hist = fast_hist(gt_masked, pred_masked, 13)
            results[bucket_name]["hist"] += hist
    
    # Compute mIoU per bucket
    for bucket_name in results:
        hist = results[bucket_name]["hist"]
        ious = per_class_iu(hist)
        results[bucket_name]["miou"] = np.nanmean(ious)
        results[bucket_name]["ious"] = ious
    
    return results

def main():
    baseline_ckpt = "experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth"
    compressed_ckpt = "experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.02.pth"
    
    print("Loading baseline model...")
    baseline_model = load_model_and_ckpt(baseline_ckpt)
    
    print("Running baseline inference on DensePASS val...")
    baseline_preds = infer_densepass_val(baseline_model)
    del baseline_model
    torch.cuda.empty_cache()
    
    print("Loading compressed model...")
    compressed_model = load_model_and_ckpt(compressed_ckpt)
    
    print("Running compressed inference on DensePASS val...")
    compressed_preds = infer_densepass_val(compressed_model)
    del compressed_model
    torch.cuda.empty_cache()
    
    print("Computing spatial bucket mIoU...")
    baseline_results = spatial_bucket_miou(baseline_preds, "data/DensePASS", "edges_vs_center")
    compressed_results = spatial_bucket_miou(compressed_preds, "data/DensePASS", "edges_vs_center")
    
    # Save results
    out_dir = "experiments/tecma/diagnostics"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "spatial_buckets.json"), "w") as f:
        json.dump({
            "baseline": {k: {"miou": float(v["miou"])} for k, v in baseline_results.items()},
            "compressed": {k: {"miou": float(v["miou"])} for k, v in compressed_results.items()}
        }, f, indent=2)
    
    # Plot
    buckets = list(baseline_results.keys())
    baseline_mious = [baseline_results[b]["miou"] for b in buckets]
    compressed_mious = [compressed_results[b]["miou"] for b in buckets]
    deltas = [compressed_mious[i] - baseline_mious[i] for i in range(len(buckets))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(buckets))
    width = 0.35
    ax1.bar(x - width/2, baseline_mious, width, label="Baseline (60.26)", color="skyblue")
    ax1.bar(x + width/2, compressed_mious, width, label="MTC+SMR (60.02)", color="coral")
    ax1.set_ylabel("mIoU")
    ax1.set_title("Spatial mIoU by horizontal position")
    ax1.set_xticks(x)
    ax1.set_xticklabels(buckets, rotation=15, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    
    colors = ["red" if d < -0.5 else "orange" if d < 0 else "lightgreen" for d in deltas]
    ax2.bar(buckets, deltas, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Δ mIoU (compressed - baseline)")
    ax2.set_title("mIoU degradation by position")
    ax2.set_xticklabels(buckets, rotation=15, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("experiments/tecma/figures/spatial_seam_diagnostic.png", dpi=150, bbox_inches="tight")
    print("Saved: experiments/tecma/figures/spatial_seam_diagnostic.png")
    
    # Summary
    print("\n=== Spatial Distribution Diagnostic ===")
    for b in buckets:
        print(f"{b}: baseline={baseline_results[b]['miou']:.2f}, compressed={compressed_results[b]['miou']:.2f}, Δ={deltas[buckets.index(b)]:.2f}")

if __name__ == "__main__":
    main()
