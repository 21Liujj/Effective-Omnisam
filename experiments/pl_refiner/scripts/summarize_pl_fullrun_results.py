#!/usr/bin/env python3
import json
import re
import csv
import glob
from pathlib import Path
from statistics import mean

ROOT = Path("/root/autodl-tmp/OmniSAM-main")
OUT_TABLE = ROOT / "experiments/pl_refiner/tables"
OUT_NOTE = ROOT / "experiments/pl_refiner/notes"
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_NOTE.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "name": "R0_identity",
        "method": "Efficient + Identity PL",
        "fullrun_glob": "experiments/pl_refiner/fullruns/R0_identity_*/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl",
        "log_glob": "experiments/pl_refiner/notes/fullrun_R0_identity_*.log",
        "diag_glob": None,
        "baseline": None,
    },
    {
        "name": "R2_class_balanced",
        "method": "Efficient + R2 Class-balanced PL",
        "fullrun_glob": "experiments/pl_refiner/fullruns/R2_class_balanced_*/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl",
        "log_glob": "experiments/pl_refiner/notes/fullrun_R2_class_balanced_*.log",
        "diag_glob": "experiments/pl_refiner/results/r2_class_balanced_fullrun_diag.jsonl",
        "baseline": "R0_identity",
    },
    {
        "name": "R2R4_proto_gate",
        "method": "Efficient + R2 Class-balanced + R4 Proto-gate",
        "fullrun_glob": "experiments/pl_refiner/fullruns/R2R4_proto_gate_*/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl",
        "log_glob": "experiments/pl_refiner/notes/fullrun_R2R4_proto_gate_*.log",
        "diag_glob": "experiments/pl_refiner/results/r2r4_proto_gate_fullrun_diag.jsonl",
        "baseline": "R2_class_balanced",
    },
]

CLASS_ORDER = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "car"
]

def latest_file(pattern: str):
    files = [Path(p) for p in glob.glob(str(ROOT / pattern))]
    files = [p for p in files if p.exists()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def read_last_jsonl(path: Path):
    if not path or not path.exists():
        return {}
    lines = [x.strip() for x in path.read_text(errors="ignore").splitlines() if x.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}

def parse_bfscore_and_perclass(log_path: Path):
    result = {
        "BFScore": None,
        "FWDice": None,
        "per_class_iou": {},
        "per_class_dice": {},
        "train_steps": None,
        "val_steps": None,
        "finished_at": None,
        "done": False,
    }
    if not log_path or not log_path.exists():
        return result

    text = log_path.read_text(errors="ignore")

    # BFScore / FWDice
    bf_matches = re.findall(r"\[ValSummary\]\s*BFScore=([0-9.]+)", text)
    if bf_matches:
        result["BFScore"] = float(bf_matches[-1])

    fwdice_matches = re.findall(r"\[ValSummary\]\s*FWDice=([0-9.]+)", text)
    if fwdice_matches:
        result["FWDice"] = float(fwdice_matches[-1])

    # finished_at / done
    fin_matches = re.findall(r"finished_at\s*:\s*([^\n\r]+)", text)
    if fin_matches:
        result["finished_at"] = fin_matches[-1].strip()
    result["done"] = ("Training Finished." in text) and ("[done]" in text)

    # train/val progress final counts
    train_matches = re.findall(r"Epoch\s+1\s+\[Train\].*?(\d+)/(\d+)", text)
    if train_matches:
        result["train_steps"] = f"{train_matches[-1][0]}/{train_matches[-1][1]}"
    val_matches = re.findall(r"Epoch\s+1\s+\[Val\].*?(\d+)/(\d+)", text)
    if val_matches:
        result["val_steps"] = f"{val_matches[-1][0]}/{val_matches[-1][1]}"

    # per-class IoU block
    iou_block_match = re.search(
        r"=+\s*per-class IoU \(0-100\)\s*=+\n(.*?)\n=+",
        text,
        flags=re.S
    )
    if iou_block_match:
        block = iou_block_match.group(1)
        for line in block.splitlines():
            m = re.match(r"\s*(.+?)\s*:\s*([0-9.]+)\s*$", line)
            if m:
                cls = m.group(1).strip()
                val = float(m.group(2))
                result["per_class_iou"][cls] = val

    # per-class Dice block
    dice_block_match = re.search(
        r"=+\s*per-class Dice/F1 \(0-100\)\s*=+\n(.*?)\n=+",
        text,
        flags=re.S
    )
    if dice_block_match:
        block = dice_block_match.group(1)
        for line in block.splitlines():
            m = re.match(r"\s*(.+?)\s*:\s*([0-9.]+)\s*$", line)
            if m:
                cls = m.group(1).strip()
                val = float(m.group(2))
                result["per_class_dice"][cls] = val

    return result

def parse_diag_from_log(log_path: Path, mode_keyword: str):
    """
    Parse lines like:
    [refiner_diag] step=4498 mode=class_balanced total_vr=0.4704 ...
    [proto_sim_diag] step=4498 mean=0.8931 std=0.0693 ...
    """
    if not log_path or not log_path.exists():
        return {}

    text = log_path.read_text(errors="ignore")
    ref_rows = []
    proto_rows = []

    for line in text.splitlines():
        if "[refiner_diag]" in line and f"mode={mode_keyword}" in line:
            row = {}
            for k, v in re.findall(r"([A-Za-z_]+)=([0-9.]+)", line):
                try:
                    row[k] = float(v)
                except Exception:
                    pass
            if row:
                ref_rows.append(row)

        if "[proto_sim_diag]" in line:
            row = {}
            for k, v in re.findall(r"([A-Za-z_]+)=([0-9.]+)", line):
                try:
                    row[k] = float(v)
                except Exception:
                    pass
            if row:
                proto_rows.append(row)

    def avg(rows, key):
        vals = [r[key] for r in rows if key in r]
        return mean(vals) if vals else None

    out = {}
    out["refiner_records"] = len(ref_rows)
    out["proto_records"] = len(proto_rows)

    for key in ["total_vr", "common_vr", "rare_vr", "ext_rare_vr", "pseudo_vr", "ignore_r", "loss_sup", "loss_unsup"]:
        out[f"mean_{key}"] = avg(ref_rows, key)

    for key in ["mean", "std", "valid_ratio", "nan"]:
        out[f"proto_mean_{key}"] = avg(proto_rows, key)

    return out

def fmt(x, nd=2):
    if x is None:
        return "N/A"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"

def fmt4(x):
    if x is None:
        return "N/A"
    return f"{x:.4f}"

def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})

def write_md_table(path, rows, header):
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in header) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    all_results = {}
    main_rows = []
    per_class_rows = []
    diag_rows = []

    for run in RUNS:
        metrics_path = latest_file(run["fullrun_glob"])
        log_path = latest_file(run["log_glob"])

        metrics = read_last_jsonl(metrics_path)
        log_info = parse_bfscore_and_perclass(log_path)

        if run["name"] == "R0_identity":
            diag = parse_diag_from_log(log_path, "unknown")
        elif run["name"] == "R2_class_balanced":
            diag = parse_diag_from_log(log_path, "class_balanced")
        else:
            diag = parse_diag_from_log(log_path, "proto_gate")

        result = {
            "run": run["name"],
            "method": run["method"],
            "metrics_path": str(metrics_path.relative_to(ROOT)) if metrics_path else "N/A",
            "log_path": str(log_path.relative_to(ROOT)) if log_path else "N/A",
            "mIoU": metrics.get("val_mIoU"),
            "mAcc": metrics.get("val_mAcc"),
            "PixAcc": metrics.get("val_PixAcc"),
            "FWIoU": metrics.get("val_FWIoU"),
            "mDice": metrics.get("val_mDice"),
            "BFScore": log_info.get("BFScore"),
            "FWDice": log_info.get("FWDice"),
            "step_time": metrics.get("step_time"),
            "epoch_time": metrics.get("epoch_time"),
            "epoch_train_time": metrics.get("epoch_train_time"),
            "gpu_mem_peak": metrics.get("gpu_mem_peak"),
            "memory_tokens_before": metrics.get("memory_tokens_before"),
            "memory_tokens_after": metrics.get("memory_tokens_after"),
            "memory_token_ratio": metrics.get("memory_token_ratio"),
            "compressor_mode": metrics.get("compressor_mode"),
            "compress_location": metrics.get("compress_location"),
            "train_steps": log_info.get("train_steps"),
            "val_steps": log_info.get("val_steps"),
            "finished_at": log_info.get("finished_at"),
            "done": log_info.get("done"),
            "per_class_iou": log_info.get("per_class_iou", {}),
            "per_class_dice": log_info.get("per_class_dice", {}),
            "diag": diag,
        }
        all_results[run["name"]] = result

    # Main table with deltas
    for run in RUNS:
        r = all_results[run["name"]]
        base = all_results.get(run["baseline"]) if run["baseline"] else None

        def delta(key):
            if not base:
                return None
            if r.get(key) is None or base.get(key) is None:
                return None
            return r[key] - base[key]

        row = {
            "Run": r["run"],
            "Method": r["method"],
            "Memory-efficient": "✓",
            "PL Refiner": (
                "identity" if r["run"] == "R0_identity"
                else "class-balanced" if r["run"] == "R2_class_balanced"
                else "class-balanced + proto-gate"
            ),
            "mIoU": fmt(r["mIoU"]),
            "ΔmIoU": fmt(delta("mIoU")),
            "mAcc": fmt(r["mAcc"]),
            "ΔmAcc": fmt(delta("mAcc")),
            "PixAcc": fmt(r["PixAcc"]),
            "FWIoU": fmt(r["FWIoU"]),
            "mDice": fmt(r["mDice"]),
            "BFScore": fmt(r["BFScore"]),
            "ΔBFScore": fmt(delta("BFScore")),
            "step_time(s)": fmt(r["step_time"], 3),
            "epoch_time(s)": fmt(r["epoch_time"], 1),
            "peak_mem(GB)": fmt(r["gpu_mem_peak"], 2),
            "train_steps": r["train_steps"] or "N/A",
            "val_steps": r["val_steps"] or "N/A",
            "done": str(r["done"]),
        }
        main_rows.append(row)

    main_header = [
        "Run", "Method", "Memory-efficient", "PL Refiner",
        "mIoU", "ΔmIoU", "mAcc", "ΔmAcc", "PixAcc", "FWIoU", "mDice",
        "BFScore", "ΔBFScore", "step_time(s)", "epoch_time(s)", "peak_mem(GB)",
        "train_steps", "val_steps", "done"
    ]

    write_csv(OUT_TABLE / "pl_fullrun_main_metrics.csv", main_rows, main_header)
    write_md_table(OUT_TABLE / "pl_fullrun_main_metrics.md", main_rows, main_header)

    # Per-class IoU table
    for cls in CLASS_ORDER:
        row = {"Class": cls}
        for run in RUNS:
            r = all_results[run["name"]]
            row[run["name"]] = fmt(r["per_class_iou"].get(cls))
        r0 = all_results["R0_identity"]["per_class_iou"].get(cls)
        r2 = all_results["R2_class_balanced"]["per_class_iou"].get(cls)
        r24 = all_results["R2R4_proto_gate"]["per_class_iou"].get(cls)
        row["ΔR2-R0"] = fmt((r2 - ris not None and r0 is not None else None)
        row["ΔR2R4-R2"] = fmt((r24 - r2) if r24 is not None and r2 is not None else None)
        row["ΔR2R4-R0"] = fmt((r24 - r0) if r24 is not None and r0 is not None else None)
        per_class_rows.append(row)

    pc_header = ["Class"] + [r["name"] for r in RUNS] + ["ΔR2-R0", "ΔR2R4-R2", "ΔR2R4-R0"]
    write_csv(OUT_TABLE / "pl_fullrun_per_class_iou.csv", per_class_rows, pc_header)
    write_md_table(OUT_TABLE / "pl_fullrun_per_class_iou.md", per_class_rows, pc_header)
    write_md_table(OUT_TABLE / "pl_fullrun_per_class_delta.md", per_class_rows, pc_header)

    # Diagnostics summary
    for run in RUNS:
        r = all_results[run["name"]]
        d = r["diag"]
        row = {
            "Run": r["run"],
            "Method": r["method"],
            "refiner_records": d.get("refiner_records", "N/A"),
            "proto_records": d.get("proto_records", "N/A"),
            "mean_total_vr": fmt4(d.get("mean_total_vr")),
            "mean_common_vr": fmt4(d.get("mean_common_vr")),
            "mean_rare_vr": fmt4(d.get("mean_rare_vr")),
            "mean_ext_rare_vr": fmt4(d.get("mean_ext_rare_vr")),
            "mean_pseudo_vr": fmt4(d.get("mean_pseudo_vr")),
            "mean_ignore_r": fmt4(d.get("mean_ignore_r")),
            "mean_loss_sup": fmt4(d.get("mean_loss_sup")),
            "mean_loss_unsup": fmt4(d.get("mean_loss_unsup")),
            "proto_sim_mean": fmt4(d.get("proto_mean_mean")),
            "proto_sim_std": fmt4(d.get("proto_mean_std")),
            "proto_valid_ratio": fmt4(d.get("proto_mean_valid_ratio")),
            "proto_nan_mean": fmt4(d.get("proto_mean_nan")),
        }
        diag_rows.append(row)

    diag_header = [
        "Run", "Method", "refiner_records", "proto_records",
        "mean_total_vr", "mean_common_vr", "mean_rare_vr", "mean_ext_rare_vr",
        "mean_pseudo_vr", "mean_ignore_r", "mean_loss_sup", "mean_loss_unsup",
        "proto_sim_mean", "proto_sim_std", "proto_valid_ratio", "proto_nan_mean"
    ]
    write_csv(OUT_TABLE / "pl_fullrun_diag_summary.csv", diag_rows, diag_header)
    write_md_table(OUT_TABLE / "pl_fullrun_diag_summary.md", diag_rows, diag_header)

    # Final markdown report
    r0 = all_results["R0_identity"]
    r2 = all_results["R2_class_balanced"]
    r24 = all_results["R2R4_proto_gate"]

    def dval(a, b, key):
        if a.get(key) is None or b.get(key) is None:
            return None
        return a[key] - b[key]

    report = f"""# Pseudo-label Refiner Full-run Summary

## Scope

This report summarizes the full-run validation of the pseudo-label refinement line under the memory-efficient setting.

Runs:
- R0_identity: Efficient + identity pseudo-label refiner.
- R2_class_balanced: Efficient + class-balanced pseudo-label filtering.
- R2R4_proto_gate: Efficient + class-balanced filtering + prototype gate.

## Main Result

| Method | mIoU | mAcc | PixAcc | FWIoU | mDice | BFScore |
|---|---:|---:|---:|---:|---:|---:|
| R0 identity | {fmt(r0['mIoU'])} | {fmt(r0['mAcc'])} | {fmt(r0['PixAcc'])} | {fmt(r0['FWIoU'])} | {fmt(r0['mDice'])} | {fmt(r0['BFScore'])} |
| R2 class-balanced | {fmt(r2['mIoU'])} | {fmt(r2['mAcc'])} | {fmt(r2['PixAcc'])} | {fmt(r2['FWIoU'])} | {fmt(r2['mDice'])} | {fmt(r2['BFScore'])} |
| R2 + R4 proto-gate | {fmt(r24['mIoU'])} | {fmt(r24['mAcc'])} | {fmt(r24['PixAcc'])} | {fmt(r24['FWIoU'])} | {fmt(r24['mDice'])} | {fmt(r24['BFScore'])} |

## Delta

| Comparison | ΔmIoU | ΔmAcc | ΔBFScore |
|---|---:|---:|---:|
| R2 - R0 | {fmt(dval(r2, r0, 'mIoU'))} | {fmt(dval(r2, r0, 'mAcc'))} | {fmt(dval(r2, r0, 'BFScore'))} |
| R2R4 - R2 | {fmt(dval(r24, r2, 'mIoU'))} | {fmt(dval(r24, r2, 'mAcc'))} | {fmt(dval(r24, r2, 'BFScore'))} |
| R2R4 - R0 | {fmt(dval(r24, r0, 'mIoU'))} | {fmt(dval(r24, r0, 'mAcc'))} | {fmt(dval(r24, r0, 'BFScore'))} |

## Efficiency

| Method | step_time(s) | epoch_time(s) | peak_mem(GB) | train_steps | val_steps |
|---|---:|---:|---:|---:|---:|
| R0 identity | {fmt(r0['step_time'], 3)} | {fmt(r0['epoch_time'], 1)} | {fmt(r0['gpu_mem_peak'], 2)} | {r0['train_steps']} | {r0['val_steps']} |
| R2 class-balanced | {fmt(r2['step_time'], 3)} | {fmt(r2['epoch_time'], 1)} | {fmt(r2['gpu_mem_peak'], 2)} | {r2['train_steps']} | {r2['val_steps']} |
| R2 + R4 proto-gate | {fmt(r24['step_time'], 3)} | {fmt(r24['epoch_time'], 1)} | {fmt(r24['gpu_mem_peak'], 2)} | {r24['train_steps']} | {r24['val_steps']} |

## Evidence Paths

| Run | Metrics | Log |
|---|---|---|
| R0 | `{r0['metrics_path']}` | `{r0['log_path']}` |
| R2 | `{r2['metrics_path']}` | `{r2['log_path']}` |
| R2R4 | `{r24['metrics_path']}` | `{r24['log_path']}` |

## Initial Interpretation

- R2 class-balanced filtering is the main effective pseudo-label refinement module.
- R2 improves mAcc and BFScore while maintaining mIoU under the memory-efficient setting.
- R4 prototype gate further improves BFScore slightly but does not further improve mIoU.
- Therefore, R2 should be treated as the primary PL-refinement component, while R4 is better positioned as a boundary-oriented auxiliary component.

## Generated Tables

- `experiments/pl_refiner/tables/pl_fullrun_main_metrics.md`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.md`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_delta.md`
- `experiments/pl_refiner/tables/pl_fullrun_diag_summary.md`
"""
    (OUT_NOTE / "step_17_pl_fullrun_result_summary.md").write_text(report, encoding="utf-8")

    print("DONE: pseudo-label full-run summary generated.")
    print("Main metrics:", OUT_TABLE / "pl_fullrun_main_metrics.md")
    print("Per-class IoU:", OUT_TABLE / "pl_per_class_iou.md")
    print("Diagnostics:", OUT_TABLE / "pl_fullrun_diag_summary.md")
    print("Report:", OUT_NOTE / "step_17_pl_fullrun_result_summary.md")

if __name__ == "__main__":
    main()
