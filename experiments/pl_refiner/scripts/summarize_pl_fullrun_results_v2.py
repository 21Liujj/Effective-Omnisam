#!/usr/bin/env python3
import json
import re
import glob
import csv
from pathlib import Path
from statistics import mean

ROOT = Path("/root/autodl-tmp/OmniSAM-main")
TABLE_DIR = ROOT / "experiments/pl_refiner/tables"
NOTE_DIR = ROOT / "experiments/pl_refiner/notes"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
NOTE_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    {
        "id": "R0",
        "tag": "R0_identity",
        "method": "Efficient + Identity PL",
        "metrics_glob": "experiments/pl_refiner/fullruns/R0_identity_*/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl",
        "log_glob": "experiments/pl_refiner/notes/fullrun_R0_identity_*.log",
        "mode": "unknown",
    },
    {
        "id": "R2",
        "tag": "R2_class_balanced",
        "method": "Efficient + R2 Class-balanced PL",
        "metrics_glob": "experiments/pl_refiner/fullruns/R2_class_balanced_*/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl",
        "log_glob": "experiments/pl_refiner/notes/fullrun_R2_class_balanced_*.log",
        "mode": "class_balanced",
    },
    {
        "id": "R2R4",
        "tag": "R2R4_proto_gate",
        "method": "Efficient + R2 Class-balanced + R4 Proto-gate",
        "metrics_glob": "experiments/pl_refiner/fullruns/R2R4_proto_gate_*/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl",
        "log_glob": "experiments/pl_refiner/notes/fullrun_R2R4_proto_gate_*.log",
        "mode": "proto_gate",
    },
]

CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "car"
]

def latest(pattern):
    files = [Path(p) for p in glob.glob(str(ROOT / pattern))]
    files = [p for p in files if p.exists()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def read_last_json(path):
    if not path or not path.exists():
        return {}
    lines = [x.strip() for x in path.read_text(errors="ignore").splitlines() if x.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except Exception:
            pass
    return {}

def parse_log(path):
    out = {
        "BFScore": None,
        "FWDice": None,
        "per_class_iou": {},
        "train_steps": None,
        "val_steps": None,
        "finished_at": None,
        "done": False,
    }
    if not path or not path.exists():
        return out

    text = path.read_text(errors="ignore")

    m = re.findall(r"\[ValSummary\]\s*BFScore=([0-9.]+)", text)
    if m:
        out["BFScore"] = float(m[-1])

    m = re.findall(r"\[ValSummary\]\s*FWDice=([0-9.]+)", text)
    if m:
        out["FWDice"] = float(m[-1])

    m = re.findall(r"finished_at\s*:\s*([^\n\r]+)", text)
    if m:
        out["finished_at"] = m[-1].strip()

    out["done"] = ("Training Finished." in text and "[done]" in text)

    m = re.findall(r"Epoch\s+1\s+\[Train\].*?(\d+)/(\d+)", text)
    if m:
        out["train_steps"] = f"{m[-1][0]}/{m[-1][1]}"

    m = re.findall(r"Epoch\s+1\s+\[Val\].*?(\d+)/(\d+)", text)
    if m:
        out["val_steps"] = f"{m[-1][0]}/{m[-1][1]}"

    block = re.search(r"=+\s*per-class IoU \(0-100\)\s*=+\n(.*?)\n=+", text, flags=re.S)
    if block:
        for line in block.group(1).splitlines():
            mm = re.match(r"\s*(.+?)\s*:\s*([0-9.]+)\s*$", line)
            if mm:
                out["per_class_iou"][mm.group(1).strip()] = float(mm.group(2))

    return out

def parse_diag_from_log(path, mode):
    out = {}
    if not path or not path.exists():
        return out
    rows = []
    proto_rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if "[refiner_diag]" in line and f"mode={mode}" in line:
            row = {}
            for k, v in re.findall(r"([A-Za-z_]+)=([0-9.]+)", line):
                try:
                    row[k] = float(v)
                except Exception:
                    pass
            if row:
                rows.append(row)
        if "[proto_sim_diag]" in line:
            row = {}
            for k, v in re.findall(r"([A-Za-z_]+)=([0-9.]+)", line):
                try:
                    row[k] = float(v)
                except Exception:
                    pass
            if row:
                proto_rows.append(row)

    def avg(data, key):
        vals = [r[key] for r in data if key in r]
        return mean(vals) if vals else None

    out["refiner_records"] = len(rows)
    out["proto_records"] = len(proto_rows)
    for key in ["total_vr", "common_vr", "rare_vr", "ext_rare_vr", "pseudo_vr", "ignore_r", "loss_sup", "loss_unsup"]:
        out[key] = avg(rows, key)
    for key in ["mean", "std", "valid_ratio", "nan"]:
        out["proto_" + key] = avg(proto_rows, key)
    return out

def f2(x):
    return "N/A" if x is None else f"{x:.2f}"

def f3(x):
    return "N/A" if x is None else f"{x:.3f}"

def f4(x):
    return "N/A" if x is None else f"{x:.4f}"

def write_md(path, header, rows):
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in header) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in header})

def delta(a, b, key):
    if a is None or b is None:
        return None
    if a.get(key) is None or b.get(key) is None:
        return None
    return a[key] - b[key]

results = {}

for run in RUNS:
    mp = latest(run["metrics_glob"])
    lp = latest(run["log_glob"])
    metrics = read_last_json(mp)
    log = parse_log(lp)
    diag = parse_diag_from_log(lp, run["mode"])

    results[run["id"]] = {
        "run_id": run["id"],
        "tag": run["tag"],
        "method": run["method"],
        "metrics_path": str(mp.relative_to(ROOT)) if mp else "N/A",
        "log_path": str(lp.relative_to(ROOT)) if lp else "N/A",
        "mIoU": metrics.get("val_mIoU"),
        "mAcc": metrics.get("val_mAcc"),
        "PixAcc": metrics.get("val_PixAcc"),
        "FWIoU": metrics.get("val_FWIoU"),
        "mDice": metrics.get("val_mDice"),
        "BFScore": log.get("BFScore"),
        "FWDice": log.get("FWDice"),
        "step_time": metrics.get("step_time"),
        "epoch_time": metrics.get("epoch_time"),
        "epoch_train_time": metrics.get("epoch_train_time"),
        "gpu_mem_peak": metrics.get("gpu_mem_peak"),
        "memory_tokens_before": metrics.get("memory_tokens_before"),
        "memory_tokens_after": metrics.get("memory_tokens_after"),
        "memory_token_ratio": metrics.get("memory_token_ratio"),
        "compressor_mode": metrics.get("compressor_mode"),
        "compress_location": metrics.get("compress_location"),
        "train_steps": log.get("train_steps"),
        "val_steps": log.get("val_steps"),
        "finished_at": log.get("finished_at"),
        "done": log.get("done"),
        "per_class_iou": log.get("per_class_iou", {}),
        "diag": diag,
    }

r0 = results["R0"]
r2 = results["R2"]
r24 = results["R2R4"]

main_rows = [
    {
        "Run": "R0",
        "Method": r0["method"],
        "Memory-efficient": "yes",
        "PL Refiner": "identity",
        "mIoU": f2(r0["mIoU"]),
        "d_mIoU": "N/A",
        "mAcc": f2(r0["mAcc"]),
        "d_mAcc": "N/A",
        "PixAcc": f2(r0["PixAcc"]),
        "FWIoU": f2(r0["FWIoU"]),
        "mDice": f2(r0["mDice"]),
        "BFScore": f2(r0["BFScore"]),
        "d_BFScore": "N/A",
        "step_time_s": f3(r0["step_time"]),
        "epoch_time_s": f2(r0["epoch_time"]),
        "peak_mem_GB": f2(r0["gpu_mem_peak"]),
        "train_steps": r0["train_steps"],
        "val_steps": r0["val_steps"],
        "done": str(r0["done"]),
    },
    {
        "Run": "R2",
        "Method": r2["method"],
        "Memory-efficient": "yes",
        "PL Refiner": "class-balanced",
        "mIoU": f2(r2["mIoU"]),
        "d_mIoU": f2(delta(r2, r0, "mIoU")),
        "mAcc": f2(r2["mAcc"]),
        "d_mAcc": f2(delta(r2, r0, "mAcc")),
        "PixAcc": f2(r2["PixAcc"]),
        "FWIoU": f2(r2["FWIoU"]),
        "mDice": f2(r2["mDice"]),
        "BFScore": f2(r2["BFScore"]),
        "d_BFScore": f2(delta(r2, r0, "BFScore")),
        "step_time_s": f3(r2["step_time"]),
        "epoch_time_s": f2(r2["epoch_time"]),
        "peak_mem_GB": f2(r2["gpu_mem_peak"]),
        "train_steps": r2["train_steps"],
        "val_steps": r2["val_steps"],
        "done": str(r2["done"]),
    },
    {
        "Run": "R2R4",
        "Method": r24["method"],
        "Memory-efficient": "yes",
        "PL Refiner": "class-balanced + proto-gate",
        "mIoU": f2(r24["mIoU"]),
        "d_mIoU": f2(delta(r24, r2, "mIoU")),
        "mAcc": f2(r24["mAcc"]),
        "d_mAcc": f2(delta(r24, r2, "mAcc")),
        "PixAcc": f2(r24["PixAcc"]),
        "FWIoU": f2(r24["FWIoU"]),
        "mDice": f2(r24["mDice"]),
        "BFScore": f2(r24["BFScore"]),
        "d_BFScore": f2(delta(r24, r2, "BFScore")),
        "step_time_s": f3(r24["step_time"]),
        "epoch_time_s": f2(r24["epoch_time"]),
        "peak_mem_GB": f2(r24["gpu_mem_peak"]),
        "train_steps": r24["train_steps"],
        "val_steps": r24["val_steps"],
        "done": str(r24["done"]),
    },
]

main_header = [
    "Run", "Method", "Memory-efficient", "PL Refiner",
    "mIoU", "d_mIoU", "mAcc", "d_mAcc", "PixAcc", "FWIoU", "mDice",
    "BFScore", "d_BFScore", "step_time_s", "epoch_time_s", "peak_mem_GB",
    "train_steps", "val_steps", "done"
]
write_md(TABLE_DIR / "pl_fullrun_main_metrics.md", main_header, main_rows)
write_csv(TABLE_DIR / "pl_fullrun_main_metrics.csv", main_header, main_rows)

pc_rows = []
for cls in CLASSES:
    a = r0["per_class_iou"].get(cls)
    b = r2["per_class_iou"].get(cls)
    c = r24["per_class_iou"].get(cls)
    pc_rows.append({
        "Class": cls,
        "R0": f2(a),
        "R2": f2(b),
        "R2R4": f2(c),
        "d_R2_minus_R0": f2((b - a) if a is not None and b is not None else None),
        "d_R2R4_minus_R2": f2((c - b) if b is not None and c is not None else None),
        "d_R2R4_minus_R0": f2((c - a) if a is not None and c is not None else None),
    })

pc_header = ["Class", "R0", "R2", "R2R4", "d_R2_minus_R0", "d_R2R4_minus_R2", "d_R2R4_minus_R0"]
write_md(TABLE_DIR / "pl_fullrun_per_class_iou.md", pc_header, pc_rows)
write_csv(TABLE_DIR / "pl_fullrun_per_class_iou.csv", pc_header, pc_rows)
write_md(TABLE_DIR / "pl_fullrun_per_class_delta.md", pc_header, pc_rows)

diag_rows = []
for k in ["R0", "R2", "R2R4"]:
    rr = results[k]
    d = rr["diag"]
    diag_rows.append({
        "Run": k,
        "Method": rr["method"],
        "refiner_records": d.get("refiner_records", "N/A"),
        "proto_records": d.get("proto_records", "N/A"),
        "mean_total_vr": f4(d.get("total_vr")),
        "mean_common_vr": f4(d.get("common_vr")),
        "mean_rare_vr": f4(d.get("rare_vr")),
        "mean_ext_rare_vr": f4(d.get("ext_rare_vr")),
        "mean_pseudo_vr": f4(d.get("pseudo_vr")),
        "mean_ignore_r": f4(d.get("ignore_r")),
        "mean_loss_sup": f4(d.get("loss_sup")),
        "mean_loss_unsup": f4(d.get("loss_unsup")),
        "proto_sim_mean": f4(d.get("proto_mean")),
        "proto_sim_std": f4(d.get("proto_std")),
        "proto_valid_ratio": f4(d.get("proto_valid_ratio")),
        "proto_nan": f4(d.get("proto_nan")),
    })

diag_header = [
    "Run", "Method", "refiner_records", "proto_records",
    "mean_total_vr", "mean_common_vr", "mean_rare_vr", "mean_ext_rare_vr",
    "mean_pseudo_vr", "mean_ignore_r", "mean_loss_sup", "mean_loss_unsup",
    "proto_sim_mean", "proto_sim_std", "proto_valid_ratio", "proto_nan"
]
write_md(TABLE_DIR / "pl_fullrun_diag_summary.md", diag_header, diag_rows)
write_csv(TABLE_DIR / "pl_fullrun_diag_summary.csv", diag_header, diag_rows)

report = f"""# Pseudo-label Refiner Full-run Result Summary

## Main Results

| Method | mIoU | mAcc | BFScore |
|---|---:|---:|---:|
| Efficient + Identity PL | {f2(r0['mIoU'])} | {f2(r0['mAcc'])} | {f2(r0['BFScore'])} |
| Efficient + R2 Class-balanced PL | {f2(r2['mIoU'])} | {f2(r2['mAcc'])} | {f2(r2['BFScore'])} |
| Efficient + R2 Class-balanced + R4 Proto-gate | {f2(r24['mIoU'])} | {f2(r24['mAcc'])} | {f2(r24['BFScore'])} |

## Delta Summary

| Comparison | d_mIoU | d_mAcc | d_BFScore |
|---|---:|---:|---:|
| R2 - R0 | {f2(delta(r2, r0, 'mIoU'))} | {f2(delta(r2, r0, 'mAcc'))} | {f2(delta(r2, r0, 'BFScore'))} |
| R2R4 - R2 | {f2(delta(r24, r2, 'mIoU'))} | {f2(delta(r24, r2, 'mAcc'))} | {f2(delta(r24, r2, 'BFScore'))} |
| R2R4 - R0 | {f2(delta(r24, r0, 'mIoU'))} | {f2(delta(r24, r0, 'mAcc'))} | {f2(delta(r24, r0, 'BFScore'))} |

## Interpretation

- R2 class-balanced filtering is the primary effective PL refinement module.
- R2 maintains mIoU and improves mAcc / BFScore under the memory-efficient setting.
- R4 proto-gate slightly improves BFScore over R2 but does not further improve mIoU.
- Therefore, R2 should be treated as the main PL module, and R4 should be positioned as a boundary-oriented auxiliary diagnostic/refinement module.

## Evidence Paths

| Run | Metrics Path | Log Path |
|---|---|---|
| R0 | `{r0['metrics_path']}` | `{r0['log_path']}` |
| R2 | `{r2['metrics_path']}` | `{r2['log_path']}` |
| R2R4 | `{r24['metrics_path']}` | `{r24['log_path']}` |

## Generated Files

- `experiments/pl_refiner/tables/pl_fullrun_main_metrics.md`
- `experiments/pl_refiner/tables/pl_fullrun_main_metrics.csv`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.md`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.csv`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_delta.md`
- `experiments/pl_refiner/tables/pl_fullrun_diag_summary.md`
- `experiments/pl_refiner/tables/pl_fullrun_diag_summary.csv`
"""
(NOTE_DIR / "step_17_pl_fullrun_result_summary.md").write_text(report, encoding="utf-8")

print("DONE")
print(TABLE_DIR / "pl_fullrun_main_metrics.md")
print(TABLE_DIR / "pl_fullrun_per_class_delta.md")
print(TABLE_DIR / "pl_fullrun_diag_summary.md")
print(NOTE_DIR / "step_17_pl_fullrun_result_summary.md")
