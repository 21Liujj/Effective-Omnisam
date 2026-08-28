#!/usr/bin/env python3
"""
experiments/tecma/scripts/check_configs.py

Dry-run config sanity check for TeCMA experiments.
Reads all YAMLs under experiments/tecma/configs/, prints A/D/E switch matrix,
validates apply_* consistency, and writes a Markdown report.

Usage (run from repo root):
    python experiments/tecma/scripts/check_configs.py
    python experiments/tecma/scripts/check_configs.py --config-dir experiments/tecma/configs
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found. Run: pip install pyyaml")


# ── field extractors ────────────────────────────────────────────────────────

def _get(d, *keys, default=None):
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def parse_config(path):
    """Load one YAML and return a flat summary dict."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    plugins = raw.get("plugins", {})
    ccpm  = plugins.get("pseudo_conf_mask",     {})
    fsmc  = plugins.get("efficient_tam_memory",  {})
    sgsmr = plugins.get("efficient_sam2_sparse", {})

    return {
        "name":         path.stem,
        # dataset / pd_root live in CLI args, not in plugin-only YAMLs
        "dataset":       raw.get("dataset", "(CLI arg)"),
        "pd_root":       raw.get("pd_root", "(CLI arg)"),
        # A
        "A_enabled":     bool(_get(ccpm,  "enabled",        default=False)),
        # D
        "D_enabled":     bool(_get(fsmc,  "enabled",        default=False)),
        "D_train":       bool(_get(fsmc,  "apply_in_train", default=False)),
        "D_pl":          bool(_get(fsmc,  "apply_in_pl",    default=False)),
        "D_compress":    _get(fsmc,  "compress_ratio",      default="—"),
        "D_pool_stride": _get(fsmc,  "pool_stride",         default="—"),
        # E
        "E_enabled":     bool(_get(sgsmr, "enabled",        default=False)),
        "E_train":       bool(_get(sgsmr, "apply_in_train", default=False)),
        "E_pl":          bool(_get(sgsmr, "apply_in_pl",    default=False)),
        "E_keep_ratio":  _get(sgsmr, "keep_ratio",          default="—"),
        "E_min_tokens":  _get(sgsmr, "min_keep_tokens",     default="—"),
    }


# ── validation rules ─────────────────────────────────────────────────────────

def validate(row):
    """Return list of ERROR strings; empty means OK."""
    errors = []
    if not row["D_enabled"]:
        if row["D_train"]:
            errors.append("D disabled but apply_in_train=true")
        if row["D_pl"]:
            errors.append("D disabled but apply_in_pl=true")
    if not row["E_enabled"]:
        if row["E_train"]:
            errors.append("E disabled but apply_in_train=true")
        if row["E_pl"]:
            errors.append("E disabled but apply_in_pl=true")
    return errors


# ── formatting helpers ────────────────────────────────────────────────────────

def yn(v):
    return "ON " if v else "off"

def ynsym(v):
    return "✅" if v else "❌"


def fmt_row(row, errors):
    status = "✅ OK" if not errors else "❌ ERRORS: " + "; ".join(errors)
    lines = [
        f"### `{row['name']}`",
        "",
        "| Field              | Value                        |",
        "|--------------------|------------------------------|",
        f"| dataset            | {row['dataset']}             |",
        f"| pd_root            | {row['pd_root']}             |",
        f"| A (CCPM) enabled   | {ynsym(row['A_enabled'])} {yn(row['A_enabled'])}|",
        f"| D (FSMC) enabled   | {ynsym(row['D_enabled'])} {yn(row['D_enabled'])}|",
        f"| D apply_in_train   | {ynsym(row['D_train'])} {yn(row['D_train'])}|",
        f"| D apply_in_pl      | {ynsym(row['D_pl'])} {yn(row['D_pl'])}|",
        f"| D compress_ratio   | {row['D_compress']}          |",
        f"| D pool_stride      | {row['D_pool_stride']}       |",
        f"| E (SG-SMR) enabled | {ynsym(row['E_enabled'])} {yn(row['E_enabled'])}|",
        f"| E apply_in_train   | {ynsym(row['E_train'])} {yn(row['E_train'])}|",
        f"| E apply_in_pl      | {ynsym(row['E_pl'])} {yn(row['E_pl'])}|",
        f"| E keep_ratio       | {row['E_keep_ratio']}        |",
        f"| E min_keep_tokens  | {row['E_min_tokens']}        |",
        f"| **Validation**     | {status}                    |",
        "",
    ]
    return "\n".join(lines)


def fmt_matrix(rows):
    hdr = "| Config | A | D_en | D_train | D_PL | E_en | E_train | E_PL |"
    sep = "|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    lines = [hdr, sep]
    for r in rows:
        s = ynsym
        lines.append(
            f"| `{r['name']}` "
            f"| {s(r['A_enabled'])} "
            f"| {s(r['D_enabled'])} "
            f"| {s(r['D_train'])} "
            f"| {s(r['D_pl'])} "
            f"| {s(r['E_enabled'])} "
            f"| {s(r['E_train'])} "
            f"| {s(r['E_pl'])} |"
        )
    return "\n".join(lines)


# ── acceptance checks ─────────────────────────────────────────────────────────

ACCEPTANCE = {
    "A_Dtrain_Etrain": {
        "A_enabled": True,
        "D_train":   True,
        "E_train":   True,
        "D_pl":      False,
        "E_pl":      False,
    },
    "baseline_omnisam": {
        "A_enabled": False,
        "D_enabled": False,
        "E_enabled": False,
    },
    "A_Dtrain_Etrain_PLon": {
        "A_enabled": True,
        "D_train":   True,
        "E_train":   True,
        "D_pl":      True,
        "E_pl":      True,
    },
    "A_ccpm": {
        "A_enabled": True,
        "D_enabled": False,
        "E_enabled": False,
    },
    "Dtrain_fsmc": {
        "A_enabled": False,
        "D_enabled": True,
        "D_train":   True,
        "D_pl":      False,
        "E_enabled": False,
    },
    "Etrain_sgsmr": {
        "A_enabled": False,
        "D_enabled": False,
        "E_enabled": True,
        "E_train":   True,
        "E_pl":      False,
    },
    "A_Etrain_keep075": {
        "A_enabled":    True,
        "D_enabled":    False,
        "E_enabled":    True,
        "E_train":      True,
        "E_pl":         False,
        "E_keep_ratio": 0.75,
    },
    "A_Etrain_keep025": {
        "A_enabled":    True,
        "D_enabled":    False,
        "E_enabled":    True,
        "E_train":      True,
        "E_pl":         False,
        "E_keep_ratio": 0.25,
    },
}


def run_acceptance(rows):
    by_name = {r["name"]: r for r in rows}
    results = []
    for cfg_name, checks in ACCEPTANCE.items():
        if cfg_name not in by_name:
            results.append(f"❌ MISSING config: `{cfg_name}`")
            continue
        row = by_name[cfg_name]
        for field, expected in checks.items():
            actual = row.get(field)
            ok = actual == expected
            sym = "✅" if ok else "❌"
            fail = "" if ok else " ← **FAIL**"
            results.append(
                f"{sym} `{cfg_name}`.{field} = `{actual}` (expected `{expected}`){fail}"
            )
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TeCMA config dry-run check")
    parser.add_argument(
        "--config-dir",
        default="experiments/tecma/configs",
        help="Directory containing TeCMA YAML configs",
    )
    parser.add_argument(
        "--out",
        default="experiments/tecma/notes/step_03_config_check_result.md",
        help="Output Markdown report path",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        sys.exit(f"Config dir not found: {config_dir}")

    yamls = sorted(config_dir.glob("*.yaml"))
    if not yamls:
        sys.exit(f"No YAML files found in {config_dir}")

    rows = []
    print(f"Checking {len(yamls)} config(s) in {config_dir} ...\n")
    for p in yamls:
        row = parse_config(p)
        errors = validate(row)
        rows.append((row, errors))
        tag = "OK  " if not errors else "FAIL"
        print(
            f"  [{tag}] {row['name']:<35s}"
            f"  A={yn(row['A_enabled']):<4}"
            f"  D_train={yn(row['D_train']):<4}  D_PL={yn(row['D_pl']):<4}"
            f"  E_train={yn(row['E_train']):<4}  E_PL={yn(row['E_pl']):<4}"
            + (f"  ← {'; '.join(errors)}" if errors else "")
        )

    row_dicts = [r for r, _ in rows]
    acceptance = run_acceptance(row_dicts)

    total_errors  = sum(1 for _, e in rows if e)
    accept_fails  = sum(1 for a in acceptance if "FAIL" in a or a.startswith("❌"))

    print("\n── Acceptance checks ──────────────────────────")
    for a in acceptance:
        clean = a.replace("✅", "[OK]").replace("❌", "[FAIL]").replace("**", "")
        print(" ", clean)

    print("\n── Summary ─────────────────────────────────────")
    print(f"  {len(rows)} configs checked, {total_errors} with validation errors")
    print(f"  {len(acceptance)} acceptance checks, {accept_fails} failed")
    if total_errors == 0 and accept_fails == 0:
        print("  [PASS] All checks passed.")
    else:
        print("  [FAIL] Some checks failed — review report.")

    # ── Markdown report ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md += [
        "# TeCMA Config Check Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"Config dir: `{config_dir}`  ",
        f"Configs checked: {len(rows)}  ",
        "",
        "---",
        "",
        "## Overview Matrix",
        "",
        fmt_matrix(row_dicts),
        "",
        "---",
        "",
        "## Acceptance Checks",
        "",
        "> Key configs validated against expected switch state.",
        "",
    ]
    md += [f"- {a}" for a in acceptance]
    md += [
        "",
        "---",
        "",
        "## Per-Config Details",
        "",
    ]
    for row, errors in rows:
        md.append(fmt_row(row, errors))

    result_line = (
        "✅ All checks passed."
        if total_errors == 0 and accept_fails == 0
        else f"❌ {total_errors} validation error(s), {accept_fails} acceptance failure(s)."
    )
    md += [
        "---",
        "",
        "## Summary",
        "",
        f"| Item | Count |",
        f"|------|-------|",
        f"| Configs checked | {len(rows)} |",
        f"| Configs with errors | {total_errors} |",
        f"| Acceptance checks | {len(acceptance)} |",
        f"| Acceptance failures | {accept_fails} |",
        "",
        f"**Result: {result_line}**",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    main()
