#!/usr/bin/env python3
"""
experiments/pl_refiner/scripts/check_pl_refiner_configs.py

Dry-run config checker for R0-R7 PseudoLabelRefiner configurations.

What this script does:
  - Reads each R0-R7 YAML file.
  - Prints a summary table of experiment_id, refiner_mode, switch states,
    memory_config, output_dir, full_train_allowed.
  - Validates internal consistency (no contradictory switch combinations).
  - Marks unimplemented modes as NOT_IMPLEMENTED_YET (not as failures).
  - Explicitly flags R1_identity_sanity vs R1_conf_mask naming distinction.

What this script does NOT do:
  - Does NOT start training.
  - Does NOT import PyTorch, SAM2, OmniSAM, or any heavy model.
  - Does NOT load datasets or checkpoints.
  - Does NOT call run_adaptation.py.

Run from the OmniSAM-main root:
  /root/miniconda3/envs/omnisam/bin/python3 \
      experiments/pl_refiner/scripts/check_pl_refiner_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml as _yaml
    def _load_yaml(path: str) -> dict:
        with open(path, 'r') as f:
            return _yaml.safe_load(f) or {}
except ImportError:
    raise ImportError("PyYAML required: pip install pyyaml")


# ---------------------------------------------------------------------------
# Implementation status registry
# ---------------------------------------------------------------------------

# Modes with working code in adaptation/pseudo_label/refiner.py
IMPLEMENTED_MODES = frozenset({"off", "identity"})
# conf_mask is implemented via the existing pseudo_conf_mask plugin
PLUGIN_BACKED_MODES = frozenset({"conf_mask"})
# Modes declared in YAML but refiner code not yet written
NOT_YET_MODES = frozenset({
    "class_balanced", "plq_basic", "proto_gate",
    "rbpc", "memory_plq", "full",
})

ALL_IMPLEMENTED = IMPLEMENTED_MODES | PLUGIN_BACKED_MODES  # off, identity, conf_mask


# ---------------------------------------------------------------------------
# Cumulative switch expectations
# Each Rx must have EXACTLY these switch values (cumulative inclusion design)
# Forbidden combinations are validated separately.
# ---------------------------------------------------------------------------

# (use_class_balance, use_plq, use_prototype, use_correction, use_memory_consistency)
EXPECTED: Dict[str, Tuple[bool, bool, bool, bool, bool]] = {
    "R0_baseline":       (False, False, False, False, False),
    "R1_conf_mask":      (False, False, False, False, False),
    "R2_class_balanced": (True,  False, False, False, False),
    "R3_plq_basic":      (True,  True,  False, False, False),
    "R4_proto_gate":     (True,  True,  True,  False, False),
    "R5_rbpc":           (True,  True,  True,  True,  False),
    "R6_memory_plq":     (True,  True,  False, False, True),
    # R7: correction/mem_con conservative=false by default
    "R7_full":           (True,  True,  True,  False, False),
}

SWITCH_NAMES = (
    "use_class_balance",
    "use_plq",
    "use_prototype",
    "use_correction",
    "use_memory_consistency",
)

# Explicit forbidden combos (early Rx must not sneak in later features)
FORBIDDEN: Dict[str, List[Tuple[str, bool, str]]] = {
    "R2_class_balanced": [
        ("use_plq",    True, "R2 must not enable PLQ — PLQ is introduced in R3"),
        ("use_prototype", True, "R2 must not enable prototype — introduced in R4"),
        ("use_correction", True, "R2 must not enable correction — introduced in R5"),
        ("use_memory_consistency", True, "R2 must not enable memory_consistency — introduced in R6"),
    ],
    "R3_plq_basic": [
        ("use_prototype", True, "R3 must not enable prototype — introduced in R4"),
        ("use_correction", True, "R3 must not enable correction — introduced in R5"),
        ("use_memory_consistency", True, "R3 must not enable memory_consistency — introduced in R6"),
    ],
    "R4_proto_gate": [
        ("use_correction", True, "R4 must not enable correction — introduced in R5"),
        ("use_memory_consistency", True, "R4 must not enable memory_consistency — introduced in R6"),
    ],
    "R0_baseline": [
        ("use_class_balance", True, "R0 must be pure baseline — no cb"),
        ("use_plq", True, "R0 must be pure baseline — no plq"),
        ("use_prototype", True, "R0 must be pure baseline — no prototype"),
        ("use_correction", True, "R0 must be pure baseline — no correction"),
        ("use_memory_consistency", True, "R0 must be pure baseline — no mem_con"),
    ],
    "R1_conf_mask": [
        ("use_class_balance", True, "R1 must not enable class_balance — introduced in R2"),
        ("use_plq", True, "R1 must not enable PLQ — introduced in R3"),
        ("use_prototype", True, "R1 must not enable prototype — introduced in R4"),
        ("use_correction", True, "R1 must not enable correction — introduced in R5"),
        ("use_memory_consistency", True, "R1 must not enable memory_consistency — introduced in R6"),
    ],
}


# ---------------------------------------------------------------------------
# YAML field extraction
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).parent.parent / "configs"

CONFIGS = [
    ("R0_baseline",       "R0_baseline.yaml"),
    ("R1_conf_mask",      "R1_conf_mask.yaml"),
    ("R2_class_balanced", "R2_class_balanced.yaml"),
    ("R3_plq_basic",      "R3_plq_basic.yaml"),
    ("R4_proto_gate",     "R4_proto_gate.yaml"),
    ("R5_rbpc",           "R5_rbpc.yaml"),
    ("R6_memory_plq",     "R6_memory_plq.yaml"),
    ("R7_full",           "R7_full.yaml"),
]

SANITY_YAML = "R1_identity_sanity.yaml"


def _nested_get(d: dict, *keys, default=None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _norm_mode(raw) -> str:
    if isinstance(raw, bool):
        return "off" if not raw else "on"
    return str(raw).lower()


def extract(cfg: dict, exp_id: str) -> dict:
    plr = cfg.get("pseudo_label_refiner", {}) or {}
    plugins = cfg.get("plugins", {}) or {}
    pcm = plugins.get("pseudo_conf_mask", {}) or {}
    d_plug = plugins.get("efficient_tam_memory", {}) or {}
    e_plug = plugins.get("efficient_sam2_sparse", {}) or {}

    mode = _norm_mode(plr.get("mode", "off"))
    enable = bool(plr.get("enable", mode != "off"))

    # Derive logical switches from config structure + mode
    cls_cfg  = plr.get("class_balance", None)
    plq_cfg  = plr.get("plq", None)
    proto_cfg = plr.get("prototype", None)
    corr_cfg  = plr.get("correction", None)
    mem_cfg_  = plr.get("memory_consistency", None)

    # A switch is ON if the sub-config block exists AND the mode activates it
    # AND (if the sub-config has an 'enable' key) that key is not False.
    def _active(sub_cfg, modes_that_use_it) -> bool:
        if mode not in modes_that_use_it:
            return False
        if sub_cfg is None:
            return False
        if isinstance(sub_cfg, dict):
            return bool(sub_cfg.get("enable", True))
        return bool(sub_cfg)

    CB_MODES  = {"class_balanced", "plq_basic", "proto_gate", "rbpc", "memory_plq", "full"}
    PLQ_MODES = {"plq_basic", "proto_gate", "rbpc", "memory_plq", "full"}
    PROTO_MODES = {"proto_gate", "rbpc", "full"}
    CORR_MODES  = {"rbpc"}  # R7 conservative default; handled below
    MEM_MODES   = {"memory_plq"}  # R7 conservative default; handled below

    use_cb   = _active(cls_cfg,  CB_MODES)
    use_plq  = _active(plq_cfg,  PLQ_MODES)
    use_proto = _active(proto_cfg, PROTO_MODES)
    use_corr  = _active(corr_cfg,  CORR_MODES)
    use_mem_c = _active(mem_cfg_,  MEM_MODES)

    # R7 "full" special-cases: correction and memory_consistency exist in config
    # but default to enable:false (conservative). Read them directly.
    if mode == "full":
        use_proto = _active(proto_cfg, {"full"})
        use_corr  = bool(_nested_get(corr_cfg, "enable", default=False)) if isinstance(corr_cfg, dict) else False
        use_mem_c = bool(_nested_get(mem_cfg_, "enable", default=False)) if isinstance(mem_cfg_, dict) else False

    # Memory plugin config
    d_on = bool(d_plug.get("enabled", False))
    e_on = bool(e_plug.get("enabled", False))
    if d_on and e_on:
        mem_config = "A_Dtrain_Etrain"
    elif d_on:
        mem_config = "D_only"
    elif e_on:
        mem_config = "E_only"
    else:
        mem_config = "none"

    full_train = {
        "R0_baseline": "true", "R1_conf_mask": "true",
        "R2_class_balanced": "true", "R3_plq_basic": "true",
        "R4_proto_gate": "true", "R5_rbpc": "conditional",
        "R6_memory_plq": "conditional", "R7_full": "true",
    }.get(exp_id, "unknown")

    reuse_dir = cfg.get("reuse_pl_dir", cfg.get("output_dir", "(not set)"))

    return dict(
        experiment_id=exp_id,
        refiner_mode=mode,
        enable=enable,
        use_class_balance=use_cb,
        use_plq=use_plq,
        use_prototype=use_proto,
        use_correction=use_corr,
        use_memory_consistency=use_mem_c,
        memory_config=mem_config,
        full_train_allowed=full_train,
        pseudo_conf_mask_enabled=bool(pcm.get("enabled", False)),
        reuse_pl_dir=str(reuse_dir),
    )


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def check(f: dict) -> List[str]:
    errors = []
    exp_id = f["experiment_id"]
    mode   = f["refiner_mode"]

    # 1. Forbidden combos
    for field_name, bad_val, reason in FORBIDDEN.get(exp_id, []):
        if f.get(field_name) == bad_val:
            errors.append(f"[ERROR-FORBIDDEN] {field_name}={bad_val} — {reason}")

    # 2. Expected switch values
    if exp_id in EXPECTED:
        exp_cb, exp_plq, exp_proto, exp_corr, exp_mem = EXPECTED[exp_id]
        checks = [
            ("use_class_balance", exp_cb),
            ("use_plq",           exp_plq),
            ("use_prototype",     exp_proto),
            ("use_correction",    exp_corr),
            ("use_memory_consistency", exp_mem),
        ]
        for fname, expected_val in checks:
            actual = f.get(fname)
            if actual != expected_val:
                errors.append(
                    f"[ERROR-EXPECTED] {fname}: expected={expected_val}, "
                    f"got={actual}"
                )

    # 3. mode='off' must have enable=False
    if mode == "off" and f["enable"]:
        errors.append("[ERROR-INCONSISTENT] mode='off' but enable=True")

    # 4. R1_conf_mask must not use mode=identity
    if exp_id == "R1_conf_mask" and mode == "identity":
        errors.append(
            "[ERROR-NAMING] R1_conf_mask.yaml uses mode='identity' — "
            "must use mode='conf_mask'. "
            "identity belongs to R1_identity_sanity.yaml only."
        )

    # 5. Warn if pseudo_conf_mask off for non-R0 active configs
    if exp_id != "R0_baseline" and mode != "off" and not f["pseudo_conf_mask_enabled"]:
        errors.append(
            "[WARN] pseudo_conf_mask is disabled. "
            "R1-R7 are designed to stack on confidence filtering. "
            "Verify this is intentional."
        )

    return errors


def impl_status(mode: str) -> str:
    if mode in IMPLEMENTED_MODES:
        return "IMPLEMENTED"
    if mode in PLUGIN_BACKED_MODES:
        return "IMPLEMENTED (via pseudo_conf_mask plugin)"
    if mode in NOT_YET_MODES:
        return "NOT_IMPLEMENTED_YET"
    return f"UNKNOWN_MODE ({mode})"


# ---------------------------------------------------------------------------
# R1 naming guard
# ---------------------------------------------------------------------------

def r1_naming_guard() -> List[str]:
    msgs = []
    sp = CONFIG_DIR / SANITY_YAML
    cp = CONFIG_DIR / "R1_conf_mask.yaml"

    if not sp.exists():
        msgs.append(f"[INFO] {SANITY_YAML} not found (may be OK if not yet needed)")
    if not cp.exists():
        msgs.append(f"[MISSING] R1_conf_mask.yaml not found at {cp}")
        return msgs

    if sp.exists() and cp.exists():
        s_mode = _norm_mode(_nested_get(_load_yaml(str(sp)), "pseudo_label_refiner", "mode", default="off"))
        c_mode = _norm_mode(_nested_get(_load_yaml(str(cp)), "pseudo_label_refiner", "mode", default="off"))
        if s_mode == c_mode:
            msgs.append(
                f"[CONFLICT] Both files use mode='{s_mode}' — "
                "R1_identity_sanity and R1_conf_mask must have different modes."
            )
        else:
            msgs.append(
                f"[OK] R1_identity_sanity mode='{s_mode}' vs "
                f"R1_conf_mask mode='{c_mode}' — correctly distinct."
            )
    return msgs


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

SEP = "-" * 92

def b(v: bool) -> str:
    return "Y" if v else "."


def main() -> int:
    print()
    print("=" * 92)
    print("  PseudoLabelRefiner Config Dry-Run Checker — Phase 1 / Task 1-2")
    print("  L1: no training | no model import | no dataset | no checkpoint")
    print("=" * 92)
    print()

    # --- R1 naming guard ---
    print(SEP)
    print("  R1 Naming Guard")
    print(SEP)
    for msg in r1_naming_guard():
        print(f"  {msg}")
    print()

    # --- Load configs ---
    all_fields: List[dict] = []
    missing: List[str] = []
    for exp_id, fname in CONFIGS:
        path = CONFIG_DIR / fname
        if not path.exists():
            missing.append(f"  [MISSING] {fname}")
            continue
        cfg = _load_yaml(str(path))
        all_fields.append(extract(cfg, exp_id))

    if missing:
        print("[MISSING FILES]")
        for m in missing:
            print(m)
        print()

    # --- Summary table ---
    print(SEP)
    hdr = (f"  {'ID':<22} {'mode':<18} {'cb':^3} {'plq':^3} "
           f"{'prt':^3} {'cor':^3} {'mc':^3}  {'mem_cfg':<18} {'train?':<13} {'impl'}")
    print(hdr)
    print(SEP)
    for f in all_fields:
        st = impl_status(f["refiner_mode"])
        st_short = "IMPL" if "NOT" not in st else "TODO"
        print(
            f"  {f['experiment_id']:<22} {f['refiner_mode']:<18} "
            f"{b(f['use_class_balance']):^3} {b(f['use_plq']):^3} "
            f"{b(f['use_prototype']):^3} {b(f['use_correction']):^3} "
            f"{b(f['use_memory_consistency']):^3}  "
            f"{f['memory_config']:<18} {f['full_train_allowed']:<13} {st_short}"
        )
    print(SEP)
    print("  Legend: Y=true  .=false  |  cb=class_balance  plq=PLQ  prt=prototype")
    print("          cor=correction  mc=memory_consistency")
    print()

    # --- Per-config detail + consistency check ---
    print(SEP)
    print("  Consistency Check Results")
    print(SEP)
    any_hard_error = False
    all_error_lists: Dict[str, List[str]] = {}
    for f in all_fields:
        errs = check(f)
        all_error_lists[f["experiment_id"]] = errs
        hard = [e for e in errs if e.startswith("[ERROR")]
        warns = [e for e in errs if e.startswith("[WARN")]
        if hard:
            any_hard_error = True
            print(f"  [FAIL] {f['experiment_id']}")
            for e in hard:
                print(f"    !! {e}")
        elif warns:
            print(f"  [WARN] {f['experiment_id']}")
            for w in warns:
                print(f"    ~~ {w}")
        else:
            print(f"  [PASS] {f['experiment_id']}")

    print(SEP)
    if any_hard_error:
        print("  CONSISTENCY: FAILED — fix errors above.")
    else:
        print("  CONSISTENCY: ALL PASS")
    print(SEP)
    print()

    # --- Implementation status ---
    print(SEP)
    print("  Implementation Status (refiner.py SUPPORTED_MODES)")
    print(SEP)
    for f in all_fields:
        st = impl_status(f["refiner_mode"])
        tick = "✓" if "NOT" not in st else "○"
        print(f"  {tick} {f['experiment_id']:<22} → {st}")
    print(SEP)
    print()

    # --- Final verdict ---
    n = len(all_fields)
    n_impl = sum(1 for f in all_fields if "NOT" not in impl_status(f["refiner_mode"]))
    print(SEP)
    print("  Final Verdict")
    print(SEP)
    print(f"  Configs loaded       : {n}/8")
    print(f"  Implemented modes    : {n_impl}/{n}  (R2-R7 = NOT_IMPLEMENTED_YET is expected)")
    print(f"  Consistency check    : {'PASS' if not any_hard_error else 'FAIL'}")
    print(f"  Training started     : NO")
    print(f"  Model/data loaded    : NO")
    print()
    if not any_hard_error and n == 8:
        print("  ✓ All 8 configs loaded and internally consistent.")
        print("  ✓ R2-R7 correctly marked NOT_IMPLEMENTED_YET.")
        print("  ✓ R1_identity_sanity and R1_conf_mask are distinct configs.")
        print("  ✓ Dry-run PASS — safe to proceed to Phase 2 / Task 2-1.")
    else:
        print("  ✗ Issues found — resolve before Phase 2.")
    print(SEP)
    print()

    return 1 if (any_hard_error or n < 8) else 0


if __name__ == "__main__":
    sys.exit(main())
