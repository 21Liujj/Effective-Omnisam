#!/usr/bin/env python3
"""experiments/pl_refiner/scripts/sanity_identity_refiner.py

Sanity check for PseudoLabelRefiner mode=off vs mode=identity.

Usage (on the remote server, in conda env omnisam):
  cd /root/autodl-tmp/OmniSAM-main
  python experiments/pl_refiner/scripts/sanity_identity_refiner.py

What this does:
  1. L1 dry-run: verifies refiner config is readable and mode is correct.
  2. Imports PseudoLabelRefiner and calls refine() with dummy tensors.
  3. Checks mode=off returns None.
  4. Checks mode=identity returns identical label + mask.
  5. Prints PASS/FAIL for each assertion.

This script does NOT train. For L3 short-run, use run_adaptation.py with
  --config experiments/pl_refiner/configs/R0_baseline.yaml  (mode=off)
  --config experiments/pl_refiner/configs/R1_identity_sanity.yaml  (mode=identity)
and compare the first 5 iterations' loss_unsup and pseudo_valid_ratio.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch

# -- L1 dry-run: import check ------------------------------------------
print('[L1] Importing PseudoLabelRefiner ...')
try:
    from adaptation.pseudo_label import PseudoLabelRefiner, PseudoLabelRefineResult
    print('[L1] Import OK')
except Exception as e:
    print(f'[L1] FAIL: {e}')
    sys.exit(1)

# -- mode=off: must return None ----------------------------------------
print('[L1] Testing mode=off ...')
r_off = PseudoLabelRefiner(cfg={'mode': 'off'})
assert not r_off.is_active(), 'FAIL: is_active() should be False for mode=off'
B, H, W = 2, 64, 64
pl = torch.randint(0, 13, (B, H, W))
conf = torch.rand(B, H, W)
result = r_off.refine(pseudo_label=pl, confidence=conf)
assert result is None, f'FAIL: mode=off must return None, got {result}'
print('[L1] mode=off PASS: returns None')

# -- mode=identity: must return unchanged tensors ----------------------
print('[L1] Testing mode=identity ...')
r_id = PseudoLabelRefiner(cfg={'mode': 'identity'})
assert r_id.is_active(), 'FAIL: is_active() should be True for mode=identity'
valid = (pl != 255)
result = r_id.refine(pseudo_label=pl, confidence=conf, valid_mask=valid)
assert result is not None, 'FAIL: mode=identity must not return None'
assert isinstance(result, PseudoLabelRefineResult), 'FAIL: wrong return type'
assert (result.refined_label is pl), 'FAIL: refined_label must be same object'
assert (result.valid_mask is valid), 'FAIL: valid_mask must be same object'
assert result.pl_weight is None, 'FAIL: pl_weight must be None in identity mode'
assert 'valid_ratio' in result.diagnostics, 'FAIL: diagnostics missing valid_ratio'
assert 'ignore_ratio' in result.diagnostics, 'FAIL: diagnostics missing ignore_ratio'
assert result.diagnostics['mode'] == 'identity'
expected_valid_ratio = float(valid.float().mean().item())
assert abs(result.diagnostics['valid_ratio'] - expected_valid_ratio) < 1e-6, \
    f'FAIL: valid_ratio mismatch {result.diagnostics["valid_ratio"]} vs {expected_valid_ratio}'
print('[L1] mode=identity PASS: label/mask unchanged, diagnostics correct')

# -- mode=unknown: must raise ValueError --------------------------------
print('[L1] Testing unknown mode raises ValueError ...')
try:
    PseudoLabelRefiner(cfg={'mode': 'class_balanced'})
    print('[L1] FAIL: should have raised ValueError')
    sys.exit(1)
except ValueError as e:
    print(f'[L1] ValueError raised correctly: {str(e)[:80]}')

# -- from_args factory --------------------------------------------------
print('[L1] Testing from_args factory ...')
class FakeArgs:
    pseudo_label_refiner = {'mode': 'identity'}
r_fa = PseudoLabelRefiner.from_args(FakeArgs(), num_classes=13)
assert r_fa.mode == 'identity'
class FakeArgsOff:
    pseudo_label_refiner = None
r_fa2 = PseudoLabelRefiner.from_args(FakeArgsOff(), num_classes=13)
assert r_fa2.mode == 'off', f'FAIL: default mode should be off, got {r_fa2.mode}'
print('[L1] from_args factory PASS')

print()
print('======================================')
print('ALL L1 ASSERTIONS PASSED')
print('======================================')
print()
print('Next: L3 short-run (5 iters each, run by user):')
print('  # R0 mode=off (baseline):')
print('  python run_adaptation.py --config experiments/pl_refiner/configs/R0_baseline.yaml')
print('  # R1 mode=identity:')
print('  python run_adaptation.py --config experiments/pl_refiner/configs/R1_identity_sanity.yaml')
print('Compare: loss_unsup and pseudo_valid_ratio in first 5 iterations.')
print('Both should be numerically identical (same PL, same mask_t, same loss path).')
