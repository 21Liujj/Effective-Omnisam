# TeCMA Config Check Report

Generated: 2026-06-17 10:31:27  
Config dir: `experiments/tecma/configs`  
Configs checked: 11  

---

## Overview Matrix

| Config | A | D_en | D_train | D_PL | E_en | E_train | E_PL |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `A_Dtrain` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `A_Dtrain_Etrain` | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `A_Dtrain_Etrain_PLon` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `A_Etrain` | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `A_Etrain_keep025` | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `A_Etrain_keep075` | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `A_ccpm` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `Dtrain_Etrain` | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| `Dtrain_fsmc` | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `Etrain_sgsmr` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `baseline_omnisam` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## Acceptance Checks

> Key configs validated against expected switch state.

- ✅ `A_Dtrain_Etrain`.A_enabled = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain`.D_train = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain`.E_train = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain`.D_pl = `False` (expected `False`)
- ✅ `A_Dtrain_Etrain`.E_pl = `False` (expected `False`)
- ✅ `baseline_omnisam`.A_enabled = `False` (expected `False`)
- ✅ `baseline_omnisam`.D_enabled = `False` (expected `False`)
- ✅ `baseline_omnisam`.E_enabled = `False` (expected `False`)
- ✅ `A_Dtrain_Etrain_PLon`.A_enabled = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain_PLon`.D_train = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain_PLon`.E_train = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain_PLon`.D_pl = `True` (expected `True`)
- ✅ `A_Dtrain_Etrain_PLon`.E_pl = `True` (expected `True`)
- ✅ `A_ccpm`.A_enabled = `True` (expected `True`)
- ✅ `A_ccpm`.D_enabled = `False` (expected `False`)
- ✅ `A_ccpm`.E_enabled = `False` (expected `False`)
- ✅ `Dtrain_fsmc`.A_enabled = `False` (expected `False`)
- ✅ `Dtrain_fsmc`.D_enabled = `True` (expected `True`)
- ✅ `Dtrain_fsmc`.D_train = `True` (expected `True`)
- ✅ `Dtrain_fsmc`.D_pl = `False` (expected `False`)
- ✅ `Dtrain_fsmc`.E_enabled = `False` (expected `False`)
- ✅ `Etrain_sgsmr`.A_enabled = `False` (expected `False`)
- ✅ `Etrain_sgsmr`.D_enabled = `False` (expected `False`)
- ✅ `Etrain_sgsmr`.E_enabled = `True` (expected `True`)
- ✅ `Etrain_sgsmr`.E_train = `True` (expected `True`)
- ✅ `Etrain_sgsmr`.E_pl = `False` (expected `False`)
- ✅ `A_Etrain_keep075`.A_enabled = `True` (expected `True`)
- ✅ `A_Etrain_keep075`.D_enabled = `False` (expected `False`)
- ✅ `A_Etrain_keep075`.E_enabled = `True` (expected `True`)
- ✅ `A_Etrain_keep075`.E_train = `True` (expected `True`)
- ✅ `A_Etrain_keep075`.E_pl = `False` (expected `False`)
- ✅ `A_Etrain_keep075`.E_keep_ratio = `0.75` (expected `0.75`)
- ✅ `A_Etrain_keep025`.A_enabled = `True` (expected `True`)
- ✅ `A_Etrain_keep025`.D_enabled = `False` (expected `False`)
- ✅ `A_Etrain_keep025`.E_enabled = `True` (expected `True`)
- ✅ `A_Etrain_keep025`.E_train = `True` (expected `True`)
- ✅ `A_Etrain_keep025`.E_pl = `False` (expected `False`)
- ✅ `A_Etrain_keep025`.E_keep_ratio = `0.25` (expected `0.25`)

---

## Per-Config Details

### `A_Dtrain`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ✅ ON |
| D apply_in_train   | ✅ ON |
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ❌ off|
| E apply_in_train   | ❌ off|
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `A_Dtrain_Etrain`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ✅ ON |
| D apply_in_train   | ✅ ON |
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `A_Dtrain_Etrain_PLon`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ✅ ON |
| D apply_in_train   | ✅ ON |
| D apply_in_pl      | ✅ ON |
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ✅ ON |
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `A_Etrain`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ❌ off|
| D apply_in_train   | ❌ off|
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `A_Etrain_keep025`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ❌ off|
| D apply_in_train   | ❌ off|
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.25        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `A_Etrain_keep075`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ❌ off|
| D apply_in_train   | ❌ off|
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.75        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `A_ccpm`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ✅ ON |
| D (FSMC) enabled   | ❌ off|
| D apply_in_train   | ❌ off|
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ❌ off|
| E apply_in_train   | ❌ off|
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `Dtrain_Etrain`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ❌ off|
| D (FSMC) enabled   | ✅ ON |
| D apply_in_train   | ✅ ON |
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `Dtrain_fsmc`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ❌ off|
| D (FSMC) enabled   | ✅ ON |
| D apply_in_train   | ✅ ON |
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ❌ off|
| E apply_in_train   | ❌ off|
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `Etrain_sgsmr`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ❌ off|
| D (FSMC) enabled   | ❌ off|
| D apply_in_train   | ❌ off|
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ✅ ON |
| E apply_in_train   | ✅ ON |
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

### `baseline_omnisam`

| Field              | Value                        |
|--------------------|------------------------------|
| dataset            | (CLI arg)             |
| pd_root            | (CLI arg)             |
| A (CCPM) enabled   | ❌ off|
| D (FSMC) enabled   | ❌ off|
| D apply_in_train   | ❌ off|
| D apply_in_pl      | ❌ off|
| D compress_ratio   | 0.5          |
| D pool_stride      | 2       |
| E (SG-SMR) enabled | ❌ off|
| E apply_in_train   | ❌ off|
| E apply_in_pl      | ❌ off|
| E keep_ratio       | 0.5        |
| E min_keep_tokens  | 64        |
| **Validation**     | ✅ OK                    |

---

## Summary

| Item | Count |
|------|-------|
| Configs checked | 11 |
| Configs with errors | 0 |
| Acceptance checks | 38 |
| Acceptance failures | 0 |

**Result: ✅ All checks passed.**

