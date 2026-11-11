# Archived Experiment Scripts (Provenance Record)

These scripts are **byte-identical copies of scripts recovered from the temporary scratch folders** of two earlier AI-assisted working sessions (9–10 Nov 2026). They are kept unchanged as a historical audit trail. Their link to specific checkpoints and JSON results in `artifacts/` is **attributed**, not independently verified: it rests on the output filenames written in the code and on local file timestamps (see [Evidence status](#evidence-status)). The scripts were not re-executed.

> [!WARNING]
> **Do not execute these scripts as-is.**
> - They contain hardcoded absolute Windows paths (`D:\PAEP_AI_Program\...`, `C:\Users\...\.gemini\antigravity\brain\...\scratch`) and import `experiment_framework` from the original scratch folder via `sys.path.append`.
> - Several scripts write directly into this repository's `artifacts/` folder and would **overwrite the published checkpoints and JSON results**.
> - Some scripts evaluate on the **Test split** (marked below); re-running them would further expose the Test set.
> - They were **not re-executed** after recovery, so it is not proven that these exact file versions created the artifacts attributed to them.

## Data dependency

All scripts read the classroom copy of the Kermany et al. (2018) pediatric CXR dataset prepared by the course kit (`Pneumonia_Chest_XRay_Kit/data/chest_xray_3class`, created by the kit's `prepare_three_classes.py`). The dataset is not distributed with this repository.

## Script → artifact map

### `session_89408523/` (9–10 Nov)

| Script | Role | Attributed output (from code) | Touches Test split |
| :--- | :--- | :--- | :---: |
| `experiment_framework.py` | Training procedure for MobileNetV2 variants (`run_experiment`: frozen head, class weights, cosine LR, top-N feature-block fine-tuning, early stopping on validation loss) | `artifacts/model_<exp_name>.pt` (relative path) | No |
| `multi_seed_validation.py` | Seeds 42/123/456 for Exp 1, Exp 2, Exp 5 (configs at lines 10–43) | `model_exp1_*_s*.pt`, `model_exp2_*_s*.pt`, `model_exp5_*_s*.pt` | No |
| `run_exp6.py` | Exp 6: stage-2 fine-tuning starting from Exp 2 | `model_exp6_stage2_finetune.pt` | No |
| `aggregate_multiseed.py`, `audit_and_recalc_metrics.py`, `patient_clustered_stats.py`, `comprehensive_clinical_audit.py`, `run_rigorous_stats.py` | Multi-seed aggregation, metric recomputation and patient-group-clustered statistics on validation | Console output | No |
| `verify_true_patients.py`, `check_*.py`, `inspect_*.py`, `audit_patient_patterns.py`, `compare_pair_images.py`, `fast_inspect_overlap.py`, `debug_baseline_cm.py` | Filename-based patient-group and split-overlap checks; checkpoint-metadata inspection; validation-set debugging | Console output | Test filenames only, no Test inference (except `inspect_normal_discrepancy.py`, listed below) |
| `eval_test.py` | Test-set evaluation of single checkpoints | Console output | **Yes** |
| `inspect_normal_discrepancy.py` | Analysis of the validation–test Normal-recall gap | Console output | **Yes** |

The driver that launched the original single-seed runs (Exp 0–Exp 5 unsuffixed checkpoints, including Exp 3 and Exp 4a) was not recovered; `experiment_framework.run_experiment` defines their training procedure.

### `session_560df2be/` (10 Nov)

| Script | Role | Attributed output (from code) | Touches Test split |
| :--- | :--- | :--- | :---: |
| `run_experiment_a.py` | Exp A: Control vs. CLAHE training (3 seeds) | `exp_a_control_s*.pt`, `exp_a_clahe_s*.pt` | No |
| `eval_and_bootstrap_exp_a.py` | Exp A evaluation and paired patient-group bootstrap | `experiment_a_validated_results.json` | No |
| `run_experiment_b.py` | Exp B: focal loss (γ = 2) vs. weighted CE (3 seeds) and bootstrap | `exp_b_focal_s*.pt`, `experiment_b_focal_results.json` | No |
| `run_experiment_c.py` | Exp C: frozen DenseNet-121 vs. frozen MobileNetV2 (3 seeds) | `exp_c_densenet_s*.pt` | No |
| `eval_and_bootstrap_exp_c.py`, `verify_and_reconcile_exp_c.py` | Exp C evaluation, bootstrap and reconciliation (last writer of the JSON) | `experiment_c_densenet_results.json` | No |
| `eval_test_multiseed.py` | Test-set evaluation of multi-seed checkpoints and ensembles | Console output | **Yes** |
| `test_threshold_calibration.py` | Class-probability reweighting fitted on validation, then evaluated on Test. **Not adopted**: the published models use plain argmax. | Console output | **Yes** |

## Patient-group definition used by the bootstrap

`get_true_patient_id` / `get_patient_id` (e.g. `session_560df2be/eval_and_bootstrap_exp_a.py`, lines 48–59) groups pneumonia images by `person<ID>_bacteria` or `person<ID>_virus` and Normal images by the filename without its last `-` segment. Applied to the classroom split this yields 2,837 / 714 / 432 groups (train / val / test). These are filename-derived groups, not verified patient identifiers.

## Test-set results

Test-set figures quoted in the main README are attributed to the console output of the **Yes** scripts above and of the original notebook. They were not saved as artifacts and cannot be checked against stored evidence.

## Evidence status

**Independently verified (2026-11-11, without re-running any script):**
- Each archived file is byte-identical (SHA-256) to its recovered scratch-folder original.
- The output filenames written in the code match the artifact names listed above.
- The patient-group function, applied to the classroom split's filenames, reproduces the reported 2,837 / 714 / 432 groups.
- The Exp A control checkpoints (`exp_a_control_s*.pt`) contain weights identical to `model_exp2_class_weighted_head_s*.pt` for all three seeds.

**Attributed only (not verified):**
- That these exact file versions produced the attributed checkpoints and JSON results.
- All Test-set figures.

**Supporting filesystem timestamps (weak evidence).** Recorded on 2026-11-11 from the local filesystem. They are mutable and Git does not preserve them; in every pair the script's last modification precedes the artifact's, which is consistent with the attribution but does not prove it.

| Script (last modified) | Attributed artifact (last modified) |
| :--- | :--- |
| `experiment_framework.py` (11-09 02:55) | `model_exp5_finetune_top4blocks.pt` (11-09 03:11) |
| `run_exp6.py` (11-09 03:12) | `model_exp6_stage2_finetune.pt` (11-09 03:15) |
| `multi_seed_validation.py` (11-09 03:39) | `model_exp5_finetune_top4blocks_s42.pt` (11-09 16:26) |
| `run_experiment_a.py` (11-10 08:02) | `exp_a_clahe_s42.pt` (11-10 08:15) |
| `eval_and_bootstrap_exp_a.py` (11-10 08:23) | `experiment_a_validated_results.json` (11-10 08:24) |
| `run_experiment_b.py` (11-10 08:50) | `experiment_b_focal_results.json` (11-10 08:57) |
| `run_experiment_c.py` (11-10 15:22) | `exp_c_densenet_s42.pt` (11-10 15:28) |
| `verify_and_reconcile_exp_c.py` (11-10 16:40) | `experiment_c_densenet_results.json` (11-10 16:42) |

## Privacy note

The scripts contain hardcoded local paths that reveal the author's Windows username, local directory layout and the AI tool's scratch-folder path. They contain no credentials, tokens or personal data beyond this. The paths are retained deliberately so that the files stay byte-identical to the recovered originals.
