# CVS-Act Processed Data

This directory contains the processed CVS-Act labels, splits, synthetic labels, and surgeon-validation sample used by the final pipeline.

## Human Annotation

Final trained-annotator version: [`cvs_act_annotations/v1/audit_v11/`](cvs_act_annotations/v1/audit_v11/)

Converted action labels:

- [`../../../notebooks/artifacts/audit_v11_simple_action_gt/audit_v11_simple_actions.json`](../../../notebooks/artifacts/audit_v11_simple_action_gt/audit_v11_simple_actions.json)
- [`../../../notebooks/artifacts/audit_v11_simple_action_gt/audit_v11_simple_actions_flat.csv`](../../../notebooks/artifacts/audit_v11_simple_action_gt/audit_v11_simple_actions_flat.csv)

Counts:

| Source | Videos | Rows | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|
| `audit_v11_simple_actions_flat.csv` action rows | 30 | 285 | 81 | 135 | 69 |
| Hugging Face `sages_trained_annotator/test.jsonl` criterion records | 30 | 90 | 26 | 44 | 20 |

Action rows by actor in `audit_v11_simple_actions_flat.csv`:

| Actor | Rows |
|---|---:|
| right | 126 |
| left | 83 |
| camera | 72 |
| other | 4 |

Previous audit versions are kept because later annotation rounds depend on earlier ones:

| Version | JSON files |
|---|---:|
| `audit_v1` | 21 |
| `audit_v2` | 30 |
| `audit_v3` | 30 |
| `audit_v4` | 6 |
| `audit_v5` | 30 |
| `audit_v6` | 30 |
| `audit_v7` | 30 |
| `audit_v8` | 30 |
| `audit_v9` | 30 |
| `audit_v10` | 30 |
| `audit_v10_camera_segments` | 30 |
| `audit_v10_right_tools` | 10 |
| `audit_v11` | 30 |

## Human Dev/Test Split

Split file: [`../../../notebooks/artifacts/qwen3.5_audit_v11_current_best_prompt_summary/validation_test_video_split_seed20260610.csv`](../../../notebooks/artifacts/qwen3.5_audit_v11_current_best_prompt_summary/validation_test_video_split_seed20260610.csv)

Split seed: `20260610`

| Split | Videos | Action rows | HF criterion records | C1 records | C2 records | C3 records |
|---|---:|---:|---:|---:|---:|---:|
| validation/dev | 10 | 95 | 30 | 6 | 18 | 6 |
| test | 20 | 190 | 60 | 20 | 26 | 14 |

The final human-vs-synthetic correlation evaluation uses the 20-video test split.

## Synthetic Labels

Final synthetic version: [`cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/`](cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/)

| File | Videos | Records | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|
| `train_full/synthetic_audit_v11_simple_actions.json` | 432 | 1312 | 359 | 560 | 393 |
| `test_full/synthetic_audit_v11_simple_actions.json` | 168 | 481 | 129 | 244 | 108 |
| `train_val_split/train.json` | 367 | 1119 | 310 | 475 | 334 |
| `train_val_split/val.json` | 65 | 193 | 49 | 85 | 59 |

Train/validation split metadata:

- [`cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/train_val_split/split_summary.json`](cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/train_val_split/split_summary.json)
- random seed: `3407`
- validation fraction: `0.15`

## Hugging Face Export

Local export: [`../../../hf_repos/cvs-act/`](../../../hf_repos/cvs-act/)

| Config | Split | Videos | Records | C1 | C2 | C3 |
|---|---|---:|---:|---:|---:|---:|
| `sages_trained_annotator` | test | 30 | 90 | 26 | 44 | 20 |
| `sages_synthetic` | train | 432 | 1312 | 359 | 560 | 393 |
| `sages_synthetic` | test | 168 | 481 | 129 | 244 | 108 |

Export entry point:

- [`../../../scripts/export_hf_cvs_act.py`](../../../scripts/export_hf_cvs_act.py)

## Expert / Surgeon Validation

Surgeon-validation data: [`cvs_act_surgeon_annotations/`](cvs_act_surgeon_annotations/)

The surgeon sample is separate from the human dev/test split.

| File | Videos | Rows | Notes |
|---|---:|---:|---|
| [`cvs_act_surgeon_annotations/sample_30.csv`](cvs_act_surgeon_annotations/sample_30.csv) | 30 | 30 | sampled videos |
| [`cvs_act_surgeon_annotations/selected_video_clips.csv`](cvs_act_surgeon_annotations/selected_video_clips.csv) | 30 | 30 | one selected coarse clip per sampled video |
| [`cvs_act_surgeon_annotations/clip_manifest.csv`](cvs_act_surgeon_annotations/clip_manifest.csv) | 30 | 168 | full candidate clip manifest |

Selected surgeon clips by criterion:

| Criterion | Clips |
|---|---:|
| C1 | 6 |
| C2 | 15 |
| C3 | 9 |

Selected clip seed: `20260617`

Asset-generation entry point:

- [`../../../scripts/data/prepare_surgeon_annotation_assets.py`](../../../scripts/data/prepare_surgeon_annotation_assets.py)

Generated frame/video assets are local-only and ignored by Git:

- `cvs_act_surgeon_annotations/aws_upload_assets/`
- `cvs_act_surgeon_annotations/aws_upload_assets.zip`

## Final Correlation Artifact

Final 20-video test correlation artifact:

- [`../../../notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_six_system_human_synthetic_final_score_correlation_test20.csv`](../../../notebooks/artifacts/cvs_act_human_synthetic_correlation_recomputed_details/recomputed_six_system_human_synthetic_final_score_correlation_test20.csv)

Reported final-score correlation over 6 evaluated systems:

| Metric | Value |
|---|---:|
| Pearson | 0.947 |
| Spearman | 0.886 |
