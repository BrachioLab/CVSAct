---
pretty_name: CVS-Act
license: other
tags:
  - v1.0.0
language:
  - en
task_categories:
  - text-classification
  - token-classification
size_categories:
  - 1K<n<10K
configs:
  - config_name: sages_trained_annotator
    data_files:
      - split: test
        path: sages_trained_annotator/test.jsonl
  - config_name: sages_synthetic
    data_files:
      - split: test
        path: sages_synthetic/test.jsonl
      - split: train
        path: sages_synthetic/train.jsonl
---

# CVS-Act: Action Recommendation for Critical View of Safety Assessment

## Dataset Description

`CVS-Act` is a surgical action recommendation dataset for laparoscopic cholecystectomy grounded in Critical View of Safety (CVS) assessment. Each example corresponds to a CVS transition example and contains structured action recommendations over the current task label space for the left instrument, right instrument, and camera, with the original `other` actor field preserved when present.

The current release packages the task-ready simple-action examples that the project notebooks and the current SurGent/CoT evaluation use, rather than the raw nested audit annotation files.

Release `v1.0.0` was released 2026-05-31, generated with Qwen3.6-35B-A3B pipeline.

## Configs / Provenance Subsets

The configs in this repo are **annotation provenance + source-dataset subsets**, not ordinary train/test splits:

- `sages_trained_annotator`: labels derived from the current `audit_v11` trained-annotator audit examples from `SAGES_CVS_Challenge_2024`
- `sages_synthetic`: synthetic labels from the v3.1 Qwen3.6-35B-A3B pipeline export from `SAGES_CVS_Challenge_2024`

The current release does **not** include a `surgeon_annotated` config because no surgeon-annotated export exists in the inspected workspace. A future release can add that config without changing the overall repo structure.

Intended naming convention for this repo:

- `sages_synthetic`: contains `train` and `test`; may later add `validation`
- `sages_trained_annotator`: currently `test`
- `sages_surgeon_annotated`: future `test`

## Splits

`sages_synthetic` exposes `train` and `test` splits from the v3.1 synthetic export. `sages_trained_annotator` currently exposes `test` only because the trained-annotator examples come from the `SAGES_CVS_Challenge_2024` test portion and cover `first_30_videos_from_test`.

## Current Release Summary

- `sages_trained_annotator/test`: 90 examples from 30 videos
- `sages_synthetic/train`: 1312 examples from 432 videos
- `sages_synthetic/test`: 481 examples from 168 videos

Criterion counts:

- `sages_trained_annotator/test`: {'C1': 26, 'C2': 44, 'C3': 20}
- `sages_synthetic/train`: {'C1': 359, 'C2': 560, 'C3': 393}
- `sages_synthetic/test`: {'C1': 129, 'C2': 244, 'C3': 108}

## Files

```text
taxonomy/action_taxonomy.json
sages_trained_annotator/test.jsonl
sages_synthetic/train.jsonl
sages_synthetic/test.jsonl
```

## Schema

Each JSONL row contains:

- `example_id`: stable example identifier
- `video_id`: source video identifier
- `criterion`: CVS criterion, currently one of `C1`, `C2`, `C3`
- `label_source`: provenance label, currently `trained_annotator` or `synthetic`
- `config_name`: exported Hugging Face config name for this row
- `taxonomy_version`: current reduced task taxonomy version for this release
- `internal_source_release`: internal source tag, currently `audit_v11`
- `frame_start`, `frame_end`: frame span for the example
- `mind_change`: CVS transition label when available
- `source_dataset`, `source_dataset_name`, `source_split`, `source_subset_note`: source-corpus provenance fields
- `left_actions`, `right_actions`, `camera_actions`, `other_actions`: per-actor structured action segments
- `source_file`: local source artifact path used for export
- `source_annotation_path`: original audit annotation path when available
- `synthetic_generation_method`, `synthetic_generation_model`, `synthetic_generation_notebook`, `paired_trained_annotator_source`: synthetic provenance fields for the synthetic config

The actor action lists preserve the original nested fields from the current simple-action exports, including tool type, action code, target structure, target context, rank, confidence, evidence, and description when those fields exist.

For provenance clarity, this public release is explicitly linked to the internal source release tag `audit_v11` in each row and in the taxonomy metadata.

## Taxonomy

The reduced task taxonomy is stored at `taxonomy/action_taxonomy.json`. This is **not** the full `taxonomy_v10.json`. It contains only the current left/right/camera label space used by the simple-action notebooks and by the current SurGent/CoT evaluation setup.

Current taxonomy summary:

- `left.retraction_direction_code`: ['KEEP_RETRACT_LATERAL', 'KEEP_RETRACT_MEDIAL', 'KEEP_RETRACT_UPWARD', 'RETRACT_LATERAL', 'RETRACT_MEDIAL', 'RETRACT_UPWARD', 'RETRACT_LATERAL_TO_MEDIAL', 'RETRACT_LATERAL_TO_UPWARD', 'RETRACT_MEDIAL_TO_LATERAL', 'RETRACT_MEDIAL_TO_UPWARD', 'RETRACT_UPWARD_TO_LATERAL', 'RETRACT_UPWARD_TO_MEDIAL']
- `right.action_code`: ['CLIP', 'COAGULATE_HEMOSTASIS', 'COUNTERTRACTION_ASSIST', 'DISSECT', 'IRRIGATOR_ASPIRATE', 'RETRACT_DOWNWARD', 'SWEEPING', 'TOOL_WITHDRAW_UNBLOCKS_VIEW']
- `camera.action_code`: ['CAMERA_NO_CHANGE', 'CAMERA_REPOSITION', 'CAMERA_UNCERTAIN', 'CAMERA_ZOOM_IN', 'CAMERA_ZOOM_OUT']

## Intended Use

This dataset is intended for research on:

- surgical video understanding
- CVS-grounded safety assessment
- action recommendation and structured decision support

## Limitations

- The current release contains trained-annotator and synthetic labels only; it does not contain surgeon-annotated evaluation data.
- The current release includes synthetic `train` and `test` splits, but trained-annotator labels remain test-only.
- Synthetic labels reflect the current notebook export pipeline and may inherit model-specific biases or taxonomy simplifications.
- This dataset is for research use and benchmarking only.

## Ethical Use / Restrictions

This dataset must not be used for direct clinical deployment, autonomous intraoperative decision-making, or real-time patient care. Any model trained on this dataset should be treated as a research artifact requiring careful human oversight and external validation.

## Loading Examples

Local path:

```python
from datasets import load_dataset

trained = load_dataset("/absolute/path/to/hf_repos/cvs-act", "sages_trained_annotator")
synthetic = load_dataset("/absolute/path/to/hf_repos/cvs-act", "sages_synthetic")
```

After upload to Hugging Face:

```python
from datasets import load_dataset

trained = load_dataset("BrachioLab/cvs-act", "sages_trained_annotator", revision="v1.0.0")
synthetic = load_dataset("BrachioLab/cvs-act", "sages_synthetic", revision="v1.0.0")
```

## Citation

TODO: add project citation / paper citation.

## License

TODO: replace `other` with the actual redistribution license once confirmed.
