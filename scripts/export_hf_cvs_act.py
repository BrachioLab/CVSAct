#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = ROOT / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1"
TRAINED_ANNOTATOR_SIMPLE_PATH = (
    ROOT / "notebooks/artifacts/audit_v11_simple_action_gt/audit_v11_simple_actions.json"
)
SYNTHETIC_TEST_SIMPLE_PATH = (
    ROOT
    / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/test_full/synthetic_audit_v11_simple_actions.json"
)
SYNTHETIC_TRAIN_SIMPLE_PATH = (
    ROOT
    / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_synthetic_data/audit_v11_cvsctx_v3_1/train_full/synthetic_audit_v11_simple_actions.json"
)
SYNTHETIC_NOTEBOOK_PATH = (
    ROOT
    / "notebooks/qwen3.5_audit_v11_seg_extract_spec_eval/qwen3.5_audit_v11_seg_extract_spec_eval_cvsctx_v3_1.ipynb"
)
CURRENT_TAXONOMY_NOTEBOOK_PATH = (
    ROOT
    / "notebooks/qwen3.5_audit_v11_seg_extract_spec_eval/qwen3.5_audit_v11_current_taxonomy_latex.ipynb"
)
AUDIT_V11_DIR = ANNOTATION_ROOT / "audit_v11"
OUT_DIR = ROOT / "hf_repos/cvs-act"

DATASET_TITLE = "CVS-Act: Action Recommendation for Critical View of Safety Assessment"
DATASET_NAME = "CVS-Act"
DATASET_REPO_NAME = "cvs-act"
PUBLIC_RELEASE_VERSION = "v1.0.0"
PUBLIC_RELEASE_DATE = "2026-05-31"
TAXONOMY_VERSION = "cvs_act_current_simple_v1"
INTERNAL_SOURCE_RELEASE = "audit_v11"
SYNTHETIC_DEFAULT_METHOD = "structured_prediction_cvs_context"
SYNTHETIC_GENERATION_MODEL = "Qwen3.6-35B-A3B pipeline"
SOURCE_DATASET = "CVS_Challenge_SAGES_v1"
SOURCE_DATASET_NAME = "SAGES_CVS_Challenge_2024"
SOURCE_CONFIG_PREFIX = "sages"
TRAINED_ANNOTATOR_SOURCE_SPLIT = "test"
TRAINED_ANNOTATOR_SOURCE_SUBSET_NOTE = "first_30_videos_from_test"
SYNTHETIC_SPLITS = {
    "train": {
        "path": SYNTHETIC_TRAIN_SIMPLE_PATH,
        "source_split": "train",
        "source_subset_note": "v3_1_train_full",
    },
    "test": {
        "path": SYNTHETIC_TEST_SIMPLE_PATH,
        "source_split": "test",
        "source_subset_note": "v3_1_test_full",
    },
}

LEFT_RETRACTION_DIRECTION_CODES = [
    "KEEP_RETRACT_LATERAL",
    "KEEP_RETRACT_MEDIAL",
    "KEEP_RETRACT_UPWARD",
    "RETRACT_LATERAL",
    "RETRACT_MEDIAL",
    "RETRACT_UPWARD",
    "RETRACT_LATERAL_TO_MEDIAL",
    "RETRACT_LATERAL_TO_UPWARD",
    "RETRACT_MEDIAL_TO_LATERAL",
    "RETRACT_MEDIAL_TO_UPWARD",
    "RETRACT_UPWARD_TO_LATERAL",
    "RETRACT_UPWARD_TO_MEDIAL",
]

RIGHT_TARGET_CONTEXT_NORMALIZATION = {
    "between cystic artery and liver bed": "between cystic artery and cystic plate",
    "on the side near the cystic plate": "near the cystic plate",
    "near cystic duct": "near the cystic duct",
}

RIGHT_TARGET_CONTEXT_ORDER = [
    "between presumed cystic duct and presumed cystic artery",
    "between cystic artery and cystic plate",
    "near the base of the hepatocystic triangle",
    "close to the gallbladder neck",
    "near the cystic duct",
    "near the cystic plate",
    "(not set)",
]

ACTION_FIELD_DEFAULTS = {
    "start_frame": None,
    "end_frame": None,
    "retraction_direction_code": None,
    "changed": None,
    "start_direction": None,
    "end_direction": None,
    "tool_type": None,
    "action_code": None,
    "target_structure": None,
    "triplet": [],
    "target_context_1": None,
    "target_context_2": None,
    "description": None,
    "rank": None,
    "confidence": None,
    "evidence": None,
    "actor_role": None,
    "camera_movement": None,
}

ACTION_CODE_ALIASES = {
    "IRRIGATOR_COUNTERTRACTION_ASSIST": "COUNTERTRACTION_ASSIST",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def relpath(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_right_target_context(value: Any) -> str:
    text = str(value if value is not None else "(not set)")
    return RIGHT_TARGET_CONTEXT_NORMALIZATION.get(text, text)


def canonical_action_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    return ACTION_CODE_ALIASES.get(text, text)


def action_sort_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action.get("start_frame") is None,
        action.get("start_frame", -1),
        action.get("end_frame", -1),
        json.dumps(action, sort_keys=True),
    )


def build_reduced_taxonomy(simple_records: list[dict[str, Any]]) -> dict[str, Any]:
    left_codes = list(LEFT_RETRACTION_DIRECTION_CODES)
    right_tool_types = sorted(
        {
            seg["tool_type"]
            for record in simple_records
            for seg in record.get("right", [])
            if seg.get("tool_type")
        }
    )
    right_action_codes = sorted(
        {
            canonical_action_code(seg["action_code"])
            for record in simple_records
            for seg in record.get("right", [])
            if seg.get("action_code")
        }
    )
    right_target_structures = sorted(
        {
            seg["target_structure"]
            for record in simple_records
            for seg in record.get("right", [])
            if seg.get("target_structure")
        }
    )
    observed_contexts = {
        normalize_right_target_context(seg.get("target_context_1", "(not set)"))
        for record in simple_records
        for seg in record.get("right", [])
    }
    observed_contexts.update(
        normalize_right_target_context(seg.get("target_context_2", "(not set)"))
        for record in simple_records
        for seg in record.get("right", [])
    )
    right_target_contexts = [value for value in RIGHT_TARGET_CONTEXT_ORDER if value in observed_contexts]
    right_target_contexts.extend(
        sorted(value for value in observed_contexts if value not in set(right_target_contexts))
    )
    camera_action_codes = sorted(
        {
            str(seg["action_code"]).upper()
            for record in simple_records
            for seg in record.get("camera", [])
            if seg.get("action_code")
        }
    )

    return {
        "dataset_name": DATASET_NAME,
        "taxonomy_version": TAXONOMY_VERSION,
        "description": (
            "Reduced current task taxonomy used by the left/right/camera simple-action setup "
            "for audit_v11, SurGent, and CoT evaluation."
        ),
        "derived_from": {
            "internal_source_release": INTERNAL_SOURCE_RELEASE,
            "base_taxonomy_path": relpath(ANNOTATION_ROOT / "taxonomy_v10.json"),
            "simple_gt_path": relpath(TRAINED_ANNOTATOR_SIMPLE_PATH),
            "current_taxonomy_notebook": relpath(CURRENT_TAXONOMY_NOTEBOOK_PATH),
        },
        "actors": {
            "left": {
                "components": {
                    "retraction_direction_code": left_codes,
                }
            },
            "right": {
                "components": {
                    "tool_type": right_tool_types,
                    "action_code": right_action_codes,
                    "target_structure": right_target_structures,
                    "target_context": right_target_contexts,
                }
            },
            "camera": {
                "components": {
                    "action_code": camera_action_codes,
                }
            },
        },
    }


def clean_source_path(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    marker = "/cvs_act_annotations/v1/"
    if marker in text:
        return text.split(marker, 1)[1]
    return text


def build_example(
    row: dict[str, Any],
    *,
    label_source: str,
    config_name: str,
    source_file: Path,
    source_split: str,
    source_subset_note: str,
    paired_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame_range = row.get("frame_range") or (paired_row or {}).get("frame_range") or [None, None]
    mind_change = row.get("mind_change") or (paired_row or {}).get("mind_change")
    source_path = row.get("source_path") or (paired_row or {}).get("source_path")
    source_annotation_path = clean_source_path(source_path)
    source_artifact = relpath(source_file)

    def normalize_action_segment(action: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(ACTION_FIELD_DEFAULTS)
        normalized.update(action)
        if normalized.get("triplet") is None:
            normalized["triplet"] = []
        return normalized

    output = {
        "example_id": row["example_id"],
        "video_id": row["video_id"],
        "criterion": row["criterion"],
        "label_source": label_source,
        "config_name": config_name,
        "taxonomy_version": TAXONOMY_VERSION,
        "internal_source_release": INTERNAL_SOURCE_RELEASE,
        "frame_start": frame_range[0] if len(frame_range) > 0 else None,
        "frame_end": frame_range[1] if len(frame_range) > 1 else None,
        "mind_change": mind_change,
        "source_dataset": SOURCE_DATASET,
        "source_dataset_name": SOURCE_DATASET_NAME,
        "source_split": source_split,
        "source_subset_note": source_subset_note,
        "left_actions": [
            normalize_action_segment(action)
            for action in sorted(row.get("left", []), key=action_sort_key)
        ],
        "right_actions": [
            normalize_action_segment(action)
            for action in sorted(row.get("right", []), key=action_sort_key)
        ],
        "camera_actions": [
            normalize_action_segment(action)
            for action in sorted(row.get("camera", []), key=action_sort_key)
        ],
        "other_actions": [
            normalize_action_segment(action)
            for action in sorted(row.get("other", []), key=action_sort_key)
        ],
        "source_file": source_artifact,
        "source_annotation_path": source_annotation_path,
    }
    if label_source == "synthetic":
        output["synthetic_generation_method"] = SYNTHETIC_DEFAULT_METHOD
        output["synthetic_generation_model"] = SYNTHETIC_GENERATION_MODEL
        output["synthetic_generation_notebook"] = relpath(SYNTHETIC_NOTEBOOK_PATH)
        output["paired_trained_annotator_source"] = relpath(TRAINED_ANNOTATOR_SIMPLE_PATH)
    return output


def build_dataset_rows() -> dict[str, dict[str, list[dict[str, Any]]]]:
    trained_records = read_json(TRAINED_ANNOTATOR_SIMPLE_PATH)
    trained_by_id = {row["example_id"]: row for row in trained_records}
    trained_config = f"{SOURCE_CONFIG_PREFIX}_trained_annotator"
    synthetic_config = f"{SOURCE_CONFIG_PREFIX}_synthetic"

    trained_rows = [
        build_example(
            row,
            label_source="trained_annotator",
            config_name=trained_config,
            source_file=TRAINED_ANNOTATOR_SIMPLE_PATH,
            source_split=TRAINED_ANNOTATOR_SOURCE_SPLIT,
            source_subset_note=TRAINED_ANNOTATOR_SOURCE_SUBSET_NOTE,
        )
        for row in trained_records
    ]
    synthetic_rows_by_split = {}
    for split_name, split_info in SYNTHETIC_SPLITS.items():
        source_file = split_info["path"]
        synthetic_records = read_json(source_file)
        synthetic_rows_by_split[split_name] = [
            build_example(
                row,
                label_source="synthetic",
                config_name=synthetic_config,
                source_file=source_file,
                source_split=split_info["source_split"],
                source_subset_note=split_info["source_subset_note"],
                paired_row=trained_by_id.get(row["example_id"]),
            )
            for row in synthetic_records
        ]
    return {
        trained_config: {TRAINED_ANNOTATOR_SOURCE_SPLIT: trained_rows},
        synthetic_config: synthetic_rows_by_split,
    }


def validate_jsonl(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            count += 1
    return count


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = Counter(row["criterion"] for row in rows)
    return {
        "examples": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "criteria": dict(sorted(criteria.items())),
    }


def build_readme(dataset_rows: dict[str, dict[str, list[dict[str, Any]]]], taxonomy: dict[str, Any]) -> str:
    trained_config = f"{SOURCE_CONFIG_PREFIX}_trained_annotator"
    synthetic_config = f"{SOURCE_CONFIG_PREFIX}_synthetic"
    future_surgeon_config = f"{SOURCE_CONFIG_PREFIX}_surgeon_annotated"
    trained_summary = subset_summary(dataset_rows[trained_config]["test"])
    synthetic_train_summary = subset_summary(dataset_rows[synthetic_config]["train"])
    synthetic_test_summary = subset_summary(dataset_rows[synthetic_config]["test"])
    configs_yaml = f"""configs:
  - config_name: {trained_config}
    data_files:
      - split: test
        path: {trained_config}/test.jsonl
  - config_name: {synthetic_config}
    data_files:
      - split: test
        path: {synthetic_config}/test.jsonl
      - split: train
        path: {synthetic_config}/train.jsonl"""
    readme = f"""---
pretty_name: {DATASET_NAME}
license: other
tags:
  - {PUBLIC_RELEASE_VERSION}
language:
  - en
task_categories:
  - text-classification
  - token-classification
size_categories:
  - 1K<n<10K
{configs_yaml}
---

# {DATASET_TITLE}

## Dataset Description

`{DATASET_NAME}` is a surgical action recommendation dataset for laparoscopic cholecystectomy grounded in Critical View of Safety (CVS) assessment. Each example corresponds to a CVS transition example and contains structured action recommendations over the current task label space for the left instrument, right instrument, and camera, with the original `other` actor field preserved when present.

The current release packages the task-ready simple-action examples that the project notebooks and the current SurGent/CoT evaluation use, rather than the raw nested audit annotation files.

Release `{PUBLIC_RELEASE_VERSION}` was released {PUBLIC_RELEASE_DATE}, generated with Qwen3.6-35B-A3B pipeline.

## Configs / Provenance Subsets

The configs in this repo are **annotation provenance + source-dataset subsets**, not ordinary train/test splits:

- `{trained_config}`: labels derived from the current `audit_v11` trained-annotator audit examples from `{SOURCE_DATASET_NAME}`
- `{synthetic_config}`: synthetic labels from the v3.1 Qwen3.6-35B-A3B pipeline export from `{SOURCE_DATASET_NAME}`

The current release does **not** include a `surgeon_annotated` config because no surgeon-annotated export exists in the inspected workspace. A future release can add that config without changing the overall repo structure.

Intended naming convention for this repo:

- `{synthetic_config}`: contains `train` and `test`; may later add `validation`
- `{trained_config}`: currently `test`
- `{future_surgeon_config}`: future `test`

## Splits

`{synthetic_config}` exposes `train` and `test` splits from the v3.1 synthetic export. `{trained_config}` currently exposes `test` only because the trained-annotator examples come from the `{SOURCE_DATASET_NAME}` test portion and cover `{TRAINED_ANNOTATOR_SOURCE_SUBSET_NOTE}`.

## Current Release Summary

- `{trained_config}/test`: {trained_summary['examples']} examples from {trained_summary['videos']} videos
- `{synthetic_config}/train`: {synthetic_train_summary['examples']} examples from {synthetic_train_summary['videos']} videos
- `{synthetic_config}/test`: {synthetic_test_summary['examples']} examples from {synthetic_test_summary['videos']} videos

Criterion counts:

- `{trained_config}/test`: {trained_summary['criteria']}
- `{synthetic_config}/train`: {synthetic_train_summary['criteria']}
- `{synthetic_config}/test`: {synthetic_test_summary['criteria']}

## Files

```text
taxonomy/action_taxonomy.json
{trained_config}/test.jsonl
{synthetic_config}/train.jsonl
{synthetic_config}/test.jsonl
```

## Schema

Each JSONL row contains:

- `example_id`: stable example identifier
- `video_id`: source video identifier
- `criterion`: CVS criterion, currently one of `C1`, `C2`, `C3`
- `label_source`: provenance label, currently `trained_annotator` or `synthetic`
- `config_name`: exported Hugging Face config name for this row
- `taxonomy_version`: current reduced task taxonomy version for this release
- `internal_source_release`: internal source tag, currently `{INTERNAL_SOURCE_RELEASE}`
- `frame_start`, `frame_end`: frame span for the example
- `mind_change`: CVS transition label when available
- `source_dataset`, `source_dataset_name`, `source_split`, `source_subset_note`: source-corpus provenance fields
- `left_actions`, `right_actions`, `camera_actions`, `other_actions`: per-actor structured action segments
- `source_file`: local source artifact path used for export
- `source_annotation_path`: original audit annotation path when available
- `synthetic_generation_method`, `synthetic_generation_model`, `synthetic_generation_notebook`, `paired_trained_annotator_source`: synthetic provenance fields for the synthetic config

The actor action lists preserve the original nested fields from the current simple-action exports, including tool type, action code, target structure, target context, rank, confidence, evidence, and description when those fields exist.

For provenance clarity, this public release is explicitly linked to the internal source release tag `{INTERNAL_SOURCE_RELEASE}` in each row and in the taxonomy metadata.

## Taxonomy

The reduced task taxonomy is stored at `taxonomy/action_taxonomy.json`. This is **not** the full `taxonomy_v10.json`. It contains only the current left/right/camera label space used by the simple-action notebooks and by the current SurGent/CoT evaluation setup.

Current taxonomy summary:

- `left.retraction_direction_code`: {taxonomy['actors']['left']['components']['retraction_direction_code']}
- `right.action_code`: {taxonomy['actors']['right']['components']['action_code']}
- `camera.action_code`: {taxonomy['actors']['camera']['components']['action_code']}

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
"""
    return readme

def export_dataset(out_dir: Path) -> dict[str, Any]:
    dataset_rows = build_dataset_rows()
    taxonomy_records = read_json(TRAINED_ANNOTATOR_SIMPLE_PATH)
    for split_info in SYNTHETIC_SPLITS.values():
        taxonomy_records.extend(read_json(split_info["path"]))
    taxonomy = build_reduced_taxonomy(taxonomy_records)

    for subset, split_rows in dataset_rows.items():
        for split_name, rows in split_rows.items():
            write_jsonl(out_dir / subset / f"{split_name}.jsonl", rows)
    write_json(out_dir / "taxonomy" / "action_taxonomy.json", taxonomy)
    write_text(out_dir / "README.md", build_readme(dataset_rows, taxonomy))
    dataset_script = out_dir / "cvs_act.py"
    if dataset_script.exists():
        dataset_script.unlink()

    counts = {}
    for subset, split_rows in dataset_rows.items():
        counts[subset] = {}
        for split_name in split_rows:
            counts[subset][split_name] = validate_jsonl(out_dir / subset / f"{split_name}.jsonl")
    return {
        "out_dir": relpath(out_dir),
        "counts": counts,
        "taxonomy_path": relpath(out_dir / "taxonomy" / "action_taxonomy.json"),
    }


def validate_dataset(out_dir: Path) -> None:
    expected_splits = {
        f"{SOURCE_CONFIG_PREFIX}_trained_annotator": ["test"],
        f"{SOURCE_CONFIG_PREFIX}_synthetic": ["train", "test"],
    }
    print("JSONL validation:")
    for subset, splits in expected_splits.items():
        for split_name in splits:
            jsonl_path = out_dir / subset / f"{split_name}.jsonl"
            count = validate_jsonl(jsonl_path)
            print(f"  {subset}/{split_name}: {count} rows OK -> {relpath(jsonl_path)}")

    try:
        from datasets import load_dataset
    except Exception as exc:
        print("")
        print("Skipping datasets.load_dataset validation because `datasets` is not installed.")
        print(f"Missing dependency detail: {type(exc).__name__}: {exc}")
        return

    print("")
    print("datasets.load_dataset validation:")
    for subset, splits in expected_splits.items():
        ds = load_dataset(str(out_dir), subset)
        split_names = list(ds.keys())
        print(f"  config={subset} splits={split_names}")
        for split_name in splits:
            first_example = ds[split_name][0]
            print(f"    split={split_name} rows={len(ds[split_name])}")
            print(f"    first example keys: {sorted(first_example.keys())}")
            print(f"    first example id: {first_example['example_id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the current CVS-Act HF dataset package.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    required_paths = [
        TRAINED_ANNOTATOR_SIMPLE_PATH,
        SYNTHETIC_TRAIN_SIMPLE_PATH,
        SYNTHETIC_TEST_SIMPLE_PATH,
        AUDIT_V11_DIR,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input paths:\n{missing_text}")

    summary = export_dataset(args.out_dir)
    print("Exported CVS-Act package:")
    print(json.dumps(summary, indent=2))

    dataset_rows = build_dataset_rows()
    print("")
    print("Counts per subset/config and split:")
    for subset, split_rows in dataset_rows.items():
        for split_name, rows in split_rows.items():
            info = subset_summary(rows)
            print(
                f"  {subset}/{split_name}: examples={info['examples']} videos={info['videos']} "
                f"criteria={info['criteria']}"
            )

    if args.validate:
        print("")
        validate_dataset(args.out_dir)


if __name__ == "__main__":
    main()
