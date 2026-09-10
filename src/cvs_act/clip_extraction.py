"""Audit-compatible clip extraction for surgeon-validation CVS-Act clips.

This module reuses the existing transition logic from :mod:`cvs_act.extraction`.
The audit_v11 mechanism recovered in this repo is:

* labels are loaded from ``*/frame.csv`` with raw score = mean of 3 raters;
* raw scores are Gaussian-smoothed with ``sigma=1.0``, nearest mode,
  ``truncate=1.0``;
* selected clips are ascending smoothed runs where raw crosses
  ``<0.5 -> >0.5``;
* coarse clip IDs use ``{video}__C?__avg__c_{start}_{end}``;
* fine regions are adjacent keyframe intervals where raw crosses
  ``<0.5 -> >0.5`` and use frame step 30.

The serializer in ``scripts/generate_train_transition_clip_records.py`` created
the audit-style generated records. Existing audit_v11 JSON files preserve those
IDs and boundaries, with human annotations filled in.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from .extraction import build_clip_records, find_selected_ascending_clips, load_video_data
from .schemas import FRAME_STEP
from .surgeon_sampling import REPO_ROOT, SEED, SURGEON_ANNOTATION_ROOT


RAW_TEST_LABELS_DIR = REPO_ROOT / "data/raw/CVS_Challenge_SAGES_v1/test/labels"
PROCESSED_TEST_FRAMES_DIR = REPO_ROOT / "data/processed/CVS_Challenge_SAGES_v1/frames/test"
AUDIT_V11_DIR = (
    REPO_ROOT
    / "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1/audit_v11"
)
SURGEON_CLIPS_ROOT = SURGEON_ANNOTATION_ROOT / "clips"
Granularity = Literal["coarse", "fine", "both"]


def _frame_path(split: str, vid: str, frame: int) -> str:
    return f"{split}/{vid}/frame_{int(frame):06d}"


def _record_from_clip(video_data: dict, clip_record: dict, crit: str, split: str) -> dict:
    vid = str(clip_record["video_id"])
    coarse_frames = [int(x) for x in clip_record["coarse_frames"]]
    sf = coarse_frames[0]
    ef = coarse_frames[-1]
    mid = coarse_frames[len(coarse_frames) // 2]
    si = int(clip_record["start_idx"])
    ei = int(clip_record["end_idx"])
    raw = video_data[vid][crit]["raw"]
    raters_start = round(float(raw[si]) * 3)
    raters_end = round(float(raw[ei]) * 3)
    example_id = f"{vid}__{crit.upper()}__avg__c_{sf:06d}_{ef:06d}"
    fine_entries = []
    for fine in clip_record["fine_regions_merged"]:
        if crit not in fine.get("criteria", [crit]):
            continue
        fs = int(fine["key_start"])
        fe = int(fine["key_end"])
        fine_entries.append(
            {
                "fine_id": f"{vid}__{crit.upper()}__avg__f_{fs:06d}_{fe:06d}",
                "start_frame": fs,
                "end_frame": fe,
                "keyframes": {
                    "start": _frame_path(split, vid, fs),
                    "end": _frame_path(split, vid, fe),
                },
                "annotation": {"actions_ranked": []},
            }
        )
    return {
        "schema_version": "cvs_act.v1.generated_transition_clips",
        "example_id": example_id,
        "video_id": vid,
        "criterion": crit.upper(),
        "rater_id": "avg",
        "mind_change": f"{raters_start}/3->{raters_end}/3",
        "granularity": "coarse",
        "generated_clip_source": {
            "method": "cvs_act.extraction.find_selected_ascending_clips",
            "split": split,
            "start_idx": si,
            "end_idx": ei,
            "frame_step": FRAME_STEP,
        },
        "coarse": {
            "start_frame": sf,
            "mid_frame": mid,
            "end_frame": ef,
            "video_id": vid,
            "keyframes": {
                "start": _frame_path(split, vid, sf),
                "mid": _frame_path(split, vid, mid),
                "end": _frame_path(split, vid, ef),
            },
            "annotation": {"actions_ranked": []},
        },
        "fine": fine_entries,
    }


def _build_records_for_videos(
    video_names: Sequence[str],
    labels_dir: Path = RAW_TEST_LABELS_DIR,
    frames_dir: Path = PROCESSED_TEST_FRAMES_DIR,
    split: str = "test",
) -> list[dict]:
    video_set = set(map(str, video_names))
    video_data = load_video_data(str(labels_dir))
    selected = find_selected_ascending_clips(video_data)
    clip_records = build_clip_records(selected, video_data, str(frames_dir))
    out: list[dict] = []
    for clip_record in clip_records:
        vid = str(clip_record["video_id"])
        if vid not in video_set:
            continue
        for crit in clip_record["criteria"]:
            out.append(_record_from_clip(video_data, clip_record, str(crit), split))
    return sorted(out, key=lambda r: (r["video_id"], r["coarse"]["start_frame"], r["coarse"]["end_frame"], r["criterion"]))


def extract_clips(
    sample_df: pd.DataFrame,
    granularity: Granularity = "both",
    labels_dir: Path = RAW_TEST_LABELS_DIR,
    frames_dir: Path = PROCESSED_TEST_FRAMES_DIR,
) -> pd.DataFrame:
    """Extract audit-compatible coarse/fine clip rows for sampled videos."""

    if "video_name" not in sample_df:
        raise ValueError("sample_df must contain video_name")
    records = _build_records_for_videos(sample_df["video_name"].astype(str).tolist(), labels_dir, frames_dir)
    meta_cols = [
        col
        for col in [
            "video_name",
            "stratum",
            "country",
            "country_collapsed",
            "ioc",
            "icg",
            "robotic",
            "device",
            "weight",
        ]
        if col in sample_df.columns
    ]
    meta = sample_df[meta_cols].drop_duplicates("video_name").set_index("video_name")
    rows: list[dict] = []
    for record in records:
        vid = record["video_id"]
        coarse = record["coarse"]
        if granularity in ("coarse", "both"):
            rows.append(
                {
                    "clip_id": record["example_id"],
                    "video_name": vid,
                    "criterion": record["criterion"],
                    "granularity": "coarse",
                    "start": int(coarse["start_frame"]),
                    "mid": int(coarse["mid_frame"]),
                    "end": int(coarse["end_frame"]),
                }
            )
        if granularity in ("fine", "both"):
            for fine in record.get("fine", []):
                rows.append(
                    {
                        "clip_id": fine.get("fine_id")
                        or f"{vid}__{record['criterion']}__avg__f_{int(fine['start_frame']):06d}_{int(fine['end_frame']):06d}",
                        "coarse_clip_id": record["example_id"],
                        "video_name": vid,
                        "criterion": record["criterion"],
                        "granularity": "fine",
                        "start": int(fine["start_frame"]),
                        "mid": None,
                        "end": int(fine["end_frame"]),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in meta.columns:
        df[col] = df["video_name"].map(meta[col])
    return df.sort_values(["video_name", "start", "end", "criterion", "granularity"]).reset_index(drop=True)


def write_clips(
    sample_df: pd.DataFrame,
    out_root: Path = SURGEON_CLIPS_ROOT,
    labels_dir: Path = RAW_TEST_LABELS_DIR,
    frames_dir: Path = PROCESSED_TEST_FRAMES_DIR,
    copy_frames: bool = False,
) -> pd.DataFrame:
    """Write audit-style JSON files and a clip manifest under ``out_root``.

    JSON files are placed in ``out_root/json/{video_name}.json`` and never
    collide with audit_v11. The manifest is returned and also written as
    ``out_root/clip_manifest.csv``.
    """

    out_root.mkdir(parents=True, exist_ok=True)
    json_dir = out_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    records = _build_records_for_videos(sample_df["video_name"].astype(str).tolist(), labels_dir, frames_dir)
    by_video: dict[str, list[dict]] = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for old in json_dir.glob("*.json"):
        old.unlink()
    for vid, recs in sorted(by_video.items()):
        recs = sorted(recs, key=lambda r: (r["coarse"]["start_frame"], r["coarse"]["end_frame"], r["criterion"]))
        (json_dir / f"{vid}.json").write_text(json.dumps(recs, indent=2) + "\n")

    if copy_frames:
        frame_out = out_root / "frames"
        for vid in sorted(by_video):
            src = frames_dir / vid
            dst = frame_out / vid
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

    manifest = extract_clips(sample_df, "both", labels_dir, frames_dir)
    manifest.to_csv(out_root / "clip_manifest.csv", index=False)
    return manifest


def select_one_clip_per_video(
    clip_manifest: pd.DataFrame,
    seed: int = SEED,
    granularity: Granularity = "coarse",
) -> pd.DataFrame:
    """Select one clip per video deterministically at random.

    By default, selection is from coarse clips because those are the primary
    audit examples; fine clips remain available in the full manifest.
    """

    if clip_manifest.empty:
        return clip_manifest.copy()
    if granularity != "both":
        candidates = clip_manifest.loc[clip_manifest["granularity"] == granularity].copy()
    else:
        candidates = clip_manifest.copy()
    if candidates.empty:
        raise ValueError(f"No clips available for granularity={granularity!r}")
    rng = np.random.default_rng(seed)
    rows = []
    for video_name, group in candidates.groupby("video_name", sort=True):
        ordered = group.sort_values(["start", "end", "criterion", "clip_id"]).reset_index(drop=True)
        pick = int(rng.integers(0, len(ordered)))
        row = ordered.iloc[pick].copy()
        row["selected_clip_seed"] = int(seed)
        row["selected_clip_granularity_pool"] = granularity
        rows.append(row)
    return pd.DataFrame(rows).sort_values("video_name").reset_index(drop=True)


def _boundary_signature(record: dict) -> tuple:
    fine = tuple(
        sorted(
            (int(item.get("start_frame", -1)), int(item.get("end_frame", -1)))
            for item in record.get("fine", [])
        )
    )
    coarse = record["coarse"]
    return (
        record.get("example_id"),
        int(coarse.get("start_frame")),
        int(coarse.get("mid_frame")),
        int(coarse.get("end_frame")),
        fine,
    )


def load_audit_v11_boundary_signatures(
    audit_dir: Path = AUDIT_V11_DIR,
    video_names: Iterable[str] | None = None,
) -> set[tuple]:
    """Load audit_v11 example/boundary signatures from existing JSON files."""

    wanted = set(map(str, video_names)) if video_names is not None else None
    signatures: set[tuple] = set()
    for path in sorted(audit_dir.glob("*.json")):
        if wanted is not None and path.stem not in wanted:
            continue
        records = json.loads(path.read_text())
        for record in records:
            signatures.add(_boundary_signature(record))
    return signatures


def assert_audit_v11_consistency(
    audit_video_names: Sequence[str] | None = None,
    labels_dir: Path = RAW_TEST_LABELS_DIR,
    frames_dir: Path = PROCESSED_TEST_FRAMES_DIR,
    audit_dir: Path = AUDIT_V11_DIR,
) -> None:
    """Assert recovered clip mechanism reproduces audit_v11 IDs/boundaries."""

    if audit_video_names is None:
        audit_video_names = sorted(path.stem for path in audit_dir.glob("*.json"))[:5]
    expected = load_audit_v11_boundary_signatures(audit_dir, audit_video_names)
    generated_records = _build_records_for_videos(audit_video_names, labels_dir, frames_dir)
    generated = {_boundary_signature(record) for record in generated_records}
    missing = expected - generated
    extra = generated - expected
    if missing or extra:
        raise AssertionError(
            "Recovered clip extraction does not match audit_v11 boundaries. "
            f"missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
        )
