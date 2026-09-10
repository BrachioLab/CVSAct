"""Blind surgeon annotation storage for CVS-Act action labels."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CVS_ROOT = REPO_ROOT / "data" / "processed" / "CVS_Challenge_SAGES_v1"
RAW_TEST_LABELS_DIR = REPO_ROOT / "data" / "raw" / "CVS_Challenge_SAGES_v1" / "test" / "labels"
DEFAULT_ANNOTATION_DIR = CVS_ROOT / "surgeon_validation" / "annotations"
DEFAULT_ANNOTATION_PATH = DEFAULT_ANNOTATION_DIR / "cvs_act_action_annotations.jsonl"
DEFAULT_TAXONOMY_PATH = CVS_ROOT / "cvs_act_annotations" / "v1" / "taxonomy_v10.json"
CVS_CRITERIA = ("c1", "c2", "c3")

ACTION_FIELDS = (
    "actor_role",
    "tool_type",
    "action_code",
    "target_structure",
    "target_context",
    "intention",
)

_CURRENT_MANIFEST: list["ClipManifestRow"] = []


@dataclasses.dataclass(frozen=True)
class ClipManifestRow:
    clip_id: str
    video_name: str
    granularity: str
    start_frame: int
    end_frame: int
    manifest_source: str
    split: str | None = None
    criterion: str | None = None
    mind_change: str | None = None
    existing_cvs_labels: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    frame_paths: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class AnnotationRecord:
    clip_id: str
    video_name: str
    granularity: str
    annotator_id: str
    actor_role: str | None = None
    tool_type: str | None = None
    action_code: str | None = None
    target_structure: str | None = None
    target_context: str | None = None
    intention: str | None = None
    note: str = ""
    observed_differs_from_recommendation: bool = False
    subclips: tuple[Mapping[str, Any], ...] = ()
    timestamp: str = ""

    def to_json(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["timestamp"] = out["timestamp"] or _utc_now()
        return out


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_action_schema(path: str | Path = DEFAULT_TAXONOMY_PATH) -> dict[str, list[str]]:
    """Load the existing CVS-Act taxonomy as field -> allowed values."""
    raw = _read_json(Path(path))
    fields = raw.get("fields") or {}
    schema: dict[str, list[str]] = {}
    for field in ACTION_FIELDS:
        values = [str(item["value"]) for item in fields.get(field, []) if "value" in item]
        if "(not set)" in values:
            values = ["(not set)"] + [v for v in values if v != "(not set)"]
        schema[field] = values
    missing = [field for field, values in schema.items() if not values]
    if missing:
        raise ValueError(f"Taxonomy is missing required fields: {missing}")
    return schema


def load_clip_manifest(path: str | Path) -> list[ClipManifestRow]:
    """Load an audit_v11-style clip directory/file into a flat manifest.

    The loader accepts either a directory of per-video JSON files or a JSON/JSONL/CSV
    manifest. It never reads model outputs and keeps CVS labels only as hidden
    metadata for analysis.
    """
    manifest_path = Path(path)
    if manifest_path.is_dir():
        rows = _load_audit_clip_dir(manifest_path)
    elif manifest_path.suffix.lower() == ".jsonl":
        rows = [_row_from_mapping(json.loads(line), str(manifest_path)) for line in manifest_path.read_text().splitlines() if line.strip()]
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open(newline="") as f:
            rows = [_row_from_mapping(row, str(manifest_path)) for row in csv.DictReader(f)]
    elif manifest_path.suffix.lower() == ".json":
        raw = _read_json(manifest_path)
        if isinstance(raw, list):
            rows = [_row_from_mapping(row, str(manifest_path)) for row in raw]
        elif isinstance(raw, dict) and "clips" in raw:
            rows = [_row_from_mapping(row, str(manifest_path)) for row in raw["clips"]]
        else:
            raise ValueError(f"Unsupported JSON manifest shape in {manifest_path}")
    else:
        raise ValueError(f"Unsupported manifest path: {manifest_path}")

    global _CURRENT_MANIFEST
    _CURRENT_MANIFEST = rows
    return rows


def save_annotation(
    record: AnnotationRecord | Mapping[str, Any],
    path: str | Path = DEFAULT_ANNOTATION_PATH,
    schema_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> dict[str, Any]:
    """Append one blind annotation revision to JSONL storage."""
    payload = _record_to_payload(record)
    _validate_payload(payload, load_action_schema(schema_path))
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def load_annotations(
    path: str | Path = DEFAULT_ANNOTATION_PATH,
    latest_only: bool = True,
) -> list[dict[str, Any]]:
    """Load annotations. By default returns the latest record per clip/annotator."""
    ann_path = Path(path)
    if not ann_path.exists():
        return []
    rows = [json.loads(line) for line in ann_path.read_text().splitlines() if line.strip()]
    if not latest_only:
        return rows
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row["clip_id"]), str(row["annotator_id"]))] = row
    return list(latest.values())


def get_progress(
    annotator_id: str,
    manifest: Iterable[ClipManifestRow] | None = None,
    annotations_path: str | Path = DEFAULT_ANNOTATION_PATH,
) -> dict[str, Any]:
    rows = list(manifest if manifest is not None else _CURRENT_MANIFEST)
    done_ids = {
        row["clip_id"]
        for row in load_annotations(annotations_path)
        if row.get("annotator_id") == annotator_id
    }
    all_ids = [row.clip_id for row in rows]
    remaining = [clip_id for clip_id in all_ids if clip_id not in done_ids]
    return {"done": len(done_ids & set(all_ids)), "remaining": len(remaining), "remaining_clip_ids": remaining}


def export_for_analysis(
    manifest: Iterable[ClipManifestRow] | None = None,
    annotator_ids: Iterable[str] | None = None,
    annotations_path: str | Path = DEFAULT_ANNOTATION_PATH,
):
    """Return a tidy DataFrame with one row per clip x annotator."""
    import pandas as pd

    clips = list(manifest if manifest is not None else _CURRENT_MANIFEST)
    annotations = {
        (row["clip_id"], row["annotator_id"]): row
        for row in load_annotations(annotations_path)
    }
    if annotator_ids is None:
        annotator_ids = sorted({key[1] for key in annotations})

    out: list[dict[str, Any]] = []
    for clip in clips:
        for annotator_id in annotator_ids:
            row = {
                "clip_id": clip.clip_id,
                "video_name": clip.video_name,
                "granularity": clip.granularity,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "annotator_id": annotator_id,
            }
            ann = annotations.get((clip.clip_id, annotator_id))
            for field in ACTION_FIELDS:
                row[field] = ann.get(field) if ann else None
            row["note"] = ann.get("note") if ann else None
            row["observed_differs_from_recommendation"] = (
                ann.get("observed_differs_from_recommendation") if ann else None
            )
            row["timestamp"] = ann.get("timestamp") if ann else None
            row["subclips"] = ann.get("subclips") if ann else None
            out.append(row)
    return pd.DataFrame(out)


def _load_audit_clip_dir(path: Path) -> list[ClipManifestRow]:
    rows: list[ClipManifestRow] = []
    for json_path in sorted(path.glob("*.json")):
        records = _read_json(json_path)
        if not isinstance(records, list):
            continue
        for record in records:
            rows.extend(_rows_from_audit_record(record, json_path))
    return rows


def _rows_from_audit_record(record: Mapping[str, Any], source: Path) -> list[ClipManifestRow]:
    video_id = str(record.get("video_id") or source.stem)
    coarse = record.get("coarse") or {}
    split = (record.get("generated_clip_source") or {}).get("split") or _infer_split(coarse)
    common = {
        "video_name": video_id,
        "manifest_source": str(source),
        "split": split,
        "criterion": record.get("criterion"),
        "mind_change": record.get("mind_change"),
    }
    out: list[ClipManifestRow] = []
    if coarse:
        coarse_frames = _keyframes_from_mapping(coarse.get("keyframes") or {})
        out.append(
            ClipManifestRow(
                clip_id=f"{record.get('example_id', video_id)}__coarse",
                granularity="coarse",
                start_frame=int(coarse["start_frame"]),
                end_frame=int(coarse["end_frame"]),
                existing_cvs_labels=_existing_cvs_labels(
                    video_id=video_id,
                    criterion=record.get("criterion"),
                    mind_change=record.get("mind_change"),
                    keyframes=coarse_frames,
                ),
                frame_paths=tuple((coarse.get("keyframes") or {}).values()),
                **common,
            )
        )
    for idx, fine in enumerate(record.get("fine") or []):
        start = int(fine["start_frame"])
        end = int(fine["end_frame"])
        fine_frames = _keyframes_from_mapping(fine.get("keyframes") or {})
        out.append(
            ClipManifestRow(
                clip_id=f"{record.get('example_id', video_id)}__fine_{idx:02d}_{start:06d}_{end:06d}",
                granularity="fine",
                start_frame=start,
                end_frame=end,
                existing_cvs_labels=_existing_cvs_labels(
                    video_id=video_id,
                    criterion=record.get("criterion"),
                    mind_change=record.get("mind_change"),
                    keyframes=fine_frames,
                ),
                frame_paths=tuple((fine.get("keyframes") or {}).values()),
                **common,
            )
        )
    return out


def _infer_split(clip: Mapping[str, Any]) -> str | None:
    for value in (clip.get("keyframes") or {}).values():
        parts = str(value).split("/")
        if parts and parts[0] in {"train", "test"}:
            return parts[0]
    return None


def _row_from_mapping(row: Mapping[str, Any], source: str) -> ClipManifestRow:
    start = row.get("start_frame")
    if start is None:
        start = row.get("start")
    end = row.get("end_frame")
    if end is None:
        end = row.get("end")
    video_name = str(row.get("video_name") or row.get("video_id"))
    keyframes = _csv_keyframes(row, start, end)
    existing_cvs_labels = dict(row.get("existing_cvs_labels") or {})
    existing_cvs_labels.update(
        _existing_cvs_labels(
            video_id=video_name,
            criterion=row.get("criterion"),
            mind_change=row.get("mind_change"),
            keyframes=keyframes,
        )
    )
    return ClipManifestRow(
        clip_id=str(row["clip_id"]),
        video_name=video_name,
        granularity=str(row["granularity"]),
        start_frame=int(start),
        end_frame=int(end),
        manifest_source=str(row.get("manifest_source") or source),
        split=row.get("split"),
        criterion=row.get("criterion"),
        mind_change=row.get("mind_change"),
        existing_cvs_labels=existing_cvs_labels,
        frame_paths=tuple(row.get("frame_paths") or ()),
    )


def _csv_keyframes(row: Mapping[str, Any], start: Any, end: Any) -> dict[str, int]:
    frames: dict[str, int] = {"start": int(start)}
    mid = row.get("mid") or row.get("mid_frame")
    if mid not in (None, "", "nan"):
        frames["mid"] = int(float(mid))
    frames["end"] = int(end)
    return frames


def _keyframes_from_mapping(keyframes: Mapping[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label, value in keyframes.items():
        frame_id = _frame_id_from_path(value)
        if frame_id is not None:
            out[str(label)] = frame_id
    return out


def _frame_id_from_path(value: Any) -> int | None:
    stem = Path(str(value)).name
    if stem.startswith("frame_"):
        stem = stem.removeprefix("frame_")
    try:
        return int(stem)
    except ValueError:
        return None


def _existing_cvs_labels(
    video_id: str,
    criterion: Any = None,
    mind_change: Any = None,
    keyframes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    labels: dict[str, Any] = {
        "criterion": criterion,
        "mind_change": mind_change,
    }
    frame_labels = _load_cvs_keyframe_labels(video_id, keyframes or {})
    if frame_labels:
        labels["keyframes"] = frame_labels
    return {key: value for key, value in labels.items() if value not in (None, "", {})}


def _load_cvs_keyframe_labels(video_id: str, keyframes: Mapping[str, int]) -> list[dict[str, Any]]:
    if not keyframes:
        return []
    label_path = RAW_TEST_LABELS_DIR / str(video_id) / "frame.csv"
    if not label_path.exists():
        return []
    wanted = {int(frame_id) for frame_id in keyframes.values()}
    if not wanted:
        return []
    label_by_frame: dict[int, dict[str, Any]] = {}
    with label_path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame_id = int(row["frame_id"])
            if frame_id not in wanted:
                continue
            criteria: dict[str, dict[str, Any]] = {}
            for crit in CVS_CRITERIA:
                votes = [int(float(row.get(f"{crit}_rater{idx}", 0) or 0)) for idx in (1, 2, 3)]
                criteria[crit.upper()] = {
                    "votes": sum(votes),
                    "total": len(votes),
                    "raters": votes,
                }
            label_by_frame[frame_id] = {"frame": frame_id, "criteria": criteria}
    out: list[dict[str, Any]] = []
    for role, frame_id in keyframes.items():
        row = label_by_frame.get(int(frame_id))
        if row:
            out.append({"role": str(role), **row})
    return out


def _record_to_payload(record: AnnotationRecord | Mapping[str, Any]) -> dict[str, Any]:
    payload = record.to_json() if isinstance(record, AnnotationRecord) else dict(record)
    payload["timestamp"] = payload.get("timestamp") or _utc_now()
    payload["note"] = payload.get("note") or ""
    payload["observed_differs_from_recommendation"] = bool(
        payload.get("observed_differs_from_recommendation", False)
    )
    return payload


def _validate_payload(payload: Mapping[str, Any], schema: Mapping[str, list[str]]) -> None:
    required = {"clip_id", "video_name", "granularity", "annotator_id"}
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise ValueError(f"Annotation is missing required fields: {missing}")
    for field in ACTION_FIELDS:
        if not payload.get(field):
            continue
        value = str(payload[field])
        if value not in schema[field]:
            raise ValueError(f"{field}={value!r} is not in the CVS-Act taxonomy")
