#!/usr/bin/env python3
"""Prepare frame and video assets for CVS-Act surgeon annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvs_act.annotation_store import load_clip_manifest
from cvs_act.surgeon_annotation_widget import (
    DEFAULT_FRAMES_ROOT,
    DEFAULT_SURGEON_ASSET_ROOT,
    prepare_annotation_assets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data"
    / "processed"
    / "CVS_Challenge_SAGES_v1"
    / "cvs_act_surgeon_annotations"
    / "selected_video_clips.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Selected surgeon clip manifest CSV/JSON/JSONL.",
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help="Optional pre-extracted frame root. Missing frames are extracted from raw mp4s.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_SURGEON_ASSET_ROOT,
        help="Output directory for per-clip frames, interval videos, and asset manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clips = load_clip_manifest(args.manifest)
    summary = prepare_annotation_assets(
        clips,
        frames_root=args.frames_root,
        assets_root=args.assets_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary.get("missing_frames") or summary.get("failed_videos"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
