"""Data loading, blurring, clip selection, deduplication, and sampling."""

import glob
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .schemas import CRITERIA, RATERS, GAUSSIAN_SIGMA, FRAME_STEP


def load_video_data(labels_dir, criteria=CRITERIA, raters=RATERS, sigma=GAUSSIAN_SIGMA):
    """Load all videos, compute continuous scores and blurred versions.

    Returns dict: video_id -> {"df": DataFrame, crit: {"raw": array, "blurred": array}}
    """
    frame_files = sorted(glob.glob(os.path.join(labels_dir, "*/frame.csv")))

    video_data = {}
    for fpath in frame_files:
        video_id = os.path.basename(os.path.dirname(fpath))
        df = pd.read_csv(fpath)
        video_data[video_id] = {"df": df}
        for crit in criteria:
            rater_cols = [f"{crit}_{r}" for r in raters]
            raw_scores = df[rater_cols].sum(axis=1).values / len(raters)
            blurred = gaussian_filter1d(raw_scores, sigma=sigma, mode="nearest", truncate=1.0)
            video_data[video_id][crit] = {"raw": raw_scores, "blurred": blurred}

    return video_data


def segment_blurred(blurred, eps=1e-9):
    """Segment blurred sequence into ascending/flat/descending runs.
    Returns list of (start_idx, end_idx, seg_type) where end_idx is EXCLUSIVE."""
    n = len(blurred)
    if n <= 1:
        return [(0, 1, 0)]
    frame_dirs = np.zeros(n, dtype=int)
    for i in range(n - 1):
        diff = blurred[i + 1] - blurred[i]
        if diff > eps:
            frame_dirs[i] = 1
        elif diff < -eps:
            frame_dirs[i] = -1
    frame_dirs[n - 1] = frame_dirs[n - 2] if n >= 2 else 0
    segments = []
    run_start = 0
    for i in range(1, n):
        if frame_dirs[i] != frame_dirs[run_start]:
            segments.append((run_start, i, int(frame_dirs[run_start])))
            run_start = i
    segments.append((run_start, n, int(frame_dirs[run_start])))
    return segments


def find_selected_ascending_clips(video_data, criteria=CRITERIA):
    """Find ascending segments (0->1 only) where raw score crosses 0.5.
    Returns dict: video_id -> crit -> list of (start_idx, end_idx) inclusive."""
    selected = {}
    for video_id, vdata in video_data.items():
        df = vdata["df"]
        n = len(df)
        selected[video_id] = {}
        for crit in criteria:
            blurred = vdata[crit]["blurred"]
            raw = vdata[crit]["raw"]
            segments = segment_blurred(blurred)
            clips = []
            for si, ei, stype in segments:
                if stype != 1:
                    continue
                end_inclusive = ei
                if end_inclusive >= n:
                    continue
                if raw[si] < 0.5 and raw[end_inclusive] > 0.5:
                    clips.append((si, end_inclusive))
            selected[video_id][crit] = clips
    return selected


def extract_frames_v1(start_idx, end_idx, df, raw, frame_dir, frame_step=FRAME_STEP):
    """Extract coarse and fine-grained frames (V1: label flip at 0.5).
    Coarse: all key frames in clip. Fine: where raw crosses 0.5."""
    frame_ids = df["frame_id"].values.astype(int)
    coarse_frames = [int(frame_ids[i]) for i in range(start_idx, end_idx + 1)]
    fine_regions = []
    for i in range(start_idx, end_idx):
        if raw[i] < 0.5 and raw[i + 1] > 0.5:
            key_start = int(frame_ids[i])
            key_end = int(frame_ids[i + 1])
            all_frames = list(range(key_start, key_end + 1, frame_step))
            existing = [f for f in all_frames
                        if os.path.isfile(os.path.join(frame_dir, f"frame_{f:06d}.png"))]
            fine_regions.append({
                "key_start": key_start, "key_end": key_end,
                "key_start_idx": i, "key_end_idx": i + 1,
                "raw_start": raw[i], "raw_end": raw[i + 1],
                "all_frames": existing if existing else all_frames,
            })
    return coarse_frames, fine_regions


def build_clip_records(selected_clips, video_data, frames_dir, frame_step=FRAME_STEP):
    """Build deduplicated clip records with merged fine-grained regions.

    Returns list of clip record dicts.
    """
    dedup_clips = {}  # key: (vid, coarse_tuple) -> dict

    for vid, crit_clips in selected_clips.items():
        df_v = video_data[vid]["df"]
        frame_ids = df_v["frame_id"].values.astype(int)
        frame_dir = os.path.join(frames_dir, vid)

        for crit, clips_list in crit_clips.items():
            for si, ei in clips_list:
                coarse = tuple(int(frame_ids[i]) for i in range(si, ei + 1))
                key = (vid, coarse)

                if key not in dedup_clips:
                    dedup_clips[key] = {
                        "video_id": vid,
                        "criteria": [],
                        "start_idx": si,
                        "end_idx": ei,
                        "coarse_frames": list(coarse),
                        "fine_regions_by_criterion": {},
                    }

                dedup_clips[key]["criteria"].append(crit)

                # Compute fine regions for this criterion
                raw = video_data[vid][crit]["raw"]
                _, fine_regions = extract_frames_v1(si, ei, df_v, raw, frame_dir, frame_step)
                dedup_clips[key]["fine_regions_by_criterion"][crit] = fine_regions

    # Merge fine regions: deduplicate by (key_start, key_end), tag with criteria
    clip_records = []
    for key, rec in dedup_clips.items():
        fine_merged = {}
        for crit, regions in rec["fine_regions_by_criterion"].items():
            for region in regions:
                fkey = (region["key_start"], region["key_end"])
                if fkey not in fine_merged:
                    fine_merged[fkey] = {**region, "criteria": [crit]}
                else:
                    if crit not in fine_merged[fkey]["criteria"]:
                        fine_merged[fkey]["criteria"].append(crit)

        clip_records.append({
            "video_id": rec["video_id"],
            "criteria": sorted(set(rec["criteria"])),
            "start_idx": rec["start_idx"],
            "end_idx": rec["end_idx"],
            "coarse_frames": rec["coarse_frames"],
            "fine_regions_merged": list(fine_merged.values()),
        })

    return clip_records


def sample_clips(clip_records, frames_dir, criteria=CRITERIA, n_per_crit=2, seed=42):
    """Sample n_per_crit examples per criterion (deduplicated across criteria).

    Returns list of dicts with keys: clip_idx, primary_criterion, record.
    """
    rng = random.Random(seed)

    sampled_clips = []
    sampled_keys = set()

    for crit in criteria:
        # Filter clips where this criterion is in the criteria list, not already sampled
        candidates = [(i, r) for i, r in enumerate(clip_records)
                      if crit in r["criteria"] and i not in sampled_keys]

        # Prefer clips with frames on disk
        candidates_with_frames = [(i, r) for i, r in candidates
                                  if os.path.isdir(os.path.join(frames_dir, r["video_id"]))]
        if len(candidates_with_frames) >= n_per_crit:
            candidates = candidates_with_frames

        # Sample and keep
        pair = rng.sample(candidates, min(n_per_crit, len(candidates)))
        for idx, rec in pair:
            sampled_clips.append({
                "clip_idx": idx,
                "primary_criterion": crit,
                "record": rec,
            })
            sampled_keys.add(idx)

    return sampled_clips
