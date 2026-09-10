#!/usr/bin/env python3
"""Onset-pointwise evaluation for CVS-Act synthetic validation predictions.

This is the training-side companion to
``evaluate_cvs_act_v1_onset_pointwise.py``. It reads the synthetic validation
``val.json`` created by the training split notebook, evaluates predictions at
actor action onsets, and writes the CVS-Act final score plus the separate
canonical, presence, and conditional-label diagnostic summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from evaluate_audit_v11_simple_recommendation import (  # noqa: E402
    gt_records_by_video,
    parse_modified_since,
)
from evaluate_cvs_act_v1_onset_pointwise import (  # noqa: E402
    ACTORS,
    GRANULARITIES,
    collect_eval_points,
    evaluate_points,
    fv,
    inventory,
    macro_actor_rows,
    summarize_labels,
    summarize_presence,
    write_csv,
    write_report,
)
from evaluate_cvs_act_v1_synthetic_val import (  # noqa: E402
    DEFAULT_VAL_JSON,
    collect_method_frames,
    load_synthetic_val_records,
)
from cvs_act.action_segment_eval import write_json  # noqa: E402
from cvs_act.evaluation import (  # noqa: E402
    DEFAULT_FINAL_SCORE_COMPONENTS,
    score_onset_pointwise_final_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-json",
        default=str(DEFAULT_VAL_JSON.relative_to(ROOT_DIR)),
        help="Synthetic validation JSON created by the train/val split notebook.",
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        required=True,
        help="One or more prediction output directories to evaluate.",
    )
    parser.add_argument(
        "--metrics-base",
        default="metrics_cvs_act_v1_synthetic_val_onset_pointwise",
        help="Output metrics directory, relative to the repo root unless absolute.",
    )
    parser.add_argument("--models", nargs="*", default=[], help="Optional model id filter.")
    parser.add_argument("--taxonomy", nargs="*", default=[], help="Optional taxonomy version filter.")
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--common-only", action="store_true")
    parser.add_argument(
        "--modified-since",
        default="",
        help="Optional local timestamp cutoff like '2026-05-24 17:00:00'.",
    )
    parser.add_argument(
        "--default-clip-level",
        choices=["coarse", "fine"],
        default="coarse",
        help="Clip level assigned to synthetic records that do not already contain clip_level.",
    )
    parser.add_argument(
        "--canonical-point-source",
        default="coarse_actor_onset",
        help="Point source used for the headline canonical score.",
    )
    parser.add_argument(
        "--canonical-granularity",
        choices=list(GRANULARITIES),
        default="exact",
        help="Label granularity used for the headline canonical score.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT_DIR / path


def _prepare_records(records: Sequence[dict], default_clip_level: str) -> list[dict]:
    out = []
    for record in records:
        item = dict(record)
        item.setdefault("clip_level", default_clip_level)
        out.append(item)
    return out


def _point_correct(row: dict, granularity: str) -> bool | None:
    if row["gt_presence_excluded_uncertain"] or row["gt_presence_unknown"] or row["pred_presence_unknown"]:
        return None
    gt_present = bool(row["gt_present"])
    pred_present = bool(row["pred_present"])
    if not gt_present:
        return not pred_present
    return pred_present and bool(row.get(f"label_match_{granularity}"))


def summarize_canonical_scores(rows: Sequence[dict], granularity: str) -> list[dict]:
    keys = ["method", "method_label", "variant_label", "model_short", "clip_level", "point_source", "actor"]
    grouped: dict[tuple, list[bool]] = defaultdict(list)
    for row in rows:
        correct = _point_correct(row, granularity)
        if correct is None:
            continue
        grouped[tuple(row.get(key) for key in keys)].append(correct)

    out = []
    for key, values in sorted(grouped.items()):
        item = dict(zip(keys, key))
        item.update(
            {
                "granularity": granularity,
                "n": len(values),
                "n_correct": sum(values),
                "canonical_exact_action_score": mean(values) if values else None,
            }
        )
        out.append(item)
    out.extend(
        macro_actor_rows(
            out,
            ["canonical_exact_action_score"],
            include_granularity=True,
        )
    )
    return out


def select_headline_rows(
    canonical_rows: Sequence[dict],
    *,
    point_source: str,
    granularity: str,
) -> list[dict]:
    return [
        row
        for row in canonical_rows
        if row.get("actor") == "macro_actor"
        and row.get("point_source") == point_source
        and row.get("granularity") == granularity
    ]


def write_compact_summary(
    path: Path,
    headline_rows: Sequence[dict],
    *,
    point_source: str,
    granularity: str,
) -> None:
    lines = [
        "## Synthetic Val Onset-Pointwise Compact Summary",
        "",
        f"Canonical score: macro-actor exact action score at `{point_source}` / `{granularity}`.",
        "",
        "| Method | n | Canonical score |",
        "| --- | ---: | ---: |",
    ]
    for row in sorted(headline_rows, key=lambda r: str(r.get("method"))):
        score = row.get("canonical_exact_action_score")
        score_text = f"{score:.3f}" if score is not None else "nan"
        lines.append(f"| {row.get('method')} | {row.get('n')} | {score_text} |")
    path.write_text("\n".join(lines) + "\n")


def write_canonical_report(
    path: Path,
    headline_rows: Sequence[dict],
    *,
    point_source: str,
    granularity: str,
) -> None:
    lines = [
        "# Synthetic Val Onset-Pointwise Canonical Scores",
        "",
        f"Canonical score: macro-actor exact action score at `{point_source}` / `{granularity}`.",
        "",
        "A point is correct when GT absence is predicted absent, or when GT presence is predicted present with the exact actor label matched.",
        "",
        "| Method | n | Canonical score |",
        "| --- | ---: | ---: |",
    ]
    for row in sorted(headline_rows, key=lambda r: str(r.get("method"))):
        lines.append(
            f"| {row.get('method')} | {row.get('n')} | {fv(row.get('canonical_exact_action_score'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_final_score_report(path: Path, final_rows: Sequence[dict]) -> None:
    policy = "+".join(
        f"{actor}/{granularity}" for actor, granularity in DEFAULT_FINAL_SCORE_COMPONENTS
    )
    lines = [
        "# Synthetic Val Onset-Pointwise Final Scores",
        "",
        f"Final score: mean clip-averaged conditional label macro-F1 over `{policy}`.",
        "",
        "Rows use `coarse` clips at `coarse_actor_onset`; each component is "
        "scored only on GT-present, non-unknown/non-uncertain points, then "
        "averaged over clips.",
        "",
        "| Method | Left exact | Right medium | Camera coarse | Final score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(final_rows, key=lambda r: str(r.get("method"))):
        lines.append(
            f"| {row.get('method')} | "
            f"{fv(row.get('left_exact'))} | "
            f"{fv(row.get('right_medium'))} | "
            f"{fv(row.get('camera_coarse'))} | "
            f"{fv(row.get('final_score'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    val_json = _resolve_repo_path(args.val_json)
    pred_roots = [_resolve_repo_path(path) for path in args.output_dirs]
    metrics_dir = _resolve_repo_path(args.metrics_base)

    gt_records = _prepare_records(load_synthetic_val_records(val_json), args.default_clip_level)
    gt_by_video = gt_records_by_video(gt_records)
    modified_since = parse_modified_since(args.modified_since)
    model_filter = set(args.models) if args.models else None
    taxonomy_filter = set(args.taxonomy) if args.taxonomy else None

    frames_by_method, skipped = collect_method_frames(
        pred_roots,
        gt_by_video,
        include_partial=args.include_partial,
        modified_since=modified_since,
        model_filter=model_filter,
        taxonomy_filter=taxonomy_filter,
        verbose=args.verbose,
    )

    if args.common_only and frames_by_method:
        common_video_ids = set.intersection(*(set(by_video) for by_video in frames_by_method.values()))
        frames_by_method = {
            method: {
                video_id: frames
                for video_id, frames in by_video.items()
                if video_id in common_video_ids
            }
            for method, by_video in frames_by_method.items()
        }
        print(f"Common videos across {len(frames_by_method)} group(s): {len(common_video_ids)}")

    eval_points = collect_eval_points(gt_records)
    eval_rows = evaluate_points(frames_by_method, eval_points)
    presence = summarize_presence(eval_rows)
    labels = summarize_labels(eval_rows)
    canonical = summarize_canonical_scores(eval_rows, args.canonical_granularity)
    headline = select_headline_rows(
        canonical,
        point_source=args.canonical_point_source,
        granularity=args.canonical_granularity,
    )
    final_scores = score_onset_pointwise_final_components(
        eval_rows,
        clip_level=args.default_clip_level,
        point_source=args.canonical_point_source,
    )

    inv = inventory(gt_records, frames_by_method, eval_rows)
    inv.update(
        {
            "val_json": str(val_json),
            "output_dirs": [str(path) for path in pred_roots],
            "modified_since": args.modified_since,
            "common_only": args.common_only,
            "default_clip_level": args.default_clip_level,
            "canonical_point_source": args.canonical_point_source,
            "canonical_granularity": args.canonical_granularity,
            "final_score_components": list(DEFAULT_FINAL_SCORE_COMPONENTS),
            "skipped": skipped,
            "n_gt_records": len(gt_records),
            "n_gt_videos": len(gt_by_video),
            "n_eval_points_defined": len(eval_points),
            "n_eval_rows": len(eval_rows),
            "n_pred_videos_by_method": {
                method: len(by_video) for method, by_video in sorted(frames_by_method.items())
            },
        }
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "onset_pointwise_details.csv", eval_rows)
    write_csv(metrics_dir / "presence_summary.csv", presence)
    write_csv(metrics_dir / "conditional_label_summary.csv", labels)
    write_csv(metrics_dir / "canonical_score_summary.csv", canonical)
    write_csv(metrics_dir / "canonical_headline.csv", headline)
    write_csv(metrics_dir / "final_score_summary.csv", final_scores)
    write_json(metrics_dir / "inventory.json", inv)
    write_report(metrics_dir / "onset_pointwise_report.md", inv, presence, labels)
    write_canonical_report(
        metrics_dir / "canonical_score_report.md",
        headline,
        point_source=args.canonical_point_source,
        granularity=args.canonical_granularity,
    )
    write_compact_summary(
        metrics_dir / "compact_summary.txt",
        headline,
        point_source=args.canonical_point_source,
        granularity=args.canonical_granularity,
    )
    write_final_score_report(metrics_dir / "final_score_report.md", final_scores)

    print(f"GT records: {len(gt_records)}  videos: {len(gt_by_video)}")
    for method, by_video in sorted(frames_by_method.items()):
        print(f"{method}: {len(by_video)} video(s)")
    print(f"Defined evaluation points: {len(eval_points)}")
    print(f"Evaluated rows: {len(eval_rows)}")
    print(
        "Canonical:",
        f"point_source={args.canonical_point_source}",
        f"granularity={args.canonical_granularity}",
    )
    for row in sorted(headline, key=lambda r: str(r.get("method"))):
        print(
            f"{row.get('method')}: "
            f"{fv(row.get('canonical_exact_action_score'))} "
            f"(n={row.get('n')})"
        )
    print(
        "Final score policy:",
        "+".join(f"{actor}/{granularity}" for actor, granularity in DEFAULT_FINAL_SCORE_COMPONENTS),
    )
    for row in sorted(final_scores, key=lambda r: str(r.get("method"))):
        print(f"{row.get('method')}: final_score={fv(row.get('final_score'))}")
    print(f"Wrote {metrics_dir / 'canonical_headline.csv'}")
    print(f"Wrote {metrics_dir / 'final_score_summary.csv'}")
    print(f"Wrote {metrics_dir / 'canonical_score_summary.csv'}")
    print(f"Wrote {metrics_dir / 'presence_summary.csv'}")
    print(f"Wrote {metrics_dir / 'conditional_label_summary.csv'}")
    print(f"Wrote {metrics_dir / 'onset_pointwise_details.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
