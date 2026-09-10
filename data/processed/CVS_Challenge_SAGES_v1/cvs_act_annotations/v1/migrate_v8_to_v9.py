#!/usr/bin/env python3
"""Migrate audit_v8 -> audit_v9 and taxonomy_v8 -> taxonomy_v9.

v9 removes all "maintain" intention actions from annotations:
  - "maintain exposure of the hepatocystic triangle"
  - "maintain exposure of the cystic plate"
  - "maintain exposure of the two tubular structures"

Rationale: "maintain" actions describe passive holding (stable traction) rather
than active surgical manoeuvres. Removing them focuses the taxonomy on
*changes* the surgeon makes to advance toward CVS.

Applies to:
  - audit_v8/          -> audit_v9/          (full annotation set)
  - train/audit_v8/    -> train/audit_v9/    (train split)

Also produces:
  - taxonomy_v9.json   (updated counts, removed maintain intentions)
  - taxonomy_annotation_guide_v9.md
"""

import copy
import json
import glob
import os
import sys
from collections import Counter
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MAINTAIN_INTENTIONS = {
    "maintain exposure of the hepatocystic triangle",
    "maintain exposure of the cystic plate",
    "maintain exposure of the two tubular structures",
}


# ---------------------------------------------------------------------------
# Annotation migration
# ---------------------------------------------------------------------------

def filter_maintain_actions(actions_ranked: list) -> list:
    """Remove actions whose intention is a maintain intention.

    Re-ranks remaining actions starting from 1.
    """
    filtered = [a for a in actions_ranked if a.get("intention") not in MAINTAIN_INTENTIONS]
    for i, a in enumerate(filtered):
        a["rank"] = i + 1
    return filtered


def migrate_annotation_file(src_path: str, dst_path: str, stats: dict):
    """Migrate a single annotation JSON file from v8 to v9."""
    with open(src_path) as f:
        data = json.load(f)

    out = []
    for entry in data:
        entry = copy.deepcopy(entry)

        # Filter coarse actions
        coarse = entry.get("coarse", {})
        annotation = coarse.get("annotation", {})
        actions = annotation.get("actions_ranked", [])
        before = len(actions)
        filtered = filter_maintain_actions(actions)
        stats["total_actions"] += before
        stats["removed_actions"] += before - len(filtered)
        annotation["actions_ranked"] = filtered

        # Filter fine actions
        for fine_block in entry.get("fine", []):
            fine_ann = fine_block.get("annotation", {})
            fine_actions = fine_ann.get("actions_ranked", [])
            before_f = len(fine_actions)
            filtered_f = filter_maintain_actions(fine_actions)
            stats["total_actions"] += before_f
            stats["removed_actions"] += before_f - len(filtered_f)
            fine_ann["actions_ranked"] = filtered_f

        out.append(entry)
        stats["entries"] += 1

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    stats["files"] += 1


def migrate_dir(src_dir: str, dst_dir: str, label: str) -> dict:
    """Migrate all JSON files in a directory."""
    stats = {"files": 0, "entries": 0, "total_actions": 0, "removed_actions": 0}
    src_files = sorted(glob.glob(os.path.join(src_dir, "*.json")))
    if not src_files:
        print(f"  [{label}] No files found in {src_dir}")
        return stats
    for src_path in src_files:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        migrate_annotation_file(src_path, dst_path, stats)
    print(f"  [{label}] {stats['files']} files, "
          f"{stats['total_actions']} actions -> removed {stats['removed_actions']} maintain "
          f"({stats['total_actions'] - stats['removed_actions']} remaining)")
    return stats


# ---------------------------------------------------------------------------
# Taxonomy migration
# ---------------------------------------------------------------------------

def build_taxonomy_v9(v8_path: str, dst_path: str, all_audit_dirs: list):
    """Build taxonomy_v9.json from v8, removing maintain intentions and re-counting."""
    with open(v8_path) as f:
        v8 = json.load(f)

    v9 = copy.deepcopy(v8)

    # Update meta
    v9["_meta"] = {
        "version": "v9",
        "description": (
            "CVS-ACT action taxonomy v9: remove 'maintain' intentions "
            "(maintain exposure of the hepatocystic triangle, maintain exposure "
            "of the cystic plate, maintain exposure of the two tubular structures). "
            "Focuses taxonomy on active surgical manoeuvres only."
        ),
        "created_at": datetime.now().isoformat(),
        "base_version": "v8",
    }

    # Re-count all fields from the migrated v9 audit files
    field_counts = {
        "actor_role": Counter(),
        "tool_type": Counter(),
        "action_code": Counter(),
        "target_structure": Counter(),
        "target_context": Counter(),
        "intention": Counter(),
    }

    for audit_dir in all_audit_dirs:
        v9_dir = audit_dir.replace("audit_v8", "audit_v9")
        for fpath in sorted(glob.glob(os.path.join(v9_dir, "*.json"))):
            data = json.load(open(fpath))
            for entry in data:
                for block in [entry.get("coarse", {})] + entry.get("fine", []):
                    for a in block.get("annotation", {}).get("actions_ranked", []):
                        field_counts["actor_role"][a.get("actor_role", "(not set)")] += 1
                        field_counts["tool_type"][a.get("tool_type", "(not set)")] += 1
                        field_counts["action_code"][a.get("action_code", "(not set)")] += 1
                        field_counts["target_structure"][a.get("target_structure", "(not set)")] += 1
                        for tc_field in ("target_context_1", "target_context_2"):
                            tc = a.get(tc_field, "(not set)")
                            if tc and tc != "(not set)":
                                field_counts["target_context"][tc] += 1
                        field_counts["intention"][a.get("intention", "(not set)")] += 1

    # Update field counts in taxonomy
    for field_name in ("actor_role", "tool_type", "action_code", "target_structure", "intention"):
        if field_name not in v9.get("fields", {}):
            continue
        counts = field_counts[field_name]
        updated = []
        for item in v9["fields"][field_name]:
            val = item["value"]
            c = counts.get(val, 0)
            if c > 0:
                item_copy = copy.deepcopy(item)
                item_copy["count"] = c
                updated.append(item_copy)
        # Sort by count descending
        updated.sort(key=lambda x: -x["count"])
        v9["fields"][field_name] = updated

    # Update target_context
    if "target_context" in v9.get("fields", {}):
        tc_counts = field_counts["target_context"]
        updated_tc = []
        # Keep existing entries with updated counts
        existing_vals = set()
        for item in v9["fields"]["target_context"]:
            val = item["value"]
            existing_vals.add(val)
            c = tc_counts.get(val, 0)
            if c > 0:
                item_copy = copy.deepcopy(item)
                item_copy["count"] = c
                updated_tc.append(item_copy)
        # Add any new ones
        for val, c in tc_counts.items():
            if val not in existing_vals and c > 0:
                updated_tc.append({"label": val, "value": val, "count": c})
        updated_tc.sort(key=lambda x: -x["count"])
        v9["fields"]["target_context"] = updated_tc

    # Update top-level intentions list (mirror of fields.intention)
    v9["intentions"] = [i for i in v9["fields"]["intention"]]

    # Add v9 rules
    v9["v9_rules"] = {
        "removed_intentions": sorted(MAINTAIN_INTENTIONS),
        "rationale": (
            "Maintain actions describe passive holding (stable traction) rather "
            "than active surgical manoeuvres. Removing them focuses the taxonomy "
            "on changes the surgeon makes to advance toward CVS."
        ),
    }

    with open(dst_path, "w") as f:
        json.dump(v9, f, indent=2, ensure_ascii=False)
    print(f"  Taxonomy v9 saved to {dst_path}")

    # Print summary
    total = sum(field_counts["intention"].values())
    print(f"  Total v9 actions: {total}")
    print(f"  Intentions ({len(field_counts['intention'])} unique):")
    for intent, c in field_counts["intention"].most_common():
        print(f"    {c:4d}  {intent}")


# ---------------------------------------------------------------------------
# Annotation guide
# ---------------------------------------------------------------------------

def build_annotation_guide_v9(taxonomy_path: str, guide_path: str):
    """Generate a markdown annotation guide from taxonomy_v9."""
    with open(taxonomy_path) as f:
        tax = json.load(f)

    fields = tax["fields"]
    lines = [
        "# CVS-Act Action Taxonomy v9 \u2014 Annotation Guide",
        "",
        "## Overview",
        "",
        "v9 builds on v8 by **removing all \u201cmaintain\u201d intentions**. The taxonomy now",
        "covers only *active* surgical actions that change the state of the operative",
        "field. Passive holding / stable traction is implicitly assumed whenever a",
        "grasper is visible and retracted.",
        "",
        "### Removed intentions",
        "",
    ]
    for intent in sorted(MAINTAIN_INTENTIONS):
        lines.append(f"- ~~{intent}~~")
    lines += [
        "",
        "If the *only* action in a frame interval is passive holding, the",
        "`actions_ranked` list will be empty for that interval.",
        "",
        "---",
        "",
        "## Fields",
        "",
    ]

    field_order = [
        ("actor_role", "Actor Role", "Who performs the action."),
        ("tool_type", "Tool Type", "Which instrument is used."),
        ("action_code", "Action Code", "The dominant surgical action."),
        ("target_structure", "Target Structure", "The anatomical structure physically manipulated."),
        ("target_context", "Target Context", "Spatial context describing where in the anatomy the action occurs."),
        ("intention", "Intention", "The surgical goal of this action."),
    ]

    for field_key, field_title, field_desc in field_order:
        items = fields.get(field_key, [])
        lines.append(f"### {field_title}")
        lines.append("")
        lines.append(field_desc)
        lines.append("")
        lines.append("| Value | Count | Description |")
        lines.append("|-------|------:|-------------|")
        for item in items:
            val = item["value"]
            count = item.get("count", 0)
            desc = item.get("description", "")
            lines.append(f"| `{val}` | {count} | {desc} |")
        lines.append("")

    # Action code natural-language mappings
    nl = tax.get("nl_mappings", {})
    if nl.get("action_text"):
        lines += [
            "---",
            "",
            "## Action Code \u2192 Natural Language",
            "",
            "| Code | Phrase |",
            "|------|--------|",
        ]
        for code, phrase in sorted(nl["action_text"].items()):
            lines.append(f"| `{code}` | {phrase} |")
        lines.append("")

    # Presets
    presets = tax.get("presets", [])
    if presets:
        lines += [
            "---",
            "",
            "## Common Presets",
            "",
            "| Name | Actor | Tool | Action | Target | Intention |",
            "|------|-------|------|--------|--------|-----------|",
        ]
        for p in presets:
            lines.append(
                f"| {p.get('name', '')} | {p.get('actor_role', '')} | "
                f"{p.get('tool_type', '')} | {p.get('action_code', '')} | "
                f"{p.get('target_structure', '')} | {p.get('intention', '')} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## Annotation Rules",
        "",
        "1. Each frame interval gets an `actions_ranked` list ordered by importance.",
        "2. If the only visible action is passive holding / stable traction, leave",
        "   `actions_ranked` empty.",
        "3. Pick **one** `action_code` per action entry (the dominant action).",
        "4. `target_context_1` and `target_context_2` are optional spatial qualifiers.",
        "   Use `(not set)` when not applicable.",
        "5. Camera actions: `actor_role=camera`, `tool_type=camera`.",
        "   Intention is determined by `action_code`:",
        "   - `CAMERA_ZOOM_OUT` \u2192 \"obtain a contextualized view\"",
        "   - `CAMERA_ZOOM_IN` \u2192 \"obtain a close-up focused view\"",
        "   - `CAMERA_REPOSITION` \u2192 \"adjust the viewing angle\"",
        "",
        f"*Generated {datetime.now().strftime('%Y-%m-%d')} from taxonomy_v9.json*",
        "",
    ]

    with open(guide_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Annotation guide saved to {guide_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Migrating v8 -> v9 (removing maintain intentions)")
    print("=" * 70)

    # Directories to migrate
    audit_pairs = [
        (os.path.join(BASE_DIR, "audit_v8"), os.path.join(BASE_DIR, "audit_v9")),
        (os.path.join(BASE_DIR, "train", "audit_v8"), os.path.join(BASE_DIR, "train", "audit_v9")),
    ]

    # Check source dirs exist
    for src, dst in audit_pairs:
        if not os.path.isdir(src):
            print(f"ERROR: Source dir not found: {src}", file=sys.stderr)
            sys.exit(1)

    # Migrate annotations
    print("\n--- Migrating annotations ---")
    for src, dst in audit_pairs:
        label = os.path.relpath(src, BASE_DIR)
        migrate_dir(src, dst, label)

    # Build taxonomy
    print("\n--- Building taxonomy_v9.json ---")
    v8_taxonomy = os.path.join(BASE_DIR, "taxonomy_v8.json")
    v9_taxonomy = os.path.join(BASE_DIR, "taxonomy_v9.json")
    all_audit_dirs = [src for src, _ in audit_pairs]
    build_taxonomy_v9(v8_taxonomy, v9_taxonomy, all_audit_dirs)

    # Build annotation guide
    print("\n--- Building annotation guide ---")
    guide_path = os.path.join(BASE_DIR, "taxonomy_annotation_guide_v9.md")
    build_annotation_guide_v9(v9_taxonomy, guide_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
