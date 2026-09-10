import base64
import copy
import datetime as dt
import io
import json
import time
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ipywidgets as widgets
from PIL import Image

from .extraction import load_video_data
from .schemas import CRITERIA

CAMERA_ACTION_CODES = [
    "CAMERA_ZOOM_IN",
    "CAMERA_ZOOM_OUT",
    "CAMERA_REPOSITION",
    "CAMERA_UNCERTAIN",
]

RETRACTION_CHANGE_OPTIONS = ["unsure", "yes", "no"]
RETRACTION_DIRECTION_OPTIONS = [
    "unsure",
    "left",
    "right",
    "upward",
    "downward",
    "not_retracted",
    "unclear",
]

EXTRA_ACTION_CODES = [
    "sweeping",
    "CLIP",
    "KEEP_RETRACT_LATERAL",
    "KEEP_RETRACT_UPWARD",
    "KEEP_RETRACT_MEDIAL",
    "RETRACT_LATERAL_TO_UPWARD",
    "RETRACT_UPWARD_TO_LATERAL",
    "RETRACT_MEDIAL_TO_UPWARD",
    "RETRACT_UPWARD_TO_MEDIAL",
]

EXTRA_TOOL_TYPES = [
    "clipper",
]


def _find_repo_root() -> str:
    cur = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(cur, "data")) and os.path.isdir(os.path.join(cur, "src")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Could not find repo root containing data/ and src/")
        cur = parent


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_version(version) -> int:
    if isinstance(version, int):
        return version
    if isinstance(version, str):
        m = re.search(r"(\d+)", version)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not parse version from {version!r}")


def _version_name(prefix: str, version: int) -> str:
    return f"{prefix}_v{int(version)}"


def _safe_get(dct, *keys, default=None):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coarse_key(record: dict) -> tuple:
    coarse = record.get("coarse", {})
    return (
        record.get("criterion"),
        int(coarse.get("start_frame", -1)),
        int(coarse.get("mid_frame", -1)),
        int(coarse.get("end_frame", -1)),
    )


def _img_to_data_url(path: str, max_height: int = 190) -> str:
    img = Image.open(path)
    if img.height > max_height:
        ratio = max_height / img.height
        img = img.resize((max(1, int(img.width * ratio)), max_height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _frame_block_html(frame_dir: str, frame_ids: List[int], title: str = "", max_height: int = 190) -> str:
    parts = []
    if title:
        parts.append(
            f'<div style="font-weight:700;font-size:13px;color:#0f172a;margin:0 0 6px 0;">{title}</div>'
        )
    parts.append('<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;">')
    for frame_id in frame_ids:
        frame_path = os.path.join(frame_dir, f"frame_{int(frame_id):06d}.png")
        if os.path.isfile(frame_path):
            url = _img_to_data_url(frame_path, max_height=max_height)
            parts.append(
                '<div style="text-align:center;">'
                f'<img src="{url}" style="height:{max_height}px;border:1px solid #cbd5e1;border-radius:6px;" />'
                f'<div style="font-size:11px;color:#475569;margin-top:2px;">f{int(frame_id):06d}</div>'
                '</div>'
            )
        else:
            parts.append(
                '<div style="display:flex;align-items:center;justify-content:center;'
                f'height:{max_height}px;width:180px;background:#f8fafc;border:1px dashed #cbd5e1;'
                'border-radius:6px;color:#94a3b8;font-size:12px;">'
                f"missing f{int(frame_id):06d}"
                "</div>"
            )
    parts.append("</div>")
    return "".join(parts)


def _label_color(value: float) -> str:
    if value >= 0.8:
        return "#065f46"
    if value >= 0.5:
        return "#a16207"
    return "#b91c1c"


def _frame_metrics_html(metrics: Optional[dict]) -> str:
    if not metrics:
        return ""
    pills = []
    for crit in CRITERIA:
        info = metrics.get(crit)
        if not info:
            continue
        val = float(info["raw"])
        color = _label_color(val)
        pills.append(
            f'<span style="padding:1px 5px;border-radius:999px;'
            f'background:{color}18;color:{color};font-size:10px;font-weight:700;">'
            f'{crit.upper()} {val:.2f}</span>'
        )
    return (
        '<div style="display:flex;gap:4px;justify-content:center;flex-wrap:wrap;'
        'margin-top:3px;max-width:210px;">'
        + "".join(pills)
        + "</div>"
    )


def _get_frame_metrics(video_data: dict, video_id: str, frame_id: int) -> Optional[dict]:
    if video_id not in video_data:
        return None
    vdata = video_data[video_id]
    df = vdata["df"]
    matches = df.index[df["frame_id"] == int(frame_id)].tolist()
    if not matches:
        return None
    row_idx = matches[0]
    out = {}
    for crit in CRITERIA:
        rater_cols = [f"{crit}_rater{r}" for r in range(1, 4)]
        out[crit] = {"raw": float(df.iloc[row_idx][rater_cols].mean())}
    return out


def _frame_block_html_with_metrics(
    frame_dir: str,
    frame_ids: List[int],
    video_data: dict,
    video_id: str,
    title: str = "",
    max_height: int = 190,
) -> str:
    parts = []
    if title:
        parts.append(
            f'<div style="font-weight:700;font-size:13px;color:#0f172a;margin:0 0 6px 0;">{title}</div>'
        )
    parts.append('<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start;">')
    for frame_id in frame_ids:
        frame_path = os.path.join(frame_dir, f"frame_{int(frame_id):06d}.png")
        metrics_html = _frame_metrics_html(_get_frame_metrics(video_data, video_id, int(frame_id)))
        if os.path.isfile(frame_path):
            url = _img_to_data_url(frame_path, max_height=max_height)
            parts.append(
                '<div style="text-align:center;">'
                f'<img src="{url}" style="height:{max_height}px;border:1px solid #cbd5e1;border-radius:6px;" />'
                f'<div style="font-size:11px;color:#475569;margin-top:2px;">f{int(frame_id):06d}</div>'
                f"{metrics_html}"
                '</div>'
            )
        else:
            parts.append(
                '<div style="text-align:center;">'
                '<div style="display:flex;align-items:center;justify-content:center;'
                f'height:{max_height}px;width:180px;background:#f8fafc;border:1px dashed #cbd5e1;'
                'border-radius:6px;color:#94a3b8;font-size:12px;">'
                f"missing f{int(frame_id):06d}"
                "</div>"
                f"{metrics_html}"
                "</div>"
            )
    parts.append("</div>")
    return "".join(parts)


def _camera_panel_html(frame_dir: str, camera_record: Optional[dict]) -> str:
    if not camera_record:
        return (
            '<div style="padding:10px 12px;border:1px solid #e2e8f0;border-radius:8px;'
            'background:#f8fafc;color:#64748b;font-size:13px;">No camera sidecar record for this clip.</div>'
        )

    panels = [
        '<div style="padding:10px 12px;border:1px solid #dbeafe;border-radius:8px;background:#eff6ff;">'
        '<div style="font-weight:700;color:#1d4ed8;margin-bottom:6px;">Camera sidecar</div>'
    ]
    if camera_record.get("notes"):
        panels.append(
            f'<div style="font-size:12px;color:#334155;margin-bottom:6px;"><b>Notes:</b> {camera_record["notes"]}</div>'
        )
    if camera_record.get("mark_uncertain"):
        panels.append(
            '<div style="font-size:12px;color:#b45309;margin-bottom:6px;"><b>Marked uncertain</b></div>'
        )

    for idx, seg in enumerate(camera_record.get("camera_segments", []), start=1):
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        mid = (start + end) // 2
        action_codes = ", ".join(seg.get("action_codes", [])) or "(none)"
        intention_map = seg.get("intention_by_action", {}) or {}
        intentions = "".join(
            f"<li><b>{code}</b>: {text}</li>" for code, text in intention_map.items()
        ) or "<li>(none)</li>"
        panels.append(
            '<div style="margin:10px 0;padding:10px;border:1px solid #bfdbfe;border-radius:8px;background:#ffffff;">'
            f'<div style="font-weight:600;color:#1e3a8a;margin-bottom:6px;">Segment {idx}: '
            f"f{start:06d} - f{end:06d}</div>"
            f'<div style="font-size:12px;color:#334155;margin-bottom:6px;"><b>Camera actions:</b> {action_codes}</div>'
            f'<ul style="margin:0 0 8px 18px;font-size:12px;color:#334155;">{intentions}</ul>'
            f'{_frame_block_html(frame_dir, [start, mid, end], title="", max_height=150)}'
            "</div>"
        )
    panels.append("</div>")
    return "".join(panels)


def _camera_panel_html_with_metrics(
    frame_dir: str,
    video_data: dict,
    video_id: str,
    camera_record: Optional[dict],
) -> str:
    if not camera_record:
        return (
            '<div style="padding:10px 12px;border:1px solid #e2e8f0;border-radius:8px;'
            'background:#f8fafc;color:#64748b;font-size:13px;">No camera sidecar record for this clip.</div>'
        )

    panels = [
        '<div style="padding:10px 12px;border:1px solid #dbeafe;border-radius:8px;background:#eff6ff;">'
        '<div style="font-weight:700;color:#1d4ed8;margin-bottom:6px;">Camera sidecar</div>'
    ]
    if camera_record.get("notes"):
        panels.append(
            f'<div style="font-size:12px;color:#334155;margin-bottom:6px;"><b>Notes:</b> {camera_record["notes"]}</div>'
        )
    if camera_record.get("mark_uncertain"):
        panels.append(
            '<div style="font-size:12px;color:#b45309;margin-bottom:6px;"><b>Marked uncertain</b></div>'
        )

    for idx, seg in enumerate(camera_record.get("camera_segments", []), start=1):
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        mid = (start + end) // 2
        action_codes = ", ".join(seg.get("action_codes", [])) or "(none)"
        intention_map = seg.get("intention_by_action", {}) or {}
        intentions = "".join(
            f"<li><b>{code}</b>: {text}</li>" for code, text in intention_map.items()
        ) or "<li>(none)</li>"
        panels.append(
            '<div style="margin:10px 0;padding:10px;border:1px solid #bfdbfe;border-radius:8px;background:#ffffff;">'
            f'<div style="font-weight:600;color:#1e3a8a;margin-bottom:6px;">Segment {idx}: '
            f"f{start:06d} - f{end:06d}</div>"
            f'<div style="font-size:12px;color:#334155;margin-bottom:6px;"><b>Camera actions:</b> {action_codes}</div>'
            f'<ul style="margin:0 0 8px 18px;font-size:12px;color:#334155;">{intentions}</ul>'
            f'{_frame_block_html_with_metrics(frame_dir, [start, mid, end], video_data, video_id, title="", max_height=150)}'
            "</div>"
        )
    panels.append("</div>")
    return "".join(panels)


def _frame_range(start_frame: int, end_frame: int, step: int = 30) -> List[int]:
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    return list(range(int(start_frame), int(end_frame) + 1, int(step)))


def _existing_clip_frames(frame_dir: str, start_frame: int, end_frame: int, step: int = 30) -> List[int]:
    frame_ids = _frame_range(start_frame, end_frame, step=step)
    existing = [
        frame_id
        for frame_id in frame_ids
        if os.path.isfile(os.path.join(frame_dir, f"frame_{int(frame_id):06d}.png"))
    ]
    return existing or frame_ids


def _coarse_preview_frames(frame_dir: str, start_frame: int, end_frame: int) -> List[int]:
    # Existing extracted frames are 1 fps (every 30 raw frames). For coarse preview,
    # show a sparser 5-second cadence by default and let the user expand to all 1 fps frames.
    preview = _existing_clip_frames(frame_dir, start_frame, end_frame, step=150)
    anchors = [int(start_frame), int(end_frame)]
    out = []
    for frame_id in sorted(set(preview + anchors)):
        frame_path = os.path.join(frame_dir, f"frame_{int(frame_id):06d}.png")
        if os.path.isfile(frame_path) or frame_id in anchors:
            out.append(frame_id)
    return sorted(set(out))


def _options_from_field(items: List[dict]) -> List[str]:
    values = [item["value"] for item in items]
    if "(not set)" in values:
        values = ["(not set)"] + [v for v in values if v != "(not set)"]
    return values


def _dropdown(options: List[str], value: Optional[str], width: str = "280px") -> widgets.Dropdown:
    clean_options = list(options)
    if value not in clean_options:
        clean_options = clean_options + [value] if value else clean_options
    chosen = value if value in clean_options else clean_options[0]
    return widgets.Dropdown(
        options=clean_options,
        value=chosen,
        layout=widgets.Layout(width=width),
    )


def _text(value: str = "", width: str = "280px") -> widgets.Text:
    return widgets.Text(value=value or "", layout=widgets.Layout(width=width))


def _textarea(value: str = "", width: str = "100%") -> widgets.Textarea:
    return widgets.Textarea(
        value=value or "",
        layout=widgets.Layout(width=width, height="70px"),
    )


def _short_textarea(value: str = "", width: str = "100%", height: str = "52px") -> widgets.Textarea:
    return widgets.Textarea(
        value=value or "",
        layout=widgets.Layout(width=width, height=height),
    )


def _default_action(rank: int) -> dict:
    return {
        "rank": rank,
        "actor_role": "(not set)",
        "tool_type": "(not set)",
        "action_code": "(not set)",
        "target_structure": "(not set)",
        "target_context_1": "(not set)",
        "target_context_2": "(not set)",
        "intention": "(not set)",
        "one_sentence": "",
        "original_sentence": "",
        "generated_sentence": "",
        "rationale": "",
        "confidence": 0.0,
        "extra_info": "",
    }


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_set(value) -> bool:
    return value not in (None, "", "(not set)")


def _has_interval(start_frame: int, end_frame: int) -> bool:
    return bool(start_frame or end_frame)


def _pop_numbered_fields(out: dict, prefixes: List[str]) -> None:
    for key in list(out):
        if any(key.startswith(prefix) for prefix in prefixes):
            out.pop(key, None)


def _schema_reference_html() -> str:
    return (
        '<div style="padding:8px 10px;background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:8px;margin-bottom:10px;font-size:12px;color:#334155;">'
        '<b>Timestamped action schema:</b> '
        'Camera rows save movement action + frame span + description. '
        'Left rows save retraction action + frame span + changed/start direction/end direction/description. '
        'Right rows save active tool + action + target + target context + frame span + description.'
        "</div>"
    )


@dataclass
class ActionEditor:
    parent: "RecordEditor"
    action: dict
    taxonomy_options: Dict[str, List[str]]
    include_cb: widgets.Checkbox = field(init=False)
    actor_dd: widgets.Dropdown = field(init=False)
    tool_dd: widgets.Dropdown = field(init=False)
    tool_box: widgets.VBox = field(init=False)
    action_dd: widgets.Dropdown = field(init=False)
    single_action_box: widgets.VBox = field(init=False)
    camera_action_widgets: List[dict] = field(init=False, default_factory=list)
    camera_actions_box: widgets.VBox = field(init=False)
    camera_rows_box: widgets.VBox = field(init=False)
    add_camera_action_btn: widgets.Button = field(init=False)
    left_action_widgets: List[dict] = field(init=False, default_factory=list)
    left_actions_box: widgets.VBox = field(init=False)
    left_rows_box: widgets.VBox = field(init=False)
    add_left_action_btn: widgets.Button = field(init=False)
    right_action_widgets: List[dict] = field(init=False, default_factory=list)
    right_actions_box: widgets.VBox = field(init=False)
    right_rows_box: widgets.VBox = field(init=False)
    add_right_action_btn: widgets.Button = field(init=False)
    generic_timing_box: widgets.VBox = field(init=False)
    generic_start_input: widgets.IntText = field(init=False)
    generic_end_input: widgets.IntText = field(init=False)
    legacy_target_box: widgets.VBox = field(init=False)
    target1_dd: widgets.Dropdown = field(init=False)
    target2_dd: widgets.Dropdown = field(init=False)
    target3_dd: widgets.Dropdown = field(init=False)
    ctx1_dd: widgets.Dropdown = field(init=False)
    ctx2_dd: widgets.Dropdown = field(init=False)
    intention_dd: widgets.Dropdown = field(init=False)
    confidence_input: widgets.FloatText = field(init=False)
    sentence_input: widgets.Textarea = field(init=False)
    rationale_input: widgets.Textarea = field(init=False)
    extra_input: widgets.Text = field(init=False)
    remove_btn: widgets.Button = field(init=False)
    box: widgets.VBox = field(init=False)

    def __post_init__(self) -> None:
        self.include_cb = widgets.Checkbox(value=True, description="Include", indent=False)
        self.actor_dd = _dropdown(self.taxonomy_options["actor_role"], self.action.get("actor_role"))
        self.actor_dd.observe(self._actor_changed, names="value")
        self.tool_dd = _dropdown(self.taxonomy_options["tool_type"], self.action.get("tool_type"))
        self.action_dd = _dropdown(self.taxonomy_options["action_code"], self.action.get("action_code"))
        self.single_action_box = widgets.VBox([widgets.HTML("<b>Action</b>"), self.action_dd])
        generic_start = self.action.get("action_start_frame", self.action.get("start_frame", 0))
        generic_end = self.action.get("action_end_frame", self.action.get("end_frame", 0))
        self.generic_start_input = widgets.IntText(
            value=int(_safe_float(generic_start, 0.0) or 0),
            layout=widgets.Layout(width="120px"),
        )
        self.generic_end_input = widgets.IntText(
            value=int(_safe_float(generic_end, 0.0) or 0),
            layout=widgets.Layout(width="120px"),
        )
        self.generic_timing_box = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700;color:#0f172a;margin-top:4px;'>Timestamp</div>"
                    "<div style='font-size:12px;color:#475569;'>Use for actions without a camera/left/right actor row.</div>"
                ),
                widgets.HBox(
                    [
                        widgets.HTML("<b>Start</b>"),
                        self.generic_start_input,
                        widgets.HTML("<b>End</b>"),
                        self.generic_end_input,
                    ]
                ),
            ]
        )
        self.camera_actions_box = self._build_camera_actions_box()
        self.left_actions_box = self._build_left_actions_box()
        self.right_actions_box = self._build_right_actions_box()
        target_values = self._initial_target_values()
        self.target1_dd = _dropdown(self.taxonomy_options["target_structure"], target_values[0])
        self.target2_dd = _dropdown(self.taxonomy_options["target_structure"], target_values[1])
        self.target3_dd = _dropdown(self.taxonomy_options["target_structure"], target_values[2])
        self.ctx1_dd = _dropdown(
            self.taxonomy_options["target_context"], self.action.get("target_context_1", "(not set)")
        )
        self.ctx2_dd = _dropdown(
            self.taxonomy_options["target_context"], self.action.get("target_context_2", "(not set)")
        )
        self.intention_dd = _dropdown(self.taxonomy_options["intention"], self.action.get("intention"))
        self.confidence_input = widgets.FloatText(
            value=_safe_float(self.action.get("confidence", 0.0), 0.0),
            layout=widgets.Layout(width="160px"),
        )
        sentence = (
            self.action.get("one_sentence")
            or self.action.get("generated_sentence")
            or self.action.get("original_sentence")
            or ""
        )
        self.sentence_input = _textarea(sentence)
        self.rationale_input = _textarea(self.action.get("rationale", ""))
        self.extra_input = _text(self.action.get("extra_info", ""), width="360px")
        self.remove_btn = widgets.Button(
            description="Remove action",
            button_style="danger",
            layout=widgets.Layout(width="130px"),
        )
        self.remove_btn.on_click(self._remove)

        original_sentence = self.action.get("one_sentence") or "(no sentence)"
        original_html = widgets.HTML(
            '<div style="padding:8px 10px;border-left:3px solid #94a3b8;background:#f8fafc;'
            'font-size:12px;color:#334155;">'
            f"<b>Prefill from {self.parent.source_label}:</b> {original_sentence}"
            "</div>"
        )

        row1 = widgets.HBox(
            [
                self.include_cb,
                self.remove_btn,
                widgets.HTML("<b>Confidence</b>"),
                self.confidence_input,
            ]
        )
        self.tool_box = widgets.VBox([widgets.HTML("<b>Tool</b>"), self.tool_dd])
        row2 = widgets.HBox(
            [
                widgets.VBox([widgets.HTML("<b>Actor</b>"), self.actor_dd]),
                self.tool_box,
                self.single_action_box,
            ]
        )
        row3 = widgets.HBox(
            [
                widgets.VBox([widgets.HTML("<b>Target 1</b>"), self.target1_dd]),
                widgets.VBox([widgets.HTML("<b>Target 2</b>"), self.target2_dd]),
                widgets.VBox([widgets.HTML("<b>Target 3</b>"), self.target3_dd]),
            ]
        )
        row4 = widgets.HBox(
            [
                widgets.VBox([widgets.HTML("<b>Target Context 1</b>"), self.ctx1_dd]),
                widgets.VBox([widgets.HTML("<b>Target Context 2</b>"), self.ctx2_dd]),
                widgets.VBox([widgets.HTML("<b>Intention</b>"), self.intention_dd]),
            ]
        )
        self.legacy_target_box = widgets.VBox([row3, row4])

        self.box = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="font-weight:700;color:#0f172a;margin-top:6px;">Action editor</div>'
                ),
                original_html,
                row1,
                row2,
                self.generic_timing_box,
                self.camera_actions_box,
                self.left_actions_box,
                self.right_actions_box,
                self.legacy_target_box,
                widgets.HTML("<b>Sentence</b>"),
                self.sentence_input,
                widgets.HTML("<b>Rationale</b>"),
                self.rationale_input,
                widgets.HTML("<b>Extra info</b>"),
                self.extra_input,
            ],
            layout=widgets.Layout(
                border="1px solid #e2e8f0", padding="10px", margin="8px 0", border_radius="8px"
            ),
        )
        self._update_camera_mode()

    def _remove(self, _btn) -> None:
        self.parent.remove_action(self)

    def _initial_target_values(self) -> List[str]:
        values = [
            self.action.get("target_structure_1"),
            self.action.get("target_structure_2"),
            self.action.get("target_structure_3"),
        ]
        legacy = self.action.get("target_structure")
        if isinstance(legacy, list):
            for idx, value in enumerate(legacy[:3]):
                if idx < len(values) and not values[idx]:
                    values[idx] = value
        elif legacy and not values[0]:
            values[0] = legacy
        normalized = [(value or "(not set)") for value in values]
        while len(normalized) < 3:
            normalized.append("(not set)")
        return normalized[:3]

    def _initial_camera_segments(self) -> List[dict]:
        existing = self.action.get("camera_action_segments")
        if isinstance(existing, list) and existing:
            rows = []
            for seg in existing:
                rows.append(
                    {
                        "action_code": seg.get("action_code", "(not set)"),
                        "start_frame": _safe_float(seg.get("start_frame"), 0.0),
                        "end_frame": _safe_float(seg.get("end_frame"), 0.0),
                        "description": seg.get("description", ""),
                    }
                )
            return rows

        rows = []
        if self.action.get("actor_role") == "camera" and self.parent.camera_record:
            for seg in self.parent.camera_record.get("camera_segments", []):
                for code in seg.get("action_codes", []):
                    rows.append(
                        {
                            "action_code": code,
                            "start_frame": int(seg.get("start_frame", 0)),
                            "end_frame": int(seg.get("end_frame", 0)),
                            "description": seg.get("description", ""),
                        }
                    )

        if not rows and self.action.get("actor_role") == "camera":
            default_start = int(self.parent.section_data.get("start_frame", 0))
            default_end = int(self.parent.section_data.get("end_frame", default_start))
            rows.append(
                {
                    "action_code": self.action.get("action_code", "(not set)"),
                    "start_frame": default_start,
                    "end_frame": default_end,
                    "description": self.action.get("one_sentence", ""),
                }
            )

        while len(rows) < 1:
            rows.append({"action_code": "(not set)", "start_frame": 0, "end_frame": 0, "description": ""})
        return rows

    def _build_camera_actions_box(self) -> widgets.VBox:
        self.camera_action_widgets = []
        self.camera_rows_box = widgets.VBox()
        self.add_camera_action_btn = widgets.Button(description="Add camera action", button_style="info")
        self.add_camera_action_btn.on_click(lambda _btn: self._append_camera_action_row())
        box = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700;color:#1d4ed8;margin-top:4px;'>Camera actions</div>"
                    "<div style='font-size:12px;color:#475569;'>List camera actions, each with its own frame span.</div>"
                ),
                self.camera_rows_box,
                self.add_camera_action_btn,
            ]
        )
        for seg in self._initial_camera_segments():
            self._append_camera_action_row(seg)
        return box

    def _append_camera_action_row(self, seg: Optional[dict] = None) -> None:
        seg = seg or {"action_code": "(not set)", "start_frame": 0, "end_frame": 0, "description": ""}
        idx = len(self.camera_action_widgets) + 1
        action_dd = _dropdown(CAMERA_ACTION_CODES + ["(not set)"], seg.get("action_code", "(not set)"), width="220px")
        start_input = widgets.IntText(value=int(seg.get("start_frame", 0) or 0), layout=widgets.Layout(width="140px"))
        end_input = widgets.IntText(value=int(seg.get("end_frame", 0) or 0), layout=widgets.Layout(width="140px"))
        description_input = _short_textarea(seg.get("description", ""), width="720px")
        clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="80px"))
        row_state = {
            "action_dd": action_dd,
            "start_input": start_input,
            "end_input": end_input,
            "description_input": description_input,
        }
        clear_btn.on_click(lambda _btn, row=row_state: self._clear_camera_action_row(row))
        row_widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.HTML(f"<b>Camera Action {idx}</b>"),
                        action_dd,
                        widgets.HTML("<b>Start</b>"),
                        start_input,
                        widgets.HTML("<b>End</b>"),
                        end_input,
                        clear_btn,
                    ]
                ),
                widgets.HBox([widgets.HTML("<b>Camera movement description</b>"), description_input]),
            ],
            layout=widgets.Layout(margin="4px 0 8px 0"),
        )
        row_state["row_widget"] = row_widget
        self.camera_action_widgets.append(row_state)
        self.camera_rows_box.children = tuple(row["row_widget"] for row in self.camera_action_widgets)

    def _clear_camera_action_row(self, row: dict) -> None:
        row["action_dd"].value = "(not set)"
        row["start_input"].value = 0
        row["end_input"].value = 0
        row["description_input"].value = ""

    def _default_interval(self) -> tuple[int, int]:
        start = int(_safe_float(self.parent.section_data.get("start_frame", 0), 0.0))
        end = int(_safe_float(self.parent.section_data.get("end_frame", start), 0.0))
        return start, end

    def _initial_left_segments(self) -> List[dict]:
        existing = self.action.get("left_action_segments")
        if isinstance(existing, list) and existing:
            rows = []
            for seg in existing:
                retraction = seg.get("grasper_retraction") or {}
                rows.append(
                    {
                        "action_code": seg.get("action_code", "(not set)"),
                        "start_frame": _safe_float(seg.get("start_frame"), 0.0),
                        "end_frame": _safe_float(seg.get("end_frame"), 0.0),
                        "changed": seg.get("changed", retraction.get("changed", "unsure")),
                        "start_direction": seg.get(
                            "start_direction", retraction.get("start_direction", "unsure")
                        ),
                        "end_direction": seg.get("end_direction", retraction.get("end_direction", "unsure")),
                        "description": seg.get("description", retraction.get("description", "")),
                    }
                )
            return rows

        default_start = int(
            _safe_float(
                self.action.get("left_action_start_frame", self.parent.section_data.get("start_frame", 0)),
                0.0,
            )
        )
        default_end = int(
            _safe_float(
                self.action.get("left_action_end_frame", self.parent.section_data.get("end_frame", default_start)),
                0.0,
            )
        )
        rows = []
        if self.action.get("actor_role") == "left_instrument":
            rows.append(
                {
                    "action_code": self.action.get("action_code", "(not set)"),
                    "start_frame": default_start,
                    "end_frame": default_end,
                    "changed": self.action.get("left_retraction_changed", "unsure"),
                    "start_direction": self.action.get("left_retraction_start_direction", "unsure"),
                    "end_direction": self.action.get("left_retraction_end_direction", "unsure"),
                    "description": self.action.get("one_sentence", ""),
                }
            )
        while len(rows) < 1:
            rows.append(
                {
                    "action_code": "(not set)",
                    "start_frame": 0,
                    "end_frame": 0,
                    "changed": "unsure",
                    "start_direction": "unsure",
                    "end_direction": "unsure",
                    "description": "",
                }
            )
        return rows

    def _build_left_actions_box(self) -> widgets.VBox:
        self.left_action_widgets = []
        self.left_rows_box = widgets.VBox()
        self.add_left_action_btn = widgets.Button(description="Add left action", button_style="info")
        self.add_left_action_btn.on_click(lambda _btn: self._append_left_action_row())
        box = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700;color:#0f172a;margin-top:4px;'>Left</div>"
                    "<div style='font-size:12px;color:#475569;'>List left-instrument action intervals. Tool is omitted for left actions.</div>"
                ),
                self.left_rows_box,
                self.add_left_action_btn,
            ]
        )
        for seg in self._initial_left_segments():
            self._append_left_action_row(seg)
        return box

    def _append_left_action_row(self, seg: Optional[dict] = None) -> None:
        seg = seg or {
            "action_code": "(not set)",
            "start_frame": 0,
            "end_frame": 0,
            "changed": "unsure",
            "start_direction": "unsure",
            "end_direction": "unsure",
            "description": "",
        }
        idx = len(self.left_action_widgets) + 1
        action_dd = _dropdown(self.taxonomy_options["action_code"], seg.get("action_code", "(not set)"), width="220px")
        start_input = widgets.IntText(value=int(seg.get("start_frame", 0) or 0), layout=widgets.Layout(width="120px"))
        end_input = widgets.IntText(value=int(seg.get("end_frame", 0) or 0), layout=widgets.Layout(width="120px"))
        changed_dd = _dropdown(RETRACTION_CHANGE_OPTIONS, seg.get("changed", "unsure"), width="110px")
        start_direction_dd = _dropdown(RETRACTION_DIRECTION_OPTIONS, seg.get("start_direction", "unsure"), width="150px")
        end_direction_dd = _dropdown(RETRACTION_DIRECTION_OPTIONS, seg.get("end_direction", "unsure"), width="150px")
        description_input = _short_textarea(seg.get("description", ""), width="720px")
        clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="80px"))
        row_state = {
            "action_dd": action_dd,
            "start_input": start_input,
            "end_input": end_input,
            "changed_dd": changed_dd,
            "start_direction_dd": start_direction_dd,
            "end_direction_dd": end_direction_dd,
            "description_input": description_input,
        }
        clear_btn.on_click(lambda _btn, row=row_state: self._clear_left_action_row(row))
        row_widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.HTML(f"<b>Left {idx}</b>"),
                        widgets.HTML("<b>Action</b>"),
                        action_dd,
                        widgets.HTML("<b>Start</b>"),
                        start_input,
                        widgets.HTML("<b>End</b>"),
                        end_input,
                        clear_btn,
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.HTML("<b>Changed</b>"),
                        changed_dd,
                        widgets.HTML("<b>Start direction</b>"),
                        start_direction_dd,
                        widgets.HTML("<b>End direction</b>"),
                        end_direction_dd,
                    ]
                ),
                widgets.HBox([widgets.HTML("<b>Retraction description</b>"), description_input]),
            ],
            layout=widgets.Layout(margin="4px 0 8px 0"),
        )
        row_state["row_widget"] = row_widget
        self.left_action_widgets.append(row_state)
        self.left_rows_box.children = tuple(row["row_widget"] for row in self.left_action_widgets)

    def _clear_left_action_row(self, row: dict) -> None:
        row["action_dd"].value = "(not set)"
        row["start_input"].value = 0
        row["end_input"].value = 0
        row["changed_dd"].value = "unsure"
        row["start_direction_dd"].value = "unsure"
        row["end_direction_dd"].value = "unsure"
        row["description_input"].value = ""

    def _initial_right_segments(self) -> List[dict]:
        existing = self.action.get("right_action_segments")
        if isinstance(existing, list) and existing:
            rows = []
            for seg in existing:
                rows.append(
                    {
                        "tool_type": seg.get("tool_type", self.action.get("tool_type", "(not set)")),
                        "action_code": seg.get("action_code", self.action.get("action_code", "(not set)")),
                        "target_structure": seg.get("target_structure", "(not set)"),
                        "target_context_1": seg.get("target_context_1", self.action.get("target_context_1", "(not set)")),
                        "target_context_2": seg.get("target_context_2", self.action.get("target_context_2", "(not set)")),
                        "start_frame": _safe_float(seg.get("start_frame"), 0.0),
                        "end_frame": _safe_float(seg.get("end_frame"), 0.0),
                        "description": seg.get("description", ""),
                    }
                )
            return rows

        start, end = self._default_interval()
        ctx1 = self.action.get("target_context_1", "(not set)")
        ctx2 = self.action.get("target_context_2", "(not set)")
        targets = [value for value in self._initial_target_values() if value and value != "(not set)"]
        rows = []
        if self.action.get("actor_role") == "right_instrument":
            if not targets:
                targets = [self.action.get("target_structure", "(not set)") or "(not set)"]
            for target in targets:
                rows.append(
                    {
                        "tool_type": self.action.get("tool_type", "(not set)"),
                        "action_code": self.action.get("action_code", "(not set)"),
                        "target_structure": target,
                        "target_context_1": ctx1,
                        "target_context_2": ctx2,
                        "start_frame": start,
                        "end_frame": end,
                        "description": self.action.get("one_sentence", ""),
                    }
                )
        while len(rows) < 1:
            rows.append(self._blank_right_segment())
        return rows

    def _blank_right_segment(self) -> dict:
        return {
            "tool_type": "(not set)",
            "action_code": "(not set)",
            "target_structure": "(not set)",
            "target_context_1": "(not set)",
            "target_context_2": "(not set)",
            "start_frame": 0,
            "end_frame": 0,
            "description": "",
        }

    def _build_right_actions_box(self) -> widgets.VBox:
        self.right_action_widgets = []
        self.right_rows_box = widgets.VBox()
        self.add_right_action_btn = widgets.Button(description="Add right action", button_style="info")
        self.add_right_action_btn.on_click(lambda _btn: self._append_right_action_row())
        box = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700;color:#0f172a;margin-top:4px;'>Right</div>"
                    "<div style='font-size:12px;color:#475569;'>List right-tool intervals with tool, action, target, and target context.</div>"
                ),
                self.right_rows_box,
                self.add_right_action_btn,
            ]
        )
        for seg in self._initial_right_segments():
            self._append_right_action_row(seg)
        return box

    def _append_right_action_row(self, seg: Optional[dict] = None) -> None:
        seg = seg or self._blank_right_segment()
        idx = len(self.right_action_widgets) + 1
        tool_dd = _dropdown(self.taxonomy_options["tool_type"], seg.get("tool_type", "(not set)"), width="180px")
        action_dd = _dropdown(self.taxonomy_options["action_code"], seg.get("action_code", "(not set)"), width="190px")
        target_dd = _dropdown(self.taxonomy_options["target_structure"], seg.get("target_structure", "(not set)"), width="220px")
        ctx1_dd = _dropdown(self.taxonomy_options["target_context"], seg.get("target_context_1", "(not set)"), width="240px")
        ctx2_dd = _dropdown(self.taxonomy_options["target_context"], seg.get("target_context_2", "(not set)"), width="240px")
        start_input = widgets.IntText(value=int(seg.get("start_frame", 0) or 0), layout=widgets.Layout(width="110px"))
        end_input = widgets.IntText(value=int(seg.get("end_frame", 0) or 0), layout=widgets.Layout(width="110px"))
        description_input = _short_textarea(seg.get("description", ""), width="720px")
        clear_btn = widgets.Button(description="Clear", layout=widgets.Layout(width="80px"))
        row_state = {
            "tool_dd": tool_dd,
            "action_dd": action_dd,
            "target_dd": target_dd,
            "ctx1_dd": ctx1_dd,
            "ctx2_dd": ctx2_dd,
            "start_input": start_input,
            "end_input": end_input,
            "description_input": description_input,
        }
        clear_btn.on_click(lambda _btn, row=row_state: self._clear_right_action_row(row))
        row_widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.HTML(f"<b>Right {idx}</b>"),
                        widgets.HTML("<b>Tool</b>"),
                        tool_dd,
                        widgets.HTML("<b>Action</b>"),
                        action_dd,
                        widgets.HTML("<b>Target</b>"),
                        target_dd,
                        widgets.HTML("<b>Start</b>"),
                        start_input,
                        widgets.HTML("<b>End</b>"),
                        end_input,
                        clear_btn,
                    ]
                ),
                widgets.HBox(
                    [
                        widgets.HTML("<b>Target Context 1</b>"),
                        ctx1_dd,
                        widgets.HTML("<b>Target Context 2</b>"),
                        ctx2_dd,
                    ]
                ),
                widgets.HBox([widgets.HTML("<b>Active tool description</b>"), description_input]),
            ],
            layout=widgets.Layout(margin="4px 0 8px 0"),
        )
        row_state["row_widget"] = row_widget
        self.right_action_widgets.append(row_state)
        self.right_rows_box.children = tuple(row["row_widget"] for row in self.right_action_widgets)

    def _clear_right_action_row(self, row: dict) -> None:
        row["tool_dd"].value = "(not set)"
        row["action_dd"].value = "(not set)"
        row["target_dd"].value = "(not set)"
        row["ctx1_dd"].value = "(not set)"
        row["ctx2_dd"].value = "(not set)"
        row["start_input"].value = 0
        row["end_input"].value = 0
        row["description_input"].value = ""

    def _update_camera_mode(self) -> None:
        is_camera = self.actor_dd.value == "camera"
        is_left = self.actor_dd.value == "left_instrument"
        is_right = self.actor_dd.value == "right_instrument"
        self.tool_box.layout.display = "none" if is_left or is_right else ""
        self.single_action_box.layout.display = "none" if is_camera or is_left or is_right else ""
        self.generic_timing_box.layout.display = "none" if is_camera or is_left or is_right else ""
        self.camera_actions_box.layout.display = "" if is_camera else "none"
        self.left_actions_box.layout.display = "" if is_left else "none"
        self.right_actions_box.layout.display = "" if is_right else "none"
        self.legacy_target_box.layout.display = "none" if is_left or is_right else ""

    def _actor_changed(self, _change) -> None:
        self._update_camera_mode()
        self.parent.refresh_layout()

    def serialize(self, rank: int) -> Optional[dict]:
        if not self.include_cb.value:
            return None
        out = copy.deepcopy(self.action)
        out["rank"] = rank
        out["actor_role"] = self.actor_dd.value
        out["tool_type"] = self.tool_dd.value
        if self.actor_dd.value == "camera":
            camera_segments = []
            previous_code = None
            _pop_numbered_fields(
                out,
                [
                    "camera_action_code_",
                    "camera_action_start_frame_",
                    "camera_action_end_frame_",
                ],
            )
            for idx, row in enumerate(self.camera_action_widgets, start=1):
                code = row["action_dd"].value
                start_frame = int(row["start_input"].value or 0)
                end_frame = int(row["end_input"].value or 0)
                description = row["description_input"].value.strip()
                if not _is_set(code) and previous_code and _has_interval(start_frame, end_frame):
                    code = previous_code
                out[f"camera_action_code_{idx}"] = code
                out[f"camera_action_start_frame_{idx}"] = start_frame
                out[f"camera_action_end_frame_{idx}"] = end_frame
                if _is_set(code):
                    camera_segments.append(
                        {
                            "action_code": code,
                            "camera_movement": code,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "description": description,
                        }
                    )
                    previous_code = code
            out["camera_action_segments"] = camera_segments
            out["action_code"] = camera_segments[0]["action_code"] if camera_segments else "(not set)"
        else:
            out["action_code"] = self.action_dd.value
            out["camera_action_segments"] = []
        if self.actor_dd.value not in {"camera", "left_instrument", "right_instrument"}:
            start_frame = int(self.generic_start_input.value or 0)
            end_frame = int(self.generic_end_input.value or 0)
            out["start_frame"] = start_frame
            out["end_frame"] = end_frame
            out["action_start_frame"] = start_frame
            out["action_end_frame"] = end_frame
        if self.actor_dd.value == "left_instrument":
            left_segments = []
            previous_code = None
            _pop_numbered_fields(
                out,
                [
                    "left_action_code_",
                    "left_action_start_frame_",
                    "left_action_end_frame_",
                ],
            )
            for idx, row in enumerate(self.left_action_widgets, start=1):
                code = row["action_dd"].value
                start_frame = int(row["start_input"].value or 0)
                end_frame = int(row["end_input"].value or 0)
                changed = row["changed_dd"].value
                start_direction = row["start_direction_dd"].value
                end_direction = row["end_direction_dd"].value
                description = row["description_input"].value.strip()
                if not _is_set(code) and previous_code and _has_interval(start_frame, end_frame):
                    code = previous_code
                out[f"left_action_code_{idx}"] = code
                out[f"left_action_start_frame_{idx}"] = start_frame
                out[f"left_action_end_frame_{idx}"] = end_frame
                if _is_set(code):
                    left_segments.append(
                        {
                            "action_code": code,
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "changed": changed,
                            "start_direction": start_direction,
                            "end_direction": end_direction,
                            "description": description,
                            "grasper_retraction": {
                                "changed": changed,
                                "start_direction": start_direction,
                                "end_direction": end_direction,
                                "description": description,
                            },
                        }
                    )
                    previous_code = code
            out["left_action_segments"] = left_segments
            if left_segments:
                out["action_code"] = left_segments[0]["action_code"]
                out["left_action_start_frame"] = left_segments[0]["start_frame"]
                out["left_action_end_frame"] = left_segments[0]["end_frame"]
                out["left_retraction_changed"] = left_segments[0]["changed"]
                out["left_retraction_start_direction"] = left_segments[0]["start_direction"]
                out["left_retraction_end_direction"] = left_segments[0]["end_direction"]
            else:
                out["left_action_start_frame"] = 0
                out["left_action_end_frame"] = 0
        else:
            out["left_action_segments"] = []
            out.pop("left_action_start_frame", None)
            out.pop("left_action_end_frame", None)
        if self.actor_dd.value == "right_instrument":
            right_segments = []
            previous = {}
            _pop_numbered_fields(
                out,
                [
                    "right_tool_type_",
                    "right_action_code_",
                    "right_target_structure_",
                    "right_target_context_1_",
                    "right_target_context_2_",
                    "right_action_start_frame_",
                    "right_action_end_frame_",
                    "target_structure_",
                ],
            )
            for idx, row in enumerate(self.right_action_widgets, start=1):
                tool = row["tool_dd"].value
                code = row["action_dd"].value
                target = row["target_dd"].value
                ctx1 = row["ctx1_dd"].value
                ctx2 = row["ctx2_dd"].value
                start_frame = int(row["start_input"].value or 0)
                end_frame = int(row["end_input"].value or 0)
                description = row["description_input"].value.strip()
                row_has_values = (
                    _is_set(tool)
                    or _is_set(code)
                    or _is_set(target)
                    or _is_set(ctx1)
                    or _is_set(ctx2)
                    or _has_interval(start_frame, end_frame)
                    or bool(description)
                )
                if row_has_values and previous:
                    if not _is_set(tool):
                        tool = previous["tool_type"]
                    if not _is_set(code):
                        code = previous["action_code"]
                    if not _is_set(target):
                        target = previous["target_structure"]
                out[f"right_tool_type_{idx}"] = tool
                out[f"right_action_code_{idx}"] = code
                out[f"right_target_structure_{idx}"] = target
                out[f"right_target_context_1_{idx}"] = ctx1
                out[f"right_target_context_2_{idx}"] = ctx2
                out[f"right_action_start_frame_{idx}"] = start_frame
                out[f"right_action_end_frame_{idx}"] = end_frame
                if not row_has_values:
                    continue
                segment = {
                    "tool_type": tool,
                    "action_code": code,
                    "target_structure": target,
                    "target_context_1": ctx1,
                    "target_context_2": ctx2,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "description": description,
                    "active_tool": {
                        "tool_type": tool,
                        "action": code,
                        "action_code": code,
                        "target": target,
                        "target_structure": target,
                        "description": description,
                    },
                }
                right_segments.append(segment)
                previous = segment
            out["right_action_segments"] = right_segments
            if right_segments:
                first = right_segments[0]
                out["tool_type"] = first["tool_type"]
                out["action_code"] = first["action_code"]
                out["target_structure"] = first["target_structure"]
                out["target_context_1"] = first["target_context_1"]
                out["target_context_2"] = first["target_context_2"]
                for idx, seg in enumerate(right_segments, start=1):
                    out[f"target_structure_{idx}"] = seg["target_structure"]
        else:
            out["right_action_segments"] = []
        target_values = [self.target1_dd.value, self.target2_dd.value, self.target3_dd.value]
        if self.actor_dd.value != "right_instrument":
            out["target_structure"] = target_values[0]
            out["target_structure_1"] = target_values[0]
            out["target_structure_2"] = target_values[1]
            out["target_structure_3"] = target_values[2]
            out["target_context_1"] = self.ctx1_dd.value
            out["target_context_2"] = self.ctx2_dd.value
        out["intention"] = self.intention_dd.value
        out["confidence"] = _safe_float(self.confidence_input.value, 0.0)
        out["one_sentence"] = self.sentence_input.value.strip()
        out["original_sentence"] = self.sentence_input.value.strip()
        out["generated_sentence"] = self.sentence_input.value.strip()
        out["rationale"] = self.rationale_input.value.strip()
        out["extra_info"] = self.extra_input.value.strip()
        return out


@dataclass
class ClipSectionEditor:
    parent: "RecordEditor"
    section_data: dict
    section_kind: str
    section_label: str
    frame_title: str
    source_label: str
    include_camera_panel: bool = False
    camera_record: Optional[dict] = None
    action_editors: List[ActionEditor] = field(default_factory=list)
    actions_box: widgets.VBox = field(init=False)
    add_btn: widgets.Button = field(init=False)
    post_actions_box: widgets.VBox = field(init=False)
    box: widgets.VBox = field(init=False)
    frame_preview_html: widgets.HTML = field(init=False)
    frame_expanded_html: widgets.HTML = field(init=False)
    toggle_btn: Optional[widgets.Button] = field(init=False, default=None)
    expanded_visible: bool = field(init=False, default=False)

    @property
    def interface(self):
        return self.parent.interface

    def __post_init__(self) -> None:
        actions = _safe_get(self.section_data, "annotation", "actions_ranked", default=[]) or []
        if not actions:
            actions = [_default_action(1)]
        self.actions_box = widgets.VBox()
        self.post_actions_box = widgets.VBox()
        self.add_btn = widgets.Button(description=f"Add {self.section_label} action", button_style="info")
        self.add_btn.on_click(self._add_action_clicked)

        frame_widgets = self._build_frame_widgets()
        header = widgets.HTML(
            '<div style="font-size:15px;font-weight:800;color:#0f172a;margin:4px 0 6px 0;">'
            f"{self.section_label}</div>"
        )
        self.box = widgets.VBox(
            [],
            layout=widgets.Layout(
                border="1px solid #cbd5e1",
                padding="10px",
                margin="10px 0",
                border_radius="8px",
            ),
        )
        self._frame_widgets = [header, *frame_widgets]
        for action in actions:
            self._append_action(copy.deepcopy(action))
        self.refresh_layout()

    def _build_frame_widgets(self) -> List[widgets.Widget]:
        frame_dir = os.path.join(self.interface.frames_dir, self.parent.video_id)
        start_frame = int(self.section_data.get("start_frame", 0))
        end_frame = int(self.section_data.get("end_frame", start_frame))

        widgets_out: List[widgets.Widget] = []
        if self.section_kind == "coarse":
            preview_frames = _coarse_preview_frames(frame_dir, start_frame, end_frame)
            expanded_frames = _existing_clip_frames(frame_dir, start_frame, end_frame, step=30)
            self.frame_preview_html = widgets.HTML(
                _frame_block_html_with_metrics(
                    frame_dir,
                    preview_frames,
                    self.interface.video_data,
                    self.parent.video_id,
                    title=f"{self.frame_title} | default preview at 5-second cadence",
                )
            )
            self.frame_expanded_html = widgets.HTML(
                _frame_block_html_with_metrics(
                    frame_dir,
                    expanded_frames,
                    self.interface.video_data,
                    self.parent.video_id,
                    title="All 1fps frames in coarse interval",
                    max_height=150,
                )
            )
            self.frame_expanded_html.layout.display = "none"
            self.toggle_btn = widgets.Button(description="Show all 1fps frames")
            self.toggle_btn.on_click(self._toggle_expanded_frames)

            left_stack = widgets.VBox([self.frame_preview_html, self.toggle_btn, self.frame_expanded_html])
            widgets_out.append(left_stack)
        else:
            fine_frames = _existing_clip_frames(frame_dir, start_frame, end_frame, step=30)
            self.frame_preview_html = widgets.HTML(
                _frame_block_html_with_metrics(
                    frame_dir,
                    fine_frames,
                    self.interface.video_data,
                    self.parent.video_id,
                    title=f"{self.frame_title} | all 1fps frames in fine interval",
                    max_height=150,
                )
            )
            self.frame_expanded_html = widgets.HTML("")
            widgets_out.append(self.frame_preview_html)
        return widgets_out

    def _toggle_expanded_frames(self, _btn) -> None:
        self.expanded_visible = not self.expanded_visible
        self.frame_expanded_html.layout.display = "" if self.expanded_visible else "none"
        if self.toggle_btn:
            self.toggle_btn.description = (
                "Hide all 1fps frames" if self.expanded_visible else "Show all 1fps frames"
            )

    def _append_action(self, action: dict) -> None:
        editor = ActionEditor(self, action, self.interface.taxonomy_options)
        self.action_editors.append(editor)
        self.refresh_layout()

    def _add_action_clicked(self, _btn) -> None:
        self._append_action(_default_action(len(self.action_editors) + 1))

    def remove_action(self, action_editor: ActionEditor) -> None:
        self.action_editors = [ae for ae in self.action_editors if ae is not action_editor]
        if not self.action_editors:
            self._append_action(_default_action(1))
        else:
            self.refresh_layout()

    def serialize_actions(self) -> List[dict]:
        actions = []
        for idx, editor in enumerate(self.action_editors, start=1):
            action = editor.serialize(idx)
            if action:
                actions.append(action)
        return actions

    def _camera_panel_widget(self) -> widgets.HTML:
        frame_dir = os.path.join(self.interface.frames_dir, self.parent.video_id)
        return widgets.HTML(
            _camera_panel_html_with_metrics(
                frame_dir,
                self.interface.video_data,
                self.parent.video_id,
                self.camera_record,
            )
        )

    def refresh_layout(self) -> None:
        action_widgets: List[widgets.Widget] = []
        camera_found = False
        for editor in self.action_editors:
            action_widgets.append(editor.box)
            if self.include_camera_panel and self.camera_record and editor.actor_dd.value == "camera":
                action_widgets.append(self._camera_panel_widget())
                camera_found = True
        self.actions_box.children = tuple(action_widgets)

        post_widgets: List[widgets.Widget] = [self.add_btn]
        if self.include_camera_panel and self.camera_record and not camera_found:
            post_widgets.append(self._camera_panel_widget())
        self.post_actions_box.children = tuple(post_widgets)
        self.box.children = tuple(self._frame_widgets + [self.actions_box, self.post_actions_box])


@dataclass
class RecordEditor:
    interface: "AuditVersionInterface"
    video_id: str
    record: dict
    camera_record: Optional[dict]
    source_label: str
    clip_editors: List[ClipSectionEditor] = field(default_factory=list)
    card: widgets.VBox = field(init=False)

    def __post_init__(self) -> None:
        criterion = self.record.get("criterion")
        coarse = self.record.get("coarse", {})
        coarse_editor = ClipSectionEditor(
            parent=self,
            section_data=coarse,
            section_kind="coarse",
            section_label="Coarse-grained actions",
            frame_title=(
                f"{criterion} | {self.record.get('mind_change')} | "
                f"{self.record.get('example_id')} | source: {self.source_label}"
            ),
            source_label=self.source_label,
            include_camera_panel=True,
            camera_record=self.camera_record,
        )
        self.clip_editors.append(coarse_editor)

        for idx, fine in enumerate(self.record.get("fine", []), start=1):
            fine_editor = ClipSectionEditor(
                parent=self,
                section_data=fine,
                section_kind="fine",
                section_label=f"Fine-grained actions {idx}",
                frame_title=(
                    f"{fine.get('fine_id')} | frames {int(fine.get('start_frame', 0)):06d} - "
                    f"{int(fine.get('end_frame', 0)):06d}"
                ),
                source_label=self.source_label,
            )
            self.clip_editors.append(fine_editor)

        states_html = widgets.HTML(self._build_states_html())
        self.card = widgets.VBox(
            [
                widgets.HTML(
                    '<div style="font-size:16px;font-weight:800;color:#0f172a;margin:8px 0 4px 0;">'
                    f"Record {self.record.get('example_id')}"
                    "</div>"
                ),
                states_html,
                *[editor.box for editor in self.clip_editors],
            ],
            layout=widgets.Layout(
                border="2px solid #94a3b8",
                padding="12px",
                margin="12px 0",
                border_radius="10px",
            ),
        )

    def _build_states_html(self) -> str:
        ann = _safe_get(self.record, "coarse", "annotation", default={}) or {}
        start_state = (ann.get("start_state") or {}).get("what_is_visible", [])
        mid_state = (ann.get("mid_state") or {}).get("what_is_visible", [])
        end_state = (ann.get("end_state") or {}).get("what_is_visible", [])

        def bullet_block(title: str, items: List[str]) -> str:
            if not items:
                items = ["(none)"]
            bullets = "".join(f"<li>{item}</li>" for item in items)
            return (
                '<div style="flex:1;min-width:260px;padding:8px 10px;background:#f8fafc;'
                'border:1px solid #e2e8f0;border-radius:8px;">'
                f'<div style="font-weight:700;color:#334155;margin-bottom:4px;">{title}</div>'
                f'<ul style="margin:0 0 0 18px;color:#475569;font-size:12px;">{bullets}</ul>'
                "</div>"
            )

        return (
            '<div style="display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 10px 0;">'
            f"{bullet_block('Start state', start_state)}"
            f"{bullet_block('Mid state', mid_state)}"
            f"{bullet_block('End state', end_state)}"
            "</div>"
        )

    def serialize_record(self) -> dict:
        out = copy.deepcopy(self.record)
        if self.clip_editors:
            out.setdefault("coarse", {}).setdefault("annotation", {})["actions_ranked"] = self.clip_editors[
                0
            ].serialize_actions()
        fine_records = out.get("fine", [])
        for fine_idx, clip_editor in enumerate(self.clip_editors[1:]):
            if fine_idx < len(fine_records):
                fine_records[fine_idx].setdefault("annotation", {})["actions_ranked"] = (
                    clip_editor.serialize_actions()
                )
        audit_meta = out.setdefault("audit", {})
        audit_meta["version"] = self.interface.target_name.replace("audit_", "")
        audit_meta[f"{self.interface.target_name}_source"] = self.interface.source_name.replace("audit_", "")
        audit_meta[f"{self.interface.target_name}_saved_at"] = _iso_now()
        return out


class AuditVersionInterface:
    def __init__(self, source_version=10, target_version=None) -> None:
        self.root_dir = _find_repo_root()
        self.annotations_dir = os.path.join(
            self.root_dir, "data/processed/CVS_Challenge_SAGES_v1/cvs_act_annotations/v1"
        )
        self.source_version = _normalize_version(source_version)
        self.target_version = (
            self._next_available_version(self.source_version) if target_version is None else _normalize_version(target_version)
        )
        if self.target_version == self.source_version:
            raise ValueError("target_version must differ from source_version")

        self.source_name = _version_name("audit", self.source_version)
        self.target_name = _version_name("audit", self.target_version)
        self.camera_name = _version_name("audit", self.source_version) + "_camera_segments"
        self.taxonomy_name = _version_name("taxonomy", self.source_version) + ".json"

        self.source_dir = os.path.join(self.annotations_dir, self.source_name)
        self.target_dir = os.path.join(self.annotations_dir, self.target_name)
        self.camera_dir = os.path.join(self.annotations_dir, self.camera_name)
        self.taxonomy_path = os.path.join(self.annotations_dir, self.taxonomy_name)
        self.frames_dir = os.path.join(self.root_dir, "data/processed/CVS_Challenge_SAGES_v1/frames/test")
        self.labels_dir = os.path.join(self.root_dir, "data/raw/CVS_Challenge_SAGES_v1/test/labels")
        os.makedirs(self.target_dir, exist_ok=True)

        if not os.path.isdir(self.source_dir):
            raise RuntimeError(f"Missing source audit dir: {self.source_dir}")
        if not os.path.isfile(self.taxonomy_path):
            raise RuntimeError(f"Missing source taxonomy file: {self.taxonomy_path}")

        self.taxonomy = _read_json(self.taxonomy_path)
        self.taxonomy_options = {
            field: _options_from_field(items) for field, items in self.taxonomy["fields"].items()
        }
        for code in EXTRA_ACTION_CODES:
            if code not in self.taxonomy_options["action_code"]:
                self.taxonomy_options["action_code"].append(code)
        for tool_type in EXTRA_TOOL_TYPES:
            if tool_type not in self.taxonomy_options["tool_type"]:
                self.taxonomy_options["tool_type"].append(tool_type)
        self.video_data = load_video_data(self.labels_dir)
        self.source_records_by_video = self._load_json_dir(self.source_dir)
        self.target_records_by_video = self._load_json_dir(self.target_dir)
        self.camera_records_by_video = self._load_json_dir(self.camera_dir)
        self.video_ids = sorted(self.source_records_by_video)
        if not self.video_ids:
            raise RuntimeError(f"No videos found in {self.source_dir}")
        self.current_idx = self._initial_video_index()
        self.record_editors: List[RecordEditor] = []
        self.video_elapsed_seconds: Dict[str, float] = {}
        self.video_first_opened_at: Dict[str, str] = {}
        self.current_video_started_monotonic: Optional[float] = None
        self.current_video_opened_at_iso: Optional[str] = None

        self.status_html = widgets.HTML()
        self.header_html = widgets.HTML()
        self.main_box = widgets.VBox()
        self.prev_btn = widgets.Button(description="Previous video")
        self.next_btn = widgets.Button(description="Next video")
        self.jump_index_input = widgets.BoundedIntText(
            value=self.current_idx + 1,
            min=1,
            max=len(self.video_ids),
            description="Jump to",
            layout=widgets.Layout(width="180px"),
        )
        self.jump_btn = widgets.Button(description="Load index")
        self.save_btn = widgets.Button(description=f"Save {self.target_name}", button_style="success")
        self.save_next_btn = widgets.Button(description="Save & next", button_style="success")
        self.reload_btn = widgets.Button(description="Reload video")

        self.prev_btn.on_click(self._prev_video)
        self.next_btn.on_click(self._next_video)
        self.jump_btn.on_click(self._jump_to_index)
        self.save_btn.on_click(self._save_current)
        self.save_next_btn.on_click(self._save_and_next)
        self.reload_btn.on_click(self._reload_current)

    def _next_available_version(self, source_version: int) -> int:
        present = set()
        for name in os.listdir(self.annotations_dir):
            m = re.fullmatch(r"audit_v(\d+)", name)
            if m:
                present.add(int(m.group(1)))
        candidate = source_version + 1
        while candidate in present:
            candidate += 1
        return candidate

    def _load_json_dir(self, directory: str) -> Dict[str, list]:
        out = {}
        if not os.path.isdir(directory):
            return out
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(directory, name)
            try:
                out[os.path.splitext(name)[0]] = _read_json(path)
            except Exception as exc:
                print(f"Skipping unreadable JSON {path}: {exc}")
        return out

    def _initial_video_index(self) -> int:
        for idx, video_id in enumerate(self.video_ids):
            if video_id not in self.target_records_by_video:
                return idx
        return 0

    def _target_path_for_video(self, video_id: str) -> str:
        return os.path.join(self.target_dir, f"{video_id}.json")

    def _refresh_target_video_from_disk(self, video_id: str) -> None:
        path = self._target_path_for_video(video_id)
        if os.path.isfile(path):
            self.target_records_by_video[video_id] = _read_json(path)
        else:
            self.target_records_by_video.pop(video_id, None)

    def _source_records_for_video(self, video_id: str) -> tuple[list, str]:
        self._refresh_target_video_from_disk(video_id)
        if video_id in self.target_records_by_video:
            return copy.deepcopy(self.target_records_by_video[video_id]), f"saved {self.target_name}"
        return copy.deepcopy(self.source_records_by_video[video_id]), self.source_name

    def _camera_lookup_for_video(self, video_id: str) -> Dict[tuple, dict]:
        records = self.camera_records_by_video.get(video_id, [])
        return {_coarse_key(rec): rec for rec in records}

    def _progress_counts(self) -> tuple[int, int]:
        saved = sum(1 for vid in self.video_ids if vid in self.target_records_by_video)
        return saved, len(self.video_ids)

    def _current_video_id(self) -> str:
        return self.video_ids[self.current_idx]

    def _extract_saved_timing(self, video_id: str) -> tuple[Optional[str], float]:
        records = self.target_records_by_video.get(video_id)
        if not records:
            return None, 0.0
        audit = (records[0] or {}).get("audit", {}) or {}
        started = audit.get("review_started_at")
        duration = _safe_float(audit.get("review_duration_seconds", 0.0), 0.0)
        return started, duration

    def _accumulate_current_video_time(self) -> None:
        if self.current_video_started_monotonic is None:
            return
        video_id = self._current_video_id()
        elapsed = max(0.0, time.monotonic() - self.current_video_started_monotonic)
        self.video_elapsed_seconds[video_id] = self.video_elapsed_seconds.get(video_id, 0.0) + elapsed
        self.current_video_started_monotonic = None

    def _start_current_video_timer(self) -> None:
        video_id = self._current_video_id()
        saved_started_at, _saved_duration = self._extract_saved_timing(video_id)
        if video_id not in self.video_first_opened_at:
            self.video_first_opened_at[video_id] = saved_started_at or _iso_now()
        self.current_video_opened_at_iso = _iso_now()
        self.current_video_started_monotonic = time.monotonic()

    def _render(self) -> None:
        video_id = self.video_ids[self.current_idx]
        source_records, source_label = self._source_records_for_video(video_id)
        camera_lookup = self._camera_lookup_for_video(video_id)
        self.record_editors = [
            RecordEditor(
                interface=self,
                video_id=video_id,
                record=record,
                camera_record=camera_lookup.get(_coarse_key(record)),
                source_label=source_label,
            )
            for record in source_records
        ]

        saved, total = self._progress_counts()
        self.header_html.value = (
            '<div style="padding:10px 12px;background:#eff6ff;border:1px solid #bfdbfe;'
            'border-radius:8px;margin-bottom:10px;">'
            f'<div style="font-size:18px;font-weight:800;color:#1e3a8a;">'
            f'{self.source_name} -> {self.target_name} interface</div>'
            f'<div style="font-size:13px;color:#334155;margin-top:4px;">'
            f'Video {self.current_idx + 1} / {len(self.video_ids)}: <b>{video_id}</b> | '
            f'Saved {self.target_name} videos: <b>{saved}</b> / {total} | Current source: <b>{source_label}</b>'
            "</div>"
            f'<div style="font-size:12px;color:#64748b;margin-top:4px;">'
            f'Taxonomy: {os.path.basename(self.taxonomy_path)} | '
            f'Camera sidecar: {self.camera_name if os.path.isdir(self.camera_dir) else "(none)"} | '
            f'Output dir: {self.target_dir}'
            "</div>"
            "</div>"
            + _schema_reference_html()
        )
        self.prev_btn.disabled = self.current_idx == 0
        self.next_btn.disabled = self.current_idx >= len(self.video_ids) - 1
        self.jump_index_input.value = self.current_idx + 1
        self.main_box.children = tuple([editor.card for editor in self.record_editors])
        self._start_current_video_timer()

    def _save_records(self) -> str:
        self._accumulate_current_video_time()
        video_id = self.video_ids[self.current_idx]
        out_records = [editor.serialize_record() for editor in self.record_editors]
        saved_started_at, saved_duration = self._extract_saved_timing(video_id)
        review_started_at = self.video_first_opened_at.get(video_id) or saved_started_at or _iso_now()
        total_duration = saved_duration + self.video_elapsed_seconds.get(video_id, 0.0)
        last_opened_at = self.current_video_opened_at_iso or _iso_now()
        for record in out_records:
            audit_meta = record.setdefault("audit", {})
            audit_meta["review_started_at"] = review_started_at
            audit_meta["last_opened_at"] = last_opened_at
            audit_meta["review_duration_seconds"] = round(total_duration, 3)
        save_path = os.path.join(self.target_dir, f"{video_id}.json")
        _write_json(save_path, out_records)
        self.target_records_by_video[video_id] = copy.deepcopy(out_records)
        self.video_elapsed_seconds[video_id] = 0.0
        self.video_first_opened_at[video_id] = review_started_at
        self.current_video_opened_at_iso = None
        return save_path

    def _set_status(self, text: str, color: str = "#0f172a") -> None:
        self.status_html.value = (
            f'<div style="padding:8px 10px;color:{color};font-size:13px;font-weight:600;">{text}</div>'
        )

    def _prev_video(self, _btn) -> None:
        if self.current_idx > 0:
            self._accumulate_current_video_time()
            self.current_idx -= 1
            self._render()
            self._set_status("Moved to previous video.", "#475569")

    def _next_video(self, _btn) -> None:
        if self.current_idx < len(self.video_ids) - 1:
            self._accumulate_current_video_time()
            self.current_idx += 1
            self._render()
            self._set_status("Moved to next video. Unsaved edits on the prior video were not persisted.", "#b45309")

    def _jump_to_index(self, _btn) -> None:
        target_idx = int(self.jump_index_input.value) - 1
        if target_idx < 0 or target_idx >= len(self.video_ids):
            self._set_status(f"Index must be between 1 and {len(self.video_ids)}.", "#b91c1c")
            return
        self._accumulate_current_video_time()
        self.current_idx = target_idx
        video_id = self._current_video_id()
        self._render()
        self._set_status(
            f"Loaded video {target_idx + 1} / {len(self.video_ids)} ({video_id}). "
            f"Saved {self.target_name} data was loaded if present.",
            "#475569",
        )

    def _reload_current(self, _btn) -> None:
        self._accumulate_current_video_time()
        self._render()
        self._set_status("Reloaded current video from disk.", "#475569")

    def _save_current(self, _btn) -> None:
        path = self._save_records()
        self._render()
        self._set_status(f"Saved {self.target_name} edits to {path}", "#166534")

    def _save_and_next(self, _btn) -> None:
        path = self._save_records()
        if self.current_idx < len(self.video_ids) - 1:
            self.current_idx += 1
        self._render()
        self._set_status(f"Saved {self.target_name} edits to {path} and advanced to the next video.", "#166534")

    def launch(self) -> widgets.VBox:
        self._render()
        controls = widgets.HBox(
            [
                self.prev_btn,
                self.next_btn,
                self.jump_index_input,
                self.jump_btn,
                self.save_btn,
                self.save_next_btn,
                self.reload_btn,
            ]
        )
        return widgets.VBox([self.header_html, controls, self.status_html, self.main_box])


class AuditV11Interface(AuditVersionInterface):
    def __init__(self) -> None:
        super().__init__(source_version=10, target_version=11)
