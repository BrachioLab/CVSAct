"""Thin Jupyter widget interface for blind CVS-Act surgeon annotation."""

from __future__ import annotations

import base64
import html
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import ipywidgets as widgets
from IPython.display import display
from PIL import Image

from .annotation_store import (
    ClipManifestRow,
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_TAXONOMY_PATH,
    load_annotations,
    load_clip_manifest,
    save_annotation,
    _load_cvs_keyframe_labels,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_ROOT = REPO_ROOT / "data" / "processed" / "CVS_Challenge_SAGES_v1" / "frames"
DEFAULT_RAW_VIDEO_ROOT = REPO_ROOT / "data" / "raw" / "CVS_Challenge_SAGES_v1"
DEFAULT_VIDEO_CACHE = REPO_ROOT / "data" / "processed" / "CVS_Challenge_SAGES_v1" / "surgeon_validation" / "video_cache"
DEFAULT_SIMPLE_TAXONOMY_PATH = REPO_ROOT / "hf_repos" / "cvs-act" / "taxonomy" / "action_taxonomy.json"
DEFAULT_SURGEON_ASSET_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "CVS_Challenge_SAGES_v1"
    / "cvs_act_surgeon_annotations"
    / "aws_upload_assets"
)


def launch_annotation_interface(
    manifest_path: str | Path,
    frames_root: str | Path = DEFAULT_FRAMES_ROOT,
    annotation_path: str | Path = DEFAULT_ANNOTATION_PATH,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    annotator_ids: Iterable[str] = ("surgeon_1", "surgeon_2", "surgeon_3"),
):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() and not manifest_path.is_absolute():
        rooted = REPO_ROOT / manifest_path
        if rooted.exists():
            manifest_path = rooted
    clips = load_clip_manifest(manifest_path)
    ui = SurgeonAnnotationWidget(
        clips=clips,
        frames_root=Path(frames_root),
        annotation_path=Path(annotation_path),
        taxonomy_path=Path(taxonomy_path),
        annotator_ids=tuple(annotator_ids),
    )
    ui.display()
    return ui


def launch_video_menu_interface(
    manifest_path: str | Path,
    frames_root: str | Path = DEFAULT_FRAMES_ROOT,
    annotation_path: str | Path = DEFAULT_ANNOTATION_PATH,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
    assets_root: str | Path = DEFAULT_SURGEON_ASSET_ROOT,
    annotator_ids: Iterable[str] = ("surgeon_1", "surgeon_2", "surgeon_3"),
):
    """Launch a video-level menu for the 30 selected surgeon annotation clips."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() and not manifest_path.is_absolute():
        rooted = REPO_ROOT / manifest_path
        if rooted.exists():
            manifest_path = rooted
    clips = load_clip_manifest(manifest_path)
    ui = SurgeonVideoMenuWidget(
        clips=clips,
        frames_root=Path(frames_root),
        annotation_path=Path(annotation_path),
        taxonomy_path=Path(taxonomy_path),
        assets_root=Path(assets_root),
        annotator_ids=tuple(annotator_ids),
    )
    ui.display()
    return ui


def prepare_annotation_assets(
    clips: Iterable[ClipManifestRow],
    frames_root: str | Path = DEFAULT_FRAMES_ROOT,
    assets_root: str | Path = DEFAULT_SURGEON_ASSET_ROOT,
) -> dict[str, int]:
    """Copy selected clip videos and frames to one upload-ready assets folder."""
    frames_root = Path(frames_root)
    assets_root = Path(assets_root)
    prepared = 0
    missing_frames = 0
    failed_videos = 0
    for clip in clips:
        result = _ensure_clip_assets(clip, frames_root=frames_root, assets_root=assets_root)
        prepared += 1
        missing_frames += int(result.get("missing_frames", 0))
        failed_videos += int(not result.get("video_path"))
    return {
        "clips": prepared,
        "missing_frames": missing_frames,
        "failed_videos": failed_videos,
    }


class SurgeonAnnotationWidget:
    def __init__(
        self,
        clips: list[ClipManifestRow],
        frames_root: Path,
        annotation_path: Path,
        taxonomy_path: Path,
        annotator_ids: tuple[str, ...],
    ) -> None:
        self.clips = clips
        self.frames_root = frames_root
        self.annotation_path = annotation_path
        self.taxonomy_path = taxonomy_path
        self.simple_taxonomy = _load_simple_taxonomy()
        self.annotator = widgets.Dropdown(options=list(annotator_ids), description="Annotator")
        self.progress = widgets.IntProgress(min=0, max=len(clips), value=0, description="Progress")
        self.status = widgets.HTML()
        self.clip_slider = widgets.IntSlider(min=0, max=max(0, len(clips) - 1), value=0, description="Clip")
        self.prev_btn = widgets.Button(description="Prev")
        self.next_btn = widgets.Button(description="Next")
        self.jump_btn = widgets.Button(description="Jump to unannotated")
        self.save_btn = widgets.Button(description="Save annotation", button_style="primary")
        self.video_box = widgets.VBox()
        self.frame_box = widgets.HTML()
        self.meta_box = widgets.HTML()
        self.cvs_label_box = widgets.HTML()
        self.add_subclip_btn = widgets.Button(description="Add subclip")
        self.subclip_box = widgets.VBox()
        self.subclip_editors: list[SubclipEditor] = []
        self.note = widgets.Textarea(description="Note", layout=widgets.Layout(width="720px", height="70px"))
        self.diff_flag = widgets.Checkbox(value=False, description="Observed differs from what I would recommend", indent=False)
        self.annotator.observe(lambda _: self._render(), names="value")
        self.clip_slider.observe(lambda _: self._render(), names="value")
        self.prev_btn.on_click(lambda _: self._move(-1))
        self.next_btn.on_click(lambda _: self._move(1))
        self.jump_btn.on_click(lambda _: self._jump_unannotated())
        self.save_btn.on_click(lambda _: self._save())
        self.add_subclip_btn.on_click(lambda _: self._add_subclip())
        self.root = widgets.VBox(
            [
                widgets.HTML("<h3>CVS-Act Observed Action Annotation</h3>"),
                widgets.HBox([self.annotator, self.progress]),
                widgets.HBox([self.prev_btn, self.next_btn, self.jump_btn, self.clip_slider]),
                self.status,
                self.meta_box,
                self.cvs_label_box,
                self.video_box,
                self.frame_box,
                self.add_subclip_btn,
                self.subclip_box,
                self.diff_flag,
                self.note,
                self.save_btn,
            ]
        )
        self._render()

    def display(self) -> None:
        display(self.root)

    def _current_clip(self) -> ClipManifestRow:
        return self.clips[int(self.clip_slider.value)]

    def _annotations_by_key(self) -> dict[tuple[str, str], dict]:
        return {
            (row["clip_id"], row["annotator_id"]): row
            for row in load_annotations(self.annotation_path)
        }

    def _render(self) -> None:
        if not self.clips:
            self.status.value = "<b>No clips loaded.</b>"
            return
        clip = self._current_clip()
        annotations = self._annotations_by_key()
        key = (clip.clip_id, self.annotator.value)
        ann = annotations.get(key)
        self.note.value = ann.get("note", "") if ann else ""
        self.diff_flag.value = bool(ann.get("observed_differs_from_recommendation", False)) if ann else False
        done_ids = {cid for (cid, aid), _ in annotations.items() if aid == self.annotator.value}
        self.progress.value = len(done_ids & {clip.clip_id for clip in self.clips})
        self.status.value = (
            f"<b>{self.progress.value}/{len(self.clips)}</b> complete for "
            f"{html.escape(str(self.annotator.value))}"
        )
        self.meta_box.value = (
            "<div style='margin:8px 0;font-size:13px;color:#334155;'>"
            f"<b>Clip:</b> {html.escape(clip.clip_id)}<br/>"
            f"<b>Video:</b> {html.escape(clip.video_name)} &nbsp; "
            f"<b>Granularity:</b> {html.escape(clip.granularity)} &nbsp; "
            f"<b>Frames:</b> {clip.start_frame}-{clip.end_frame}"
            "</div>"
        )
        self.cvs_label_box.value = _cvs_keyframe_label_table(clip)
        frame_dir = self.frames_root / (clip.split or "test") / clip.video_name
        self.current_frame_dir = frame_dir
        self.current_frame_ids = _frame_ids_1fps(clip.start_frame, clip.end_frame)
        self.current_cvs_frame_labels = _cvs_labels_by_frame(clip.video_name, self.current_frame_ids)
        self.video_box.children = (_video_widget_for_clip(clip),)
        self.frame_box.value = (
            "<div style='font-weight:700;margin:8px 0 6px;'>"
            "Add a subclip to select boundaries from the 1 fps frame grid."
            "</div>"
        )
        self._load_subclips(ann.get("subclips", []) if ann else [])

    def _move(self, delta: int) -> None:
        self.clip_slider.value = min(max(0, int(self.clip_slider.value) + delta), len(self.clips) - 1)

    def _jump_unannotated(self) -> None:
        annotations = self._annotations_by_key()
        done_ids = {cid for (cid, aid), _ in annotations.items() if aid == self.annotator.value}
        for idx, clip in enumerate(self.clips):
            if clip.clip_id not in done_ids:
                self.clip_slider.value = idx
                return

    def _save(self) -> None:
        clip = self._current_clip()
        record = {
            "clip_id": clip.clip_id,
            "video_name": clip.video_name,
            "granularity": clip.granularity,
            "annotator_id": str(self.annotator.value),
            "note": self.note.value,
            "observed_differs_from_recommendation": bool(self.diff_flag.value),
            "subclips": [editor.to_json() for editor in self.subclip_editors if editor.is_complete()],
        }
        save_annotation(record, path=self.annotation_path, schema_path=self.taxonomy_path)
        self._render()
        self._jump_unannotated()

    def _load_subclips(self, subclips: list[dict]) -> None:
        self.subclip_editors = []
        for subclip in subclips:
            self._add_subclip(initial=subclip, render=False)
        if not self.subclip_editors:
            self._add_subclip(render=False)
        self._render_subclips()

    def _add_subclip(self, _btn=None, initial: dict | None = None, render: bool = True) -> None:
        clip = self._current_clip()
        editor = SubclipEditor(
            parent=self,
            clip_start=int(clip.start_frame),
            clip_end=int(clip.end_frame),
            frame_dir=self.current_frame_dir,
            frame_ids=self.current_frame_ids,
            taxonomy=self.simple_taxonomy,
            initial=initial,
        )
        self.subclip_editors.append(editor)
        if render:
            self._render_subclips()

    def _remove_subclip(self, editor: "SubclipEditor") -> None:
        self.subclip_editors = [item for item in self.subclip_editors if item is not editor]
        self._render_subclips()

    def _render_subclips(self) -> None:
        if not self.subclip_editors:
            self.subclip_box.children = (widgets.HTML("<em>No subclips added.</em>"),)
            return
        for idx, editor in enumerate(self.subclip_editors, start=1):
            editor.refresh(index=idx)
        self.subclip_box.children = tuple(editor.box for editor in self.subclip_editors)

    def _other_selected_ranges(self, editor: "SubclipEditor") -> list[tuple[int, int]]:
        return [item.selected_range() for item in self.subclip_editors if item is not editor and item.is_complete()]


class SurgeonVideoMenuWidget:
    def __init__(
        self,
        clips: list[ClipManifestRow],
        frames_root: Path,
        annotation_path: Path,
        taxonomy_path: Path,
        assets_root: Path,
        annotator_ids: tuple[str, ...],
    ) -> None:
        self.clips = clips
        self.frames_root = frames_root
        self.annotation_path = annotation_path
        self.taxonomy_path = taxonomy_path
        self.assets_root = assets_root
        self.simple_taxonomy = _load_simple_taxonomy()
        self.current_clip_idx: int | None = None
        self.current_frame_dir = Path()
        self.current_frame_ids: list[int] = []
        self.current_cvs_frame_labels: dict[int, dict] = {}
        self.subclip_editors: list[SubclipEditor] = []

        options = [(str(idx + 1), annotator_id) for idx, annotator_id in enumerate(annotator_ids)]
        self.annotator = widgets.Dropdown(options=options, description="Surgeon ID")
        self.progress = widgets.IntProgress(min=0, max=len(clips), value=0, description="Progress")
        self.status = widgets.HTML()
        self.menu_box = widgets.VBox()
        self.detail_box = widgets.VBox()

        self.back_btn = widgets.Button(description="Back to menu")
        self.submit_btn = widgets.Button(description="Submit", button_style="primary")
        self.reset_btn = widgets.Button(description="Reset")
        self.bottom_back_btn = widgets.Button(description="Back to menu")
        self.bottom_submit_btn = widgets.Button(description="Submit", button_style="primary")
        self.bottom_reset_btn = widgets.Button(description="Reset")
        self.add_subclip_btn = widgets.Button(description="Add subclip")
        self.video_box = widgets.VBox()
        self.meta_box = widgets.HTML()
        self.cvs_label_box = widgets.HTML()
        self.subclip_box = widgets.VBox()
        self.note = widgets.Textarea(description="Note", layout=widgets.Layout(width="720px", height="70px"))

        self.annotator.observe(lambda _: self._show_menu(), names="value")
        self.back_btn.on_click(lambda _: self._save_and_show_menu())
        self.submit_btn.on_click(lambda _: self._submit())
        self.reset_btn.on_click(lambda _: self._reset_current_clip())
        self.bottom_back_btn.on_click(lambda _: self._save_and_show_menu())
        self.bottom_submit_btn.on_click(lambda _: self._submit())
        self.bottom_reset_btn.on_click(lambda _: self._reset_current_clip())
        self.add_subclip_btn.on_click(lambda _: self._add_subclip())

        self.root = widgets.VBox(
            [
                widgets.HTML("<h3>CVS-Act Surgeon Annotation Menu</h3>"),
                widgets.HBox([self.annotator, self.progress]),
                self.status,
                self.menu_box,
                self.detail_box,
            ]
        )
        self._show_menu()

    def display(self) -> None:
        display(self.root)

    def _annotations_by_key(self) -> dict[tuple[str, str], dict]:
        return {
            (row["clip_id"], row["annotator_id"]): row
            for row in load_annotations(self.annotation_path)
        }

    def _surgeon_annotation_path(self) -> Path:
        return self.annotation_path.parent / "by_surgeon" / f"{self.annotator.value}.jsonl"

    def _show_menu(self) -> None:
        self.current_clip_idx = None
        self.detail_box.children = ()
        annotations = self._annotations_by_key()
        complete_ids = {
            clip_id
            for (clip_id, annotator_id), _ in annotations.items()
            if annotator_id == self.annotator.value
            and _.get("annotation_status") == "submitted"
        }
        self.progress.value = len(complete_ids & {clip.clip_id for clip in self.clips})
        self.status.value = (
            "<div style='margin:8px 0;color:#334155;'>"
            f"<b>{self.progress.value}/{len(self.clips)}</b> videos complete for "
            f"<b>surgeon {html.escape(str(self.annotator.label))}</b>. "
            f"Assets folder: <code>{html.escape(str(self.assets_root))}</code>"
            "</div>"
        )
        rows = []
        for idx, clip in enumerate(self.clips, start=1):
            is_complete = clip.clip_id in complete_ids
            open_btn = widgets.Button(
                description=f"Video {idx:02d}",
                button_style="success" if is_complete else "",
                layout=widgets.Layout(width="104px"),
            )
            open_btn.on_click(lambda _, clip_idx=idx - 1: self._open_clip(clip_idx))
            badge = (
                "<span style='display:inline-block;min-width:82px;color:#166534;font-weight:700;'>complete</span>"
                if is_complete
                else "<span style='display:inline-block;min-width:82px;color:#991b1b;font-weight:700;'>incomplete</span>"
            )
            label = widgets.HTML(
                "<div style='font-size:13px;color:#334155;'>"
                f"{badge}"
                f"<b>{html.escape(clip.video_name[:8])}</b> &nbsp; "
                f"{html.escape(clip.criterion or '')} &nbsp; "
                f"f{int(clip.start_frame):06d}-f{int(clip.end_frame):06d}"
                "</div>",
                layout=widgets.Layout(width="620px"),
            )
            rows.append(widgets.HBox([open_btn, label], layout=widgets.Layout(margin="2px 0")))
        self.menu_box.children = tuple(rows)

    def _open_clip(self, clip_idx: int) -> None:
        self.current_clip_idx = clip_idx
        self.menu_box.children = ()
        clip = self.clips[clip_idx]
        annotations = self._annotations_by_key()
        ann = annotations.get((clip.clip_id, self.annotator.value))
        self.note.value = ann.get("note", "") if ann else ""

        self.current_frame_ids = _frame_ids_1fps(clip.start_frame, clip.end_frame)
        self.current_cvs_frame_labels = _cvs_labels_by_frame(clip.video_name, self.current_frame_ids)
        asset_info = _ensure_clip_assets(clip, frames_root=self.frames_root, assets_root=self.assets_root)
        self.current_frame_dir = Path(asset_info["frame_dir"])

        self.meta_box.value = (
            "<div style='margin:8px 0;font-size:13px;color:#334155;'>"
            f"<b>Video:</b> {clip_idx + 1}/{len(self.clips)} &nbsp; "
            f"<b>ID:</b> {html.escape(clip.video_name)}<br/>"
            f"<b>Clip:</b> {html.escape(clip.clip_id)} &nbsp; "
            f"<b>Frames:</b> {int(clip.start_frame)}-{int(clip.end_frame)} &nbsp; "
            f"<b>Assets:</b> <code>{html.escape(str(asset_info['asset_dir']))}</code>"
            "</div>"
        )
        self.cvs_label_box.value = _cvs_keyframe_label_table(clip)
        video_path = asset_info.get("video_path")
        if video_path:
            self.video_box.children = (widgets.Video.from_file(str(video_path), format="mp4", width=760, height=430),)
        else:
            self.video_box.children = (_video_widget_for_clip(clip),)
        self._load_subclips(ann.get("subclips", []) if ann else [])
        self.detail_box.children = (
            widgets.HBox([self.back_btn, self.submit_btn, self.reset_btn]),
            self.meta_box,
            self.cvs_label_box,
            self.video_box,
            self.subclip_box,
            self.add_subclip_btn,
            self.note,
            widgets.HBox([self.bottom_back_btn, self.bottom_submit_btn, self.bottom_reset_btn]),
        )

    def _current_record(self, annotation_status: str) -> dict | None:
        if self.current_clip_idx is None:
            return None
        clip = self.clips[self.current_clip_idx]
        return {
            "clip_id": clip.clip_id,
            "video_name": clip.video_name,
            "granularity": clip.granularity,
            "annotator_id": str(self.annotator.value),
            "note": self.note.value,
            "observed_differs_from_recommendation": False,
            "annotation_status": annotation_status,
            "subclips": [editor.to_json() for editor in self.subclip_editors if editor.is_complete()],
        }

    def _save_current_record(self, annotation_status: str) -> dict | None:
        record = self._current_record(annotation_status)
        if record is None:
            return None
        saved = save_annotation(record, path=self.annotation_path, schema_path=self.taxonomy_path)
        save_annotation(saved, path=self._surgeon_annotation_path(), schema_path=self.taxonomy_path)
        return saved

    def _save_and_show_menu(self) -> None:
        self._save_current_record("draft")
        self._show_menu()

    def _submit(self, _btn=None) -> None:
        if self.current_clip_idx is None:
            return
        clip = self.clips[self.current_clip_idx]
        saved = self._save_current_record("submitted")
        if saved is None:
            return
        self.status.value = (
            "<div style='margin:8px 0;color:#166534;font-weight:700;'>"
            f"Saved {html.escape(clip.video_name)} for surgeon {html.escape(str(self.annotator.label))}."
            "</div>"
        )
        self._show_menu()

    def _reset_current_clip(self) -> None:
        if self.current_clip_idx is None:
            return
        self._open_clip(self.current_clip_idx)

    def _load_subclips(self, subclips: list[dict]) -> None:
        self.subclip_editors = []
        for subclip in subclips:
            self._add_subclip(initial=subclip, render=False)
        if not self.subclip_editors:
            self._add_subclip(render=False)
        self._render_subclips()

    def _add_subclip(self, _btn=None, initial: dict | None = None, render: bool = True) -> None:
        if self.current_clip_idx is None:
            return
        clip = self.clips[self.current_clip_idx]
        editor = SubclipEditor(
            parent=self,
            clip_start=int(clip.start_frame),
            clip_end=int(clip.end_frame),
            frame_dir=self.current_frame_dir,
            frame_ids=self.current_frame_ids,
            taxonomy=self.simple_taxonomy,
            initial=initial,
        )
        self.subclip_editors.append(editor)
        if render:
            self._render_subclips()

    def _remove_subclip(self, editor: "SubclipEditor") -> None:
        self.subclip_editors = [item for item in self.subclip_editors if item is not editor]
        self._render_subclips()

    def _render_subclips(self) -> None:
        if not self.subclip_editors:
            self.subclip_box.children = (widgets.HTML("<em>No subclips added.</em>"),)
            return
        for idx, editor in enumerate(self.subclip_editors, start=1):
            editor.refresh(index=idx)
        self.subclip_box.children = tuple(editor.box for editor in self.subclip_editors)

    def _other_selected_ranges(self, editor: "SubclipEditor") -> list[tuple[int, int]]:
        return [item.selected_range() for item in self.subclip_editors if item is not editor and item.is_complete()]


class SubclipEditor:
    def __init__(
        self,
        parent: SurgeonAnnotationWidget,
        clip_start: int,
        clip_end: int,
        frame_dir: Path,
        frame_ids: list[int],
        taxonomy: dict,
        initial: dict | None = None,
    ) -> None:
        self.parent = parent
        self.frame_dir = frame_dir
        self.frame_ids = frame_ids
        self.index = 0
        initial = initial or {}
        self.start_frame = int(initial["start_frame"]) if initial.get("start_frame") is not None else None
        self.end_frame = int(initial["end_frame"]) if initial.get("end_frame") is not None else None
        self.header = widgets.HTML()
        self.boundary_status = widgets.HTML()
        self.frame_grid = widgets.GridBox()
        self._frame_card_widgets: dict[int, widgets.VBox] = {}
        self._frame_card_index: int | None = None
        left = initial.get("left") or {}
        right = initial.get("right") or {}
        camera = initial.get("camera") or {}

        self.left_action = widgets.Dropdown(
            options=["(none)", *taxonomy["left"]["retraction_direction_code"]],
            value=left.get("retraction_direction_code", "(none)"),
            description="Left",
        )
        self.right_tool = widgets.Dropdown(
            options=["(none)", *taxonomy["right"]["tool_type"]],
            value=right.get("tool_type", "(none)"),
            description="R tool",
        )
        self.right_action = widgets.Dropdown(
            options=["(none)", *taxonomy["right"]["action_code"]],
            value=right.get("action_code", "(none)"),
            description="R action",
        )
        self.right_target = widgets.Dropdown(
            options=["(none)", *taxonomy["right"]["target_structure"]],
            value=right.get("target_structure", "(none)"),
            description="R target",
        )
        right_context_options = ["(none)", *taxonomy["right"]["target_context"]]
        self.right_context_1 = widgets.Dropdown(
            options=right_context_options,
            value=_valid_dropdown_value(right.get("target_context_1", right.get("target_context", "(none)")), right_context_options),
            description="R ctx 1",
        )
        self.right_context_2 = widgets.Dropdown(
            options=right_context_options,
            value=_valid_dropdown_value(right.get("target_context_2", "(none)"), right_context_options),
            description="R ctx 2",
        )
        self.camera_action = widgets.Dropdown(
            options=["(none)", *taxonomy["camera"]["action_code"]],
            value=camera.get("action_code", "(none)"),
            description="Camera",
        )
        self.note = widgets.Text(value=initial.get("note", ""), description="Note")
        self.remove_btn = widgets.Button(description="Remove")
        self.remove_btn.on_click(lambda _: self.parent._remove_subclip(self))
        self.box = widgets.VBox(
            [
                self.header,
                widgets.HBox([self.boundary_status, self.remove_btn]),
                self.frame_grid,
                widgets.HBox(
                    [
                        _actor_label("Left", LEFT_HELP_TEXT),
                        self.left_action,
                    ]
                ),
                widgets.HBox(
                    [
                        _actor_label("Right", RIGHT_HELP_TEXT),
                        widgets.VBox(
                            [
                                widgets.HBox([self.right_tool, self.right_action, self.right_target]),
                                widgets.HBox([self.right_context_1, self.right_context_2]),
                            ]
                        ),
                    ]
                ),
                widgets.HBox(
                    [
                        _actor_label("Camera", CAMERA_HELP_TEXT),
                        self.camera_action,
                    ]
                ),
                self.note,
            ],
            layout=widgets.Layout(border="1px solid #cbd5e1", padding="8px", margin="8px 0"),
        )

    def set_boundary(self, frame_id: int, boundary: str) -> None:
        if boundary == "start":
            self.start_frame = int(frame_id)
            if self.end_frame is not None and self.end_frame < self.start_frame:
                self.end_frame = self.start_frame
        else:
            self.end_frame = int(frame_id)
            if self.start_frame is not None and self.start_frame > self.end_frame:
                self.start_frame = self.end_frame
        self.parent._render_subclips()

    def is_complete(self) -> bool:
        return self.start_frame is not None and self.end_frame is not None

    def selected_range(self) -> tuple[int, int]:
        if not self.is_complete():
            raise ValueError("Subclip is incomplete")
        return (min(int(self.start_frame), int(self.end_frame)), max(int(self.start_frame), int(self.end_frame)))

    def refresh(self, index: int) -> None:
        self.index = index
        self.header.value = f"<div style='font-weight:700;'>Subclip {index}</div>"
        start_text = f"f{self.start_frame:06d}" if self.start_frame is not None else "not set"
        end_text = f"f{self.end_frame:06d}" if self.end_frame is not None else "not set"
        self.boundary_status.value = (
            "<div style='min-width:260px;font-size:13px;color:#334155;'>"
            f"<b>Start:</b> {start_text} &nbsp; <b>End:</b> {end_text}"
            "</div>"
        )
        gray_ranges = self.parent._other_selected_ranges(self)
        if self._frame_card_index != self.index or set(self._frame_card_widgets) != set(self.frame_ids):
            self._frame_card_widgets = {
                frame_id: self._frame_card(frame_id)
                for frame_id in self.frame_ids
            }
            self._frame_card_index = self.index
            self.frame_grid.children = tuple(self._frame_card_widgets[frame_id] for frame_id in self.frame_ids)
        self._update_frame_card_borders(gray_ranges)
        self.frame_grid.layout = widgets.Layout(grid_template_columns="repeat(auto-fill, 190px)")

    def _update_frame_card_borders(self, gray_ranges: list[tuple[int, int]]) -> None:
        selected = self.selected_range() if self.is_complete() else None
        for frame_id, card in self._frame_card_widgets.items():
            current = selected is not None and selected[0] <= frame_id <= selected[1]
            prior = any(start <= frame_id <= end for start, end in gray_ranges)
            card.layout.border = _frame_card_border(current=current, prior=prior)

    def _frame_card(self, frame_id: int) -> widgets.VBox:
        path = self.frame_dir / f"frame_{frame_id:06d}.png"
        if path.exists():
            image = widgets.HTML(
                _clickable_frame_image_html(
                    path,
                    frame_id=frame_id,
                    subclip_index=self.index,
                    frame_ids=self.frame_ids,
                )
            )
        else:
            image = widgets.HTML(
                "<div style='height:145px;width:180px;display:flex;align-items:flex-end;justify-content:center;"
                "color:#64748b;'>"
                f"missing f{frame_id:06d}</div>"
            )
        start_btn = widgets.Button(description="Start", layout=widgets.Layout(width="82px"))
        end_btn = widgets.Button(description="End", layout=widgets.Layout(width="82px"))
        start_btn.on_click(lambda _, fid=frame_id: self.set_boundary(fid, "start"))
        end_btn.on_click(lambda _, fid=frame_id: self.set_boundary(fid, "end"))
        return widgets.VBox(
            [
                image,
                widgets.HTML(
                    f"<div style='font-size:11px;line-height:12px;color:#475569;text-align:center;"
                    f"margin-top:1px;margin-bottom:0;'>f{frame_id:06d}</div>"
                ),
                widgets.HTML(_cvs_frame_badges(self.parent.current_cvs_frame_labels.get(frame_id))),
                widgets.HBox([start_btn, end_btn], layout=widgets.Layout(margin="4px 0 0 0")),
            ],
            layout=widgets.Layout(
                width="180px",
                border=_frame_card_border(current=False, prior=False),
                padding="3px",
                margin="0 8px 10px 0",
            ),
        )

    def to_json(self) -> dict:
        start, end = self.selected_range()
        out = {"start_frame": start, "end_frame": end, "left": {}, "right": {}, "camera": {}, "note": self.note.value}
        if self.left_action.value != "(none)":
            out["left"] = {"retraction_direction_code": self.left_action.value}
        right_values = (
            self.right_tool.value,
            self.right_action.value,
            self.right_target.value,
            self.right_context_1.value,
            self.right_context_2.value,
        )
        if any(value != "(none)" for value in right_values):
            out["right"] = {
                "tool_type": self.right_tool.value,
                "action_code": self.right_action.value,
                "target_structure": self.right_target.value,
                "target_context_1": _context_or_not_set(self.right_context_1.value),
                "target_context_2": _context_or_not_set(self.right_context_2.value),
            }
        if self.camera_action.value != "(none)":
            out["camera"] = {"action_code": self.camera_action.value}
        return out


LEFT_HELP_TEXT = (
    "Annotate the observed left instrument / grasper retraction only. Directions follow the LLM "
    "hint prompts: lateral = pulling leftward from the camera perspective, medial = pulling "
    "rightward, upward = cephalad/lifting toward the liver-bed view. KEEP_RETRACT_LATERAL means "
    "the grasper is already retracting laterally and maintains that lateral retraction. "
    "KEEP_RETRACT_MEDIAL means it maintains medial/rightward retraction. KEEP_RETRACT_UPWARD means "
    "it maintains upward/cephalad retraction. RETRACT_LATERAL means it begins or changes into "
    "lateral/leftward retraction from a non-retracted or unclear prior state. RETRACT_MEDIAL means "
    "it begins or changes into medial/rightward retraction. RETRACT_UPWARD means it begins or "
    "changes into upward/cephalad retraction. RETRACT_X_TO_Y means the grasper starts this subclip "
    "retracting in direction X and visibly changes to direction Y, e.g. RETRACT_LATERAL_TO_MEDIAL "
    "means it was pulling left/lateral and switches to pulling right/medial. Use the analogous "
    "meaning for LATERAL_TO_UPWARD, MEDIAL_TO_LATERAL, MEDIAL_TO_UPWARD, UPWARD_TO_LATERAL, and "
    "UPWARD_TO_MEDIAL. Leave (none) if the left grasper is absent, not acting, or no dropdown value "
    "matches what happened."
)
RIGHT_HELP_TEXT = (
    "Annotate the observed active right instrument only: dissection, clipping, cautery, aspiration, "
    "countertraction assist, sweeping, or withdrawing when it blocks view. Tool type should match "
    "the visible working instrument. For DISSECT in the hepatocystic region, choose "
    "target_structure by anatomic specificity. If the cystic duct and cystic artery / two tubular "
    "structures are not clearly delineated and the action is general dissection in that area, use "
    "HepatocysticTriangle. If the action is visibly operating mostly near the cystic duct or cystic "
    "artery, even if not fully skeletonized, prefer CysticDuct or CysticArtery. If the action is at "
    "the gallbladder-cystic plate / liver-bed interface, use CysticPlate. If it is at the neck or "
    "infundibulum, use GallbladderNeck_Infundibulum. Use target_context_1 and target_context_2 for "
    "up to two more precise visible locations: between presumed cystic duct and presumed cystic "
    "artery, between cystic artery and cystic plate, near the base of the hepatocystic triangle, "
    "close to the gallbladder neck, near the cystic duct, near the cystic plate, or on the outside "
    "of cystic duct. If the action targets the space between the two tubular structures rather than "
    "one structure specifically, use HepatocysticTriangle plus context 'between presumed cystic duct "
    "and presumed cystic artery'. If it targets the plane between artery and plate, use context "
    "'between cystic artery and cystic plate'. Use (none) when absent; saved empty contexts become "
    "(not set). Do not invent values outside the dropdowns."
)
CAMERA_HELP_TEXT = (
    "Annotate observed camera movement only; ignore tool motion. CAMERA_ZOOM_IN means the camera "
    "moves closer and structures appear larger. CAMERA_ZOOM_OUT means the camera moves farther away "
    "and structures appear smaller. CAMERA_REPOSITION means the camera changes angle, framing, or "
    "viewpoint without a clear zoom-only change, or the hepatocystic triangle is being re-centered. "
    "CAMERA_NO_CHANGE means the camera view stays meaningfully stable / no meaningful camera motion "
    "is visible. CAMERA_UNCERTAIN means camera motion is present or suspected but cannot be judged "
    "reliably. Leave (none) if no camera actor label should be recorded for this subclip."
)


def _actor_label(label: str, help_text: str) -> widgets.HTML:
    return widgets.HTML(
        "<span style='min-width:86px;display:inline-flex;align-items:center;gap:6px;'>"
        f"<b>{html.escape(label)}</b>"
        "<span "
        f"title='{html.escape(help_text, quote=True)}' "
        "style='display:inline-flex;align-items:center;justify-content:center;"
        "width:16px;height:16px;border:1px solid #64748b;border-radius:50%;"
        "font-size:11px;font-weight:700;color:#334155;cursor:help;'>i</span>"
        "</span>"
    )


def _context_or_not_set(value: str) -> str:
    return "(not set)" if value == "(none)" else value


def _valid_dropdown_value(value: str, options: list[str]) -> str:
    return value if value in options else "(none)"


def _cvs_labels_by_frame(video_name: str, frame_ids: list[int]) -> dict[int, dict]:
    labels = _load_cvs_keyframe_labels(
        video_name,
        {f"f{int(frame_id):06d}": int(frame_id) for frame_id in frame_ids},
    )
    return {int(item["frame"]): item for item in labels}


def _cvs_frame_badges(label: dict | None) -> str:
    if not label:
        return "<div style='height:22px;'></div>"
    badges = []
    criteria = label.get("criteria") or {}
    for crit in ("C1", "C2", "C3"):
        item = criteria.get(crit) or {}
        votes = int(item.get("votes", 0))
        total = int(item.get("total", 3)) or 3
        ratio = votes / total
        if ratio > 0.5:
            bg, fg, border = "#dcfce7", "#166534", "#86efac"
        else:
            bg, fg, border = "#fee2e2", "#991b1b", "#fecaca"
        badges.append(
            "<span "
            f"title='{crit}: {votes}/{total} raters' "
            "style='display:inline-block;margin:1px 2px;padding:1px 5px;"
            f"border:1px solid {border};border-radius:4px;background:{bg};color:{fg};"
            "font-size:11px;font-weight:700;'>"
            f"{crit} {votes}/{total}</span>"
        )
    return (
        "<div style='min-height:20px;line-height:16px;text-align:center;white-space:nowrap;"
        "margin-top:1px;margin-bottom:6px;'>"
        + "".join(badges)
        + "</div>"
    )


def _frame_card_border(current: bool, prior: bool) -> str:
    if current:
        return "4px solid #2563eb"
    if prior:
        return "4px solid #94a3b8"
    return "1px solid #cbd5e1"


def _cvs_keyframe_label_table(clip: ClipManifestRow) -> str:
    keyframes = (clip.existing_cvs_labels or {}).get("keyframes") or []
    if not keyframes:
        keyframes = _load_cvs_keyframe_labels(
            clip.video_name,
            {"start": int(clip.start_frame), "end": int(clip.end_frame)},
        )
    if not keyframes:
        criterion = (clip.existing_cvs_labels or {}).get("criterion") or clip.criterion
        mind_change = (clip.existing_cvs_labels or {}).get("mind_change") or clip.mind_change
        if not criterion and not mind_change:
            return (
                "<div style='margin:8px 0;padding:8px;border:1px solid #cbd5e1;"
                "background:#f8fafc;color:#64748b;font-size:12px;'>"
                "<b>CVS keyframe labels:</b> not found for this clip."
                "</div>"
            )
        parts = []
        if criterion:
            parts.append(f"<b>Transition criterion:</b> {html.escape(str(criterion))}")
        if mind_change:
            parts.append(f"<b>Mind change:</b> {html.escape(str(mind_change))}")
        return (
            "<div style='margin:8px 0;padding:8px;border:1px solid #cbd5e1;"
            "background:#f8fafc;color:#334155;font-size:12px;'>"
            "<b>CVS keyframe labels:</b> raw C1/C2/C3 labels not found. "
            + " &nbsp; ".join(parts)
            + "</div>"
        )

    rows = []
    for item in keyframes:
        criteria = item.get("criteria") or {}
        cells = []
        for crit in ("C1", "C2", "C3"):
            label = criteria.get(crit) or {}
            if label:
                cells.append(f"{crit} {int(label.get('votes', 0))}/{int(label.get('total', 3))}")
            else:
                cells.append(f"{crit} -")
        rows.append(
            "<tr>"
            f"<td style='padding:2px 8px 2px 0;'>{html.escape(str(item.get('role', 'key')))}</td>"
            f"<td style='padding:2px 8px 2px 0;'>f{int(item.get('frame', 0)):06d}</td>"
            f"<td style='padding:2px 8px 2px 0;'>{html.escape(' | '.join(cells))}</td>"
            "</tr>"
        )
    criterion = (clip.existing_cvs_labels or {}).get("criterion") or clip.criterion
    mind_change = (clip.existing_cvs_labels or {}).get("mind_change") or clip.mind_change
    summary = []
    if criterion:
        summary.append(f"transition {html.escape(str(criterion))}")
    if mind_change:
        summary.append(html.escape(str(mind_change)))
    title = "CVS keyframe labels"
    if summary:
        title += " (" + ", ".join(summary) + ")"
    return (
        "<div style='margin:8px 0;padding:8px;border:1px solid #cbd5e1;background:#f8fafc;color:#334155;'>"
        f"<div style='font-weight:700;margin-bottom:4px;'>{title}</div>"
        "<table style='border-collapse:collapse;font-size:12px;color:#334155;'>"
        + "".join(rows)
        + "</table></div>"
    )


def _load_simple_taxonomy(path: Path = DEFAULT_SIMPLE_TAXONOMY_PATH) -> dict[str, dict[str, list[str]]]:
    if path.exists():
        raw = json.loads(path.read_text())
        actors = raw.get("actors") or {}
        return {
            actor: dict((actors.get(actor) or {}).get("components") or {})
            for actor in ("left", "right", "camera")
        }
    return {
        "left": {
            "retraction_direction_code": [
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
        },
        "right": {
            "tool_type": ["Hook", "Irrigator", "Maryland", "Scissors", "clipper"],
            "action_code": [
                "CLIP",
                "COAGULATE_HEMOSTASIS",
                "COUNTERTRACTION_ASSIST",
                "DISSECT",
                "IRRIGATOR_ASPIRATE",
                "RETRACT_DOWNWARD",
                "SWEEPING",
                "TOOL_WITHDRAW_UNBLOCKS_VIEW",
            ],
            "target_structure": [
                "CysticArtery",
                "CysticDuct",
                "CysticPlate",
                "GallbladderNeck_Infundibulum",
                "HepatocysticTriangle",
            ],
            "target_context": [
                "between presumed cystic duct and presumed cystic artery",
                "between cystic artery and cystic plate",
                "near the base of the hepatocystic triangle",
                "close to the gallbladder neck",
                "near the cystic duct",
                "near the cystic plate",
                "(not set)",
                "on the outside of cystic duct",
            ],
        },
        "camera": {
            "action_code": [
                "CAMERA_NO_CHANGE",
                "CAMERA_REPOSITION",
                "CAMERA_UNCERTAIN",
                "CAMERA_ZOOM_IN",
                "CAMERA_ZOOM_OUT",
            ]
        },
    }


def _frame_ids_1fps(start_frame: int, end_frame: int) -> list[int]:
    step = 30
    frame_ids = list(range(int(start_frame), int(end_frame) + 1, step))
    for fid in (int(start_frame), int(end_frame)):
        if fid not in frame_ids:
            frame_ids.append(fid)
    return sorted(set(frame_ids))


def _ensure_clip_assets(clip: ClipManifestRow, frames_root: Path, assets_root: Path) -> dict[str, object]:
    split = clip.split or "test"
    asset_dir = assets_root / _safe_clip_name(clip.video_name) / _safe_clip_name(clip.clip_id)
    frame_asset_dir = asset_dir / "frames"
    video_asset_dir = asset_dir / "video"
    frame_asset_dir.mkdir(parents=True, exist_ok=True)
    video_asset_dir.mkdir(parents=True, exist_ok=True)

    source_frame_dir = frames_root / split / clip.video_name
    frame_ids = _frame_ids_1fps(clip.start_frame, clip.end_frame)
    copied_frames = []
    missing_frames = []
    for frame_id in frame_ids:
        source = source_frame_dir / f"frame_{int(frame_id):06d}.png"
        dest = frame_asset_dir / f"frame_{int(frame_id):06d}.png"
        if not source.exists():
            if dest.exists():
                copied_frames.append(str(dest))
                continue
            if _extract_frame_from_raw_video(clip, frame_id, dest):
                copied_frames.append(str(dest))
                continue
            missing_frames.append(int(frame_id))
            continue
        if not dest.exists() or dest.stat().st_size != source.stat().st_size:
            shutil.copy2(source, dest)
        copied_frames.append(str(dest))

    video_path: Path | None = None
    try:
        cached_video = _ensure_interval_video(clip)
        dest_video = video_asset_dir / cached_video.name
        if not dest_video.exists() or dest_video.stat().st_size != cached_video.stat().st_size:
            shutil.copy2(cached_video, dest_video)
        video_path = dest_video
    except Exception:
        video_path = None

    manifest = {
        "clip_id": clip.clip_id,
        "video_name": clip.video_name,
        "granularity": clip.granularity,
        "criterion": clip.criterion,
        "start_frame": int(clip.start_frame),
        "end_frame": int(clip.end_frame),
        "frames": copied_frames,
        "missing_frames": missing_frames,
        "video": str(video_path) if video_path else None,
    }
    (asset_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "asset_dir": asset_dir,
        "frame_dir": frame_asset_dir,
        "video_path": video_path,
        "missing_frames": len(missing_frames),
    }


def _extract_frame_from_raw_video(clip: ClipManifestRow, frame_id: int, dest: Path) -> bool:
    split = clip.split or "test"
    source = DEFAULT_RAW_VIDEO_ROOT / split / "videos" / f"{clip.video_name}.mp4"
    if not source.exists():
        return False
    try:
        import cv2

        cap = cv2.VideoCapture(str(source))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(dest), frame))
    except Exception:
        return False


def _clickable_frame_image_html(path: Path, frame_id: int, subclip_index: int, frame_ids: list[int]) -> str:
    thumb_url = _img_to_data_url(path, max_height=145)
    large_url = _img_to_data_url(path, max_height=820)
    ordered_frame_ids = [int(fid) for fid in frame_ids]
    frame_idx = ordered_frame_ids.index(int(frame_id))
    prev_frame_id = ordered_frame_ids[max(0, frame_idx - 1)]
    next_frame_id = ordered_frame_ids[min(len(ordered_frame_ids) - 1, frame_idx + 1)]
    modal_group = f"frame-modal-group-{int(subclip_index)}"
    modal_id = _frame_modal_id(subclip_index, frame_id)
    close_id = f"frame-modal-close-{int(subclip_index)}-{int(frame_id)}"
    prev_id = _frame_modal_id(subclip_index, prev_frame_id)
    next_id = _frame_modal_id(subclip_index, next_frame_id)
    frame_label = f"f{int(frame_id):06d}"
    focus_current_js = html.escape(_focus_frame_modal_js(modal_id), quote=True)
    select_prev_js = html.escape(_select_frame_modal_js(prev_id), quote=True)
    select_next_js = html.escape(_select_frame_modal_js(next_id), quote=True)
    prev_button = (
        f"<span onclick='event.preventDefault();event.stopPropagation();{select_prev_js}' "
        "style='position:absolute;left:24px;top:50%;transform:translateY(-50%);"
        "min-width:48px;height:48px;border-radius:24px;background:rgba(15,23,42,0.78);"
        "border:1px solid rgba(226,232,240,0.75);color:#fff;display:flex;align-items:center;"
        "justify-content:center;font-size:30px;font-weight:700;cursor:pointer;'>&lsaquo;</span>"
        if frame_idx > 0
        else ""
    )
    next_button = (
        f"<span onclick='event.preventDefault();event.stopPropagation();{select_next_js}' "
        "style='position:absolute;right:24px;top:50%;transform:translateY(-50%);"
        "min-width:48px;height:48px;border-radius:24px;background:rgba(15,23,42,0.78);"
        "border:1px solid rgba(226,232,240,0.75);color:#fff;display:flex;align-items:center;"
        "justify-content:center;font-size:30px;font-weight:700;cursor:pointer;'>&rsaquo;</span>"
        if frame_idx < len(ordered_frame_ids) - 1
        else ""
    )
    keydown_js = html.escape(
        (
            "if(event.key==='ArrowLeft'){"
            f"{_select_frame_modal_js(prev_id)}"
            "event.preventDefault();"
            "}else if(event.key==='ArrowRight'){"
            f"{_select_frame_modal_js(next_id)}"
            "event.preventDefault();"
            "}else if(event.key==='Escape'){"
            f"document.getElementById('{close_id}').checked=true;"
            "event.preventDefault();"
            "}"
        ),
        quote=True,
    )
    return (
        "<div style='height:145px;display:flex;align-items:flex-end;justify-content:center;'>"
        f"<input id='{close_id}' name='{modal_group}' type='radio' style='display:none;'/>"
        f"<input id='{modal_id}' name='{modal_group}' type='radio' style='display:none;'/>"
        f"<label for='{modal_id}' title='Open {frame_label} larger' onclick='{focus_current_js}' "
        "style='display:flex;align-items:flex-end;justify-content:center;height:145px;width:180px;cursor:zoom-in;'>"
        f"<img src='{thumb_url}' "
        "style='max-height:145px;max-width:180px;width:auto;height:auto;object-fit:contain;'/>"
        "</label>"
        "<style>"
        f"#{modal_id}:not(:checked) ~ .frame-modal-overlay {{ display:none !important; }}"
        "</style>"
        f"<label for='{close_id}' class='frame-modal-overlay' data-frame-modal='{modal_id}' tabindex='0' "
        f"onkeydown='{keydown_js}' "
        "style='position:fixed;inset:0;z-index:10000;background:rgba(15,23,42,0.86);"
        "display:flex;align-items:center;justify-content:center;padding:28px;cursor:zoom-out;'>"
        "<span style='position:absolute;top:18px;right:24px;color:#fff;font-size:28px;"
        "font-weight:700;line-height:1;'>x</span>"
        "<span style='position:absolute;right:24px;bottom:18px;color:#fff;font-size:28px;"
        "font-weight:700;line-height:1;background:rgba(0,0,0,0.65);padding:8px 10px;"
        "border-radius:2px;'>"
        f"{frame_label}"
        "</span>"
        f"<img src='{large_url}' "
        "style='max-width:96vw;max-height:92vh;width:auto;height:auto;object-fit:contain;"
        "border:1px solid #e2e8f0;background:#000;box-shadow:0 20px 80px rgba(0,0,0,0.45);'/>"
        f"{prev_button}"
        f"{next_button}"
        "</label>"
        "</div>"
    )


def _frame_modal_id(subclip_index: int, frame_id: int) -> str:
    return f"frame-modal-{int(subclip_index)}-{int(frame_id)}"


def _focus_frame_modal_js(modal_id: str) -> str:
    return (
        "setTimeout(function(){"
        f"var el=document.querySelector('[data-frame-modal=\"{modal_id}\"]');"
        "if(el){el.focus();}"
        "},0);"
    )


def _select_frame_modal_js(modal_id: str) -> str:
    return f"document.getElementById('{modal_id}').checked=true;{_focus_frame_modal_js(modal_id)}"


def _video_widget_for_clip(clip: ClipManifestRow) -> widgets.Widget:
    try:
        video_path = _ensure_interval_video(clip)
    except Exception as exc:
        return widgets.HTML(
            "<div style='padding:10px 12px;border:1px solid #fecaca;background:#fef2f2;"
            "border-radius:6px;color:#991b1b;'>"
            f"Could not load source mp4 for this clip: {html.escape(str(exc))}"
            "</div>"
        )
    return widgets.Video.from_file(str(video_path), format="mp4", width=760, height=430)


def _ensure_interval_video(clip: ClipManifestRow) -> Path:
    split = clip.split or "test"
    source = DEFAULT_RAW_VIDEO_ROOT / split / "videos" / f"{clip.video_name}.mp4"
    if not source.exists():
        raise FileNotFoundError(source)
    fps = _video_fps(source)
    start_s = max(0.0, float(clip.start_frame) / fps)
    end_s = max(start_s + 0.05, float(clip.end_frame + 1) / fps)
    out_dir = DEFAULT_VIDEO_CACHE / split / clip.video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_clip_name(clip.clip_id)}__frame_overlay_2x.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    frame_overlay = (
        "drawtext="
        "text='f%{eif\\:"
        f"{int(clip.start_frame)}+30*floor(n/30)"
        "\\:d}':"
        "x=w-tw-12:y=h-th-12:"
        "fontsize=48:"
        "fontcolor=white:"
        "box=1:"
        "boxcolor=black@0.65:"
        "boxborderw=12"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-to",
        f"{end_s:.6f}",
        "-i",
        str(source),
        "-an",
        "-vf",
        frame_overlay,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _video_fps(path: Path) -> float:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps > 0:
            return fps
    except Exception:
        pass
    return 30.0


def _safe_clip_name(clip_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in clip_id)[:220]


def _img_to_data_url(path: Path, max_height: int = 260) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(path, max_height=max_height)).decode("ascii")


def _png_bytes(path: Path, max_height: int = 260) -> bytes:
    image = Image.open(path)
    if image.height > max_height:
        ratio = max_height / float(image.height)
        image = image.resize((max(1, int(image.width * ratio)), max_height))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
