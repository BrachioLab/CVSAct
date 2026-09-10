"""Visualization helpers: frame grids, video overviews, HTML cards, PDF figures."""

import html as html_mod
import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

from .schemas import CRITERIA, COLS_PER_ROW


def parse_llm_json(raw_text):
    """Parse JSON from LLM output, handling ```json ... ``` wrapping and bare JSON."""
    if raw_text is None:
        return None
    text = raw_text.strip()
    m = re.match(r'^```(?:json)?\s*\n?(.*?)\n?\s*```$', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def plot_frame_grid(frame_ids_list, frame_dir, title, key_frame_set=None,
                    key_color="tab:blue", label_fn=None, cols_per_row=COLS_PER_ROW,
                    title_color="black"):
    """Plot a grid of frame images. Returns fig (no plt.show())."""
    n_frames = len(frame_ids_list)
    if n_frames == 0:
        return None
    nrows = int(np.ceil(n_frames / cols_per_row))
    ncols = min(n_frames, cols_per_row)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.8 * nrows), squeeze=False)
    for f_idx, fid in enumerate(frame_ids_list):
        r, c = f_idx // cols_per_row, f_idx % cols_per_row
        ax = axes[r][c]
        fpath = os.path.join(frame_dir, f"frame_{fid:06d}.png")
        if os.path.isfile(fpath):
            ax.imshow(imread(fpath))
        else:
            ax.text(0.5, 0.5, f"f{fid}\n(missing)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        title_str = f"f{fid}"
        is_key = key_frame_set and fid in key_frame_set
        if is_key:
            title_str += " [KEY]"
        if label_fn:
            title_str += f"\n{label_fn(fid)}"
        ax.set_title(title_str, fontsize=10, fontweight="bold" if is_key else "normal")
        if is_key:
            for spine in ax.spines.values():
                spine.set_edgecolor(key_color); spine.set_linewidth(3)
    for f_idx in range(n_frames, nrows * ncols):
        r, c = f_idx // cols_per_row, f_idx % cols_per_row
        axes[r][c].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", color=title_color)
    plt.tight_layout()
    return fig


def plot_video_overview(vid, vdata, frame_dir, selected_clips, criteria=CRITERIA):
    """Plot all key frames for a video with raw scores for all criteria. Returns fig."""
    df_vid = vdata["df"]
    frame_ids = df_vid["frame_id"].values.astype(int)
    valid_fids = [fid for fid in frame_ids
                  if os.path.isfile(os.path.join(frame_dir, f"frame_{fid:06d}.png"))]
    if not valid_fids:
        print(f"  (no frame images found for {vid})")
        return None
    n = len(df_vid)
    fid_to_idx = {int(df_vid.iloc[i]["frame_id"]): i for i in range(n)}
    crit_summary = []
    for crit in criteria:
        nc = len(selected_clips[vid][crit])
        if nc > 0:
            crit_summary.append(f"{crit.upper()}:{nc}")

    def overview_label(fid):
        if fid in fid_to_idx:
            idx = fid_to_idx[fid]
            return " ".join(f"{c}={vdata[c]['raw'][idx]:.2f}" for c in criteria)
        return ""

    return plot_frame_grid(
        valid_fids, frame_dir,
        f"FULL VIDEO: {vid[:36]}..  |  {len(valid_fids)} key frames  |  "
        f"Selected clips: {', '.join(crit_summary)}",
        label_fn=overview_label
    )


# ──────────────────────────────────────────────────────────────
# PDF helpers (matplotlib text figures)
# ──────────────────────────────────────────────────────────────

def _wrap(text, width=88, indent=""):
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent + "  ")


def render_action_to_fig(data, *, frame_start=None, frame_end=None,
                         score_start=None, score_end=None,
                         clip_label="", raw_json_str=None):
    """Render parsed action JSON as a matplotlib figure (for PDF)."""
    if data is None:
        fig = plt.figure(figsize=(11, 2))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.5, f"[Failed to parse JSON]\n{(raw_json_str or '')[:200]}",
                 ha="center", va="center", fontsize=10, color="red",
                 fontfamily="monospace")
        return fig

    lines = []
    if clip_label:
        lines.append(clip_label)
        lines.append("\u2500" * 70)

    crit = data.get("criterion", "?")
    mind_change = data.get("mind_change", "?").replace("->", " \u2192 ")
    scene_flags = ", ".join(data.get("scene_flags", [])) or "none"
    start_conf = data.get("start_state", {}).get("confidence", "?")
    end_conf = data.get("end_state", {}).get("confidence", "?")

    lines.append(f"{crit}: {mind_change}")
    fs = f"{frame_start} \u2192 {frame_end}" if frame_start is not None else "?"
    ss = f"{score_start:.3f} \u2192 {score_end:.3f}" if score_start is not None else "?"
    lines.append(f"Frames: {fs}  |  Score: {ss}  |  Scene: {scene_flags}")
    lines.append(f"Start conf: {start_conf}  |  End conf: {end_conf}")
    lines.append("")

    lines.append("PERCEPTION SHIFT")
    for label, key, fid in [("Start", "start_state", frame_start),
                             ("End", "end_state", frame_end)]:
        bullets = data.get(key, {}).get("what_is_visible", [])
        lines.append(f"  {label} state (frame {fid or '?'}):")
        for b in bullets:
            lines.append(_wrap(b, indent="    \u2022 "))

    mid = data.get("mid_state")
    if mid and isinstance(mid, dict) and mid.get("what_is_visible"):
        mid_fid = mid.get("evidence_frame", "?")
        mid_conf = mid.get("confidence", "?")
        lines.append(f"  Mid state (frame {mid_fid}, conf {mid_conf}):")
        for b in mid["what_is_visible"]:
            lines.append(_wrap(b, indent="    \u2022 "))
    lines.append("")

    actions = data.get("actions_ranked", [])
    lines.append("CAUSAL ACTIONS")
    if not actions:
        lines.append("  (none)")
    for a in actions:
        rank = a.get("rank", "?")
        side = (a.get("actor_side") or "?").capitalize()
        tool = a.get("tool", "?")
        verb = a.get("verb", "?")
        imp = a.get("importance", "?")
        conf = a.get("confidence", "?")
        sentence = a.get("one_sentence", "")
        rationale = a.get("rationale", "")
        lines.append(f"  #{rank} {side} \u00b7 {tool} \u00b7 {verb}"
                     f"  (importance: {imp}, conf: {conf})")
        lines.append(_wrap(f"\u2b9e {sentence}", indent="     "))
        if rationale:
            lines.append(_wrap(f"Rationale: {rationale}", indent="     "))
    lines.append("")

    track = data.get("action_track", [])
    if track:
        parts = []
        for t in track:
            side = (t.get("actor_side") or "?").capitalize()
            tool = t.get("tool", "?")
            verb = t.get("verb", "?")
            direction = t.get("direction", "")
            dir_str = f" ({direction})" if direction and direction != "unclear" else ""
            parts.append(f"{side} {tool}: {verb}{dir_str}")
        lines.append("ACTION TRACK: " + " | ".join(parts))

    text = "\n".join(lines)
    n_lines = text.count("\n") + 1
    fig_height = max(2.5, n_lines * 0.19 + 0.6)

    fig = plt.figure(figsize=(11, fig_height))
    fig.patch.set_facecolor("white")
    fig.text(0.03, 0.97, text, fontsize=7.5, fontfamily="monospace",
             verticalalignment="top", horizontalalignment="left",
             linespacing=1.5)
    return fig


def render_text_header_fig(text):
    """Create a small matplotlib figure containing header text (for PDF)."""
    n_lines = text.count("\n") + 1
    fig = plt.figure(figsize=(11, max(0.6, n_lines * 0.25 + 0.3)))
    fig.patch.set_facecolor("white")
    fig.text(0.03, 0.5, text, fontsize=9, fontfamily="monospace",
             verticalalignment="center")
    return fig


# ──────────────────────────────────────────────────────────────
# Notebook HTML rendering (with highlighted one_sentence)
# ──────────────────────────────────────────────────────────────

def render_action_html(data, *, frame_start=None, frame_end=None,
                       score_start=None, score_end=None, raw_json_str=None,
                       clip_label=""):
    """Render parsed action JSON as a styled HTML card in the notebook."""
    from IPython.display import display, HTML

    if data is None:
        display(HTML(
            '<div style="border:1px solid #fca5a5; border-radius:10px; padding:12px; '
            'background:#fef2f2; color:#991b1b; font-size:13px; margin:8px 0;">'
            '<b>Failed to parse JSON</b><br>'
            '<pre style="white-space:pre-wrap; font-size:12px; margin-top:6px;">'
            + html_mod.escape(raw_json_str or "(empty)") + '</pre></div>'
        ))
        return

    esc = html_mod.escape

    crit = data.get("criterion", "?")
    mind_change = data.get("mind_change", "?")
    scene_flags = ", ".join(data.get("scene_flags", [])) or "none"

    start_conf = data.get("start_state", {}).get("confidence", "?")
    end_conf = data.get("end_state", {}).get("confidence", "?")

    if "unsatisfied->satisfied" in mind_change:
        mc_html = ('<span style="color:#b91c1c;">unsatisfied</span> &rarr; '
                   '<span style="color:#065f46;">satisfied</span>')
    elif "satisfied->unsatisfied" in mind_change:
        mc_html = ('<span style="color:#065f46;">satisfied</span> &rarr; '
                   '<span style="color:#b91c1c;">unsatisfied</span>')
    else:
        mc_html = '<span style="color:#6b7280;">no change</span>'

    frames_str = (f"{frame_start} &rarr; {frame_end}"
                  if frame_start is not None and frame_end is not None else "?")
    score_str = (f"{score_start:.3f} &rarr; {score_end:.3f}"
                 if score_start is not None and score_end is not None else "?")

    start_bullets = data.get("start_state", {}).get("what_is_visible", [])
    end_bullets = data.get("end_state", {}).get("what_is_visible", [])
    start_li = "".join(f"<li>{esc(b)}</li>" for b in start_bullets)
    end_li = "".join(f"<li>{esc(b)}</li>" for b in end_bullets)

    # mid_state (optional)
    mid = data.get("mid_state")
    mid_html_block = ""
    if mid and isinstance(mid, dict) and mid.get("what_is_visible"):
        mid_fid = mid.get("evidence_frame", "?")
        mid_conf = mid.get("confidence", "?")
        mid_li = "".join(f"<li>{esc(b)}</li>" for b in mid["what_is_visible"])
        mid_html_block = (
            '<div style="margin-top:12px; padding:8px 10px; border-radius:8px;'
            ' background:linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);'
            ' border-left:4px solid #3b82f6;">'
            '<div style="font-size:12px; letter-spacing:.03em; color:#1e40af; text-transform:uppercase;'
            ' font-weight:600;">'
            f'Mid state (frame {mid_fid}) &middot; conf {mid_conf}</div>'
            f'<ul style="margin:6px 0 0 18px; color:#1e3a5f; font-size:13px;">{mid_li}</ul>'
            '</div>'
        )

    delta_lookup = {}
    for d in data.get("deltas", []):
        key = (d.get("verb"), d.get("actor_side"))
        if key not in delta_lookup:
            delta_lookup[key] = d

    actions = data.get("actions_ranked", [])
    actions_html = ""
    for a in actions:
        rank = a.get("rank", "?")
        side = (a.get("actor_side") or "?").capitalize()
        tool = esc(a.get("tool", "?"))
        verb = esc(a.get("verb", "?"))
        imp = a.get("importance", "?")
        conf = a.get("confidence", "?")
        sentence = esc(a.get("one_sentence", ""))
        rationale = esc(a.get("rationale", ""))

        d = delta_lookup.get((a.get("verb"), a.get("actor_side")), {})
        mechanism = esc(d.get("mechanism", ""))
        evidence = d.get("evidence_frame", "")

        mech_parts = []
        if mechanism:
            mech_parts.append(f"Mechanism: {mechanism}")
        if rationale and rationale != mechanism:
            mech_parts.append(f"Rationale: {rationale}")
        if evidence:
            mech_parts.append(f"Evidence frame: <b>{evidence}</b>")
        mech_line = (
            '<div style="margin-top:6px; color:#6b7280; font-size:12px;">'
            + " &middot; ".join(mech_parts) + '</div>' if mech_parts else ""
        )

        # highlighted one_sentence
        sentence_block = (
            '<div style="margin-top:8px; padding:6px 10px; border-radius:6px;'
            ' background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);'
            ' border-left:4px solid #f59e0b; color:#78350f;'
            ' font-size:13.5px; font-weight:600; line-height:1.4;">'
            f'{sentence}</div>'
        )

        actions_html += (
            '<div style="margin-top:10px; padding:10px; border-radius:12px;'
            ' background:#f9fafb; border:1px solid #eef2f7;">'
            '<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">'
            f'<div style="font-weight:700; color:#111827; font-size:13px;">'
            f'#{rank} {side} &middot; {tool} &middot; {verb}</div>'
            f'<div style="font-size:12px; color:#374151;">'
            f'importance <b>{imp}</b> &middot; conf <b>{conf}</b></div>'
            '</div>'
            + sentence_block
            + mech_line +
            '</div>'
        )

    if not actions:
        actions_html = ('<div style="margin-top:10px; color:#6b7280; font-size:13px;">'
                        'No actions (no_change or none identified)</div>')

    track = data.get("action_track", [])
    track_pills = ""
    for t in track:
        side = (t.get("actor_side") or "?").capitalize()
        tool = esc(t.get("tool", "?"))
        verb = esc(t.get("verb", "?"))
        direction = t.get("direction", "")
        dir_str = f" ({esc(direction)})" if direction and direction != "unclear" else ""
        track_pills += (
            f'<span style="padding:6px 10px; border-radius:999px; background:#eef2ff; '
            f'color:#1f2937; font-size:12px;">{side} {tool}: {verb}{dir_str}</span>\n'
        )

    raw_escaped = esc(raw_json_str if raw_json_str else json.dumps(data, indent=2))

    label_line = (f'<div style="font-size:12px; color:#6b7280; margin-bottom:4px;">{esc(clip_label)}</div>'
                  if clip_label else "")

    full_html = (
        '<div style="'
        'font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;'
        ' line-height: 1.35; max-width: 980px;'
        ' border: 1px solid #e5e7eb; border-radius: 14px;'
        ' padding: 16px 18px; background: #fff; margin: 10px 0;">'
        + label_line +
        '<div style="display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap;">'
        '<div>'
        '<div style="font-size:14px; color:#6b7280;">Mind change</div>'
        f'<div style="font-size:20px; font-weight:700;">{crit}: {mc_html}</div>'
        f'<div style="margin-top:6px; color:#374151; font-size:13px;">'
        f'Frames: <b>{frames_str}</b> &middot; Score: <b>{score_str}</b>'
        f' &middot; Scene: <b>{esc(scene_flags)}</b></div>'
        '</div>'
        '<div style="display:flex; gap:8px; align-items:center;">'
        f'<div style="padding:6px 10px; border-radius:999px; background:#f3f4f6;'
        f' font-size:12px; color:#374151;">Start conf: <b>{start_conf}</b></div>'
        f'<div style="padding:6px 10px; border-radius:999px; background:#f3f4f6;'
        f' font-size:12px; color:#374151;">End conf: <b>{end_conf}</b></div>'
        '</div></div>'
        '<div style="height:12px;"></div>'
        '<div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">'
        '<div style="border:1px solid #eef2f7; border-radius:12px; padding:12px;">'
        '<div style="font-weight:700; font-size:14px; color:#111827;">Perception shift</div>'
        '<div style="margin-top:10px;">'
        f'<div style="font-size:12px; letter-spacing:.03em; color:#6b7280; text-transform:uppercase;">'
        f'Start state (frame {frame_start or "?"})</div>'
        f'<ul style="margin:6px 0 0 18px; color:#111827; font-size:13px;">{start_li}</ul>'
        '</div>'
        + mid_html_block +
        '<div style="margin-top:12px;">'
        f'<div style="font-size:12px; letter-spacing:.03em; color:#6b7280; text-transform:uppercase;">'
        f'End state (frame {frame_end or "?"})</div>'
        f'<ul style="margin:6px 0 0 18px; color:#111827; font-size:13px;">{end_li}</ul>'
        '</div></div>'
        '<div style="border:1px solid #eef2f7; border-radius:12px; padding:12px;">'
        '<div style="font-weight:700; font-size:14px; color:#111827;">Causal actions</div>'
        + actions_html +
        '<div style="margin-top:12px;">'
        '<div style="font-size:12px; letter-spacing:.03em; color:#6b7280; text-transform:uppercase;">'
        'Action track</div>'
        f'<div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:8px;">{track_pills}</div>'
        '</div></div>'
        '</div>'
        '<details style="margin-top:14px;">'
        '<summary style="cursor:pointer; color:#374151; font-size:13px;">Raw JSON</summary>'
        '<pre style="margin-top:10px; padding:12px; border-radius:12px;'
        ' background:#0b1020; color:#e5e7eb; overflow:auto;'
        f' font-size:12px; white-space:pre-wrap;">{raw_escaped}</pre>'
        '</details>'
        '</div>'
    )
    display(HTML(full_html))
