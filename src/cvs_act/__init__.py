"""CVS Action Description package — extract, prompt, and visualize CVS transition actions."""

from .schemas import (
    CRITERION_DEFINITIONS,
    _CRIT_DEF_LOWER,
    VERBS,
    TOOLS,
    ITEMS_PRIORITY,
    REGIONS_PRIORITY,
    SCENE_FLAGS,
    DIRECTIONS,
    ALLOW_OTHER_ITEMS,
    ALLOW_OTHER_REGIONS,
    CRITERIA,
    RATERS,
    GAUSSIAN_SIGMA,
    FRAME_STEP,
    COLS_PER_ROW,
)
from .extraction import (
    load_video_data,
    segment_blurred,
    find_selected_ascending_clips,
    extract_frames_v1,
    build_clip_records,
    sample_clips,
)
try:
    from .prompts import describe_transition_action
except Exception:  # optional runtime dependency for notebook-only environments
    describe_transition_action = None

try:
    from .visualization import (
        parse_llm_json,
        plot_frame_grid,
        plot_video_overview,
        render_action_html,
        render_action_to_fig,
        render_text_header_fig,
    )
except Exception:  # optional plotting dependencies
    parse_llm_json = None
    plot_frame_grid = None
    plot_video_overview = None
    render_action_html = None
    render_action_to_fig = None
    render_text_header_fig = None

from .io import img_to_data_url, save_and_show, save_only


def __getattr__(name):
    if name in {"AuditVersionInterface", "AuditV11Interface"}:
        from .audit_v11_interface import AuditVersionInterface, AuditV11Interface

        return {
            "AuditVersionInterface": AuditVersionInterface,
            "AuditV11Interface": AuditV11Interface,
        }[name]
    raise AttributeError(name)
