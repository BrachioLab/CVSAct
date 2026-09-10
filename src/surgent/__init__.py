"""Surgent LangGraph agent (template)."""

from __future__ import annotations

from .schemas import CRITERION_DEFINITIONS, DEFAULT_CVS_CRITERIA, CVSRequest, CVSResult
try:
    from .graph import (
        SurgentContext,
        run_cvs_agent_sync,
        run_cvs_agent_with_memory,
        get_tool_help,
        build_agent_graph,
    )
except Exception as _graph_import_error:  # pragma: no cover - optional graph deps
    SurgentContext = None  # type: ignore[assignment]

    def _raise_graph_import_error(*args, **kwargs):
        raise ImportError(
            "surgent.graph requires langchain_core/langgraph dependencies."
        ) from _graph_import_error

    run_cvs_agent_sync = _raise_graph_import_error  # type: ignore[assignment]
    run_cvs_agent_with_memory = _raise_graph_import_error  # type: ignore[assignment]
    get_tool_help = _raise_graph_import_error  # type: ignore[assignment]
    build_agent_graph = _raise_graph_import_error  # type: ignore[assignment]

__all__ = [
    "CRITERION_DEFINITIONS",
    "DEFAULT_CVS_CRITERIA",
    "CVSRequest",
    "CVSResult",
    "build_agent_graph",
    "run_cvs_agent_sync",
    "run_cvs_agent_with_memory",
    "get_tool_help",
    "SurgentContext",
]
