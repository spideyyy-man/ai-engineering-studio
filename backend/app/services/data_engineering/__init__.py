"""Data Engineering & EDA Studio — full implementation."""
from .baseline import compute_baseline
from .diff import column_stats, stats_diff
from .history import history_dir, load_snapshot, save_snapshot, snapshot_path
from .loader import load_dataframe, save_upload
from .pipeline_export import build_pipeline_export
from .profiler import profile_dataframe, summarize_for_card
from .quality import build_quality_report
from .recommendations import build_recommendations
from .task_detector import detect_task_type
from .transformer import (
    OPERATIONS,
    apply_operation,
    list_operations,
    recommend_fix,
    suggest_operations,
)

__all__ = [
    "OPERATIONS",
    "apply_operation",
    "build_pipeline_export",
    "build_quality_report",
    "build_recommendations",
    "column_stats",
    "compute_baseline",
    "detect_task_type",
    "history_dir",
    "list_operations",
    "load_dataframe",
    "load_snapshot",
    "profile_dataframe",
    "recommend_fix",
    "save_snapshot",
    "save_upload",
    "snapshot_path",
    "stats_diff",
    "suggest_operations",
    "summarize_for_card",
]
