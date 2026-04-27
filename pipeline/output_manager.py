"""output_manager.py — Tracker-conditional output directory management.

Centralises the creation of the canonical Phase 1 / Phase 2 output
directory trees so that every module that writes files can use the same
paths without hard-coding them.

Directory layout created by :func:`setup_output_dirs`::

    <base>/
        videos/           ← annotated output videos (.mp4)
        videos/frames/    ← frame snapshots (recovered / flicker)
        logs/             ← tracking_log.json, per-video CSV
        metrics/          ← metrics.json

Canonical bases per tracker::

    SORT     → results/phase_2_sort/
    DeepSORT → results/phase_2_deepsort/
"""

from __future__ import annotations

from pathlib import Path


# Canonical root per tracker type (relative to project root)
_TRACKER_BASE: dict[str, str] = {
    "sort":     "results/phase_2_sort",
    "deepsort": "results/phase_2_deepsort",
}


def tracker_base_dir(tracker_type: str) -> Path:
    """Return the canonical base output directory for *tracker_type*.

    Args:
        tracker_type: ``"sort"`` or ``"deepsort"`` (case-insensitive).

    Returns:
        A :class:`~pathlib.Path` for the tracker's output root.

    Raises:
        ValueError: if *tracker_type* is not recognised.
    """
    key = tracker_type.lower()
    if key not in _TRACKER_BASE:
        raise ValueError(
            f"Unknown tracker_type '{tracker_type}'. "
            f"Expected one of: {list(_TRACKER_BASE)}"
        )
    return Path(_TRACKER_BASE[key])


def setup_output_dirs(base_path: str | Path) -> dict[str, Path]:
    """Create the standard output directory tree under *base_path*.

    All sub-directories are created with ``exist_ok=True`` so this
    function is safe to call multiple times.

    Args:
        base_path: Root directory for one tracker's outputs,
                   e.g. ``results/phase_2_deepsort``.

    Returns:
        Dict mapping logical names to created :class:`~pathlib.Path` objects:

        - ``"base"``    — *base_path* itself
        - ``"videos"``  — ``<base>/videos/``
        - ``"frames"``  — ``<base>/videos/frames/``
        - ``"logs"``    — ``<base>/logs/``
        - ``"metrics"`` — ``<base>/metrics/``
    """
    base = Path(base_path)
    paths: dict[str, Path] = {
        "base":    base,
        "videos":  base / "videos",
        "frames":  base / "videos" / "frames",
        "logs":    base / "logs",
        "metrics": base / "metrics",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
