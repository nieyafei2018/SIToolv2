# -*- coding: utf-8 -*-
"""
Runtime CPU utilization tracing utilities.

This module samples real system CPU utilization during one SITool run and
exports a time-series figure. The sampling is OS-level (Linux `/proc/stat`)
instead of scheduler-estimated "thread efficiency".
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pylab as plt

_ENV_TRACE_FILE = "SITOOL_EFF_TRACE_FILE"
_ENV_TOTAL_THREADS = "SITOOL_EFF_TOTAL_THREADS"
_ENV_START_TS = "SITOOL_EFF_START_TS"

_SAMPLER_LOCK = threading.Lock()
_SAMPLER_THREAD: Optional[threading.Thread] = None
_SAMPLER_STOP_EVENT: Optional[threading.Event] = None
_LATEST_CPU_PERCENT: Optional[float] = None


def _enabled() -> bool:
    return bool(os.environ.get(_ENV_TRACE_FILE))


def _trace_file() -> Optional[str]:
    path = str(os.environ.get(_ENV_TRACE_FILE, "")).strip()
    return path or None


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _emit_record(path: Path, payload: Dict[str, Any]) -> None:
    line = (json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _read_proc_stat_total_idle() -> Optional[Tuple[int, int]]:
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline().strip()
    except Exception:
        return None

    if not line.startswith("cpu "):
        return None

    parts = line.split()
    if len(parts) < 5:
        return None
    vals: List[int] = []
    for token in parts[1:]:
        try:
            vals.append(int(token))
        except Exception:
            vals.append(0)
    while len(vals) < 8:
        vals.append(0)

    user, nice, system, idle, iowait, irq, softirq, steal = vals[:8]
    idle_all = idle + iowait
    total = user + nice + system + idle + iowait + irq + softirq + steal
    return total, idle_all


def _cpu_percent_from_loadavg() -> float:
    try:
        load1 = float(os.getloadavg()[0])
        cpu_count = max(1, int(os.cpu_count() or 1))
        return max(0.0, min(100.0, 100.0 * load1 / float(cpu_count)))
    except Exception:
        return 0.0


def _sample_once(prev: Optional[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], float, str]:
    current = _read_proc_stat_total_idle()
    if current is not None and prev is not None:
        total_delta = int(current[0]) - int(prev[0])
        idle_delta = int(current[1]) - int(prev[1])
        if total_delta > 0:
            busy_delta = max(0, total_delta - max(0, idle_delta))
            pct = 100.0 * float(busy_delta) / float(total_delta)
            return current, max(0.0, min(100.0, float(pct))), "proc_stat"
    if current is not None:
        return current, _cpu_percent_from_loadavg(), "loadavg_fallback"
    return prev, _cpu_percent_from_loadavg(), "loadavg_only"


def _stop_sampler() -> None:
    global _SAMPLER_THREAD, _SAMPLER_STOP_EVENT
    with _SAMPLER_LOCK:
        stop_evt = _SAMPLER_STOP_EVENT
        thread = _SAMPLER_THREAD
        _SAMPLER_STOP_EVENT = None
        _SAMPLER_THREAD = None
    if stop_evt is not None:
        stop_evt.set()
    if thread is not None:
        try:
            thread.join(timeout=2.0)
        except Exception:
            pass


def initialize_trace(
    trace_file: Path,
    total_threads: int,
    case_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Initialize one fresh CPU trace and start one background sampler thread."""
    global _LATEST_CPU_PERCENT
    _stop_sampler()

    safe_total = max(1, int(total_threads))
    trace_path = Path(trace_file).resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")

    start_ts = float(time.time())
    os.environ[_ENV_TRACE_FILE] = str(trace_path)
    os.environ[_ENV_TOTAL_THREADS] = str(safe_total)
    os.environ[_ENV_START_TS] = f"{start_ts:.6f}"

    _emit_record(
        trace_path,
        {
            "type": "meta",
            "ts": start_ts,
            "pid": int(os.getpid()),
            "tid": int(threading.get_ident()),
            "case_name": str(case_name or ""),
            "run_id": str(run_id or ""),
            "total_threads": safe_total,
            "start_ts": start_ts,
            "sample_interval_sec": 1.0,
        },
    )

    _LATEST_CPU_PERCENT = None
    stop_evt = threading.Event()

    def _sampler_loop(path: Path, started_at: float, stop_event: threading.Event) -> None:
        global _LATEST_CPU_PERCENT
        prev = _read_proc_stat_total_idle()
        while not stop_event.wait(timeout=1.0):
            now = float(time.time())
            prev, cpu_pct, source = _sample_once(prev)
            _LATEST_CPU_PERCENT = float(cpu_pct)
            try:
                _emit_record(
                    path,
                    {
                        "type": "cpu_sample",
                        "ts": now,
                        "pid": int(os.getpid()),
                        "tid": int(threading.get_ident()),
                        "elapsed_seconds": max(0.0, now - started_at),
                        "cpu_percent": float(cpu_pct),
                        "source": source,
                    },
                )
            except Exception:
                # Tracing failures must never abort a scientific run.
                pass

    sampler = threading.Thread(
        target=_sampler_loop,
        args=(trace_path, start_ts, stop_evt),
        name="sitool-cpu-sampler",
        daemon=True,
    )
    with _SAMPLER_LOCK:
        _SAMPLER_STOP_EVENT = stop_evt
        _SAMPLER_THREAD = sampler
    sampler.start()


def disable_trace() -> None:
    """Disable runtime trace emission and stop sampler in this process."""
    global _LATEST_CPU_PERCENT
    _stop_sampler()
    _LATEST_CPU_PERCENT = None
    for key in (_ENV_TRACE_FILE, _ENV_TOTAL_THREADS, _ENV_START_TS):
        os.environ.pop(key, None)


@contextmanager
def process_busy_scope(
    module: Optional[str] = None,
    hemisphere: Optional[str] = None,
    expected_threads: Optional[int] = None,
):
    """Compatibility no-op context manager (kept for existing call sites)."""
    _ = (module, hemisphere, expected_threads)
    yield


def run_tracked_task(func, *args, **kwargs):
    """Compatibility wrapper: execute callable without task-level pseudo tracing."""
    return func(*args, **kwargs)


def _load_samples(trace_file: Path) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    meta: Dict[str, Any] = {}
    samples: List[Tuple[float, float]] = []
    path = Path(trace_file)
    if not path.exists():
        return meta, samples

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            typ = str(rec.get("type", "")).strip().lower()
            if typ == "meta" and not meta:
                meta = rec
                continue
            if typ != "cpu_sample":
                continue
            cpu_pct = max(0.0, min(100.0, _safe_float(rec.get("cpu_percent"), 0.0)))
            elapsed = _safe_float(rec.get("elapsed_seconds"), -1.0)
            if elapsed < 0.0:
                start_ts = _safe_float(meta.get("start_ts"), 0.0)
                elapsed = max(0.0, _safe_float(rec.get("ts"), start_ts) - start_ts)
            samples.append((elapsed, cpu_pct))

    samples.sort(key=lambda x: float(x[0]))
    return meta, samples


def latest_cpu_utilization_percent(trace_file: Path, fallback_percent: float = 0.0) -> float:
    """Return latest sampled real system CPU utilization percent."""
    if _LATEST_CPU_PERCENT is not None:
        return float(_LATEST_CPU_PERCENT)
    _meta, samples = _load_samples(Path(trace_file))
    if not samples:
        return float(fallback_percent)
    return float(samples[-1][1])


def latest_efficiency_percent(trace_file: Path, fallback_total_threads: int = 1) -> float:
    """Compatibility alias kept for scheduler callers."""
    _ = fallback_total_threads
    return latest_cpu_utilization_percent(trace_file=trace_file, fallback_percent=0.0)


def export_trace_outputs(
    trace_file: Path,
    output_png: Path,
    output_csv: Optional[Path] = None,
    chart_title: str = "CPU Real-time Utilization",
) -> Tuple[Path, Optional[Path]]:
    """Export one real-CPU-utilization PNG and optional sampled CSV."""
    meta, samples = _load_samples(Path(trace_file))
    if not samples:
        start_ts = _safe_float(meta.get("start_ts"), 0.0)
        elapsed = max(1.0, float(time.time()) - start_ts) if start_ts > 0.0 else 60.0
        samples = [(0.0, 0.0), (float(elapsed), 0.0)]

    png_path = Path(output_png)
    csv_path = Path(output_csv) if output_csv is not None else None
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("elapsed_seconds,cpu_percent\n")
            for elapsed, pct in samples:
                f.write(f"{elapsed:.6f},{pct:.6f}\n")

    xs = [float(row[0]) for row in samples]
    ys = [float(row[1]) for row in samples]
    if len(xs) == 1:
        xs.append(xs[0] + 1.0)
        ys.append(ys[0])
    xs_minutes = [x / 60.0 for x in xs]
    peak_cpu = max(ys) if ys else 0.0
    mean_cpu = (sum(ys) / float(len(ys))) if ys else 0.0
    requested_threads = max(1, _safe_int(meta.get("total_threads"), 1))
    title_with_meta = (
        f"{chart_title} | Threads budget: {requested_threads} "
        f"(mean: {mean_cpu:.1f}%, peak: {peak_cpu:.1f}%)"
    )

    fig = plt.figure(figsize=(10, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(xs_minutes, ys, linewidth=2.0, color="#1f77b4")
    ax.plot(xs_minutes, ys, linestyle="none", marker="o", markersize=2.2, color="#1f77b4")
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("System CPU utilization (%)")
    ax.set_title(title_with_meta)
    ax.set_ylim(0.0, 100.0)
    ax.set_xlim(0.0, max(1.0, xs_minutes[-1] if xs_minutes else 1.0))
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return png_path, csv_path
