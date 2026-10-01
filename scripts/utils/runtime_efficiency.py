# -*- coding: utf-8 -*-
"""
Runtime CPU and memory tracing utilities.

This module samples real system CPU utilization together with process-tree RSS
memory during one SITool run and exports time-series figures. CPU sampling is
OS-level (Linux `/proc/stat`) instead of scheduler-estimated "thread
efficiency".
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
_LATEST_RSS_BYTES: Optional[int] = None


# -----------------------------------------------------------------------------
# Function: `_enabled`.
# Purpose: Handle enabled workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _enabled() -> bool:
    return bool(os.environ.get(_ENV_TRACE_FILE))


# -----------------------------------------------------------------------------
# Function: `_trace_file`.
# Purpose: Handle trace file workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _trace_file() -> Optional[str]:
    path = str(os.environ.get(_ENV_TRACE_FILE, "")).strip()
    return path or None


# -----------------------------------------------------------------------------
# Function: `_safe_int`.
# Purpose: Handle safe int workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


# -----------------------------------------------------------------------------
# Function: `_safe_float`.
# Purpose: Handle safe float workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


# -----------------------------------------------------------------------------
# Function: `_emit_record`.
# Purpose: Handle emit record workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _emit_record(path: Path, payload: Dict[str, Any]) -> None:
    line = (json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


# -----------------------------------------------------------------------------
# Function: `_read_proc_stat_total_idle`.
# Purpose: Handle read proc stat total idle workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Function: `_cpu_percent_from_loadavg`.
# Purpose: Handle cpu percent from loadavg workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _cpu_percent_from_loadavg() -> float:
    try:
        load1 = float(os.getloadavg()[0])
        cpu_count = max(1, int(os.cpu_count() or 1))
        return max(0.0, min(100.0, 100.0 * load1 / float(cpu_count)))
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# Function: `_sample_once`.
# Purpose: Handle sample once workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Function: `_read_proc_children`.
# Purpose: Handle read proc children workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _read_proc_children(parent_pid: int) -> List[int]:
    children_file = Path(f"/proc/{int(parent_pid)}/task/{int(parent_pid)}/children")
    try:
        raw = children_file.read_text(encoding="utf-8").strip()
    except Exception:
        return []
    if not raw:
        return []
    out: List[int] = []
    for tok in raw.split():
        try:
            pid = int(tok)
        except Exception:
            continue
        if pid > 0:
            out.append(pid)
    return out


# -----------------------------------------------------------------------------
# Function: `_read_proc_rss_bytes`.
# Purpose: Handle read proc rss bytes workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _read_proc_rss_bytes(pid: int) -> Optional[int]:
    status_file = Path(f"/proc/{int(pid)}/status")
    try:
        with status_file.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line.startswith("VmRSS:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                rss_kib = int(parts[1])
                return max(0, rss_kib) * 1024
    except Exception:
        return None
    return None


# -----------------------------------------------------------------------------
# Function: `_sample_process_tree_rss_bytes`.
# Purpose: Handle sample process tree rss bytes workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _sample_process_tree_rss_bytes(root_pid: int) -> Tuple[int, int, str]:
    root = int(root_pid)
    visited: set = set()
    stack: List[int] = [root]
    rss_bytes_total = 0
    counted_processes = 0

    while stack:
        pid = int(stack.pop())
        if pid <= 0 or pid in visited:
            continue
        visited.add(pid)

        rss_bytes = _read_proc_rss_bytes(pid)
        if rss_bytes is not None:
            rss_bytes_total += int(rss_bytes)
            counted_processes += 1

        for child_pid in _read_proc_children(pid):
            if child_pid not in visited:
                stack.append(child_pid)

    if counted_processes > 0:
        return int(rss_bytes_total), int(counted_processes), "proc_tree_rss"

    fallback_rss = _read_proc_rss_bytes(root)
    if fallback_rss is not None:
        return int(fallback_rss), 1, "proc_root_rss"
    return 0, 0, "unavailable"


# -----------------------------------------------------------------------------
# Function: `_read_proc_cpu_ticks`.
# Purpose: Read cumulative CPU ticks (utime+stime) for one PID from
#          /proc/<pid>/stat.  Returns None on any error.
# -----------------------------------------------------------------------------
_CLK_TCK: int = 0

def _get_clk_tck() -> int:
    global _CLK_TCK
    if _CLK_TCK <= 0:
        try:
            import os as _os
            _CLK_TCK = int(_os.sysconf("SC_CLK_TCK"))
        except Exception:
            _CLK_TCK = 100
    return _CLK_TCK


def _read_proc_cpu_ticks(pid: int) -> Optional[int]:
    """Return utime + stime (clock ticks) for *pid*, or None on failure."""
    try:
        stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except Exception:
        return None
    # Field layout: pid (comm) state ppid ... utime(14) stime(15) ...
    # The comm field may contain spaces, so find the closing ')' first.
    try:
        rparen = stat_text.rfind(")")
        if rparen < 0:
            return None
        fields = stat_text[rparen + 2:].split()
        utime = int(fields[11])   # 14th field (0-indexed from after comm)
        stime = int(fields[12])   # 15th field
        return utime + stime
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Function: `_sample_worker_cpu`.
# Purpose: Walk the process tree from *root_pid*, accumulate CPU ticks for
#          every child (workers, CDO subprocesses, etc.), then convert the
#          tick-delta between two consecutive calls to a CPU-utilisation
#          percentage relative to the total available CPU capacity.
#
# Returns: (active_process_count, worker_cpu_pct, prev_ticks_map)
#   active_process_count — number of processes found in the tree (excl. root)
#   worker_cpu_pct       — [0..N*100] aggregate % across all logical cores
#   prev_ticks_map       — dict[pid -> ticks] to pass back as prev on next call
# -----------------------------------------------------------------------------
def _sample_worker_cpu(
    root_pid: int,
    prev_ticks: Dict[int, int],
    elapsed_sec: float,
) -> Tuple[int, float, Dict[int, int]]:
    """Measure per-worker CPU% by diffing /proc/<pid>/stat ticks."""
    cpu_count = max(1, int(os.cpu_count() or 1))
    clk = _get_clk_tck()

    root = int(root_pid)
    visited: set = set()
    stack: List[int] = [root]
    current_ticks: Dict[int, int] = {}
    child_count = 0

    while stack:
        pid = int(stack.pop())
        if pid <= 0 or pid in visited:
            continue
        visited.add(pid)
        ticks = _read_proc_cpu_ticks(pid)
        if ticks is not None:
            current_ticks[pid] = ticks
            if pid != root:
                child_count += 1
        for child_pid in _read_proc_children(pid):
            if child_pid not in visited:
                stack.append(child_pid)

    # Compute tick delta for all PIDs seen in both snapshots.
    dt = max(elapsed_sec, 0.1)
    total_ticks_delta = 0
    for pid, cur in current_ticks.items():
        prev_val = prev_ticks.get(pid, cur)
        delta = max(0, cur - prev_val)
        total_ticks_delta += delta

    # Convert ticks → seconds → pct of one logical core, then normalise.
    # 100 % means one full core busy; cpu_count*100 % means all cores busy.
    worker_cpu_pct = 100.0 * float(total_ticks_delta) / (float(clk) * dt * float(cpu_count))
    worker_cpu_pct = max(0.0, min(float(cpu_count) * 100.0, worker_cpu_pct))

    return child_count, worker_cpu_pct, current_ticks


# -----------------------------------------------------------------------------
# Function: `_stop_sampler`.
# Purpose: Handle stop sampler workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Function: `initialize_trace`.
# Purpose: Handle initialize trace workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def initialize_trace(
    trace_file: Path,
    total_threads: int,
    case_name: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """Initialize one fresh CPU/memory trace and start one sampler thread."""
    global _LATEST_CPU_PERCENT, _LATEST_RSS_BYTES
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
            "memory_mode": "process_tree_rss",
        },
    )

    _LATEST_CPU_PERCENT = None
    _LATEST_RSS_BYTES = None
    stop_evt = threading.Event()
    root_pid = int(os.getpid())

    # -----------------------------------------------------------------------------
    # Function: `_sampler_loop`.
    # Purpose: Handle sampler loop workflow for the current execution path.
    # Maintenance: Keep input/output schema stable for upstream and downstream callers.
    # -----------------------------------------------------------------------------
    def _sampler_loop(path: Path, started_at: float, stop_event: threading.Event) -> None:
        global _LATEST_CPU_PERCENT, _LATEST_RSS_BYTES
        prev = _read_proc_stat_total_idle()
        # Worker CPU tracking: remember previous tick snapshot and last-sample time.
        prev_worker_ticks: Dict[int, int] = {}
        prev_worker_ts: float = started_at
        while not stop_event.wait(timeout=1.0):
            now = float(time.time())
            elapsed = max(0.0, now - started_at)
            prev, cpu_pct, source = _sample_once(prev)
            _LATEST_CPU_PERCENT = float(cpu_pct)
            rss_bytes, proc_count, mem_source = _sample_process_tree_rss_bytes(root_pid)
            _LATEST_RSS_BYTES = int(rss_bytes)
            # Worker-level CPU: aggregate over all child processes in the tree.
            dt_worker = max(0.1, now - prev_worker_ts)
            worker_child_count, worker_cpu_pct, prev_worker_ticks = _sample_worker_cpu(
                root_pid, prev_worker_ticks, dt_worker
            )
            prev_worker_ts = now
            try:
                _emit_record(
                    path,
                    {
                        "type": "cpu_sample",
                        "ts": now,
                        "pid": int(os.getpid()),
                        "tid": int(threading.get_ident()),
                        "elapsed_seconds": elapsed,
                        "cpu_percent": float(cpu_pct),
                        "source": source,
                    },
                )
                _emit_record(
                    path,
                    {
                        "type": "mem_sample",
                        "ts": now,
                        "pid": int(root_pid),
                        "tid": int(threading.get_ident()),
                        "elapsed_seconds": elapsed,
                        "rss_bytes": int(rss_bytes),
                        "rss_mebibytes": float(rss_bytes) / (1024.0 * 1024.0),
                        "process_count": int(proc_count),
                        "source": mem_source,
                    },
                )
                # Worker CPU record: measures actual computation happening in child processes.
                _emit_record(
                    path,
                    {
                        "type": "worker_cpu_sample",
                        "ts": now,
                        "pid": int(root_pid),
                        "elapsed_seconds": elapsed,
                        "worker_child_count": int(worker_child_count),
                        "worker_cpu_percent": float(worker_cpu_pct),
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


# -----------------------------------------------------------------------------
# Function: `disable_trace`.
# Purpose: Handle disable trace workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def disable_trace() -> None:
    """Disable runtime trace emission and stop sampler in this process."""
    global _LATEST_CPU_PERCENT, _LATEST_RSS_BYTES
    _stop_sampler()
    _LATEST_CPU_PERCENT = None
    _LATEST_RSS_BYTES = None
    for key in (_ENV_TRACE_FILE, _ENV_TOTAL_THREADS, _ENV_START_TS):
        os.environ.pop(key, None)


# -----------------------------------------------------------------------------
# Function: `process_busy_scope`.
# Purpose: Handle process busy scope workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
@contextmanager
def process_busy_scope(
    module: Optional[str] = None,
    hemisphere: Optional[str] = None,
    expected_threads: Optional[int] = None,
):
    """Compatibility no-op context manager (kept for existing call sites)."""
    _ = (module, hemisphere, expected_threads)
    yield


# -----------------------------------------------------------------------------
# Function: `run_tracked_task`.
# Purpose: Handle run tracked task workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def run_tracked_task(func, *args, **kwargs):
    """Compatibility wrapper: execute callable without task-level pseudo tracing."""
    return func(*args, **kwargs)


# -----------------------------------------------------------------------------
# Function: `_load_trace_samples`.
# Purpose: Handle load trace samples workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _load_trace_samples(
    trace_file: Path,
) -> Tuple[Dict[str, Any], List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """Load all trace samples from *trace_file*.

    Returns:
        (meta, cpu_samples, mem_samples, worker_cpu_samples)
        cpu_samples        — list of (elapsed_sec, cpu_pct)
        mem_samples        — list of (elapsed_sec, rss_mib)
        worker_cpu_samples — list of (elapsed_sec, worker_cpu_pct, worker_child_count)
    """
    meta: Dict[str, Any] = {}
    cpu_samples: List[Tuple[float, float]] = []
    mem_samples: List[Tuple[float, float]] = []
    worker_cpu_samples: List[Tuple[float, float, float]] = []
    path = Path(trace_file)
    if not path.exists():
        return meta, cpu_samples, mem_samples, worker_cpu_samples

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
            if typ == "cpu_sample":
                cpu_pct = max(0.0, min(100.0, _safe_float(rec.get("cpu_percent"), 0.0)))
                elapsed = _safe_float(rec.get("elapsed_seconds"), -1.0)
                if elapsed < 0.0:
                    start_ts = _safe_float(meta.get("start_ts"), 0.0)
                    elapsed = max(0.0, _safe_float(rec.get("ts"), start_ts) - start_ts)
                cpu_samples.append((elapsed, cpu_pct))
                continue
            if typ == "worker_cpu_sample":
                elapsed = _safe_float(rec.get("elapsed_seconds"), -1.0)
                if elapsed < 0.0:
                    start_ts = _safe_float(meta.get("start_ts"), 0.0)
                    elapsed = max(0.0, _safe_float(rec.get("ts"), start_ts) - start_ts)
                wcpu = max(0.0, _safe_float(rec.get("worker_cpu_percent"), 0.0))
                wcount = max(0.0, _safe_float(rec.get("worker_child_count"), 0.0))
                worker_cpu_samples.append((elapsed, wcpu, wcount))
                continue
            if typ != "mem_sample":
                continue
            elapsed = _safe_float(rec.get("elapsed_seconds"), -1.0)
            if elapsed < 0.0:
                start_ts = _safe_float(meta.get("start_ts"), 0.0)
                elapsed = max(0.0, _safe_float(rec.get("ts"), start_ts) - start_ts)
            rss_mib = _safe_float(rec.get("rss_mebibytes"), -1.0)
            if rss_mib < 0.0:
                rss_bytes = _safe_float(rec.get("rss_bytes"), -1.0)
                rss_mib = max(0.0, rss_bytes) / (1024.0 * 1024.0) if rss_bytes >= 0.0 else 0.0
            mem_samples.append((elapsed, max(0.0, rss_mib)))

    cpu_samples.sort(key=lambda x: float(x[0]))
    mem_samples.sort(key=lambda x: float(x[0]))
    worker_cpu_samples.sort(key=lambda x: float(x[0]))
    return meta, cpu_samples, mem_samples, worker_cpu_samples


# -----------------------------------------------------------------------------
# Function: `_load_samples`.
# Purpose: Handle load samples workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _load_samples(trace_file: Path) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    meta, cpu_samples, _mem_samples, _worker_cpu_samples = _load_trace_samples(trace_file=trace_file)
    return meta, cpu_samples


# -----------------------------------------------------------------------------
# Function: `latest_cpu_utilization_percent`.
# Purpose: Handle latest cpu utilization percent workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def latest_cpu_utilization_percent(trace_file: Path, fallback_percent: float = 0.0) -> float:
    """Return latest sampled real system CPU utilization percent."""
    if _LATEST_CPU_PERCENT is not None:
        return float(_LATEST_CPU_PERCENT)
    _meta, samples = _load_samples(Path(trace_file))
    if not samples:
        return float(fallback_percent)
    return float(samples[-1][1])


# -----------------------------------------------------------------------------
# Function: `latest_memory_usage_mebibytes`.
# Purpose: Handle latest memory usage mebibytes workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def latest_memory_usage_mebibytes(trace_file: Path, fallback_mebibytes: float = 0.0) -> float:
    """Return latest sampled process-tree RSS memory in MiB."""
    if _LATEST_RSS_BYTES is not None:
        return float(_LATEST_RSS_BYTES) / (1024.0 * 1024.0)
    _meta, _cpu_samples, mem_samples, _worker = _load_trace_samples(Path(trace_file))
    if not mem_samples:
        return float(fallback_mebibytes)
    return float(mem_samples[-1][1])


# -----------------------------------------------------------------------------
# Function: `latest_efficiency_percent`.
# Purpose: Handle latest efficiency percent workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def latest_efficiency_percent(trace_file: Path, fallback_total_threads: int = 1) -> float:
    """Compatibility alias kept for scheduler callers."""
    _ = fallback_total_threads
    return latest_cpu_utilization_percent(trace_file=trace_file, fallback_percent=0.0)


# -----------------------------------------------------------------------------
# Function: `_aggregate_window_max`.
# Purpose: Handle aggregate window max workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def _aggregate_window_max(
    samples: List[Tuple[float, float]],
    window_seconds: float,
) -> List[Tuple[float, float]]:
    safe_window = max(1.0, float(window_seconds))
    if not samples:
        return []

    bucket_max: Dict[int, float] = {}
    for elapsed, value in samples:
        idx = int(max(0.0, float(elapsed)) // safe_window)
        current = bucket_max.get(idx)
        val = float(value)
        if current is None or val > current:
            bucket_max[idx] = val

    last_elapsed = max(0.0, float(samples[-1][0]))
    out: List[Tuple[float, float]] = []
    for idx in sorted(bucket_max.keys()):
        bucket_end = min(last_elapsed, float(idx + 1) * safe_window)
        out.append((bucket_end, float(bucket_max[idx])))
    return out


# -----------------------------------------------------------------------------
# Function: `export_trace_outputs`.
# Purpose: Handle export trace outputs workflow for the current execution path.
# Maintenance: Keep input/output schema stable for upstream and downstream callers.
# -----------------------------------------------------------------------------
def export_trace_outputs(
    trace_file: Path,
    output_png: Path,
    output_csv: Optional[Path] = None,
    chart_title: str = "CPU & Memory Real-time Utilization",
    memory_window_seconds: float = 300.0,
    override_meta: Optional[dict] = None,
) -> Tuple[Path, Optional[Path]]:
    """Export real-time CPU/memory/worker PNG (3 panels) and optional CSV.

    Panel 1: System-wide CPU utilization % (from /proc/stat).
    Panel 2: Process-tree peak RSS memory (GiB).
    Panel 3: Active child-process count and thread utilisation % of requested budget.

    override_meta: optional dict merged into the meta read from the trace file,
                   useful when the trace predates a field (e.g. total_threads).
    """
    meta, cpu_samples, mem_samples, worker_cpu_samples = _load_trace_samples(Path(trace_file))
    if override_meta:
        meta = {**meta, **override_meta}
    if not cpu_samples:
        start_ts = _safe_float(meta.get("start_ts"), 0.0)
        elapsed = max(1.0, float(time.time()) - start_ts) if start_ts > 0.0 else 60.0
        cpu_samples = [(0.0, 0.0), (float(elapsed), 0.0)]
    if not mem_samples:
        end_elapsed = float(cpu_samples[-1][0]) if cpu_samples else 60.0
        mem_samples = [(0.0, 0.0), (max(1.0, end_elapsed), 0.0)]

    png_path = Path(output_png)
    csv_path = Path(output_csv) if output_csv is not None else None
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("elapsed_seconds,cpu_percent,worker_cpu_percent,worker_child_count\n")
            wcpu_map = {row[0]: (row[1], row[2]) for row in worker_cpu_samples}
            for elapsed, pct in cpu_samples:
                wc, wn = wcpu_map.get(elapsed, (0.0, 0.0))
                f.write(f"{elapsed:.6f},{pct:.6f},{wc:.6f},{wn:.0f}\n")

    # ── Panel 1: System CPU ───────────────────────────────────────────────────
    cpu_xs = [float(row[0]) for row in cpu_samples]
    cpu_ys = [float(row[1]) for row in cpu_samples]
    if len(cpu_xs) == 1:
        cpu_xs.append(cpu_xs[0] + 1.0)
        cpu_ys.append(cpu_ys[0])
    cpu_xs_min = [x / 60.0 for x in cpu_xs]
    peak_cpu = max(cpu_ys) if cpu_ys else 0.0
    mean_cpu = sum(cpu_ys) / float(len(cpu_ys)) if cpu_ys else 0.0

    # ── Panel 2: Memory ───────────────────────────────────────────────────────
    mem_window_seconds = max(60.0, float(memory_window_seconds))
    mem_win = _aggregate_window_max(mem_samples, window_seconds=mem_window_seconds)
    if not mem_win:
        mem_win = [(0.0, 0.0)]
    mem_xs_min = [float(t) / 60.0 for t, _ in mem_win]
    mem_ys_gib = [float(v) / 1024.0 for _, v in mem_win]
    if len(mem_xs_min) == 1:
        mem_xs_min.append(mem_xs_min[0] + 1.0)
        mem_ys_gib.append(mem_ys_gib[0])
    mem_mean_gib = sum(mem_ys_gib) / float(len(mem_ys_gib)) if mem_ys_gib else 0.0
    mem_peak_gib = max(mem_ys_gib) if mem_ys_gib else 0.0
    window_minutes = mem_window_seconds / 60.0

    # ── Panel 3: Worker processes ─────────────────────────────────────────────
    have_worker = bool(worker_cpu_samples)
    if have_worker:
        wcpu_xs_min = [float(row[0]) / 60.0 for row in worker_cpu_samples]
        wcpu_ys     = [float(row[1]) for row in worker_cpu_samples]
        wcnt_ys     = [float(row[2]) for row in worker_cpu_samples]
        if len(wcpu_xs_min) == 1:
            wcpu_xs_min.append(wcpu_xs_min[0] + 1.0)
            wcpu_ys.append(wcpu_ys[0])
            wcnt_ys.append(wcnt_ys[0])
        peak_wcpu = max(wcpu_ys) if wcpu_ys else 0.0
        mean_wcpu = sum(wcpu_ys) / float(len(wcpu_ys)) if wcpu_ys else 0.0
        peak_cnt  = max(wcnt_ys) if wcnt_ys else 0.0
    else:
        wcpu_xs_min = wcpu_ys = wcnt_ys = []
        peak_wcpu = mean_wcpu = peak_cnt = 0.0

    cpu_count = max(1, int(os.cpu_count() or 1))
    requested_threads = max(1, _safe_int(meta.get("total_threads"), 1))
    title_with_meta = f"{chart_title} | Threads: {requested_threads} | CPU cores: {cpu_count}"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
    ax_cpu, ax_mem, ax_wkr = axes
    fig.suptitle(title_with_meta)

    # Panel 1 ─────────────────────────────────────────────────────────────────
    ax_cpu.fill_between(cpu_xs_min, cpu_ys, alpha=0.18, color="#1f77b4")
    ax_cpu.plot(cpu_xs_min, cpu_ys, linewidth=1.6, color="#1f77b4")
    ax_cpu.plot(cpu_xs_min, cpu_ys, linestyle="none", marker="o", markersize=2.0, color="#1f77b4")
    ax_cpu.set_ylabel("System CPU (%)")
    ax_cpu.set_title(f"System-wide CPU utilization  (mean: {mean_cpu:.1f}%  peak: {peak_cpu:.1f}%)")
    ax_cpu.set_ylim(0.0, 105.0)
    ax_cpu.axhline(100.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_cpu.grid(True, linestyle="--", alpha=0.30)

    # Panel 2 ─────────────────────────────────────────────────────────────────
    ax_mem.fill_between(mem_xs_min, mem_ys_gib, alpha=0.18, color="#ff7f0e")
    ax_mem.plot(mem_xs_min, mem_ys_gib, linewidth=1.6, color="#ff7f0e")
    ax_mem.plot(mem_xs_min, mem_ys_gib, linestyle="none", marker="o", markersize=2.4, color="#ff7f0e")
    ax_mem.set_ylabel(f"Peak RSS / {window_minutes:.0f} min (GiB)")
    ax_mem.set_title(
        f"Process-tree RSS  (bucket: {window_minutes:.0f} min  mean: {mem_mean_gib:.2f} GiB  "
        f"peak: {mem_peak_gib:.2f} GiB)"
    )
    ax_mem.grid(True, linestyle="--", alpha=0.30)

    # Panel 3: active child-process count (left) and utilisation % of requested threads (right)
    if have_worker:
        # Right axis: active processes as a percentage of the requested thread budget
        wcnt_pct_ys = [v / float(requested_threads) * 100.0 for v in wcnt_ys]
        peak_pct = max(wcnt_pct_ys) if wcnt_pct_ys else 0.0
        mean_pct = sum(wcnt_pct_ys) / float(len(wcnt_pct_ys)) if wcnt_pct_ys else 0.0

        ax_wkr2 = ax_wkr.twinx()

        # Left axis: raw active child-process count, capped at requested_threads
        ax_wkr.fill_between(wcpu_xs_min, wcnt_ys, alpha=0.18, color="#9467bd", step="mid")
        ax_wkr.step(wcpu_xs_min, wcnt_ys, where="mid", linewidth=1.6, color="#9467bd",
                    label=f"Active child processes (peak {peak_cnt:.0f})")
        ax_wkr.set_ylabel("Active child processes", color="#9467bd")
        ax_wkr.tick_params(axis="y", labelcolor="#9467bd")
        ax_wkr.set_ylim(0.0, requested_threads * 1.05)
        # Reference line at the requested thread budget
        ax_wkr.axhline(float(requested_threads), color="#9467bd", linestyle=":",
                       linewidth=0.9, alpha=0.6, label=f"Requested threads ({requested_threads})")

        # Right axis: utilisation percentage, always 0-100 %
        ax_wkr2.step(wcpu_xs_min, wcnt_pct_ys, where="mid", linewidth=1.2,
                     color="#d62728", linestyle="--", alpha=0.75,
                     label=f"Thread utilisation % (mean {mean_pct:.1f}%  peak {peak_pct:.1f}%)")
        ax_wkr2.set_ylabel("Thread utilisation (% of requested)", color="#d62728")
        ax_wkr2.tick_params(axis="y", labelcolor="#d62728")
        ax_wkr2.set_ylim(0.0, 105.0)
        ax_wkr2.axhline(100.0, color="#d62728", linestyle=":", linewidth=0.9, alpha=0.45)

        lines1, labels1 = ax_wkr.get_legend_handles_labels()
        lines2, labels2 = ax_wkr2.get_legend_handles_labels()
        ax_wkr.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper right")
    else:
        ax_wkr.text(0.5, 0.5, "No worker_cpu_sample records\n(run with newer runtime_efficiency)",
                    ha="center", va="center", transform=ax_wkr.transAxes, fontsize=10, color="gray")
    ax_wkr.set_xlabel("Runtime (minutes)")
    ax_wkr.set_title(
        f"Active child processes and thread utilisation  "
        f"(requested: {requested_threads}  peak: {peak_cnt:.0f})"
    )
    ax_wkr.grid(True, linestyle="--", alpha=0.30)

    max_runtime_minutes = max(
        1.0,
        cpu_xs_min[-1] if cpu_xs_min else 1.0,
        mem_xs_min[-1] if mem_xs_min else 1.0,
        wcpu_xs_min[-1] if wcpu_xs_min else 1.0,
    )
    ax_wkr.set_xlim(0.0, max_runtime_minutes)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(str(png_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return png_path, csv_path
