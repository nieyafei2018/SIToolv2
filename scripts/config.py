# -*- coding: utf-8 -*-
"""
Global constants and logging configuration for SIToolv2.
"""
import datetime
import logging
import os

# --- Calendar constants ---
# Days per month (non-leap year, Jan–Dec)
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Days per season: SH (DJF/summer, MAM/autumn, JJA/winter, SON/spring)
#                  NH (DJF/winter, MAM/spring, JJA/summer, SON/autumn)
DAYS_PER_SEASON = [90, 92, 92, 91]

# --- Plotting constants ---
# Line style cycle for model datasets
LINE_STYLES = ['--', '-.', ':']

# Color cycle for model datasets
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Output figure resolution (dots per inch)
DPI = 200

# 12-month abbreviations for x-axis ticks
MONTH_TICKS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']


def setup_logging(level: int = logging.INFO, log_file: str = None) -> None:
    """Configure root logger to write to console and optionally a log file.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path for the log file. The parent directory is
            created automatically if it does not exist.
    """
    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers when called more than once.
    # Explicitly close removed handlers to release file descriptors.
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
    root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Rotate the previous log file to avoid cross-run contamination when
        # a stale subprocess still holds an old file descriptor.
        if os.path.exists(log_file):
            stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            archived = f"{log_file}.{stamp}.prev"
            try:
                os.replace(log_file, archived)
            except Exception:
                # Best effort only; continue with in-place overwrite.
                pass
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
