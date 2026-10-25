"""Serialize complete NetCDF dataset lifetimes within each Python process."""

from contextlib import contextmanager
import os
import threading

import xarray as xr


NETCDF_IO_LOCK = threading.RLock()


def _reset_after_fork():
    global NETCDF_IO_LOCK
    NETCDF_IO_LOCK = threading.RLock()


def _before_fork():
    NETCDF_IO_LOCK.acquire()


def _after_fork_parent():
    NETCDF_IO_LOCK.release()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_parent,
                        after_in_child=_reset_after_fork)


@contextmanager
def open_netcdf_dataset(*args, **kwargs):
    """Protect open, lazy reads, metadata, and close with one reentrant lock."""
    with NETCDF_IO_LOCK:
        with xr.open_dataset(*args, **kwargs) as dataset:
            yield dataset


def write_netcdf_dataset(dataset, *args, **kwargs):
    """Protect writes with the same lock used for dataset reads and closes."""
    with NETCDF_IO_LOCK:
        return dataset.to_netcdf(*args, **kwargs)
