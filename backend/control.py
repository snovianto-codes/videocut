"""Pause / resume / cancel for background jobs, reaching into every subprocess a job starts."""
from __future__ import annotations

import contextvars
import functools
import os
import signal
import subprocess
import threading


class Cancelled(Exception):
    pass


class JobControl:
    def __init__(self):
        self.cancelled = threading.Event()
        self.running = threading.Event()  # cleared while paused
        self.running.set()
        self.procs: set[subprocess.Popen] = set()
        self.lock = threading.Lock()

    @property
    def paused(self) -> bool:
        return not self.running.is_set()

    def checkpoint(self):
        """Call between steps: blocks while paused, raises if cancelled."""
        while not self.running.wait(0.25):
            if self.cancelled.is_set():
                break
        if self.cancelled.is_set():
            raise Cancelled()

    def _signal(self, sig):
        with self.lock:
            for p in list(self.procs):
                if p.poll() is None:
                    try:
                        os.kill(p.pid, sig)
                    except ProcessLookupError:
                        pass

    def pause(self):
        self.running.clear()
        self._signal(signal.SIGSTOP)

    def resume(self):
        self._signal(signal.SIGCONT)
        self.running.set()

    def cancel(self):
        self.cancelled.set()
        self._signal(signal.SIGKILL)  # works on stopped processes too
        self.running.set()


CURRENT: contextvars.ContextVar[JobControl | None] = contextvars.ContextVar("job_control", default=None)


def current() -> JobControl | None:
    return CURRENT.get()


def checkpoint():
    if (ctl := CURRENT.get()) is not None:
        ctl.checkpoint()


def bind(fn):
    """Carry the caller's job control into worker threads (ThreadPoolExecutor doesn't copy context)."""
    ctl = CURRENT.get()

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        token = CURRENT.set(ctl)
        try:
            return fn(*a, **kw)
        finally:
            CURRENT.reset(token)
    return wrapper


def run(cmd: list[str], text: bool = False, check: bool = False) -> subprocess.CompletedProcess:
    """subprocess.run(capture_output=True) that the current job can pause and kill."""
    ctl = CURRENT.get()
    if ctl is None:
        return subprocess.run(cmd, capture_output=True, text=text, check=check)
    ctl.checkpoint()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
    with ctl.lock:
        ctl.procs.add(p)
    if ctl.paused:
        os.kill(p.pid, signal.SIGSTOP)
    try:
        out, err = p.communicate()
    finally:
        with ctl.lock:
            ctl.procs.discard(p)
    if ctl.cancelled.is_set():
        raise Cancelled()
    if check and p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)
