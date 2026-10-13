"""CLI subprocess transport for the GUI, independent of Tk widgets and state.

Start processes on the UI thread, read them on workers, and apply queued
messages on the UI thread. Generation completion carries process identity so
a delayed completion cannot finish a newer job.
"""

from __future__ import annotations

import queue
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .gui_helpers import PROC_DONE_PREFIX


def start_cli_process(cmd: list[str], cwd: str | Path) -> subprocess.Popen[str]:
    """Start a UTF-8 CLI stream in its own process group on POSIX."""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="backslashreplace",
        cwd=str(cwd),
        creationflags=(
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        ),
        start_new_session=(sys.platform != "win32"),
    )


def read_generation_output(
    proc: subprocess.Popen[str],
    job_id: int,
    messages: queue.Queue[str],
    on_line: Callable[[str], None] | None = None,
) -> None:
    """Stream output, optionally observe each line, and publish completion.

    ``on_line`` runs on the reader thread before the line is added to the UI
    queue, so callers can capture process-local state even if UI polling stops.
    """
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                if on_line is not None:
                    on_line(line)
                messages.put(line)
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass
        try:
            proc.wait()
        finally:
            messages.put(f"{PROC_DONE_PREFIX}{id(proc)}:{job_id}")


def read_encode_output(
    proc: subprocess.Popen[str],
    out_path: str,
    job_id: int | None,
    messages: queue.Queue[tuple],
    on_line: Callable[[str], None] | None = None,
) -> None:
    """Stream export output, optionally observe lines, and publish completion.

    ``on_line`` runs on the reader thread before the line is added to the UI
    queue, so callers can capture process-local state even if UI polling stops.
    """
    rc = -1
    error = ""
    try:
        if proc.stdout is not None:
            for line in iter(proc.stdout.readline, ""):
                if on_line is not None:
                    on_line(line)
                messages.put(("line", line, out_path))
        proc.wait()
        rc = proc.returncode
    except OSError as exc:
        error = f"Error: could not start video export: {exc}\n"
    except Exception as exc:
        error = f"Error during video export: {exc}\n"
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass
        messages.put(("done", rc, error, out_path, job_id))
