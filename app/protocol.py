"""Machine-readable progress events between the DeepDream CLI and the GUI.

The GUI runs each job as `python -m app.deepdream` and reads its stdout. Human
log lines are also printed, but the GUI used to parse them as English text, so
reworded output silently broke progress, previews and output discovery.

Structured events travel on the same stream behind a control prefix, one JSON
object per line. Anything the GUI cannot parse is still shown as a log line, and
the GUI keeps its text parsing as a fallback, so old and new CLI versions work
together in either direction.
"""

from __future__ import annotations

import json
from typing import Any

# Control character: cannot occur in ordinary log output, so a user-facing
# message can never be mistaken for an event.
EVENT_PREFIX = "\x00dd-event:"

# Event names.
EV_OCTAVE = "octave"
EV_ITERATION = "iteration"
EV_FRAME = "frame"
EV_PREVIEW = "preview"
EV_DONE = "done"
EV_STATUS = "status"


def format_event(event: str, **fields: Any) -> str:
    """Serialize one event as a prefixed single line, ready to print."""
    payload = {"event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    return EVENT_PREFIX + json.dumps(payload, separators=(",", ":"), sort_keys=True)


def parse_event(line: str) -> dict[str, Any] | None:
    """Parse a line into an event dict, or None if it is ordinary output."""
    if not line or not line.startswith(EVENT_PREFIX):
        return None
    try:
        data = json.loads(line[len(EVENT_PREFIX) :])
    except ValueError:
        return None
    if not isinstance(data, dict) or not data.get("event"):
        return None
    return data
