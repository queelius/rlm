"""Canonical opt-in debug event recording."""

from __future__ import annotations

import copy
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rlm.config import DebugConfig


class TraceRecorder:
    def __init__(self, config: DebugConfig, *, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self._lock = threading.Lock()
        self._next_id = 0
        self._closed = False
        self.events: list[dict[str, Any]] = []
        self.directory: Path | None = None
        self._file = None
        if config.enabled:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.directory = Path(config.directory).expanduser().resolve() / f"{stamp}-{run_id}"
            self.directory.mkdir(parents=True, exist_ok=False)
            self._file = (self.directory / "trace.jsonl").open("a", encoding="utf-8")

    def event(
        self,
        event_type: str,
        payload: Any,
        *,
        branch_id: str,
        depth: int,
        parent_event_id: str | None = None,
    ) -> str:
        with self._lock:
            event_id = f"evt-{self._next_id:06d}"
            self._next_id += 1
            if self._closed:
                return event_id
            event = {
                "run_id": self.run_id,
                "event_id": event_id,
                "parent_event_id": parent_event_id,
                "branch_id": branch_id,
                "depth": depth,
                "timestamp": time.time(),
                "type": event_type,
                "payload": copy.deepcopy(payload),
            }
            if self.config.enabled:
                self.events.append(event)
                assert self._file is not None
                encoded = json.dumps(event, ensure_ascii=False, default=_json_default)
                self._file.write(encoded + "\n")
                self._file.flush()
            return event_id

    def write_json(self, name: str, value: Any) -> None:
        if self.directory is None:
            return
        path = self.directory / name
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )

    def close(self, *, manifest: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.directory is None:
                return
            self.write_json("manifest.json", manifest)
            if self.config.markdown:
                (self.directory / "trace.md").write_text(self._markdown(), encoding="utf-8")
            if self._file is not None and not self._file.closed:
                self._file.close()

    def _markdown(self) -> str:
        lines = [f"# RLM trace `{self.run_id}`", ""]
        for event in self.events:
            lines.extend(
                [
                    f"## {event['event_id']} · {event['type']}",
                    "",
                    f"Branch `{event['branch_id']}`, depth {event['depth']}",
                    "",
                    "````json",
                    json.dumps(
                        event["payload"],
                        ensure_ascii=False,
                        indent=2,
                        default=_json_default,
                    ),
                    "````",
                    "",
                ]
            )
        return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        return value.to_dict()
    except AttributeError:
        return repr(value)
