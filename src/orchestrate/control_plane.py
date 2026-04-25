from __future__ import annotations

from dataclasses import dataclass
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse

from src.common.io import write_json_atomic


@dataclass(slots=True)
class ControlPlaneConfig:
    enabled: bool = True
    bind_host: str = "0.0.0.0"
    advertise_host: str | None = None
    port: int = 0
    heartbeat_seconds: float = 10.0
    node_ttl_seconds: float = 45.0


class PipelineControlPlane:
    def __init__(
        self,
        lock_dir: Path,
        *,
        workspace_root: Path,
        config: ControlPlaneConfig,
    ) -> None:
        self.lock_dir = lock_dir
        self.workspace_root = workspace_root
        self.config = config
        self.hostname = socket.gethostname()
        self.pid = 0
        self.node_id = ""
        self.endpoint = ""
        self._started_at = time.time()
        self._lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._node_path: Path | None = None
        self._node_updated_at = self._started_at
        self._last_node_persisted_at = 0.0
        self._last_persisted_payload: dict[str, Any] | None = None
        self._run_state: dict[str, Any] = {
            "status": "idle",
            "detail": "",
            "updated_at": self._started_at,
            "started_at": self._started_at,
        }
        self._capacity_state: dict[str, Any] = {
            "document_workers": 0,
            "gpu_slots": 0,
            "queued": 0,
            "local_active": 0,
            "remote_active": 0,
            "leased_total": 0,
            "completed": 0,
            "extracted": 0,
            "unresolved": 0,
            "updated_at": self._started_at,
        }
        self._stop_requested = False
        self._cancel_requests: set[str] = set()
        self._cancel_callback: Callable[[str], bool] | None = None
        self._force_callback: Callable[[str], dict[str, Any]] | None = None
        self._trigger_callback: Callable[[str], dict[str, Any]] | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        if not self.config.enabled:
            return
        nodes_dir = self.lock_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        self.pid = self._resolve_pid()
        advertise_host = self.config.advertise_host or self.hostname
        server = self._build_server()
        self._server = server
        actual_port = int(server.server_address[1])
        self.node_id = f"{self.hostname}-{self.pid}-{actual_port}"
        self.endpoint = f"http://{advertise_host}:{actual_port}"
        self._node_path = nodes_dir / f"{self.node_id}.node.json"
        self._server_thread = threading.Thread(target=server.serve_forever, name=f"control-plane-{self.node_id}", daemon=True)
        self._server_thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name=f"control-heartbeat-{self.node_id}", daemon=True)
        self._heartbeat_thread.start()
        self._persist_node(force=True)

    def stop(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self.config.heartbeat_seconds + 1.0)
        node_path = self._node_path
        if node_path is not None:
            try:
                node_path.unlink()
            except FileNotFoundError:
                pass

    def set_cancel_callback(self, callback: Callable[[str], bool]) -> None:
        with self._lock:
            self._cancel_callback = callback

    def set_force_callback(self, callback: Callable[[str], dict[str, Any]]) -> None:
        with self._lock:
            self._force_callback = callback

    def set_trigger_callback(self, callback: Callable[[str], dict[str, Any]]) -> None:
        with self._lock:
            self._trigger_callback = callback

    def request_stop(self, reason: str = "remote drain requested") -> None:
        changed = False
        with self._lock:
            changed = not self._stop_requested or self._run_state.get("detail") != reason
            if changed:
                self._stop_requested = True
                self._run_state["detail"] = reason
                self._run_state["updated_at"] = time.time()
                self._node_updated_at = self._run_state["updated_at"]
        if changed:
            self._persist_node(force=True)

    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def consume_cancel_requests(self) -> set[str]:
        changed = False
        with self._lock:
            pending = set(self._cancel_requests)
            if pending:
                self._cancel_requests.clear()
                self._node_updated_at = time.time()
                changed = True
        if changed:
            self._persist_node(force=True)
        return pending

    def queue_cancel(self, source_id: str) -> bool:
        normalized = Path(source_id.strip()).stem.replace(" ", "_").lower()
        if not normalized:
            return False
        handled = False
        changed = False
        with self._lock:
            if normalized not in self._cancel_requests:
                self._cancel_requests.add(normalized)
                self._node_updated_at = time.time()
                changed = True
            callback = self._cancel_callback
        if callback is not None:
            try:
                handled = bool(callback(normalized))
            except Exception:
                handled = False
        if changed:
            self._persist_node(force=True)
        return handled

    def force_release(self, source_id: str) -> dict[str, Any]:
        normalized = Path(source_id.strip()).stem.replace(" ", "_").lower()
        if not normalized:
            return {"ok": False, "source_id": normalized, "detail": "empty source id"}
        with self._lock:
            callback = self._force_callback
        if callback is None:
            return {"ok": False, "source_id": normalized, "detail": "force release is not available on this node"}
        result = callback(normalized)
        with self._lock:
            self._node_updated_at = time.time()
        self._persist_node(force=True)
        return result

    def trigger_source(self, source_id: str) -> dict[str, Any]:
        normalized = Path(source_id.strip()).stem.replace(" ", "_").lower()
        if not normalized:
            return {"ok": False, "source_id": normalized, "detail": "empty source id"}
        with self._lock:
            callback = self._trigger_callback
        if callback is None:
            return {"ok": False, "source_id": normalized, "detail": "trigger is not available on this node"}
        result = callback(normalized)
        with self._lock:
            self._node_updated_at = time.time()
        self._persist_node(force=True)
        return result

    def update_run_state(self, status: str, detail: str = "", **extra: Any) -> None:
        changed = False
        with self._lock:
            next_state = dict(self._run_state)
            next_state.update({"status": status, "detail": detail})
            next_state.update(extra)
            comparable_current = {key: value for key, value in self._run_state.items() if key != "updated_at"}
            comparable_next = {key: value for key, value in next_state.items() if key != "updated_at"}
            changed = comparable_current != comparable_next
            if changed:
                next_state["updated_at"] = time.time()
                self._run_state = next_state
                self._node_updated_at = next_state["updated_at"]
        if changed:
            self._persist_node(force=True)

    def update_capacity(self, **payload: Any) -> None:
        changed = False
        with self._lock:
            next_state = dict(self._capacity_state)
            next_state.update(payload)
            comparable_current = {key: value for key, value in self._capacity_state.items() if key != "updated_at"}
            comparable_next = {key: value for key, value in next_state.items() if key != "updated_at"}
            changed = comparable_current != comparable_next
            if changed:
                next_state["updated_at"] = time.time()
                self._capacity_state = next_state
                self._node_updated_at = next_state["updated_at"]
        if changed:
            self._persist_node()

    def node_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._node_payload_locked()

    def list_registered_nodes(self) -> list[dict[str, Any]]:
        nodes_dir = self.lock_dir / "nodes"
        if not nodes_dir.exists():
            return []
        now = time.time()
        entries: list[dict[str, Any]] = []
        for path in sorted(nodes_dir.glob("*.node.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            updated_at = payload.get("updated_at")
            ttl_seconds = payload.get("ttl_seconds")
            healthy = isinstance(updated_at, (int, float)) and isinstance(ttl_seconds, (int, float)) and (now - float(updated_at)) <= float(ttl_seconds)
            payload["healthy"] = healthy
            entries.append(payload)
        return entries

    def local_source_snapshot(self, source_id: str) -> dict[str, Any]:
        normalized = source_id.strip().replace(" ", "_").lower()
        status_path = self.lock_dir / "source_status" / f"{normalized}.status.json"
        lease_path = self.lock_dir / "source_active" / f"{normalized}.lease.json"
        status_payload = self._read_json(status_path)
        lease_payload = self._read_json(lease_path)
        return {
            "source_id": normalized,
            "status": status_payload,
            "lease": lease_payload,
            "observed_at": time.time(),
        }

    def local_leases_snapshot(self) -> dict[str, Any]:
        active_dir = self.lock_dir / "source_active"
        leases: list[dict[str, Any]] = []
        if active_dir.exists():
            for path in sorted(active_dir.glob("*.lease.json")):
                payload = self._read_json(path)
                if payload is None:
                    continue
                leases.append(payload)
        return {
            "hostname": self.hostname,
            "pid": self.pid,
            "leases": leases,
            "observed_at": time.time(),
        }

    def _resolve_pid(self) -> int:
        import os

        return os.getpid()

    def _build_server(self) -> ThreadingHTTPServer:
        control_plane = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                segments = [segment for segment in parsed.path.split("/") if segment]
                if parsed.path == "/healthz":
                    self._send_json({"ok": True, "node_id": control_plane.node_id or None})
                    return
                if parsed.path == "/v1/node":
                    self._send_json(control_plane.node_snapshot())
                    return
                if parsed.path == "/v1/capacity":
                    self._send_json(control_plane.node_snapshot()["capacity"])
                    return
                if parsed.path == "/v1/status":
                    snapshot = control_plane.node_snapshot()
                    self._send_json({"node": snapshot, "peers": control_plane.list_registered_nodes()})
                    return
                if parsed.path == "/v1/leases":
                    self._send_json(control_plane.local_leases_snapshot())
                    return
                if len(segments) == 3 and segments[:2] == ["v1", "sources"]:
                    self._send_json(control_plane.local_source_snapshot(segments[2]))
                    return
                if parsed.path == "/v1/peers":
                    self._send_json({"nodes": control_plane.list_registered_nodes(), "observed_at": time.time()})
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                segments = [segment for segment in parsed.path.split("/") if segment]
                if parsed.path == "/v1/control/drain":
                    control_plane.request_stop()
                    self._send_json({"ok": True, "stop_requested": True})
                    return
                if len(segments) == 4 and segments[:3] == ["v1", "control", "cancel"]:
                    handled = control_plane.queue_cancel(segments[3])
                    self._send_json({"ok": True, "source_id": segments[3], "handled_immediately": handled})
                    return
                if len(segments) == 4 and segments[:3] == ["v1", "control", "force"]:
                    self._send_json(control_plane.force_release(segments[3]))
                    return
                if len(segments) == 4 and segments[:3] == ["v1", "control", "trigger"]:
                    self._send_json(control_plane.trigger_source(segments[3]))
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return ThreadingHTTPServer((self.config.bind_host, self.config.port), Handler)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.config.heartbeat_seconds):
            with self._lock:
                self._node_updated_at = time.time()
            self._persist_node(force=True)

    def _persist_node(self, *, force: bool = False) -> None:
        node_path = self._node_path
        if not self.config.enabled or node_path is None:
            return
        with self._lock:
            payload = self._node_payload_locked()
            now = time.time()
            if not force and (now - self._last_node_persisted_at) < self.config.heartbeat_seconds:
                return
            if not force and payload == self._last_persisted_payload:
                return
            self._last_persisted_payload = json.loads(json.dumps(payload, sort_keys=True))
            self._last_node_persisted_at = now
        write_json_atomic(node_path, payload)

    def _node_payload_locked(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "pid": self.pid,
            "endpoint": self.endpoint,
            "workspace_root": str(self.workspace_root),
            "started_at": self._started_at,
            "updated_at": self._node_updated_at,
            "ttl_seconds": self.config.node_ttl_seconds,
            "run_state": dict(self._run_state),
            "capacity": dict(self._capacity_state),
            "control": {
                "stop_requested": self._stop_requested,
                "queued_cancels": sorted(self._cancel_requests),
            },
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
