from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import uuid

from src.common.io import write_json_atomic


@dataclass(slots=True)
class LeaseDenied:
    reason: str
    status: str = "skipped"


class ManagedLease:
    def __init__(
        self,
        *,
        status_path: Path,
        lease_path: Path,
        lease_key: str,
        owner: dict[str, object],
        heartbeat_seconds: float,
        min_update_interval_seconds: float,
        metadata: dict[str, object],
    ) -> None:
        self.status_path = status_path
        self.lease_path = lease_path
        self.lease_key = lease_key
        self.owner = dict(owner)
        self.heartbeat_seconds = max(1.0, heartbeat_seconds)
        self.min_update_interval_seconds = max(1.0, min_update_interval_seconds)
        self.metadata = dict(metadata)
        self._status = "running"
        self._stage = "claimed"
        self._detail = "processing claim acquired"
        self._started_at = time.time()
        self._updated_at = self._started_at
        self._finished_at: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._last_persisted_at = 0.0

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._snapshot_locked()

    def start(self) -> None:
        self._persist(force=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name=f"lease-heartbeat-{self.lease_key}", daemon=True)
        self._heartbeat_thread.start()

    def update(
        self,
        stage: str,
        detail: str = "",
        *,
        status: str | None = None,
        extra: dict[str, object] | None = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            changed = False
            if status is not None:
                changed = changed or self._status != status
                self._status = status
            changed = changed or self._stage != stage or self._detail != detail
            self._stage = stage
            self._detail = detail
            self._updated_at = time.time()
            if extra:
                self.metadata.update(extra)
                changed = True
            if force or self._should_persist_locked(changed):
                self._persist_locked(force=True)

    def complete(self, detail: str = "", *, extra: dict[str, object] | None = None) -> None:
        with self._lock:
            self._status = "completed"
            self._stage = "completed"
            self._detail = detail
            self._updated_at = time.time()
            self._finished_at = self._updated_at
            if extra:
                self.metadata.update(extra)
            self._persist_locked(force=True)
        self._shutdown(remove_lease=True)

    def fail(self, detail: str, *, extra: dict[str, object] | None = None) -> None:
        with self._lock:
            self._status = "failed"
            self._stage = "failed"
            self._detail = detail
            self._updated_at = time.time()
            self._finished_at = self._updated_at
            if extra:
                self.metadata.update(extra)
            self._persist_locked(force=True)
        self._shutdown(remove_lease=True)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_seconds):
            with self._lock:
                if self._finished_at is not None:
                    return
                self._updated_at = time.time()
                self._persist_locked(force=True)

    def _shutdown(self, *, remove_lease: bool) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=self.heartbeat_seconds + 0.5)
        if remove_lease:
            try:
                self.lease_path.unlink()
            except FileNotFoundError:
                pass

    def _persist(self, *, force: bool = False) -> None:
        with self._lock:
            self._persist_locked(force=force)

    def _persist_locked(self, *, force: bool = False) -> None:
        if not force and not self._should_persist_locked(True):
            return
        snapshot = self._snapshot_locked()
        write_json_atomic(self.status_path, snapshot)
        if self._finished_at is None:
            write_json_atomic(self.lease_path, snapshot)
        self._last_persisted_at = self._updated_at

    def _should_persist_locked(self, changed: bool) -> bool:
        if not changed:
            return False
        return (self._updated_at - self._last_persisted_at) >= self.min_update_interval_seconds

    def _snapshot_locked(self) -> dict[str, object]:
        payload = {
            "lease_key": self.lease_key,
            "status": self._status,
            "stage": self._stage,
            "detail": self._detail,
            "owner": dict(self.owner),
            "started_at": self._started_at,
            "updated_at": self._updated_at,
            "heartbeat_at": self._updated_at,
            "finished_at": self._finished_at,
        }
        payload.update(self.metadata)
        return payload


class PipelineLockManager:
    def __init__(
        self,
        root_dir: Path,
        *,
        heartbeat_seconds: float = 8.0,
        stale_after_seconds: float = 45.0,
        min_update_interval_seconds: float = 4.0,
    ) -> None:
        self.root_dir = root_dir
        self.source_status_dir = root_dir / "source_status"
        self.source_active_dir = root_dir / "source_active"
        self.merge_status_dir = root_dir / "merge_status"
        self.merge_active_dir = root_dir / "merge_active"
        self.heartbeat_seconds = heartbeat_seconds
        self.stale_after_seconds = stale_after_seconds
        self.min_update_interval_seconds = min_update_interval_seconds
        self.owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "run_id": uuid.uuid4().hex,
        }

    def ensure_dirs(self) -> None:
        for path in (self.source_status_dir, self.source_active_dir, self.merge_status_dir, self.merge_active_dir):
            path.mkdir(parents=True, exist_ok=True)

    def note_discovered(self, source_id: str, source_name: str, source_path: Path) -> None:
        status_path = self._source_status_path(source_id)
        lease_path = self._source_lease_path(source_id)
        if lease_path.exists() and not self._lease_is_stale(self._read_json(lease_path)):
            return
        existing = self._read_json(status_path)
        if existing and str(existing.get("status") or "") in {"running", "completed", "skipped", "unsupported"}:
            return
        write_json_atomic(
            status_path,
            {
                "source_id": source_id,
                "source_name": source_name,
                "source_path": str(source_path),
                "status": "queued",
                "stage": "queued",
                "detail": "awaiting dispatch",
                "updated_at": time.time(),
            },
        )

    def mark_source_state(
        self,
        source_id: str,
        source_name: str,
        source_path: Path,
        *,
        status: str,
        stage: str,
        detail: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "source_id": source_id,
            "source_name": source_name,
            "source_path": str(source_path),
            "status": status,
            "stage": stage,
            "detail": detail,
            "updated_at": time.time(),
        }
        if extra:
            payload.update(extra)
        write_json_atomic(self._source_status_path(source_id), payload)

    def list_source_statuses(self) -> dict[str, dict[str, object]]:
        statuses: dict[str, dict[str, object]] = {}
        if not self.source_status_dir.exists():
            return statuses
        for path in sorted(self.source_status_dir.glob("*.status.json")):
            payload = self._read_json(path)
            if not payload:
                continue
            source_id = payload.get("source_id")
            if isinstance(source_id, str) and source_id:
                statuses[source_id] = payload
        return statuses

    def claim_source(
        self,
        source_id: str,
        source_name: str,
        source_path: Path,
        *,
        summary_path: Path,
        csv_path: Path,
        per_source_mode: str,
    ) -> ManagedLease | LeaseDenied:
        status_path = self._source_status_path(source_id)
        lease_path = self._source_lease_path(source_id)
        existing_status = self._read_json(status_path)
        if per_source_mode == "skip" and existing_status and str(existing_status.get("status") or "") == "completed":
            return LeaseDenied("completed status already recorded")
        if per_source_mode == "skip" and (summary_path.exists() or csv_path.exists()):
            self.mark_source_state(
                source_id,
                source_name,
                source_path,
                status="completed",
                stage="existing_outputs",
                detail="existing per-source outputs found",
                extra={"summary_path": str(summary_path), "csv_path": str(csv_path)},
            )
            return LeaseDenied("existing per-source outputs found")
        for _ in range(3):
            existing_lease = self._read_json(lease_path)
            if existing_lease:
                if not self._lease_is_stale(existing_lease):
                    owner = existing_lease.get("owner", {})
                    return LeaseDenied(f"active lease held by {owner.get('host', 'unknown-host')}:{owner.get('pid', 'unknown-pid')}", status="running")
                self._remove_stale_lease(lease_path)
                self.mark_source_state(
                    source_id,
                    source_name,
                    source_path,
                    status="queued",
                    stage="reclaimed",
                    detail="stale lease reclaimed",
                    extra={"summary_path": str(summary_path), "csv_path": str(csv_path)},
                )
            lease = ManagedLease(
                status_path=status_path,
                lease_path=lease_path,
                lease_key=source_id,
                owner=self.owner,
                heartbeat_seconds=self.heartbeat_seconds,
                min_update_interval_seconds=self.min_update_interval_seconds,
                metadata={
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_path": str(source_path),
                    "summary_path": str(summary_path),
                    "csv_path": str(csv_path),
                },
            )
            try:
                _create_json_exclusive(lease_path, lease.snapshot())
            except FileExistsError:
                continue
            lease.start()
            return lease
        return LeaseDenied("lease contention detected")

    def acquire_merge_lease(self, merge_key: str, *, wait_seconds: float = 60.0, poll_seconds: float = 1.0) -> ManagedLease:
        status_path = self.merge_status_dir / f"{merge_key}.status.json"
        lease_path = self.merge_active_dir / f"{merge_key}.lease.json"
        deadline = time.time() + max(0.0, wait_seconds)
        while True:
            existing_lease = self._read_json(lease_path)
            if existing_lease:
                if self._lease_is_stale(existing_lease):
                    self._remove_stale_lease(lease_path)
                    write_json_atomic(
                        status_path,
                        {
                            "merge_key": merge_key,
                            "status": "queued",
                            "stage": "reclaimed",
                            "detail": "stale merge lease reclaimed",
                            "updated_at": time.time(),
                        },
                    )
                elif time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for merge lease `{merge_key}`")
                else:
                    time.sleep(poll_seconds)
                    continue
            lease = ManagedLease(
                status_path=status_path,
                lease_path=lease_path,
                lease_key=merge_key,
                owner=self.owner,
                heartbeat_seconds=self.heartbeat_seconds,
                min_update_interval_seconds=self.min_update_interval_seconds,
                metadata={"merge_key": merge_key},
            )
            try:
                _create_json_exclusive(lease_path, lease.snapshot())
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for merge lease `{merge_key}`")
                time.sleep(poll_seconds)
                continue
            lease.start()
            return lease

    def _source_status_path(self, source_id: str) -> Path:
        return self.source_status_dir / f"{source_id}.status.json"

    def _source_lease_path(self, source_id: str) -> Path:
        return self.source_active_dir / f"{source_id}.lease.json"

    def _lease_is_stale(self, payload: dict[str, object] | None) -> bool:
        if not payload:
            return False
        heartbeat_at = payload.get("heartbeat_at")
        if not isinstance(heartbeat_at, (int, float)):
            return True
        return (time.time() - float(heartbeat_at)) > self.stale_after_seconds

    def _remove_stale_lease(self, lease_path: Path) -> None:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass

    def _read_json(self, path: Path) -> dict[str, object] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def _create_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
