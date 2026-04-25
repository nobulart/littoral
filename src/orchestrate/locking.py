from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid

from src.common.io import write_json_atomic


@dataclass(slots=True)
class LeaseDenied:
    reason: str
    status: str = "skipped"


@dataclass(slots=True)
class ForceReleaseResult:
    source_id: str
    lease_found: bool
    status_found: bool
    removed_lease: bool
    killed_pids: list[int]
    skipped_pids: list[str]
    detail: str


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
        self._persist(force=True, write_status=True)
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
            self._persist_locked(force=True, write_status=True)
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
            self._persist_locked(force=True, write_status=True)
        self._shutdown(remove_lease=True)

    def abandon(self, *, remove_lease: bool = False) -> None:
        self._shutdown(remove_lease=remove_lease)

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

    def _persist(self, *, force: bool = False, write_status: bool = False) -> None:
        with self._lock:
            self._persist_locked(force=force, write_status=write_status)

    def _persist_locked(self, *, force: bool = False, write_status: bool = False) -> None:
        if not force and not self._should_persist_locked(True):
            return
        snapshot = self._snapshot_locked()
        if write_status:
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
        heartbeat_seconds: float = 15.0,
        stale_after_seconds: float = 45.0,
        min_update_interval_seconds: float = 10.0,
        sync_settle_seconds: float = 1.5,
        status_bridge_seconds: float = 20.0,
    ) -> None:
        self.root_dir = root_dir
        self.source_status_dir = root_dir / "source_status"
        self.source_active_dir = root_dir / "source_active"
        self.merge_status_dir = root_dir / "merge_status"
        self.merge_active_dir = root_dir / "merge_active"
        self.heartbeat_seconds = heartbeat_seconds
        self.stale_after_seconds = stale_after_seconds
        self.min_update_interval_seconds = min_update_interval_seconds
        self.sync_settle_seconds = max(0.0, sync_settle_seconds)
        self.status_bridge_seconds = max(self.sync_settle_seconds, status_bridge_seconds)
        self.owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "run_id": uuid.uuid4().hex,
        }

    def ensure_dirs(self) -> None:
        for path in (self.source_status_dir, self.source_active_dir, self.merge_status_dir, self.merge_active_dir):
            path.mkdir(parents=True, exist_ok=True)

    def cleanup_local_orphaned_processes(self, workspace_root: Path) -> list[str]:
        cleaned: list[str] = []
        lease_paths = []
        if self.source_active_dir.exists():
            lease_paths.extend(sorted(self.source_active_dir.glob("*.lease.json")))
        if self.merge_active_dir.exists():
            lease_paths.extend(sorted(self.merge_active_dir.glob("*.lease.json")))
        for lease_path in lease_paths:
            payload = self._read_json(lease_path)
            if not payload:
                continue
            owner = payload.get("owner")
            if not isinstance(owner, dict):
                continue
            if owner.get("host") != self.owner["host"]:
                continue
            pid = owner.get("pid")
            if not isinstance(pid, int) or pid == self.owner["pid"]:
                continue
            if not self._lease_is_stale(payload):
                continue
            worker_pid = payload.get("worker_pid")
            if isinstance(worker_pid, int) and worker_pid != self.owner["pid"] and self._pid_exists(worker_pid):
                worker_command = self._command_for_pid(worker_pid)
                if self._looks_like_littoral_process(worker_command, workspace_root) and self._terminate_pid(worker_pid):
                    cleaned.append(f"terminated orphaned worker pid {worker_pid} from {lease_path.name}")
            if not self._pid_exists(pid):
                self._remove_stale_lease(lease_path)
                cleaned.append(f"removed stale lease {lease_path.name} for dead pid {pid}")
                continue
            command = self._command_for_pid(pid)
            if not self._looks_like_littoral_process(command, workspace_root):
                continue
            if self._terminate_pid(pid):
                self._remove_stale_lease(lease_path)
                cleaned.append(f"terminated orphaned pid {pid} from {lease_path.name}")
        return cleaned

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

    def force_release_source(self, source_id: str, *, workspace_root: Path, reason: str = "force released by operator") -> ForceReleaseResult:
        normalized_source_id = Path(source_id.strip()).stem.replace(" ", "_").lower()
        status_path = self._source_status_path(normalized_source_id)
        lease_path = self._source_lease_path(normalized_source_id)
        lease_payload = self._read_json(lease_path)
        status_payload = self._read_json(status_path)
        killed_pids: list[int] = []
        skipped_pids: list[str] = []

        for pid_label, pid in self._candidate_pids_for_force_release(lease_payload, status_payload):
            if pid == self.owner["pid"]:
                skipped_pids.append(f"{pid_label}:{pid}:current-controller")
                continue
            if not self._pid_exists(pid):
                continue
            command = self._command_for_pid(pid)
            if command and not self._looks_like_littoral_process(command, workspace_root):
                skipped_pids.append(f"{pid_label}:{pid}:command-not-recognized")
                continue
            if self._terminate_pid(pid):
                killed_pids.append(pid)
            else:
                skipped_pids.append(f"{pid_label}:{pid}:terminate-failed")

        removed_lease = False
        try:
            lease_path.unlink()
            removed_lease = True
        except FileNotFoundError:
            pass

        now = time.time()
        base_payload = status_payload if isinstance(status_payload, dict) else {}
        base_payload.update(
            {
                "source_id": normalized_source_id,
                "status": "failed",
                "stage": "force_released",
                "detail": reason,
                "updated_at": now,
                "finished_at": now,
                "force_release": {
                    "host": self.owner["host"],
                    "pid": self.owner["pid"],
                    "run_id": self.owner["run_id"],
                    "reason": reason,
                    "killed_pids": killed_pids,
                    "skipped_pids": skipped_pids,
                    "removed_lease": removed_lease,
                    "at": now,
                },
            }
        )
        write_json_atomic(status_path, base_payload)
        return ForceReleaseResult(
            source_id=normalized_source_id,
            lease_found=lease_payload is not None,
            status_found=status_payload is not None,
            removed_lease=removed_lease,
            killed_pids=killed_pids,
            skipped_pids=skipped_pids,
            detail=reason,
        )

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
            accepted_points = _count_csv_records(csv_path)
            if (
                existing_status.get("accepted_points") != accepted_points
                or existing_status.get("summary_path") != str(summary_path)
                or existing_status.get("csv_path") != str(csv_path)
            ):
                self.mark_source_state(
                    source_id,
                    source_name,
                    source_path,
                    status="completed",
                    stage=str(existing_status.get("stage") or "existing_outputs"),
                    detail=str(existing_status.get("detail") or "completed status already recorded"),
                    extra={
                        "summary_path": str(summary_path),
                        "csv_path": str(csv_path),
                        "accepted_points": accepted_points,
                    },
                )
            return LeaseDenied("completed status already recorded")
        if per_source_mode == "skip" and (summary_path.exists() or csv_path.exists()):
            self.mark_source_state(
                source_id,
                source_name,
                source_path,
                status="completed",
                stage="existing_outputs",
                detail="existing per-source outputs found",
                extra={
                    "summary_path": str(summary_path),
                    "csv_path": str(csv_path),
                    "accepted_points": _count_csv_records(csv_path),
                },
            )
            return LeaseDenied("existing per-source outputs found")
        active_status = self._active_status_reason(existing_status)
        if active_status is not None:
            return LeaseDenied(active_status, status="running")
        if self.sync_settle_seconds > 0:
            time.sleep(self.sync_settle_seconds)
            existing_status = self._read_json(status_path)
            existing_lease = self._read_json(lease_path)
            active_lease = self._active_lease_reason(existing_lease)
            if active_lease is not None:
                return LeaseDenied(active_lease, status="running")
            active_status = self._active_status_reason(existing_status)
            if active_status is not None:
                return LeaseDenied(active_status, status="running")
        for _ in range(3):
            existing_lease = self._read_json(lease_path)
            if existing_lease:
                active_lease = self._active_lease_reason(existing_lease)
                if active_lease is not None:
                    return LeaseDenied(active_lease, status="running")
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

    def _active_lease_reason(self, payload: dict[str, object] | None) -> str | None:
        if not payload or self._lease_is_stale(payload):
            return None
        owner = payload.get("owner", {})
        if isinstance(owner, dict) and owner.get("run_id") == self.owner["run_id"]:
            return None
        host = owner.get("host", "unknown-host") if isinstance(owner, dict) else "unknown-host"
        pid = owner.get("pid", "unknown-pid") if isinstance(owner, dict) else "unknown-pid"
        return f"active lease held by {host}:{pid}"

    def _active_status_reason(self, payload: dict[str, object] | None) -> str | None:
        if not payload:
            return None
        status = str(payload.get("status") or "")
        if status != "running":
            return None
        updated_at = payload.get("updated_at")
        if not isinstance(updated_at, (int, float)):
            return "active running status recorded elsewhere"
        status_age = time.time() - float(updated_at)
        if status_age > min(self.stale_after_seconds, self.status_bridge_seconds):
            return None
        owner = payload.get("owner", {})
        if isinstance(owner, dict) and owner.get("run_id") == self.owner["run_id"]:
            return None
        host = owner.get("host", "unknown-host") if isinstance(owner, dict) else "unknown-host"
        pid = owner.get("pid", "unknown-pid") if isinstance(owner, dict) else "unknown-pid"
        return f"active status held by {host}:{pid}"

    def _remove_stale_lease(self, lease_path: Path) -> None:
        try:
            lease_path.unlink()
        except FileNotFoundError:
            pass

    def _candidate_pids_for_force_release(
        self,
        lease_payload: dict[str, object] | None,
        status_payload: dict[str, object] | None,
    ) -> list[tuple[str, int]]:
        candidates: list[tuple[str, int]] = []
        seen: set[int] = set()
        for payload in (lease_payload, status_payload):
            if not isinstance(payload, dict):
                continue
            worker_pid = payload.get("worker_pid")
            if isinstance(worker_pid, int) and worker_pid not in seen:
                candidates.append(("worker_pid", worker_pid))
                seen.add(worker_pid)
            owner = payload.get("owner")
            if isinstance(owner, dict):
                owner_host = owner.get("host")
                owner_pid = owner.get("pid")
                if owner_host == self.owner["host"] and isinstance(owner_pid, int) and owner_pid not in seen:
                    candidates.append(("owner_pid", owner_pid))
                    seen.add(owner_pid)
        return candidates

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _command_for_pid(self, pid: int) -> str:
        try:
            result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], check=True, capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip()

    def _looks_like_littoral_process(self, command: str, workspace_root: Path) -> bool:
        normalized = command.strip()
        if not normalized:
            return False
        return "run_pipeline.py" in normalized or str(workspace_root) in normalized

    def _terminate_pid(self, pid: int) -> bool:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            for _ in range(10):
                if not self._pid_exists(pid):
                    return True
                time.sleep(0.2)
        return not self._pid_exists(pid)

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


def _count_csv_records(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, csv.Error):
        return 0
