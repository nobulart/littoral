from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    observed_at: float
    cpu_count: int
    load_1m: float | None
    load_per_cpu: float | None
    memory_total_gb: float | None
    memory_available_gb: float | None
    memory_used_percent: float | None
    disk_free_gb: float
    disk_used_percent: float


@dataclass(frozen=True, slots=True)
class AutopilotDecision:
    enabled: bool
    max_active_jobs: int
    severity: str
    reason: str
    snapshot: ResourceSnapshot

    def capacity_payload(self) -> dict[str, object]:
        return {
            "autopilot_enabled": self.enabled,
            "autopilot_max_active_jobs": self.max_active_jobs,
            "autopilot_severity": self.severity,
            "autopilot_reason": self.reason,
            "resource_load_1m": self.snapshot.load_1m,
            "resource_load_per_cpu": self.snapshot.load_per_cpu,
            "resource_memory_used_percent": self.snapshot.memory_used_percent,
            "resource_memory_available_gb": self.snapshot.memory_available_gb,
            "resource_disk_free_gb": self.snapshot.disk_free_gb,
            "resource_disk_used_percent": self.snapshot.disk_used_percent,
        }


class AutopilotController:
    def __init__(
        self,
        workspace_root: Path,
        *,
        document_workers: int,
        min_interval_seconds: float = 2.0,
        memory_warn_percent: float = 82.0,
        memory_critical_percent: float = 92.0,
        load_warn_per_cpu: float = 1.4,
        load_critical_per_cpu: float = 2.0,
        disk_warn_free_gb: float = 10.0,
        disk_critical_free_gb: float = 2.0,
    ) -> None:
        self.workspace_root = workspace_root
        self.document_workers = max(1, document_workers)
        self.min_interval_seconds = max(0.25, min_interval_seconds)
        self.memory_warn_percent = memory_warn_percent
        self.memory_critical_percent = memory_critical_percent
        self.load_warn_per_cpu = load_warn_per_cpu
        self.load_critical_per_cpu = load_critical_per_cpu
        self.disk_warn_free_gb = disk_warn_free_gb
        self.disk_critical_free_gb = disk_critical_free_gb
        self._last_sampled_at = 0.0
        self._last_decision: AutopilotDecision | None = None

    def evaluate(self, *, enabled: bool, active_jobs: int, queued_jobs: int) -> AutopilotDecision:
        now = time.perf_counter()
        if self._last_decision is not None and (now - self._last_sampled_at) < self.min_interval_seconds:
            if self._last_decision.enabled == enabled:
                return self._last_decision
        snapshot = sample_resources(self.workspace_root)
        self._last_sampled_at = now
        decision = self._decide(snapshot, enabled=enabled, active_jobs=active_jobs, queued_jobs=queued_jobs)
        self._last_decision = decision
        return decision

    def _decide(self, snapshot: ResourceSnapshot, *, enabled: bool, active_jobs: int, queued_jobs: int) -> AutopilotDecision:
        if not enabled:
            return AutopilotDecision(
                enabled=False,
                max_active_jobs=self.document_workers,
                severity="off",
                reason="manual control",
                snapshot=snapshot,
            )

        limit = self.document_workers
        severity_rank = 0
        reasons: list[str] = []

        memory_used = snapshot.memory_used_percent
        if memory_used is not None and memory_used >= self.memory_critical_percent:
            limit = 0
            severity_rank = max(severity_rank, 3)
            reasons.append(f"memory {memory_used:.0f}%")
        elif memory_used is not None and memory_used >= self.memory_warn_percent:
            limit = min(limit, max(1, self.document_workers // 2))
            severity_rank = max(severity_rank, 2)
            reasons.append(f"memory {memory_used:.0f}%")

        if snapshot.disk_free_gb <= self.disk_critical_free_gb or snapshot.disk_used_percent >= 98.0:
            limit = 0
            severity_rank = max(severity_rank, 3)
            reasons.append(f"disk free {snapshot.disk_free_gb:.1f}GB")
        elif snapshot.disk_free_gb <= self.disk_warn_free_gb or snapshot.disk_used_percent >= 95.0:
            limit = min(limit, max(1, self.document_workers // 2))
            severity_rank = max(severity_rank, 2)
            reasons.append(f"disk free {snapshot.disk_free_gb:.1f}GB")

        load_per_cpu = snapshot.load_per_cpu
        if load_per_cpu is not None and load_per_cpu >= self.load_critical_per_cpu:
            limit = min(limit, max(1, self.document_workers // 3))
            severity_rank = max(severity_rank, 2)
            reasons.append(f"load/cpu {load_per_cpu:.1f}")
        elif load_per_cpu is not None and load_per_cpu >= self.load_warn_per_cpu:
            limit = min(limit, max(1, self.document_workers // 2))
            severity_rank = max(severity_rank, 1)
            reasons.append(f"load/cpu {load_per_cpu:.1f}")

        if queued_jobs == 0:
            reasons.append("queue empty")

        severity = ["nominal", "watch", "throttle", "drain"][severity_rank]
        reason = ", ".join(reasons) if reasons else "resources nominal"
        return AutopilotDecision(
            enabled=True,
            max_active_jobs=max(0, min(self.document_workers, limit)),
            severity=severity,
            reason=reason,
            snapshot=snapshot,
        )


def sample_resources(workspace_root: Path) -> ResourceSnapshot:
    cpu_count = max(1, os.cpu_count() or 1)
    load_1m = _load_1m()
    load_per_cpu = (load_1m / cpu_count) if load_1m is not None else None
    memory_total_gb, memory_available_gb = _memory()
    if memory_total_gb and memory_available_gb is not None:
        memory_used_percent = max(0.0, min(100.0, 100.0 * (1.0 - (memory_available_gb / memory_total_gb))))
    else:
        memory_used_percent = None
    disk = shutil.disk_usage(workspace_root)
    disk_total = max(1, disk.total)
    disk_free_gb = disk.free / (1024 ** 3)
    disk_used_percent = 100.0 * (1.0 - (disk.free / disk_total))
    return ResourceSnapshot(
        observed_at=time.time(),
        cpu_count=cpu_count,
        load_1m=load_1m,
        load_per_cpu=load_per_cpu,
        memory_total_gb=memory_total_gb,
        memory_available_gb=memory_available_gb,
        memory_used_percent=memory_used_percent,
        disk_free_gb=disk_free_gb,
        disk_used_percent=disk_used_percent,
    )


def _load_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def _memory() -> tuple[float | None, float | None]:
    if platform.system() == "Darwin":
        return _darwin_memory()
    return _linux_memory()


def _darwin_memory() -> tuple[float | None, float | None]:
    try:
        total_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2, stderr=subprocess.DEVNULL).strip()
        )
        vm_stat = subprocess.check_output(["vm_stat"], text=True, timeout=2, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, None
    page_size = 4096
    free_pages = 0
    for raw_line in vm_stat.splitlines():
        line = raw_line.strip().rstrip(".")
        if "page size of" in line:
            parts = [part for part in line.split() if part.isdigit()]
            if parts:
                page_size = int(parts[0])
            continue
        if ":" not in line:
            continue
        label, value_text = line.split(":", 1)
        label = label.lower()
        if label not in {"pages free", "pages inactive", "pages speculative"}:
            continue
        try:
            free_pages += int(value_text.strip().replace(".", ""))
        except ValueError:
            continue
    return total_bytes / (1024 ** 3), (free_pages * page_size) / (1024 ** 3)


def _linux_memory() -> tuple[float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values: dict[str, float] = {}
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts:
                values[key] = float(parts[0]) / (1024 ** 2)
    except (OSError, ValueError):
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable", values.get("MemFree"))
    return total, available
