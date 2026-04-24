from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
import os
import platform
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Iterator

from src.extract.settings import load_extraction_settings
from src.ontology.catalog import Ontology

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on some platforms
    fcntl = None


def auto_document_workers() -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    memory_gb = _memory_gb()
    if memory_gb >= 256:
        return min(10, max(6, cpu_count // 2))
    if memory_gb >= 64:
        return min(4, max(2, cpu_count // 3))
    return min(2, cpu_count)


def auto_gpu_slots() -> int:
    memory_gb = _memory_gb()
    return 2 if memory_gb >= 256 else 1


@dataclass(slots=True)
class PipelineRuntime:
    ontology: Ontology
    gpu_slots: int
    gpu_lock_dir: Path | None = None
    settings_cache: dict[Path, dict[str, Any]] = field(default_factory=dict)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)
    ollama_models_cache: dict[str, set[str] | None] = field(default_factory=dict)
    ollama_lock: threading.Lock = field(default_factory=threading.Lock)
    _gpu_semaphore: threading.BoundedSemaphore = field(init=False, repr=False)
    _gpu_slot_paths: tuple[Path, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gpu_semaphore = threading.BoundedSemaphore(max(1, self.gpu_slots))
        if self.gpu_lock_dir is None:
            self._gpu_slot_paths = ()
            return
        slot_dir = self.gpu_lock_dir / "gpu_slots"
        slot_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_slot_paths = tuple(slot_dir / f"slot-{index:02d}.lock" for index in range(max(1, self.gpu_slots)))

    @property
    def ontology_categories(self) -> list[str]:
        return sorted(self.ontology.categories.keys())

    def settings_for(self, source_path: Path) -> dict[str, Any]:
        key = source_path.resolve()
        with self.settings_lock:
            cached = self.settings_cache.get(key)
            if cached is not None:
                return cached
        settings = load_extraction_settings(source_path)
        with self.settings_lock:
            self.settings_cache[key] = settings
        return settings

    def can_run_ollama_model(self, api_url: str, model: str, resolver: Callable[[], set[str] | None]) -> bool:
        normalized_url = api_url.rstrip("/")
        with self.ollama_lock:
            if normalized_url in self.ollama_models_cache:
                cached = self.ollama_models_cache[normalized_url]
                return cached is not None and model in cached
        models = resolver()
        with self.ollama_lock:
            self.ollama_models_cache[normalized_url] = models
        return models is not None and model in models

    @contextmanager
    def gpu_task(self) -> Iterator[None]:
        if not self._gpu_slot_paths or fcntl is None:
            with self._gpu_semaphore:
                yield
            return

        handle = None
        acquired_path = None
        while handle is None:
            for slot_path in self._gpu_slot_paths:
                candidate = slot_path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                acquired_path = slot_path
                payload = {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "acquired_at": time.time(),
                }
                candidate.seek(0)
                candidate.truncate()
                json.dump(payload, candidate, indent=2, sort_keys=True)
                candidate.write("\n")
                candidate.flush()
                handle = candidate
                break
            if handle is None:
                time.sleep(0.1)

        try:
            yield
        finally:
            if handle is not None:
                handle.seek(0)
                handle.truncate()
                handle.flush()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            if acquired_path is not None:
                try:
                    acquired_path.touch(exist_ok=True)
                except OSError:
                    pass


def maybe_gpu_task(runtime: PipelineRuntime | None) -> Any:
    if runtime is None:
        return nullcontext()
    return runtime.gpu_task()


def hardware_profile_summary(document_workers: int, gpu_slots: int) -> str:
    memory_gb = _memory_gb()
    cpu_count = max(1, os.cpu_count() or 1)
    machine = platform.machine() or "unknown"
    return f"machine={machine}, memory_gb={memory_gb}, cpu_count={cpu_count}, document_workers={document_workers}, gpu_slots={gpu_slots}"


def _memory_gb() -> int:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=2, stderr=subprocess.DEVNULL).strip()
            return max(1, round(int(output) / (1024 ** 3)))
        except (OSError, subprocess.SubprocessError, ValueError):
            return 64
    page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    phys_pages = os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else 0
    if page_size and phys_pages:
        return max(1, round((page_size * phys_pages) / (1024 ** 3)))
    return 64
