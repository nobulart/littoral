from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
import os
import platform
import subprocess
import threading
from typing import Any, Callable, Iterator

from src.extract.settings import load_extraction_settings
from src.ontology.catalog import Ontology


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
    settings_cache: dict[Path, dict[str, Any]] = field(default_factory=dict)
    settings_lock: threading.Lock = field(default_factory=threading.Lock)
    ollama_models_cache: dict[str, set[str] | None] = field(default_factory=dict)
    ollama_lock: threading.Lock = field(default_factory=threading.Lock)
    _gpu_semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gpu_semaphore = threading.BoundedSemaphore(max(1, self.gpu_slots))

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
        with self._gpu_semaphore:
            yield


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
