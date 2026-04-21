from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.models import SamplePoint


@dataclass(slots=True)
class ExtractionResult:
    source_id: str
    summary_lines: list[str]
    sample_points: list[SamplePoint]
    unresolved_lines: list[str]


class BaseExtractor:
    supported_suffixes: tuple[str, ...] = ()

    def extract(self, source_path: Path, source_id: str) -> ExtractionResult:
        raise NotImplementedError
