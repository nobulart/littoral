from __future__ import annotations

from pathlib import Path

from src.extract.base import BaseExtractor, ExtractionResult
from src.extract.document_loader import load_document_payload
from src.extract.interpreter import interpret_document
from src.orchestrate.runtime import PipelineRuntime


class PdfExtractor(BaseExtractor):
    supported_suffixes = (".pdf",)

    def extract(self, source_path: Path, source_id: str, runtime: PipelineRuntime | None = None) -> ExtractionResult:
        payload = load_document_payload(source_path, runtime=runtime)
        return interpret_document(source_path, source_id, payload, runtime=runtime)
