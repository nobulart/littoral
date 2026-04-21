from __future__ import annotations

from pathlib import Path

from src.extract.base import BaseExtractor, ExtractionResult
from src.extract.document_loader import load_document_payload
from src.extract.interpreter import interpret_document


class TextExtractor(BaseExtractor):
    supported_suffixes = (".txt", ".md")

    def extract(self, source_path: Path, source_id: str) -> ExtractionResult:
        payload = load_document_payload(source_path)
        return interpret_document(source_path, source_id, payload)
