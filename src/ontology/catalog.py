from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Ontology:
    payload: dict

    @property
    def version(self) -> str:
        return str(self.payload["version"])

    @property
    def categories(self) -> dict:
        return dict(self.payload["primary_categories"])

    @property
    def record_classes(self) -> set[str]:
        return set(self.payload["record_classes"].keys())

    def has_category(self, category: str) -> bool:
        return category in self.payload["primary_categories"]

    def has_record_class(self, record_class: str) -> bool:
        return record_class in self.record_classes


def load_ontology(path: Path) -> Ontology:
    return Ontology(json.loads(path.read_text(encoding="utf-8")))
