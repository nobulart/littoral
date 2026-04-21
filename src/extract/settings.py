from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    "mineru": {
        "enabled": True,
        "skip_existing": True,
        "command": "mineru",
        "backend": "hybrid-auto-engine",
        "method": "auto",
        "lang": "en",
        "timeout_seconds": 3600,
    },
    "mineru_inference": {
        "enabled": True,
        "llm_enabled": True,
        "max_llm_contexts": 4,
    },
    "pdf": {
        "ocr_enabled": True,
        "ocr_max_pages": 12,
        "ocr_dpi": 200,
        "native_text_min_length": 1500,
        "min_quality_score": 0.55,
        "page_ocr_enabled": True,
        "page_ocr_max_pages": 3,
        "page_cue_pattern": "table|fig\\.|figure|plate|map|caption",
    },
    "ollama": {
        "enabled": False,
        "document_interpretation_enabled": False,
        "place_normalization_enabled": False,
        "model": "glm-4.7-flash:latest",
        "api_url": "http://localhost:11434",
        "timeout_seconds": 45,
        "max_input_chars": 12000,
        "candidate_only_when_needed": True,
    },
    "geocoding": {
        "enabled": True,
        "service": "nominatim",
        "url": "https://nominatim.openstreetmap.org/search",
        "user_agent": "LITTORAL/0.1",
        "timeout_seconds": 20,
        "limit": 3,
    },
}


def source_workspace_root(source_path: Path) -> Path:
    return source_path.resolve().parents[2]


def load_extraction_settings(source_path: Path) -> dict[str, Any]:
    workspace_root = source_workspace_root(source_path)
    settings_path = workspace_root / "config" / "extraction.json"
    if not settings_path.exists():
        return DEFAULT_SETTINGS

    loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    for top_key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(top_key), dict):
            merged[top_key].update(value)
        else:
            merged[top_key] = value
    if os.environ.get("LITTORAL_FAST_TEST") == "1":
        merged["mineru"]["enabled"] = False
        merged["mineru_inference"]["llm_enabled"] = False
        merged["ollama"]["enabled"] = False
        merged["geocoding"]["enabled"] = False
    return merged
