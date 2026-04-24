from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.extract.settings import load_extraction_settings
from src.orchestrate.runtime import PipelineRuntime, maybe_gpu_task

if TYPE_CHECKING:
    from src.extract.document_loader import DocumentPayload


class OllamaClient:
    def __init__(self, source_path: Path, runtime: PipelineRuntime | None = None) -> None:
        settings = runtime.settings_for(source_path) if runtime is not None else load_extraction_settings(source_path)
        self.settings = settings["ollama"]
        self.enabled = bool(self.settings.get("enabled", False))
        self.model = str(self.settings.get("model", "glm-4.7-flash:latest"))
        self.timeout_seconds = int(self.settings.get("timeout_seconds", 45))
        self.max_input_chars = int(self.settings.get("max_input_chars", 12000))
        self.api_url = str(self.settings.get("api_url", "http://localhost:11434")).rstrip("/")
        self.runtime = runtime

    def can_run(self) -> bool:
        if not self.enabled or not self.settings.get("document_interpretation_enabled", False):
            return False
        return self.can_run_model(self.model)

    def can_run_model(self, model: str) -> bool:
        if self.runtime is not None:
            return self.runtime.can_run_ollama_model(self.api_url, model, self._resolve_available_models)
        api_models = self._resolve_available_models()
        if api_models is not None:
            return model in api_models
        try:
            result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return model in result.stdout

    def interpret_document(self, payload: DocumentPayload, source_name: str, ontology_categories: list[str]) -> dict[str, Any] | None:
        if not self.can_run():
            return None
        prompt = self._build_prompt(payload, source_name, ontology_categories)
        return self._interpret_prompt(prompt)

    def interpret_mineru_context(self, context_label: str, context_text: str, source_name: str, ontology_categories: list[str]) -> dict[str, Any] | None:
        if not self.can_run():
            return None
        categories = ", ".join(sorted(ontology_categories))
        prompt = (
            "You are extracting structured paleo-relative sea-level evidence from one MinerU table, map, figure, or chart context. "
            "Return JSON only. Preserve records even when coordinates are not explicit; use null latitude/longitude and include a place_query/location_name from the context. "
            "Do not invent values unsupported by the context. Return at most 12 of the strongest candidate records. "
            "If the context is ancient stratigraphy, palaeontology, or process sedimentology without Quaternary/coastal relative sea-level measurements, return no candidate_records. "
            "Exception: preserve guyots, tablemounts, flat-topped seamounts, drowned islands, atolls, and drowned carbonate platforms when the context reports summit/platform depth, drowning, emergence, subsidence, or sea-level history.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "summary": "string",\n'
            '  "source_relevance": "extractable_primary_source|secondary_compilation|narrative_background|non_relevant",\n'
            '  "candidate_records": [\n'
            "    {\n"
            '      "site_name": "string|null", "sample_id": "string|null", "location_name": "string|null", "place_query": "string|null",\n'
            '      "latitude": number|null, "longitude": number|null, "coordinate_source": "reported|inferred_text|inferred_map|null", "coordinate_uncertainty_m": number|null,\n'
            '      "indicator_type": "one of: ' + categories + '", "indicator_subtype": "string|null", "record_class": "sea_level_indicator|marine_limiting|terrestrial_limiting|null",\n'
            '      "elevation_m": number|null, "elevation_reference": "MSL|LAT|unknown|null", "depth_source": "reported|SRTM15+V2|other|null",\n'
            '      "age_ka": number|string|array|null, "dating_method": "string|null", "description": "string|null",\n'
            '      "bibliographic_reference": "string|null", "doi_or_url": "string|null", "quote_or_paraphrase": "string",\n'
            '      "page": "string|null", "figure": "string|null", "table": "string|null", "notes": "string|null"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Source: {source_name}\n"
            f"Context label: {context_label}\n"
            f"Context:\n{context_text[: self.max_input_chars]}\n"
        )
        return self._interpret_prompt(prompt)

    def _interpret_prompt(self, prompt: str) -> dict[str, Any] | None:
        output = self._generate(prompt)
        if output is None:
            output = self._generate_with_cli(prompt)
        if output is None:
            return None

        output = output.strip()
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(output[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _list_models_from_api(self) -> set[str] | None:
        try:
            with urllib.request.urlopen(f"{self.api_url}/api/tags", timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None
        models = payload.get("models", [])
        if not isinstance(models, list):
            return set()
        names: set[str] = set()
        for item in models:
            if not isinstance(item, dict):
                continue
            for key in ("name", "model"):
                value = item.get(key)
                if isinstance(value, str):
                    names.add(value)
        return names

    def _generate(self, prompt: str) -> str | None:
        request_payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}/api/generate",
            data=request_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with maybe_gpu_task(self.runtime):
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None
        output = payload.get("response")
        return output if isinstance(output, str) else None

    def _generate_with_cli(self, prompt: str) -> str | None:
        try:
            with maybe_gpu_task(self.runtime):
                result = subprocess.run(
                    ["ollama", "run", self.model],
                    input=prompt,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None
        return result.stdout

    def ocr_page_image(self, image_path: Path, prompt: str, model: str = "glm-ocr:latest") -> str | None:
        if not self.can_run_model(model):
            return None
        api_text = self._generate_image(prompt, image_path, model)
        if api_text:
            return api_text
        task_prefix = _glm_ocr_task_prefix(prompt)
        try:
            with maybe_gpu_task(self.runtime):
                result = subprocess.run(
                    ["ollama", "run", model, f"{task_prefix}: {image_path}"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=max(self.timeout_seconds, 45),
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None
        text = _strip_ansi_sequences(result.stdout)
        if not text:
            return None
        return str(text).strip()

    def _generate_image(self, prompt: str, image_path: Path, model: str) -> str | None:
        try:
            image_payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        except OSError:
            return None
        request_payload = json.dumps({"model": model, "prompt": prompt, "images": [image_payload], "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_url}/api/generate",
            data=request_payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with maybe_gpu_task(self.runtime):
                with urllib.request.urlopen(request, timeout=max(self.timeout_seconds, 45)) as response:
                    payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return None
        output = payload.get("response")
        return output.strip() if isinstance(output, str) and output.strip() else None

    def _resolve_available_models(self) -> set[str] | None:
        api_models = self._list_models_from_api()
        if api_models is not None:
            return api_models
        try:
            result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return {line.split()[0] for line in result.stdout.splitlines()[1:] if line.strip()}

    def _build_prompt(self, payload: DocumentPayload, source_name: str, ontology_categories: list[str]) -> str:
        text = payload.text[: self.max_input_chars]
        categories = ", ".join(sorted(ontology_categories))
        figure_lines = _extract_labeled_lines(payload.text, ("fig", "figure", "map", "plate"))
        table_lines = _extract_labeled_lines(payload.text, ("table",))
        page_blocks = "\n\n".join(
            [f"[page {block.page_number} | cue={block.cue} | source={block.source}]\n{block.text[:2000]}" for block in payload.page_blocks[:12]]
        )
        return (
            "You are extracting paleo-relative sea-level evidence from literature. Return JSON only. Do not invent coordinates, ages, indicators, or localities not explicitly supported by the text, figures, maps, or tables. If a field is not explicit, use null. Return at most 12 of the strongest candidate records when a paper clearly discusses multiple indicators or localities. If the document is ancient stratigraphy, palaeontology, or process sedimentology without Quaternary/coastal relative sea-level measurements, return no candidate_records. Exception: preserve guyots, tablemounts, flat-topped seamounts, drowned islands, atolls, and drowned carbonate platforms when the document reports summit/platform depth, drowning, emergence, subsidence, or sea-level history.\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "summary": "string",\n'
            '  "source_relevance": "extractable_primary_source|secondary_compilation|narrative_background|non_relevant",\n'
            '  "candidate_records": [\n'
            "    {\n"
            '      "site_name": "string|null",\n'
            '      "sample_id": "string|null",\n'
            '      "location_name": "string|null",\n'
            '      "place_query": "string|null",\n'
            '      "latitude": number|null,\n'
            '      "longitude": number|null,\n'
            '      "coordinate_source": "reported|inferred_text|inferred_map|null",\n'
            '      "coordinate_uncertainty_m": number|null,\n'
            '      "indicator_type": "one of: ' + categories + '",\n'
            '      "indicator_subtype": "string|null",\n'
            '      "record_class": "sea_level_indicator|marine_limiting|terrestrial_limiting|null",\n'
            '      "elevation_m": number|null,\n'
            '      "elevation_reference": "MSL|LAT|unknown|null",\n'
            '      "depth_source": "reported|SRTM15+V2|other|null",\n'
            '      "age_ka": number|string|array|null,\n'
            '      "dating_method": "string|null",\n'
            '      "description": "string|null",\n'
            '      "bibliographic_reference": "string|null",\n'
            '      "doi_or_url": "string|null",\n'
            '      "quote_or_paraphrase": "string",\n'
            '      "page": "string|null",\n'
            '      "figure": "string|null",\n'
            '      "table": "string|null",\n'
            '      "notes": "string|null"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Source: {source_name}\n"
            f"Title: {payload.title}\n"
            f"Extraction methods: {', '.join(payload.extraction_methods)}\n"
            f"Text quality score: {payload.text_quality_score}\n\n"
            "MinerU figure/map lines:\n"
            f"{figure_lines}\n\n"
            "MinerU table lines:\n"
            f"{table_lines}\n\n"
            "MinerU structured page blocks:\n"
            f"{page_blocks}\n\n"
            "Document text:\n"
            f"{text}\n"
        )


def _glm_ocr_task_prefix(prompt: str) -> str:
    lowered = prompt.lower()
    if "table" in lowered:
        return "Table Recognition"
    if "figure" in lowered or "map" in lowered or "plate" in lowered:
        return "Figure Recognition"
    return "Text Recognition"


def _strip_ansi_sequences(text: str) -> str:
    ansi_pattern = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_pattern.sub("", text)


def _extract_labeled_lines(text: str, labels: tuple[str, ...]) -> str:
    selected: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        lowered = line.lower()
        if any(label in lowered for label in labels):
            selected.append(line)
        if len(selected) >= 20:
            break
    return "\n".join(selected)
