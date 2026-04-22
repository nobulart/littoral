from __future__ import annotations

from pathlib import Path

from src.common.models import SamplePoint
from src.extract.base import ExtractionResult
from src.extract.document_loader import DocumentPayload
from src.extract.heuristics import build_heuristic_sample_points, build_page_block_sample_points, clean_title, llm_candidate_to_sample_point, summarize_payload
from src.extract.mineru_inference import mine_mineru_outputs
from src.extract.narrative_fallback import build_narrative_fallback_sample_points
from src.extract.ollama_client import OllamaClient
from src.extract.settings import load_extraction_settings, source_workspace_root
from src.extract.text_analysis import analyze_text, build_unresolved_line
from src.ontology.catalog import load_ontology


def interpret_document(source_path: Path, source_id: str, payload: DocumentPayload) -> ExtractionResult:
    analysis = analyze_text(payload.text)
    title = clean_title(payload.title, payload.text, source_path)
    heuristic_points = build_heuristic_sample_points(source_id, source_path, payload)
    page_block_points = build_page_block_sample_points(source_id, source_path, payload)
    mineru_result = mine_mineru_outputs(source_id, source_path, payload)
    mineru_points = mineru_result.sample_points

    workspace_root = source_workspace_root(source_path)
    ontology = load_ontology(workspace_root / "config" / "categories.json")
    settings = load_extraction_settings(source_path)
    mineru_settings = settings.get("mineru_inference", {})
    ollama = OllamaClient(source_path)
    llm_payload = None
    mineru_llm_payloads: list[dict] = []
    llm_points: list[SamplePoint] = []
    narrative_fallback_result = None
    has_structured_artifacts = _contains_table_or_figure(payload.text)
    ancient_non_rsl_source = _looks_like_ancient_non_rsl_source(title, payload.text)

    should_try_llm = not ancient_non_rsl_source and ollama.can_run() and (
        not settings["ollama"].get("candidate_only_when_needed", True)
        or (analysis["source_classification"] == "candidate_review_needed" and not heuristic_points and not page_block_points and not mineru_points)
        or (has_structured_artifacts and not heuristic_points and not page_block_points and not mineru_points)
    )
    should_try_targeted_llm = not (heuristic_points or page_block_points or mineru_points)
    if (
        should_try_targeted_llm
        and not ancient_non_rsl_source
        and mineru_settings.get("enabled", True)
        and ollama.can_run()
        and mineru_settings.get("llm_enabled", True)
    ):
        for context_index, (context_label, context_text) in enumerate(mineru_result.llm_contexts[: int(mineru_settings.get("max_llm_contexts", 4))], start=1):
            context_payload = ollama.interpret_mineru_context(context_label, context_text, source_path.name, sorted(ontology.categories.keys()))
            if not context_payload:
                continue
            mineru_llm_payloads.append(context_payload)
            for candidate_index, candidate in enumerate(context_payload.get("candidate_records", []), start=1):
                if not candidate.get("sample_id"):
                    candidate["sample_id"] = f"mineru_llm_{context_index}_{candidate_index}"
                point = llm_candidate_to_sample_point(source_id, source_path, candidate, title)
                if point is not None:
                    llm_points.append(point)

    if should_try_llm:
        llm_payload = ollama.interpret_document(payload, source_path.name, sorted(ontology.categories.keys()))
        if llm_payload:
            for index, candidate in enumerate(llm_payload.get("candidate_records", []), start=1):
                if not candidate.get("sample_id"):
                    candidate["sample_id"] = f"llm_feature_{index}"
                point = llm_candidate_to_sample_point(source_id, source_path, candidate, title)
                if point is not None:
                    llm_points.append(point)

    sample_points = _deduplicate_points(heuristic_points + page_block_points + mineru_points + llm_points)
    fallback_settings = settings.get("narrative_fallback", {})
    if not sample_points and fallback_settings.get("enabled", True):
        narrative_fallback_result = build_narrative_fallback_sample_points(source_id, source_path, payload)
        sample_points = _deduplicate_points(narrative_fallback_result.sample_points)
    unresolved = []
    if not sample_points:
        unresolved.append(build_unresolved_line(source_id, source_path.name, str(analysis["source_classification"])))

    summary_lines = [
        f"# Summary for {source_path.name}",
        "",
        f"- Source ID: `{source_id}`",
        f"- File type: `{source_path.suffix}`",
        f"- Title: `{title}`",
        f"- Source classification: `{analysis['source_classification']}`",
        f"- Coordinate-like strings detected: `{len(analysis['coordinate_hits'])}`",
        f"- Age-like strings detected: `{len(analysis['age_hits'])}`",
        f"- Elevation/depth-like strings detected: `{len(analysis['elevation_hits'])}`",
        f"- RSL keyword hits: `{len(analysis['keyword_hits'])}`",
        f"- Structured SamplePoints extracted: `{len(sample_points)}`",
    ]
    summary_lines.extend(summarize_payload(payload))
    if analysis["keyword_hits"]:
        summary_lines.append(f"- Matched RSL keywords: `{', '.join(analysis['keyword_hits'][:12])}`")
    if llm_payload:
        summary_lines.append(f"- Ollama summary: `{str(llm_payload.get('summary', ''))[:300]}`")
        summary_lines.append(f"- Ollama candidate count: `{len(llm_payload.get('candidate_records', []))}`")
    if mineru_llm_payloads:
        summary_lines.append(f"- MinerU targeted Ollama contexts: `{len(mineru_llm_payloads)}`")
        summary_lines.append(f"- MinerU targeted Ollama candidates: `{sum(len(payload.get('candidate_records', [])) for payload in mineru_llm_payloads)}`")
    summary_lines.append(f"- MinerU deterministic records: `{mineru_result.deterministic_records}`")
    summary_lines.append(f"- MinerU LLM candidate contexts: `{len(mineru_result.llm_contexts)}`")
    summary_lines.append(f"- Table/figure cues detected: `{has_structured_artifacts}`")
    summary_lines.append(f"- Ancient/non-RSL LLM guard: `{ancient_non_rsl_source}`")
    summary_lines.append(f"- Page OCR candidate records: `{len(page_block_points)}`")
    if narrative_fallback_result:
        summary_lines.append(f"- Narrative fallback evidence clusters: `{narrative_fallback_result.evidence_count}`")
        summary_lines.append(f"- Narrative fallback candidate records: `{len(narrative_fallback_result.sample_points)}`")
        if narrative_fallback_result.ledger_lines:
            summary_lines.extend(["", "## Narrative fallback evidence ledger"])
            summary_lines.extend(narrative_fallback_result.ledger_lines)
    summary_lines.extend(
        [
            "",
            "## Source assessment",
            f"- Quote sample: `{' '.join(payload.text.split())[:300]}`",
        ]
    )
    return ExtractionResult(source_id=source_id, summary_lines=summary_lines, sample_points=sample_points, unresolved_lines=unresolved)


def _deduplicate_points(points: list[SamplePoint]) -> list[SamplePoint]:
    unique: dict[str, SamplePoint] = {}
    for point in points:
        unique[point.id] = point
    return list(unique.values())


def _contains_table_or_figure(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["table", "fig.", "figure", "plate", "map"])


def _looks_like_ancient_non_rsl_source(title: str, text: str) -> bool:
    lowered = f"{title}\n{text[:4000]}".lower()
    ancient_terms = [
        "permian",
        "eocene",
        "cretaceous",
        "jurassic",
        "triassic",
        "palaeontology",
        "paleontology",
        "conodont",
        "bryozoan",
        "foraminifera",
        "trace fossils",
        "stratigraphic",
        "submarine channel",
        "sedimentology",
        "basin floor",
    ]
    rsl_terms = [
        "holocene",
        "pleistocene",
        "radiocarbon",
        "raised beach",
        "submerged beach",
        "marine terrace",
        "relative sea level",
        "relative sea-level",
        "yrs b.p",
        "years b.p",
    ]
    ancient_hits = sum(1 for term in ancient_terms if term in lowered)
    rsl_hits = sum(1 for term in rsl_terms if term in lowered)
    return ancient_hits >= 2 and rsl_hits == 0
