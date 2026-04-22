from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time
from collections.abc import Callable

from src.common.io import append_log, write_csv, write_summary
from src.extract.base import BaseExtractor
from src.extract.csv_extractor import CsvExtractor
from src.extract.pdf_extractor import PdfExtractor
from src.extract.settings import load_extraction_settings
from src.extract.txt_extractor import TextExtractor
from src.merge.build import build_master_outputs
from src.normalize.elevation import apply_elevation_normalization
from src.ontology.catalog import Ontology, load_ontology
from src.validate.samplepoint import load_schema, score_confidence, validate_sample_point


@dataclass(slots=True)
class PipelineConfig:
    workspace_root: Path
    incoming_dir: Path
    staged_dir: Path
    per_source_dir: Path
    merged_dir: Path
    unresolved_log: Path
    report_path: Path
    ontology_path: Path
    schema_path: Path
    raster_path: Path
    clear_outputs: bool = False
    per_source_mode: str = "skip"
    merged_mode: str = "append"
    mineru_cache_mode: str = "reuse"
    source_ids: tuple[str, ...] | None = None
    limit: int | None = None
    progress_callback: Callable[[str], None] | None = None
    verbosity: int = 1


def default_config(workspace_root: Path) -> PipelineConfig:
    return PipelineConfig(
        workspace_root=workspace_root,
        incoming_dir=workspace_root / "data" / "incoming",
        staged_dir=workspace_root / "data" / "staged",
        per_source_dir=workspace_root / "outputs" / "per_source",
        merged_dir=workspace_root / "outputs" / "merged",
        unresolved_log=workspace_root / "logs" / "UnresolvedRecords.log",
        report_path=workspace_root / "logs" / "processing_report.md",
        ontology_path=workspace_root / "config" / "categories.json",
        schema_path=workspace_root / "config" / "schema.samplepoint.json",
        raster_path=workspace_root / "data" / "elevation" / "SRTM15+V2.tiff",
    )


def run_pipeline(config: PipelineConfig) -> None:
    pipeline_started = time.perf_counter()
    _emit_progress(config, "Loading ontology and schema")
    ontology = load_ontology(config.ontology_path)
    schema = load_schema(config.schema_path)
    extractors = _extractor_registry()
    processed_files: list[str] = []
    skipped_files: list[str] = []
    extracted_records = 0
    unresolved_count = 0

    config.per_source_dir.mkdir(parents=True, exist_ok=True)
    config.merged_dir.mkdir(parents=True, exist_ok=True)
    config.unresolved_log.parent.mkdir(parents=True, exist_ok=True)
    if config.clear_outputs:
        _emit_progress(config, "Clearing previous outputs")
        _clear_directory(config.per_source_dir)
        _clear_directory(config.merged_dir)
        config.unresolved_log.write_text("", encoding="utf-8")
    elif not config.unresolved_log.exists():
        config.unresolved_log.write_text("", encoding="utf-8")

    source_paths = _selected_input_files(config)
    _emit_progress(config, f"Processing {len(source_paths)} input file(s)")
    _emit_progress(config, f"Inputs: {config.incoming_dir}", level=2)
    _emit_progress(config, f"MinerU staged cache: {config.staged_dir}", level=2)
    _emit_progress(config, f"Per-source outputs: {config.per_source_dir}", level=2)
    for index, source_path in enumerate(source_paths, start=1):
        file_started = time.perf_counter()
        source_id = _source_id_for_path(source_path)
        _emit_progress(config, f"[{index}/{len(source_paths)}] {source_path.name}")
        summary_path, csv_path = _per_source_output_paths(config.per_source_dir, source_id)
        if config.per_source_mode == "skip" and _per_source_outputs_exist(summary_path, csv_path):
            _emit_progress(config, "  existing per-source outputs found; skipping")
            skipped_files.append(source_path.name)
            continue

        extractor = extractors.get(source_path.suffix.lower())
        if extractor is None:
            _emit_progress(config, f"  unsupported file type: {source_path.suffix}")
            append_log(config.unresolved_log, f"{source_id}\t{source_path.name}\tUnsupported file type")
            unresolved_count += 1
            continue

        if source_path.suffix.lower() == ".pdf":
            mineru_started = time.perf_counter()
            stage_dir = _mineru_stage_dir(config.staged_dir, source_id)
            if config.mineru_cache_mode == "skip":
                _emit_progress(config, "  MinerU cache bypassed")
            elif config.mineru_cache_mode == "refresh":
                _emit_progress(config, "  MinerU cache refresh requested")
            elif _mineru_artifacts_complete(stage_dir, source_id):
                _emit_progress(config, "  MinerU cache hit")
            else:
                _emit_progress(config, "  MinerU cache missing or incomplete; attempting extraction")
            mineru_result = _ensure_mineru_artifacts(source_path, source_id, config)
            if mineru_result:
                _emit_progress(config, f"  {mineru_result.split(chr(9))[-1]}")
                append_log(config.unresolved_log, mineru_result)
                unresolved_count += 1
            _emit_progress(config, f"  MinerU stage completed in {_format_elapsed(mineru_started)}", level=2)

        extract_started = time.perf_counter()
        result = extractor.extract(source_path, source_id)
        _emit_progress(config, f"  extracted {len(result.sample_points)} candidate point(s)")
        _emit_progress(config, f"  extraction/inference completed in {_format_elapsed(extract_started)}", level=2)
        validation_started = time.perf_counter()
        normalized_points = []
        validation_errors: list[str] = []
        for point in result.sample_points:
            point = apply_elevation_normalization(point, config.raster_path)
            point.confidence_score = score_confidence(point)
            point_errors = validate_sample_point(point, ontology, schema)
            if point_errors:
                validation_errors.extend(point_errors)
                _emit_progress(config, f"  rejected {point.sample_id}: {'; '.join(point_errors)}")
                append_log(
                    config.unresolved_log,
                    f"{source_id}\t{source_path.name}\tRejected candidate SamplePoint `{point.sample_id}`: {'; '.join(point_errors)}"
                )
                unresolved_count += 1
                continue
            _emit_progress(
                config,
                f"  accepted candidate {point.sample_id} ({point.site_name}; {point.latitude}, {point.longitude})",
                level=3,
            )
            normalized_points.append(point)
        _emit_progress(config, f"  validation/normalization completed in {_format_elapsed(validation_started)}", level=2)

        summary_lines = list(result.summary_lines)
        if validation_errors:
            summary_lines.extend(["", "## Validation errors"])
            summary_lines.extend([f"- {error}" for error in validation_errors])

        write_summary(summary_path, summary_lines)
        write_csv(csv_path, normalized_points)

        for line in result.unresolved_lines:
            append_log(config.unresolved_log, line)
            unresolved_count += 1

        processed_files.append(source_path.name)
        extracted_records += len(normalized_points)
        _emit_progress(config, f"  accepted {len(normalized_points)} point(s)")
        _emit_progress(config, f"  file completed in {_format_elapsed(file_started)}", level=2)

    merge_started = time.perf_counter()
    _emit_progress(config, f"Building merged outputs ({config.merged_mode})")
    csv_path, geojson_path, merged_count = build_master_outputs(config.per_source_dir, config.merged_dir, mode=config.merged_mode)
    _emit_progress(config, f"Merge completed in {_format_elapsed(merge_started)}", level=2)
    _write_processing_report(
        config.report_path,
        ontology,
        processed_files,
        skipped_files,
        extracted_records,
        merged_count,
        unresolved_count,
        csv_path,
        geojson_path,
        config.staged_dir,
        config.workspace_root,
    )
    _emit_progress(config, f"Done: {extracted_records} extracted, {len(skipped_files)} skipped, {merged_count} merged, {unresolved_count} unresolved")
    _emit_progress(config, f"Pipeline completed in {_format_elapsed(pipeline_started)}", level=2)


def _extractor_registry() -> dict[str, BaseExtractor]:
    extractors: list[BaseExtractor] = [CsvExtractor(), TextExtractor(), PdfExtractor()]
    registry: dict[str, BaseExtractor] = {}
    for extractor in extractors:
        for suffix in extractor.supported_suffixes:
            registry[suffix] = extractor
    return registry


def _ensure_mineru_artifacts(source_path: Path, source_id: str, config: PipelineConfig) -> str:
    settings = load_extraction_settings(source_path)
    mineru_settings = settings.get("mineru", {})
    if not mineru_settings.get("enabled", True):
        return ""

    stage_dir = _mineru_stage_dir(config.staged_dir, source_id)
    if config.mineru_cache_mode == "skip":
        return ""
    if config.mineru_cache_mode == "reuse" and mineru_settings.get("skip_existing", True) and _mineru_artifacts_complete(stage_dir, source_id):
        return ""
    if config.mineru_cache_mode == "refresh" and stage_dir.exists():
        _clear_directory(stage_dir)

    config.staged_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(mineru_settings.get("command", "mineru")),
        "-p",
        str(source_path),
        "-o",
        str(config.staged_dir),
    ]
    for option, flag in {
        "backend": "--backend",
        "method": "--method",
        "lang": "--lang",
    }.items():
        value = mineru_settings.get(option)
        if value:
            command.extend([flag, str(value)])

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=int(mineru_settings.get("timeout_seconds", 3600)),
        )
    except FileNotFoundError:
        return f"{source_id}\t{source_path.name}\tMinerU command not found; falling back to legacy PDF extraction"
    except subprocess.TimeoutExpired:
        return f"{source_id}\t{source_path.name}\tMinerU timed out; falling back to legacy PDF extraction"
    except subprocess.CalledProcessError as error:
        stderr = " ".join((error.stderr or error.stdout or "").split())[:500]
        return f"{source_id}\t{source_path.name}\tMinerU failed; falling back to legacy PDF extraction: {stderr}"

    if not _mineru_artifacts_complete(stage_dir, source_id):
        return f"{source_id}\t{source_path.name}\tMinerU completed but expected staged artifacts were not found"
    return ""


def _mineru_stage_dir(staged_dir: Path, source_id: str) -> Path:
    return staged_dir / source_id / "hybrid_auto"


def _mineru_artifacts_complete(stage_dir: Path, source_id: str) -> bool:
    required_paths = [
        stage_dir / f"{source_id}.md",
        stage_dir / f"{source_id}_content_list.json",
        stage_dir / f"{source_id}_middle.json",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required_paths)


def _iter_input_files(directory: Path):
    if not directory.exists():
        return []
    return [path for path in directory.iterdir() if path.is_file() and not path.name.startswith(".")]


def _selected_input_files(config: PipelineConfig) -> list[Path]:
    source_paths = sorted(_iter_input_files(config.incoming_dir))
    if config.source_ids:
        wanted = {_normalize_source_selector(selector) for selector in config.source_ids}
        source_paths = [path for path in source_paths if _source_id_for_path(path) in wanted or path.name.lower() in wanted]
    if config.limit is not None:
        source_paths = source_paths[: max(config.limit, 0)]
    return source_paths


def _clear_directory(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _per_source_output_paths(per_source_dir: Path, source_id: str) -> tuple[Path, Path]:
    return per_source_dir / f"{source_id}.summary.md", per_source_dir / f"{source_id}.csv"


def _per_source_outputs_exist(summary_path: Path, csv_path: Path) -> bool:
    return summary_path.exists() or csv_path.exists()


def _source_id_for_path(path: Path) -> str:
    return path.stem.replace(" ", "_").lower()


def _normalize_source_selector(selector: str) -> str:
    return selector.strip().replace(" ", "_").lower()


def _emit_progress(config: PipelineConfig, message: str, level: int = 1) -> None:
    if config.progress_callback is not None and config.verbosity >= level:
        config.progress_callback(message)


def _format_elapsed(started: float) -> str:
    elapsed = time.perf_counter() - started
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    minutes, seconds = divmod(elapsed, 60)
    return f"{int(minutes)}m {seconds:.1f}s"


def _write_processing_report(
    path: Path,
    ontology: Ontology,
    processed_files: list[str],
    skipped_files: list[str],
    extracted_records: int,
    merged_count: int,
    unresolved_count: int,
    csv_path: Path,
    geojson_path: Path,
    staged_dir: Path,
    workspace_root: Path,
) -> None:
    lines = [
        "# LITTORAL Processing Report",
        "",
        f"- Ontology version: `{ontology.version}`",
        f"- Files processed: `{len(processed_files)}`",
        f"- Files skipped with existing per-source outputs: `{len(skipped_files)}`",
        f"- Records extracted before merge: `{extracted_records}`",
        f"- Records in merged dataset: `{merged_count}`",
        f"- Unresolved entries logged: `{unresolved_count}`",
        f"- Master CSV: `{_display_path(csv_path, workspace_root)}`",
        f"- Master GeoJSON: `{_display_path(geojson_path, workspace_root)}`",
        f"- MinerU staged artifacts: `{_display_path(staged_dir, workspace_root)}`",
        "",
        "## Files processed",
    ]
    if processed_files:
        lines.extend([f"- `{name}`" for name in processed_files])
    else:
        lines.append("- No files were processed in this run.")
    lines.extend(["", "## Files skipped"])
    if skipped_files:
        lines.extend([f"- `{name}`" for name in skipped_files])
    else:
        lines.append("- No files were skipped due to existing per-source outputs.")
    lines.extend(
        [
            "",
            "## Processing Notes",
            "- Unsupported file types are logged to `logs/UnresolvedRecords.log`.",
            "- Records failing validation are rejected and logged to `logs/UnresolvedRecords.log`.",
            "- MinerU staged Markdown/content-list artifacts are reused for PDFs when present; MinerU runs only for missing or incomplete staged artifacts.",
            "- Legacy PDF OCR remains a fallback if MinerU cannot run or does not produce the expected cache.",
            "- Ollama-assisted interpretation is configured through `config/extraction.json`.",
            "",
            "## Scientific Review Notes",
            "- Approximate geocodes derived from title, locality, table, caption, or context text should be treated as regional locality anchors rather than surveyed coordinates.",
            "- Any unresolved coordinate, datum, age, or indicator interpretation remains in the unresolved log until specialized extraction rules or human review resolve it.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path)
