from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import time
from collections.abc import Callable

from src.common.io import append_log, write_csv, write_summary, write_text_atomic
from src.extract.base import BaseExtractor
from src.extract.csv_extractor import CsvExtractor
from src.extract.pdf_extractor import PdfExtractor
from src.extract.settings import load_extraction_settings
from src.extract.txt_extractor import TextExtractor
from src.merge.build import build_master_outputs
from src.normalize.elevation import apply_elevation_normalization
from src.ontology.catalog import Ontology, load_ontology
from src.orchestrate.locking import LeaseDenied, ManagedLease, PipelineLockManager
from src.orchestrate.progress import PipelineProgressReporter
from src.orchestrate.runtime import PipelineRuntime, auto_document_workers, auto_gpu_slots, hardware_profile_summary
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
    lock_dir: Path
    ontology_path: Path
    schema_path: Path
    raster_path: Path
    clear_outputs: bool = False
    per_source_mode: str = "skip"
    merged_mode: str = "append"
    mineru_cache_mode: str = "reuse"
    source_ids: tuple[str, ...] | None = None
    limit: int | None = None
    document_workers: int | None = None
    gpu_slots: int | None = None
    progress_ui: str = "auto"
    progress_callback: Callable[[str], None] | None = None
    verbosity: int = 1


@dataclass(slots=True)
class FileProcessingResult:
    index: int
    source_path: Path
    source_id: str
    summary_lines: list[str]
    normalized_points: list = field(default_factory=list)
    unresolved_lines: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)


def default_config(workspace_root: Path) -> PipelineConfig:
    return PipelineConfig(
        workspace_root=workspace_root,
        incoming_dir=workspace_root / "data" / "incoming",
        staged_dir=workspace_root / "data" / "staged",
        per_source_dir=workspace_root / "outputs" / "per_source",
        merged_dir=workspace_root / "outputs" / "merged",
        unresolved_log=workspace_root / "logs" / "UnresolvedRecords.log",
        report_path=workspace_root / "logs" / "processing_report.md",
        lock_dir=workspace_root / "locks",
        ontology_path=workspace_root / "config" / "categories.json",
        schema_path=workspace_root / "config" / "schema.samplepoint.json",
        raster_path=workspace_root / "data" / "elevation" / "SRTM15+V2.tiff",
    )


def run_pipeline(config: PipelineConfig) -> None:
    pipeline_started = time.perf_counter()
    reporter = PipelineProgressReporter(total_files=len(_selected_input_files(config)), mode=config.progress_ui, enabled=config.verbosity > 0)
    def emit(message: str, level: int = 1) -> None:
        if config.verbosity >= level and config.progress_callback is None:
            reporter.emit_global(message)
        _emit_progress(config, message, level=level)

    try:
        emit("Loading ontology and schema")
        ontology = load_ontology(config.ontology_path)
        schema = load_schema(config.schema_path)
        runtime = PipelineRuntime(
            ontology=ontology,
            gpu_slots=config.gpu_slots or auto_gpu_slots(),
        )
        lock_manager = PipelineLockManager(config.lock_dir)
        lock_manager.ensure_dirs()
        extractors = _extractor_registry()
        processed_files: list[str] = []
        skipped_files: list[str] = []
        extracted_records = 0
        unresolved_count = 0

        config.per_source_dir.mkdir(parents=True, exist_ok=True)
        config.merged_dir.mkdir(parents=True, exist_ok=True)
        config.unresolved_log.parent.mkdir(parents=True, exist_ok=True)
        if config.clear_outputs:
            emit("Clearing previous outputs")
            _clear_directory(config.per_source_dir)
            _clear_directory(config.merged_dir)
            config.unresolved_log.write_text("", encoding="utf-8")
        elif not config.unresolved_log.exists():
            config.unresolved_log.write_text("", encoding="utf-8")

        source_paths = _selected_input_files(config)
        emit(f"Processing {len(source_paths)} input file(s)")
        emit(f"Inputs: {config.incoming_dir}", level=2)
        emit(f"MinerU staged cache: {config.staged_dir}", level=2)
        emit(f"Per-source outputs: {config.per_source_dir}", level=2)
        document_workers = config.document_workers or auto_document_workers()
        emit(f"Concurrency profile: {hardware_profile_summary(document_workers, runtime.gpu_slots)}", level=2)
        work_items: list[tuple[int, Path, str, BaseExtractor]] = []
        source_index_map: dict[str, tuple[int, str]] = {}
        for index, source_path in enumerate(source_paths, start=1):
            source_id = _source_id_for_path(source_path)
            source_index_map[source_id] = (index, source_path.name)
            reporter.queue_file(index, source_path.name)
            lock_manager.note_discovered(source_id, source_path.name, source_path)

            extractor = extractors.get(source_path.suffix.lower())
            if extractor is None:
                reporter.mark_unsupported(index, source_path.name, f"unsupported file type: {source_path.suffix}")
                lock_manager.mark_source_state(
                    source_id,
                    source_path.name,
                    source_path,
                    status="unsupported",
                    stage="unsupported",
                    detail=f"unsupported file type: {source_path.suffix}",
                )
                append_log(config.unresolved_log, f"{source_id}\t{source_path.name}\tUnsupported file type")
                unresolved_count += 1
                continue
            work_items.append((index, source_path, source_id, extractor))

        if work_items:
            with ThreadPoolExecutor(max_workers=max(1, min(document_workers, len(work_items)))) as executor:
                pending = deque(work_items)
                active_futures: dict = {}
                while pending or active_futures:
                    _refresh_reporter_from_shared_statuses(reporter, lock_manager, source_index_map)
                    reporter.tick()
                    if reporter.abort_requested():
                        pending.clear()
                    pending = deque(
                        item for item in pending if not reporter.is_cancelled_before_dispatch(item[0])
                    )
                    while (
                        pending
                        and len(active_futures) < max(1, min(document_workers, len(work_items)))
                        and not reporter.pause_requested()
                        and not reporter.stop_requested()
                    ):
                        next_index = reporter.pick_pending_index([item[0] for item in pending])
                        if next_index is None:
                            pending.clear()
                            break
                        selected_position = next(i for i, item in enumerate(pending) if item[0] == next_index)
                        index, source_path, source_id, extractor = pending[selected_position]
                        del pending[selected_position]
                        summary_path, csv_path = _per_source_output_paths(config.per_source_dir, source_id)
                        lease = lock_manager.claim_source(
                            source_id,
                            source_path.name,
                            source_path,
                            summary_path=summary_path,
                            csv_path=csv_path,
                            per_source_mode=config.per_source_mode,
                        )
                        if isinstance(lease, LeaseDenied):
                            reporter.skip_file(index, source_path.name, lease.reason)
                            skipped_files.append(source_path.name)
                            continue
                        reporter.note_dispatch_started(index)
                        future = executor.submit(
                            _process_source_file,
                            index,
                            source_path,
                            source_id,
                            extractor,
                            config,
                            runtime,
                            ontology,
                            schema,
                            reporter,
                            lease,
                        )
                        active_futures[future] = lease
                    if not active_futures:
                        if pending and reporter.pause_requested():
                            time.sleep(0.1)
                            continue
                        if pending and reporter.stop_requested():
                            pending.clear()
                            continue
                        break
                    done, remaining = wait(set(active_futures.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                    future_map = active_futures
                    active_futures = {future: future_map[future] for future in remaining}
                    for future in done:
                        lease = future_map[future]
                        try:
                            result = future.result()
                        except Exception as error:
                            lease.fail(f"processing failed: {error.__class__.__name__}: {error}")
                            raise
                        summary_path, csv_path = _per_source_output_paths(config.per_source_dir, result.source_id)
                        lease.update("publish", "publishing per-source outputs")
                        write_summary(summary_path, result.summary_lines)
                        write_csv(csv_path, result.normalized_points)
                        for line in result.unresolved_lines:
                            append_log(config.unresolved_log, line)
                            unresolved_count += 1
                        lease.complete(
                            f"accepted {len(result.normalized_points)} point(s)",
                            extra={
                                "accepted_points": len(result.normalized_points),
                                "unresolved_entries": len(result.unresolved_lines),
                            },
                        )
                        reporter.complete_file(result.index, result.source_path.name, len(result.normalized_points))
                        processed_files.append(result.source_path.name)
                        extracted_records += len(result.normalized_points)
                _refresh_reporter_from_shared_statuses(reporter, lock_manager, source_index_map)

        merge_started = time.perf_counter()
        emit(f"Building merged outputs ({config.merged_mode})")
        merge_lease = lock_manager.acquire_merge_lease("master_dataset")
        try:
            merge_lease.update("merge", f"building merged outputs ({config.merged_mode})")
            csv_path, geojson_path, merged_count = build_master_outputs(config.per_source_dir, config.merged_dir, mode=config.merged_mode)
            merge_lease.complete(
                f"merge completed in {_format_elapsed(merge_started)}",
                extra={
                    "master_csv_path": str(csv_path),
                    "master_geojson_path": str(geojson_path),
                    "merged_count": merged_count,
                },
            )
        except Exception as error:
            merge_lease.fail(f"merge failed: {error.__class__.__name__}: {error}")
            raise
        emit(f"Merge completed in {_format_elapsed(merge_started)}", level=2)
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
        emit(f"Done: {extracted_records} extracted, {len(skipped_files)} skipped, {merged_count} merged, {unresolved_count} unresolved")
        emit(f"Pipeline completed in {_format_elapsed(pipeline_started)}", level=2)
    finally:
        reporter.close()


def _extractor_registry() -> dict[str, BaseExtractor]:
    extractors: list[BaseExtractor] = [CsvExtractor(), TextExtractor(), PdfExtractor()]
    registry: dict[str, BaseExtractor] = {}
    for extractor in extractors:
        for suffix in extractor.supported_suffixes:
            registry[suffix] = extractor
    return registry


def _ensure_mineru_artifacts(source_path: Path, source_id: str, config: PipelineConfig) -> str:
    return _ensure_mineru_artifacts_with_runtime(source_path, source_id, config, runtime=None)


def _ensure_mineru_artifacts_with_runtime(
    source_path: Path,
    source_id: str,
    config: PipelineConfig,
    runtime: PipelineRuntime | None,
) -> str:
    settings = runtime.settings_for(source_path) if runtime is not None else load_extraction_settings(source_path)
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
        context = runtime.gpu_task() if runtime is not None else _null_gpu_task()
        with context:
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


def _refresh_reporter_from_shared_statuses(
    reporter: PipelineProgressReporter,
    lock_manager: PipelineLockManager,
    source_index_map: dict[str, tuple[int, str]],
) -> None:
    for source_id, payload in lock_manager.list_source_statuses().items():
        mapping = source_index_map.get(source_id)
        if mapping is None:
            continue
        index, name = mapping
        reporter.sync_shared_state(
            index,
            name,
            status=str(payload.get("status") or "queued"),
            stage=str(payload.get("stage") or payload.get("status") or "queued"),
            detail=str(payload.get("detail") or ""),
            started_at=float(payload["started_at"]) if isinstance(payload.get("started_at"), (int, float)) else None,
            finished_at=float(payload["finished_at"]) if isinstance(payload.get("finished_at"), (int, float)) else None,
            candidates=int(payload["candidate_points"]) if isinstance(payload.get("candidate_points"), int) else None,
            accepted=int(payload["accepted_points"]) if isinstance(payload.get("accepted_points"), int) else None,
            unresolved=int(payload["unresolved_entries"]) if isinstance(payload.get("unresolved_entries"), int) else None,
        )


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
    write_text_atomic(path, "\n".join(lines).rstrip() + "\n")


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        return str(path)


def _process_source_file(
    index: int,
    source_path: Path,
    source_id: str,
    extractor: BaseExtractor,
    config: PipelineConfig,
    runtime: PipelineRuntime,
    ontology: Ontology,
    schema: dict,
    reporter: PipelineProgressReporter,
    lease: ManagedLease,
) -> FileProcessingResult:
    file_started = time.perf_counter()
    local_unresolved_lines: list[str] = []
    reporter.start_file(index, source_path.name)
    lease.update("start", "worker started")

    if source_path.suffix.lower() == ".pdf":
        mineru_started = time.perf_counter()
        stage_dir = _mineru_stage_dir(config.staged_dir, source_id)
        if config.mineru_cache_mode == "skip":
            reporter.update_stage(index, source_path.name, "mineru", "cache bypassed")
            lease.update("mineru", "cache bypassed")
        elif config.mineru_cache_mode == "refresh":
            reporter.update_stage(index, source_path.name, "mineru", "cache refresh requested")
            lease.update("mineru", "cache refresh requested")
        elif _mineru_artifacts_complete(stage_dir, source_id):
            reporter.update_stage(index, source_path.name, "mineru", "cache hit")
            lease.update("mineru", "cache hit")
        else:
            reporter.update_stage(index, source_path.name, "mineru", "cache miss; attempting extraction")
            lease.update("mineru", "cache miss; attempting extraction")
        mineru_result = _ensure_mineru_artifacts_with_runtime(source_path, source_id, config, runtime=runtime)
        if mineru_result:
            reporter.record_unresolved(index, source_path.name, detail=mineru_result.split(chr(9))[-1])
            local_unresolved_lines.append(mineru_result)
        reporter.update_stage(index, source_path.name, "mineru", f"completed in {_format_elapsed(mineru_started)}")
        lease.update("mineru", f"completed in {_format_elapsed(mineru_started)}")

    extract_started = time.perf_counter()
    reporter.update_stage(index, source_path.name, "extract", "running deterministic and inference stages")
    lease.update("extract", "running deterministic and inference stages")
    result = extractor.extract(source_path, source_id, runtime=runtime)
    reporter.update_candidates(index, source_path.name, len(result.sample_points))
    reporter.update_stage(index, source_path.name, "extract", f"completed in {_format_elapsed(extract_started)}")
    lease.update("extract", f"completed in {_format_elapsed(extract_started)}", extra={"candidate_points": len(result.sample_points)})

    validation_started = time.perf_counter()
    reporter.update_stage(index, source_path.name, "validate", "normalizing elevations and validating records")
    lease.update("validate", "normalizing elevations and validating records")
    normalized_points = []
    validation_errors: list[str] = []
    for point in result.sample_points:
        point = apply_elevation_normalization(point, config.raster_path)
        point.confidence_score = score_confidence(point)
        point_errors = validate_sample_point(point, ontology, schema)
        if point_errors:
            validation_errors.extend(point_errors)
            reporter.record_unresolved(index, source_path.name, detail=f"rejected {point.sample_id}: {'; '.join(point_errors)}")
            local_unresolved_lines.append(
                f"{source_id}\t{source_path.name}\tRejected candidate SamplePoint `{point.sample_id}`: {'; '.join(point_errors)}"
            )
            continue
        normalized_points.append(point)
    reporter.update_stage(index, source_path.name, "validate", f"completed in {_format_elapsed(validation_started)}")
    lease.update("validate", f"completed in {_format_elapsed(validation_started)}", extra={"accepted_points": len(normalized_points)})

    summary_lines = list(result.summary_lines)
    if validation_errors:
        summary_lines.extend(["", "## Validation errors"])
        summary_lines.extend([f"- {error}" for error in validation_errors])

    local_unresolved_lines.extend(result.unresolved_lines)
    if result.unresolved_lines:
        reporter.record_unresolved(index, source_path.name, count=len(result.unresolved_lines), detail="source unresolved lines logged")
    reporter.update_stage(index, source_path.name, "publish", "ready for atomic publish")
    lease.update("publish", "ready for atomic publish")

    return FileProcessingResult(
        index=index,
        source_path=source_path,
        source_id=source_id,
        summary_lines=summary_lines,
        normalized_points=normalized_points,
        unresolved_lines=local_unresolved_lines,
        validation_errors=validation_errors,
    )


def _null_gpu_task():
    from contextlib import nullcontext

    return nullcontext()
