from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.orchestrate.pipeline import (
    PipelineConfig,
    _mineru_artifacts_complete,
    _mineru_stage_dir,
    _selected_input_files,
    _source_id_for_path,
    default_config,
    run_pipeline,
)
from src.orchestrate.locking import PipelineLockManager


DEFAULT_INCOMING_DIR = Path("data/incoming")
DEFAULT_STAGED_DIR = Path("data/staged")
DEFAULT_PER_SOURCE_DIR = Path("outputs/per_source")
DEFAULT_MERGED_DIR = Path("outputs/merged")
DEFAULT_UNRESOLVED_LOG = Path("logs/UnresolvedRecords.log")
DEFAULT_REPORT_PATH = Path("logs/processing_report.md")
DEFAULT_LOCK_DIR = Path("locks")
DEFAULT_ONTOLOGY_PATH = Path("config/categories.json")
DEFAULT_SCHEMA_PATH = Path("config/schema.samplepoint.json")
DEFAULT_RASTER_PATH = Path("data/elevation/SRTM15+V2.tiff")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace_root = args.workspace_root.resolve()

    if args.fast_test:
        os.environ["LITTORAL_FAST_TEST"] = "1"

    config = _build_config(args, workspace_root)
    if args.force_kill_source:
        return _force_kill_sources(config, args.force_kill_source)
    if args.check_mineru_cache:
        return _check_mineru_cache(config)

    if _uses_default_layout(args) and _create_missing_default_dirs(config):
        return 0

    run_pipeline(config)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LITTORAL coastal and relative sea-level evidence extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("document", nargs="?", type=Path, help="Optional single source document to process instead of the default input batch.")
    parser.add_argument("--workspace-root", type=Path, default=Path("."), help="Repository/workspace root.")
    parser.add_argument("--incoming-dir", type=Path, default=DEFAULT_INCOMING_DIR, help="Directory containing source files to process.")
    parser.add_argument("--staged-dir", type=Path, default=DEFAULT_STAGED_DIR, help="Directory containing MinerU staged outputs.")
    parser.add_argument("--per-source-dir", type=Path, default=DEFAULT_PER_SOURCE_DIR, help="Directory for per-source summaries and CSVs.")
    parser.add_argument("--merged-dir", type=Path, default=DEFAULT_MERGED_DIR, help="Directory for merged CSV/GeoJSON outputs.")
    parser.add_argument("--unresolved-log", type=Path, default=DEFAULT_UNRESOLVED_LOG, help="Path for unresolved/rejected record log.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Path for the processing report.")
    parser.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR, help="Directory for non-hidden per-source and merge lock/status files.")
    parser.add_argument("--ontology-path", type=Path, default=DEFAULT_ONTOLOGY_PATH, help="Path to ontology categories JSON.")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to SamplePoint schema JSON.")
    parser.add_argument("--raster-path", type=Path, default=DEFAULT_RASTER_PATH, help="Path to elevation raster.")
    parser.add_argument("--source", action="append", dest="source_ids", help="Process only this source id or filename. May be repeated.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected input files.")
    parser.add_argument("--document-workers", type=int, help="Number of documents to process concurrently.")
    parser.add_argument("--gpu-slots", type=int, help="Maximum concurrent GPU-bound tasks across MinerU/Ollama/OCR.")
    parser.add_argument("--job-timeout-minutes", type=float, default=240.0, help="Maximum wall-clock minutes for an individual document job. Use 0 to disable.")
    parser.add_argument("--autopilot", action="store_true", help="Start with autopilot enabled. In the ncurses dashboard, press 'a' to toggle it on or off.")
    parser.add_argument("--force-kill-source", action="append", help="Force-release a source by id or filename stem, kill any local owned worker PID, remove its lease, then exit. May be repeated.")
    parser.add_argument("--no-control-api", action="store_true", help="Disable the hybrid control-plane API and filesystem node registry.")
    parser.add_argument("--control-api-bind-host", default="0.0.0.0", help="Host/interface to bind the local control-plane API server to.")
    parser.add_argument("--control-api-advertise-host", help="Hostname or IP to advertise to peer workstations in the filesystem node registry.")
    parser.add_argument("--control-api-port", type=int, default=0, help="Port for the local control-plane API server. Use 0 to auto-select.")
    parser.add_argument("--control-api-heartbeat-seconds", type=float, default=10.0, help="How often to publish the control-plane node registry heartbeat.")
    parser.add_argument("--control-api-node-ttl-seconds", type=float, default=45.0, help="How recently a node registry entry must have changed to be considered fresh by filesystem-only readers.")
    parser.add_argument("--progress-ui", choices=("auto", "plain", "ncurses"), default="auto", help="Progress display mode.")
    parser.add_argument("--no-clear", action="store_true", help="Retained for compatibility; outputs are protected by default.")
    parser.add_argument("--clear-outputs", action="store_true", help="Delete per-source and merged outputs before running.")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing per-source and merged outputs unless a step-specific mode is supplied.")
    parser.add_argument(
        "--per-source-mode",
        choices=("skip", "overwrite"),
        help="How to handle existing per-source summary/CSV outputs.",
    )
    parser.add_argument(
        "--overwrite-per-source",
        action="store_const",
        const="overwrite",
        dest="per_source_mode",
        help="Overwrite existing per-source summary/CSV outputs.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("append", "overwrite", "skip"),
        help="How to handle existing merged master outputs.",
    )
    parser.add_argument(
        "--append-merged",
        action="store_const",
        const="append",
        dest="merge_mode",
        help="Append new records to existing merged outputs.",
    )
    parser.add_argument(
        "--overwrite-merged",
        action="store_const",
        const="overwrite",
        dest="merge_mode",
        help="Rebuild and overwrite merged master outputs.",
    )
    parser.add_argument(
        "--skip-merged",
        action="store_const",
        const="skip",
        dest="merge_mode",
        help="Leave existing merged master outputs unchanged.",
    )
    parser.add_argument(
        "--mineru-cache-mode",
        choices=("reuse", "refresh", "skip"),
        help="How to manage MinerU staged artifacts for PDFs.",
    )
    parser.add_argument(
        "--refresh-mineru-cache",
        action="store_const",
        const="refresh",
        dest="mineru_cache_mode",
        help="Regenerate MinerU staged artifacts for processed PDFs.",
    )
    parser.add_argument(
        "--skip-mineru-cache",
        action="store_const",
        const="skip",
        dest="mineru_cache_mode",
        help="Do not read or create MinerU staged artifacts during this run.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output. Equivalent to --verbosity 0.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase progress detail. Use -v for timings and -vv for per-candidate diagnostics.")
    parser.add_argument("--verbosity", type=int, choices=range(0, 4), metavar="{0,1,2,3}", help="Set progress verbosity directly: 0=quiet, 1=normal, 2=timed stages, 3=per-candidate diagnostics.")
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Skip MinerU, Ollama, and geocoding calls. Useful for testing cached parsing and merge logic.",
    )
    parser.add_argument(
        "--check-mineru-cache",
        action="store_true",
        help="Report which selected PDFs have complete MinerU staged artifacts, then exit.",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace, workspace_root: Path) -> PipelineConfig:
    incoming_dir = _resolve_path(args.incoming_dir, workspace_root)
    source_ids = tuple(args.source_ids) if args.source_ids else None
    limit = args.limit
    verbosity = _resolve_verbosity(args)
    if args.document is not None:
        document_path = _resolve_path(args.document, workspace_root)
        incoming_dir = document_path.parent
        source_ids = (document_path.name.lower(),)
        limit = 1

    return PipelineConfig(
        workspace_root=workspace_root,
        incoming_dir=incoming_dir,
        staged_dir=_resolve_path(args.staged_dir, workspace_root),
        per_source_dir=_resolve_path(args.per_source_dir, workspace_root),
        merged_dir=_resolve_path(args.merged_dir, workspace_root),
        unresolved_log=_resolve_path(args.unresolved_log, workspace_root),
        report_path=_resolve_path(args.report_path, workspace_root),
        lock_dir=_resolve_path(args.lock_dir, workspace_root),
        ontology_path=_resolve_path(args.ontology_path, workspace_root),
        schema_path=_resolve_path(args.schema_path, workspace_root),
        raster_path=_resolve_path(args.raster_path, workspace_root),
        clear_outputs=args.clear_outputs,
        per_source_mode=args.per_source_mode or ("overwrite" if args.overwrite_existing else "skip"),
        merged_mode=args.merge_mode or ("overwrite" if args.overwrite_existing else "append"),
        mineru_cache_mode=args.mineru_cache_mode or "reuse",
        source_ids=source_ids,
        limit=limit,
        document_workers=args.document_workers,
        gpu_slots=args.gpu_slots,
        progress_ui=args.progress_ui,
        progress_callback=None if verbosity == 0 or _uses_dashboard_output(args.progress_ui) else _print_progress,
        verbosity=verbosity,
        job_timeout_minutes=args.job_timeout_minutes,
        autopilot_enabled=args.autopilot,
        control_api_enabled=not args.no_control_api,
        control_api_bind_host=args.control_api_bind_host,
        control_api_advertise_host=args.control_api_advertise_host,
        control_api_port=args.control_api_port,
        control_api_heartbeat_seconds=args.control_api_heartbeat_seconds,
        control_api_node_ttl_seconds=args.control_api_node_ttl_seconds,
    )


def _force_kill_sources(config: PipelineConfig, source_ids: list[str]) -> int:
    lock_manager = PipelineLockManager(config.lock_dir)
    lock_manager.ensure_dirs()
    for source_id in source_ids:
        normalized = _source_id_for_path(Path(source_id))
        result = lock_manager.force_release_source(
            normalized,
            workspace_root=config.workspace_root,
            reason="force killed by command line",
        )
        print(
            "force-killed "
            f"{result.source_id}: lease_found={result.lease_found} removed_lease={result.removed_lease} "
            f"killed_pids={result.killed_pids} skipped_pids={result.skipped_pids}"
        )
    return 0


def _resolve_verbosity(args: argparse.Namespace) -> int:
    if args.quiet:
        return 0
    if args.verbosity is not None:
        return args.verbosity
    return min(3, 1 + int(args.verbose or 0))


def _resolve_path(path: Path, workspace_root: Path) -> Path:
    if path.is_absolute():
        return path
    return workspace_root / path


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _uses_dashboard_output(mode: str) -> bool:
    if mode == "plain":
        return False
    if mode == "ncurses":
        return True
    if not sys.stdout.isatty():
        return False
    if os.environ.get("TERM", "") in {"", "dumb"}:
        return False
    try:
        import curses  # noqa: F401
    except ImportError:
        return False
    return True


def _check_mineru_cache(config: PipelineConfig) -> int:
    missing: list[str] = []
    source_paths = [path for path in _selected_input_files(config) if path.suffix.lower() == ".pdf"]
    print(f"Checking MinerU cache for {len(source_paths)} PDF file(s)")
    for source_path in source_paths:
        source_id = _source_id_for_path(source_path)
        stage_dir = _mineru_stage_dir(config.staged_dir, source_id)
        complete = _mineru_artifacts_complete(stage_dir, source_id)
        status = "ok" if complete else "missing"
        print(f"{status:7} {source_id:20} {stage_dir}")
        if not complete:
            missing.append(source_id)
    if missing:
        print(f"Missing or incomplete MinerU cache for {len(missing)} source(s): {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


def _create_missing_default_dirs(config: PipelineConfig) -> bool:
    directories = [
        config.incoming_dir,
        config.staged_dir,
        config.per_source_dir,
        config.merged_dir,
        config.unresolved_log.parent,
        config.report_path.parent,
        config.lock_dir,
    ]
    missing = [directory for directory in directories if not directory.exists()]
    if not missing:
        return False

    for directory in missing:
        directory.mkdir(parents=True, exist_ok=True)

    print("Created missing pipeline directories:")
    for directory in missing:
        print(f"- {directory}")
    print("Directory setup is complete. Add source files and staged MinerU artifacts as needed, then rerun the command.")
    return True


def _uses_default_layout(args: argparse.Namespace) -> bool:
    return (
        args.incoming_dir == DEFAULT_INCOMING_DIR
        and args.staged_dir == DEFAULT_STAGED_DIR
        and args.per_source_dir == DEFAULT_PER_SOURCE_DIR
        and args.merged_dir == DEFAULT_MERGED_DIR
        and args.unresolved_log == DEFAULT_UNRESOLVED_LOG
        and args.report_path == DEFAULT_REPORT_PATH
        and args.lock_dir == DEFAULT_LOCK_DIR
    )


if __name__ == "__main__":
    raise SystemExit(main())
