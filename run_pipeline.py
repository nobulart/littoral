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


DEFAULT_INCOMING_DIR = Path("data/incoming")
DEFAULT_STAGED_DIR = Path("data/staged")
DEFAULT_PER_SOURCE_DIR = Path("outputs/per_source")
DEFAULT_MERGED_DIR = Path("outputs/merged")
DEFAULT_UNRESOLVED_LOG = Path("logs/UnresolvedRecords.log")
DEFAULT_REPORT_PATH = Path("logs/processing_report.md")
DEFAULT_ONTOLOGY_PATH = Path("config/categories.json")
DEFAULT_SCHEMA_PATH = Path("config/schema.samplepoint.json")
DEFAULT_RASTER_PATH = Path("SRTM15+V2.tiff")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace_root = args.workspace_root.resolve()

    if args.fast_test:
        os.environ["LITTORAL_FAST_TEST"] = "1"

    config = _build_config(args, workspace_root)
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
    parser.add_argument("--ontology-path", type=Path, default=DEFAULT_ONTOLOGY_PATH, help="Path to ontology categories JSON.")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to SamplePoint schema JSON.")
    parser.add_argument("--raster-path", type=Path, default=DEFAULT_RASTER_PATH, help="Path to elevation raster.")
    parser.add_argument("--source", action="append", dest="source_ids", help="Process only this source id or filename. May be repeated.")
    parser.add_argument("--limit", type=int, help="Process only the first N selected input files.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear output directories before running.")
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
        ontology_path=_resolve_path(args.ontology_path, workspace_root),
        schema_path=_resolve_path(args.schema_path, workspace_root),
        raster_path=_resolve_path(args.raster_path, workspace_root),
        clear_outputs=not args.no_clear,
        source_ids=source_ids,
        limit=limit,
        progress_callback=None if verbosity == 0 else _print_progress,
        verbosity=verbosity,
    )


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
    )


if __name__ == "__main__":
    raise SystemExit(main())
