#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract.walis_ingest import DEFAULT_WALIS_SUMMARY, QUALITY_MODES, SOURCE_ID, ingest_walis_summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workspace_root = args.workspace_root.resolve()
    summary_csv = _resolve(args.summary_csv, workspace_root)
    per_source_dir = _resolve(args.per_source_dir, workspace_root)
    merged_dir = _resolve(args.merged_dir, workspace_root)

    result = ingest_walis_summary(
        summary_csv,
        per_source_dir,
        source_id=args.source_id,
        quality_mode=args.quality_mode,
        include_limiting=args.include_limiting,
        merge=args.merge,
        merged_dir=merged_dir,
        merge_mode=args.merge_mode,
    )

    print(f"WALIS rows seen: {result.rows_seen}")
    print(f"SamplePoints written: {result.points_written}")
    print(f"Rows skipped: {result.rows_skipped}")
    print(f"Per-source CSV: {result.per_source_csv}")
    print(f"Summary: {result.summary_path}")
    if result.merged_csv and result.merged_geojson:
        print(f"Merged CSV: {result.merged_csv}")
        print(f"Merged GeoJSON: {result.merged_geojson}")
        print(f"Merged record count: {result.merged_count}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the local WALIS post-review Summary.csv export into canonical LITTORAL SamplePoint outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace-root", type=Path, default=PROJECT_ROOT, help="Repository/workspace root.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_WALIS_SUMMARY, help="Path to WALIS Summary.csv.")
    parser.add_argument("--per-source-dir", type=Path, default=Path("outputs/per_source"), help="Directory for canonical per-source outputs.")
    parser.add_argument("--merged-dir", type=Path, default=Path("outputs/merged"), help="Directory for merged outputs when --merge is set.")
    parser.add_argument("--source-id", default=SOURCE_ID, help="LITTORAL source_id to assign to imported WALIS rows.")
    parser.add_argument("--quality-mode", choices=sorted(QUALITY_MODES), default="accepted", help="WALIS row quality filter.")
    parser.add_argument("--include-limiting", action="store_true", help="Include marine/terrestrial limiting WALIS rows.")
    parser.add_argument("--merge", action="store_true", help="Rebuild or append master merged CSV/GeoJSON after writing the WALIS per-source CSV.")
    parser.add_argument("--merge-mode", choices=("append", "overwrite", "skip"), default="append", help="Merge behavior passed to LITTORAL merge builder.")
    return parser.parse_args(argv)


def _resolve(path: Path, workspace_root: Path) -> Path:
    if path.is_absolute():
        return path
    return workspace_root / path


if __name__ == "__main__":
    raise SystemExit(main())
