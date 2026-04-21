# LITTORAL (Literature-Inferred Terrestrial and Oceanic Relative Altimetry Levels)

LITTORAL is a reproducible extraction pipeline for converting coastal, shelf, and relative sea-level literature into structured geospatial evidence records. The system is designed for scientific synthesis work where useful observations are distributed across narrative text, tables, maps, figures, captions, and OCR-derived page content.

## Scientific Scope

LITTORAL targets observations that constrain past coastal position, relative sea level, shelf exposure, inundation, and associated geomorphic or sedimentary indicators. Candidate records may include submerged shoreline deposits, marine terraces, beach ridges, reef features, wave-cut platforms, estuarine or lagoonal facies, depth-constrained geomorphic surfaces, and mapped coastal landforms.

The pipeline emphasizes provenance and uncertainty. It distinguishes reported observations from derived values, preserves source locators and quotations, records coordinate provenance, and treats geocoded coordinates as approximate unless the source provides precise coordinates.

## Methodological Overview

1. **Document staging**
   Source documents are placed in `data/incoming/`. PDFs are parsed preferentially from MinerU staged artifacts in `data/staged/<source>/hybrid_auto/`.

2. **MinerU cache reuse**
   Existing MinerU outputs are reused when the expected Markdown, content-list JSON, and middle JSON artifacts are present. MinerU is run only for missing or incomplete staged artifacts.

3. **Structured extraction**
   The loader combines MinerU Markdown, structured table and figure metadata, native PDF text, OCR fallback text, and page-level OCR blocks where needed.

4. **Targeted inference**
   Deterministic extractors mine known table and feature patterns. Optional local Ollama models interpret high-value MinerU table, map, figure, and chart contexts when deterministic parsing is insufficient.

5. **Contextual geocoding**
   When exact source coordinates are absent, LITTORAL attempts contextual geocoding from the paper title, locality names, captions, table rows, and nearby descriptive text. These coordinates are written with inferred coordinate provenance and should be interpreted as approximate locality anchors.

6. **Validation and merge**
   Candidate records are validated against the controlled vocabulary and schema before per-source outputs are merged into master CSV and GeoJSON products.

## Data Products

- `outputs/per_source/<source>.summary.md`: source-level extraction report.
- `outputs/per_source/<source>.csv`: validated records for one source.
- `outputs/merged/master_dataset.csv`: merged tabular dataset.
- `outputs/merged/master_dataset.geojson`: merged geospatial dataset.
- `logs/UnresolvedRecords.log`: unsupported files, rejected records, and unresolved extraction cases.
- `logs/processing_report.md`: run-level processing report.

## Configuration

Primary configuration files:

- `config/extraction.json`: MinerU, PDF/OCR, Ollama, and geocoding settings.
- `config/categories.json`: controlled vocabulary for record classes, coordinate sources, depth sources, indicators, and measurement semantics.
- `config/schema.samplepoint.json`: canonical SamplePoint validation schema.

Important settings:

- `mineru.skip_existing`: reuse staged MinerU artifacts when complete.
- `mineru_inference.llm_enabled`: enable targeted local LLM interpretation of MinerU contexts.
- `mineru_inference.max_llm_contexts`: cap expensive targeted LLM calls per document.
- `ollama.model`: local Ollama model used for reasoning.
- `geocoding.max_contextual_queries`: cap gazetteer attempts for inferred coordinates.
- `geocoding.min_delay_seconds`: rate-limit public gazetteer requests.

## Running the Pipeline

Default batch mode processes files in `data/incoming/`:

```bash
python3 run_pipeline.py
```

Process a single document and exit:

```bash
python3 run_pipeline.py data/incoming/brooke2017.pdf
```

Check staged MinerU cache completeness without processing:

```bash
python3 run_pipeline.py --check-mineru-cache
```

Run a fast structural test that skips MinerU, Ollama, and geocoding:

```bash
python3 run_pipeline.py --fast-test
```

The equivalent environment variable is:

```bash
LITTORAL_FAST_TEST=1 python3 run_pipeline.py
```

## Progress and Profiling

Progress output is controlled from the command line:

```bash
python3 run_pipeline.py --verbosity 0  # quiet
python3 run_pipeline.py --verbosity 1  # normal progress
python3 run_pipeline.py --verbosity 2  # timed stages
python3 run_pipeline.py --verbosity 3  # per-candidate diagnostics
```

The shorthand form is also available:

```bash
python3 run_pipeline.py -v   # timed stages
python3 run_pipeline.py -vv  # per-candidate diagnostics
```

MinerU and local LLM inference may each take ten minutes or more for table-rich or figure-rich documents. Timed verbosity is intended to make those costs observable without changing outputs.

## Coordinate Policy

LITTORAL writes exact coordinates when they are reported by the source. When exact coordinates are absent, it may infer approximate coordinates from place names and source context. Inferred coordinates are marked through `coordinate_source`, `coordinate_uncertainty_m`, `notes`, and `source_locator`.

Approximate geocodes are suitable for discovery, mapping, deduplication, and regional synthesis. They should not be treated as surveyed sample positions without human review.

## Reproducibility Notes

- MinerU artifacts are treated as reusable staged data.
- Local model behavior depends on the installed Ollama model and version.
- Public gazetteer results may change over time; coordinate provenance and query strings are retained in record notes.
- Generated outputs should be regenerated after configuration or extraction-rule changes.

## Development Status

LITTORAL is under active development. The present implementation prioritizes transparent provenance, conservative validation, and rapid iteration on MinerU-derived evidence. Specialized parsers can be added for recurring publication formats as new source families are encountered.
