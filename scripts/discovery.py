#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io import CSV_COLUMNS as LITTORAL_CSV_COLUMNS

logger = logging.getLogger("discovery")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CROSSREF_API_URL = "https://api.crossref.org/works"
DEFAULT_MIN_YEAR = 2015
DEFAULT_MAX_YEAR = 2026
DEFAULT_RESULTS_LIMIT = 100
RATE_LIMIT_DELAY = 1.0

SEARCH_KEYWORDS = {
    "coastal_landforms": [
        "raised beach",
        "submerged beach",
        "marine terrace",
        "wave-cut platform",
        "shore platform",
        "beachrock",
        "marine shell bed",
        "tidal notch",
        "coral reef",
        "guyot",
        "drowned platform",
        "tablemount",
        "flat-topped seamount",
    ],
    "sea_level_terms": [
        "sea level",
        "relative sea level",
        "paleo sea level",
        "paleoselevel",
        "sea-level change",
        "sea level rise",
        "sea level fall",
        "relative sea-level change",
    ],
    "indicator_types": [
        "marine limiting",
        "terrestrial limiting",
        "sea-level indicator",
        "paleo sea-level indicator",
        "sea level indicator",
    ],
}


def normalize_doi(doi_or_url: str) -> str | None:
    """Extract DOI from DOI or DOI URL."""
    doi_or_url = doi_or_url.strip()
    if not doi_or_url:
        return None
    
    if doi_or_url.startswith("https://doi.org/"):
        return doi_or_url.replace("https://doi.org/", "")
    if doi_or_url.startswith("http://doi.org/"):
        return doi_or_url.replace("http://doi.org/", "")
    
    return doi_or_url


def load_existing_dois(merged_csv: Path) -> set[str]:
    """Load all DOIs from LITTORAL's merged dataset."""
    existing_dois: set[str] = set()
    
    if not merged_csv.exists():
        logger.warning(f"Merged CSV not found: {merged_csv}")
        return existing_dois
    
    with merged_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            doi_or_url = row.get("doi_or_url", "").strip()
            if doi_or_url:
                doi = normalize_doi(doi_or_url)
                if doi:
                    existing_dois.add(doi.lower())
    
    logger.info(f"Loaded {len(existing_dois)} existing DOIs from {merged_csv}")
    return existing_dois


def search_crossref(
    query: str,
    min_year: int,
    max_year: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Search Crossref API for works matching query."""
    params: dict[str, Any] = {
        "query": query,
        "filter": f"from-pub-date:{min_year},until-pub-date:{max_year},type:journal-article",
        "sort": "relevance",
        "order": "desc",
        "rows": min(limit, 1000),
    }
    
    try:
        response = requests.get(CROSSREF_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        items = data.get("message", {}).get("items", [])
        logger.info(f"Crossref search for '{query}': {len(items)} results")
        
        return items
    except requests.RequestException as e:
        logger.error(f"Crossref API error for query '{query}': {e}")
        return []
    except ValueError as e:
        logger.error(f"JSON decode error for query '{query}': {e}")
        return []


def extract_work_metadata(work: dict[str, Any]) -> dict[str, Any] | None:
    """Extract metadata from Crossref work item."""
    try:
        doi = None
        doi_url = None
        
        # Check top-level DOI field first
        if work.get("DOI"):
            doi = work.get("DOI")
            doi_url = f"https://doi.org/{doi}"
        
        # Fallback: check identifier array
        if not doi:
            for obj in work.get("identifier", []):
                id_type = obj.get("type", "")
                id_value = obj.get("value", "")
                if id_type == "doi":
                    doi = id_value
                    doi_url = f"https://doi.org/{id_value}"
                    break
        
        # Fallback: check URL list
        if not doi:
            for url in work.get("URL", []):
                if "doi.org/" in url:
                    doi = url.replace("https://doi.org/", "").replace("http://doi.org/", "")
                    doi_url = url
                    break
        
        if not doi:
            logger.debug(f"Skipping work without DOI: {work.get('title', ['No title'])}")
            return None
        
        title_list = work.get("title", [])
        title = title_list[0] if title_list else "No title"
        
        author_list = work.get("author", [])
        if author_list:
            if len(author_list) <= 3:
                authors = ", ".join(f"{a.get('family', '')} {a.get('given', '')}".strip() for a in author_list)
            else:
                first_author = f"{author_list[0].get('family', '')} {author_list[0].get('given', '')}".strip()
                authors = f"{first_author} et al."
        else:
            authors = "Unknown"
        
        published = work.get("published", {})
        year = published.get("date-parts", [[None]])[0][0]
        if year is None:
            year = "Unknown"
        
        return {
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "url": doi_url,
        }
    except (KeyError, IndexError, TypeError) as e:
        logger.debug(f"Error extracting metadata from work: {e}")
        return None


def search_and_discover(
    existing_dois: set[str],
    min_year: int,
    max_year: int,
    limit: int,
    search_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all searches and return new works not in existing_dois."""
    all_new_works: list[dict[str, Any]] = []
    seen_dois: set[str] = set()
    keyword_counts: Counter = Counter()
    
    # Use custom search terms if provided, otherwise use default categories
    if search_terms:
        keywords = search_terms
        logger.info(f"Using custom search terms: {keywords}")
    else:
        keywords = []
        for category, category_keywords in SEARCH_KEYWORDS.items():
            logger.info(f"Searching category: {category}")
            keywords.extend(category_keywords)
    
    for keyword in keywords:
        query = keyword
        works = search_crossref(query, min_year, max_year, limit)
        
        for work in works:
            metadata = extract_work_metadata(work)
            if metadata is None:
                continue
            
            doi = metadata["doi"].lower()
            
            if doi in existing_dois:
                continue
            if doi in seen_dois:
                continue
            
            seen_dois.add(doi)
            metadata["search_keywords_matched"] = keyword
            keyword_counts[keyword] += 1
            all_new_works.append(metadata)
    
    logger.info(f"Discovery complete: {len(all_new_works)} new works found")
    logger.info(f"Keyword match counts: {dict(keyword_counts)}")
    
    return all_new_works


def write_discovery_csv(works: list[dict[str, Any]], output_path: Path) -> None:
    """Write discovery results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["title", "authors", "year", "doi", "url", "search_keywords_matched", "search_date"]
    
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        
        search_date = time.strftime("%Y-%m-%d")
        
        for work in works:
            row = {
                "title": work.get("title", ""),
                "authors": work.get("authors", ""),
                "year": str(work.get("year", "")),
                "doi": work.get("doi", ""),
                "url": work.get("url", ""),
                "search_keywords_matched": work.get("search_keywords_matched", ""),
                "search_date": search_date,
            }
            writer.writerow(row)
    
    logger.info(f"Wrote {len(works)} records to {output_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    
    workspace_root = args.workspace_root.resolve()
    merged_csv = _resolve_path(args.merged_csv, workspace_root)
    output_csv = _resolve_path(args.output_csv, workspace_root)
    
    existing_dois = load_existing_dois(merged_csv)
    
    # Use custom search terms if provided, otherwise use defaults
    search_terms = args.search if args.search else None
    
    works = search_and_discover(
        existing_dois=existing_dois,
        min_year=args.min_year,
        max_year=args.max_year,
        limit=args.limit,
        search_terms=search_terms,
    )
    
    write_discovery_csv(works, output_csv)
    
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Crossref for scientific literature not yet in LITTORAL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository/workspace root.",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=Path("outputs/merged/master_dataset.csv"),
        help="Path to LITTORAL's merged CSV (for existing DOIs).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/discovery.csv"),
        help="Path to save discovery results.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=DEFAULT_MIN_YEAR,
        help=f"Minimum publication year (default: {DEFAULT_MIN_YEAR}).",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=DEFAULT_MAX_YEAR,
        help=f"Maximum publication year (default: {DEFAULT_MAX_YEAR}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RESULTS_LIMIT,
        help=f"Maximum results per keyword search (default: {DEFAULT_RESULTS_LIMIT}).",
    )
    parser.add_argument(
        "--search",
        action="append",
        help="Custom search term. Can be specified multiple times. Overrides default keywords.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return parser.parse_args(argv)


def _resolve_path(path: Path, workspace_root: Path) -> Path:
    if path.is_absolute():
        return path
    return workspace_root / path


if __name__ == "__main__":
    raise SystemExit(main())
