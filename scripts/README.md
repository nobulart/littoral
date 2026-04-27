# Scripts folder

## discovery.py
Searches Crossref API for scientific literature that might contain additional data points for harvesting into LITTORAL. 
Outputs results to outputs/discovery.csv with title, authors, year, and DOI.

Usage: python3 scripts/discovery.py [--min-year 2020] [--max-year 2026] [--limit 100]

## ingest_walis.py  
Converts WALIS CSV exports to LITTORAL format.
