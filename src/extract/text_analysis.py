from __future__ import annotations

import re


COORDINATE_PATTERN = re.compile(r"[-+]?\d{1,3}(?:\.\d+)?\s*[NSEW]|[-+]?\d{1,3}\.\d+\s*,\s*[-+]?\d{1,3}\.\d+")
AGE_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:ka|kyr|ky|Ma|BP|yrs B\.P\.)\b", re.IGNORECASE)
ELEVATION_PATTERN = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:m|meters|metres|feet|ft|fathoms)\b", re.IGNORECASE)

RSL_KEYWORDS = {
    "relative sea level",
    "rsl",
    "marine terrace",
    "guyot",
    "guyots",
    "tablemount",
    "table mount",
    "flat-topped seamount",
    "flat topped seamount",
    "drowned carbonate platform",
    "drowned island",
    "seamount",
    "atoll",
    "raised beach",
    "submerged beach",
    "submerged forest",
    "beachrock",
    "coral reef",
    "tidal notch",
    "lagoonal",
    "deltaic",
    "saltmarsh",
    "mangrove",
    "shoreline",
    "shore platform",
    "wave-cut",
    "intertidal",
    "marine shell",
    "speleothem",
    "paleo sea level",
    "sea-level",
    "sea level",
    "water depth",
    "radiocarbon",
}

NARRATIVE_HINTS = {
    "chapter",
    "legend",
    "story",
    "myth",
    "biblical",
    "victorian",
    "archaeologist",
    "journey",
    "adventure",
    "royal",
    "king",
}


def analyze_text(text: str) -> dict[str, object]:
    lowered = text.lower()
    keyword_hits = sorted(keyword for keyword in RSL_KEYWORDS if keyword in lowered)
    narrative_hits = sorted(keyword for keyword in NARRATIVE_HINTS if keyword in lowered)
    coordinate_hits = COORDINATE_PATTERN.findall(text)
    age_hits = AGE_PATTERN.findall(text)
    elevation_hits = ELEVATION_PATTERN.findall(text)
    return {
        "keyword_hits": keyword_hits,
        "narrative_hits": narrative_hits,
        "coordinate_hits": coordinate_hits,
        "age_hits": age_hits,
        "elevation_hits": elevation_hits,
        "source_classification": classify_source(keyword_hits, coordinate_hits, age_hits, elevation_hits, narrative_hits),
    }


def classify_source(
    keyword_hits: list[str],
    coordinate_hits: list[str],
    age_hits: list[str],
    elevation_hits: list[str],
    narrative_hits: list[str],
) -> str:
    has_structured_signal = bool(coordinate_hits) or (len(age_hits) >= 2 and len(elevation_hits) >= 2) or len(keyword_hits) >= 3
    if has_structured_signal:
        return "candidate_review_needed"
    if narrative_hits:
        return "non_candidate_narrative_source"
    return "non_candidate_unstructured_source"


def build_unresolved_line(source_id: str, filename: str, source_classification: str) -> str:
    if source_classification == "non_candidate_narrative_source":
        reason = "No candidate RSL SamplePoints detected; source appears to be narrative/background text rather than an extractable site report."
    elif source_classification == "candidate_review_needed":
        reason = "Potential RSL-related terms detected, but no point-level coordinates/elevations/ages were extracted into validated SamplePoints."
    else:
        reason = "No structured SamplePoint records extracted from unstructured text source."
    return f"{source_id}\t{filename}\t{reason}"
