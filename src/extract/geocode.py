from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.extract.settings import load_extraction_settings


@dataclass(slots=True)
class GeocodeResult:
    query: str
    latitude: float
    longitude: float
    display_name: str
    uncertainty_m: float


_NOMINATIM_CACHE: dict[str, GeocodeResult | None] = {}
_LAST_NOMINATIM_REQUEST_AT = 0.0


def geocode_place_query(source_path: Path, query: str, context_title: str, context_text: str, normalize_with_ollama: bool = True) -> GeocodeResult | None:
    settings = load_extraction_settings(source_path)
    geocoding = settings["geocoding"]
    if not geocoding.get("enabled", True):
        return None

    normalized_query = _normalize_query_with_ollama(source_path, query, context_title, context_text) if normalize_with_ollama else None
    normalized_query = normalized_query or query
    cache_key = normalized_query.casefold()
    if not normalize_with_ollama and cache_key in _NOMINATIM_CACHE:
        return _NOMINATIM_CACHE[cache_key]
    params = urllib.parse.urlencode(
        {
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": int(geocoding.get("limit", 3)),
            "q": normalized_query,
        }
    )
    url = f"{geocoding.get('url')}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": str(geocoding.get("user_agent", "LITTORAL/0.1"))})
    try:
        _respect_geocode_rate_limit(float(geocoding.get("min_delay_seconds", 1.0)))
        with urllib.request.urlopen(request, timeout=float(geocoding.get("timeout_seconds", 20))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    if not payload:
        if not normalize_with_ollama:
            _NOMINATIM_CACHE[cache_key] = None
        return None

    best = _choose_best_geocode_result(normalized_query, payload)
    if best is None:
        if not normalize_with_ollama:
            _NOMINATIM_CACHE[cache_key] = None
        return None
    uncertainty_m = _estimate_uncertainty(normalized_query, best.get("display_name", ""))
    result = GeocodeResult(
        query=normalized_query,
        latitude=float(best["lat"]),
        longitude=float(best["lon"]),
        display_name=str(best.get("display_name", normalized_query)),
        uncertainty_m=uncertainty_m,
    )
    if not normalize_with_ollama:
        _NOMINATIM_CACHE[cache_key] = result
    return result


def geocode_contextual_location(source_path: Path, queries: list[str], context_title: str, context_text: str) -> GeocodeResult | None:
    """Try increasingly broad place queries from explicit source context."""
    settings = load_extraction_settings(source_path)
    geocoding = settings["geocoding"]
    max_queries = int(geocoding.get("max_contextual_queries", 8))
    for query in _contextual_query_variants(queries, context_title, context_text)[:max_queries]:
        result = geocode_place_query(source_path, query, context_title, context_text, normalize_with_ollama=False)
        if result is not None:
            return result
    return None


def _contextual_query_variants(queries: list[str], title: str, text: str) -> list[str]:
    variants: list[str] = []
    for query in queries:
        variants.extend(_query_variants(query))
    variants.extend(_place_cues_from_text(title))
    variants.extend(_place_cues_from_text(text[:2000]))
    if "australia" in f"{title} {text[:2000]}".lower():
        variants.extend([f"{query}, Australia" for query in queries if query and "australia" not in query.lower()])
    if "south africa" in f"{title} {text[:2000]}".lower():
        variants.extend([f"{query}, South Africa" for query in queries if query and "south africa" not in query.lower()])
    return _dedupe_queries(variants)


def _query_variants(query: str) -> list[str]:
    cleaned = " ".join(str(query).replace("–", "-").split()).strip(" ,.;:")
    if not cleaned:
        return []
    variants = [cleaned]
    variants.extend(_known_place_variants(cleaned))
    variants.append(re.sub(r"\bcontinental shelf\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:"))
    variants.append(re.sub(r"\bshelf\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:"))
    variants.append(re.sub(r"\boffshore\b", "", cleaned, flags=re.IGNORECASE).strip(" ,.;:"))
    expanded = _expand_australian_abbreviations(cleaned)
    if expanded != cleaned:
        variants.append(expanded)
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) > 1:
        variants.append(", ".join(parts[-2:]))
        variants.append(parts[-1])
    return [variant for variant in variants if variant]


def _known_place_variants(query: str) -> list[str]:
    lowered = query.lower()
    variants: list[str] = []
    known_places = {
        "mossel bay": ["Mossel Bay, South Africa"],
        "cape st blaize": ["Cape St Blaize, Mossel Bay, South Africa"],
        "cape saint blaize": ["Cape St Blaize, Mossel Bay, South Africa"],
        "groot brak": ["Groot Brak River, Western Cape, South Africa"],
        "rottnest": ["Rottnest Island, Western Australia", "Rottnest Shelf, Western Australia"],
        "recherche": ["Esperance, Western Australia", "Recherche Archipelago, Western Australia"],
        "lacepede": ["Lacepede Bay, South Australia"],
        "gippsland": ["Gippsland, Victoria, Australia"],
        "southeast australian shelf": ["Newcastle, New South Wales", "Coffs Harbour, New South Wales", "New South Wales, Australia"],
        "eastern tasmanian shelf": ["Freycinet National Park, Tasmania", "Freycinet Peninsula, Tasmania"],
        "great barrier reef": ["Great Barrier Reef, Queensland, Australia"],
        "torres strait": ["Torres Strait, Queensland, Australia"],
        "lord howe": ["Lord Howe Island, New South Wales", "Ball's Pyramid, Lord Howe Island"],
        "balls pyramid": ["Ball's Pyramid, Lord Howe Island"],
        "bonaparte": ["Joseph Bonaparte Gulf, Australia"],
        "carnarvon": ["Carnarvon, Western Australia"],
        "esperance": ["Esperance, Western Australia"],
    }
    for cue, cue_variants in known_places.items():
        if cue in lowered:
            variants.extend(cue_variants)
    return variants


def _expand_australian_abbreviations(query: str) -> str:
    replacements = {
        " SW ": " southwest ",
        " SE ": " southeast ",
        " NW ": " northwest ",
        " NE ": " northeast ",
        " WA": " Western Australia",
        " NSW": " New South Wales",
        " GBR": " Great Barrier Reef",
    }
    expanded = f" {query} "
    for source, target in replacements.items():
        expanded = expanded.replace(source, target)
    return " ".join(expanded.split())


def _place_cues_from_text(text: str) -> list[str]:
    cues: list[str] = []
    patterns = [
        r"offshore of ([A-Z][A-Za-z ]+?)(?:,|\.|\s+and|\s+near|\s+on\s+the|$)",
        r"near ([A-Z][A-Za-z ]+?)(?:,|\.|\s+on\s+the|$)",
        r"of ([A-Z][A-Za-z ]+?,\s*(?:South Africa|Australia|Western Australia|Queensland|New South Wales|Tasmania))",
        r"([A-Z][A-Za-z ]+ Bay),\s*(South Africa|Australia)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = [group.strip() for group in match.groups() if group]
            if groups:
                cues.append(", ".join(groups))
    return cues


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(str(query).split()).strip(" ,.;:")
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _respect_geocode_rate_limit(min_delay_seconds: float) -> None:
    global _LAST_NOMINATIM_REQUEST_AT
    if min_delay_seconds <= 0:
        return
    now = time.monotonic()
    wait_seconds = min_delay_seconds - (now - _LAST_NOMINATIM_REQUEST_AT)
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    _LAST_NOMINATIM_REQUEST_AT = time.monotonic()


def _estimate_uncertainty(query: str, display_name: str) -> float:
    query_lower = query.lower()
    display_lower = display_name.lower()
    if "near " in query_lower or "off " in query_lower or "between " in query_lower:
        return 10000.0
    if any(token in display_lower for token in ["france", "norway", "australia", "bermuda", "south africa"]):
        return 5000.0
    return 10000.0


def _choose_best_geocode_result(query: str, payload: list[dict]) -> dict | None:
    acceptable: list[dict] = []
    for item in payload:
        item_class = str(item.get("class") or item.get("category") or "").lower()
        item_type = str(item.get("type", "")).lower()
        item_addresstype = str(item.get("addresstype", "")).lower()
        display = str(item.get("display_name", "")).lower()
        query_tokens = [token.lower() for token in re_split_query(query) if len(token) > 2]
        if item_class in {"amenity", "shop", "building", "office", "tourism", "leisure", "highway"}:
            continue
        if item_type in {"kindergarten", "school", "office", "hotel", "restaurant", "residential", "road", "service"}:
            continue
        if item_addresstype in {"amenity", "road", "building", "office", "tourism", "shop", "house_number"}:
            continue
        if query_tokens and not all(token in display for token in query_tokens[:1]):
            continue
        acceptable.append(item)
    if not acceptable:
        return None
    acceptable.sort(key=lambda item: float(item.get("importance", 0.0)), reverse=True)
    return acceptable[0]


def re_split_query(query: str) -> list[str]:
    return [token for token in query.replace(",", " ").split() if token]


def _normalize_query_with_ollama(source_path: Path, query: str, title: str, text: str) -> str | None:
    settings = load_extraction_settings(source_path)
    ollama = settings["ollama"]
    if not ollama.get("enabled", False) or not ollama.get("place_normalization_enabled", False):
        return None
    model = str(ollama.get("model", "glm-4.7-flash:latest"))
    prompt = (
        "Return JSON only with the best explicit place query for gazetteer lookup. "
        "Do not invent places not stated in the source. Schema: {\"place_query\": \"string or null\"}.\n\n"
        f"Current query: {query}\n"
        f"Title: {title}\n"
        f"Context: {' '.join(text.split())[:1200]}\n"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            check=True,
            capture_output=True,
            text=True,
            timeout=int(ollama.get("timeout_seconds", 45)),
        )
    except Exception:
        return None

    output = result.stdout.strip()
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    place_query = payload.get("place_query")
    if not place_query:
        return None
    return str(place_query)
