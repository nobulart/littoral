from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.models import AgeModel, SamplePoint, SourceLocator, deterministic_sample_id
from src.extract.document_loader import DocumentPayload, PageOCRBlock
from src.extract.geocode import geocode_contextual_location, geocode_place_query
from src.extract.manual_geocodes import load_manual_geocodes, manual_geocode_note
from src.orchestrate.runtime import PipelineRuntime


GUYOT_KEYWORD_PATTERN = re.compile(
    r"guyots?|table\s*mounts?|flat[- ]topped\s+seamounts?|drowned\s+(?:carbonate\s+)?platforms?|drowned\s+islands?",
    re.IGNORECASE,
)
TITLE_KEYWORDS = re.compile(
    r"submerged beach|ancient beach|submerged forest|marine terrace|raised beach|"
    r"guyots?|table\s*mounts?|flat[- ]topped\s+seamounts?|drowned\s+(?:carbonate\s+)?platforms?",
    re.IGNORECASE,
)
LATITUDE_PATTERN = re.compile(r"(\d{1,2})\D+(\d{1,2})\D+(North|South)\s+Latitude", re.IGNORECASE)
LONGITUDE_PATTERN = re.compile(r"(\d{1,3})\D+(\d{1,2})\D+(West|East).{0,120}?Longitude", re.IGNORECASE | re.DOTALL)
DECIMAL_PAIR_PATTERN = re.compile(r"\b(-?\d{1,2}\.\d{2,6})\s*,\s*(-?\d{1,3}\.\d{2,6})\b")
FATHOM_PATTERN = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*fathoms", re.IGNORECASE)
METER_DEPTH_PATTERN = re.compile(r"(?:water depth|depth(?: of| at)?|located at a water depth of|minimum depth of)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*m", re.IGNORECASE)
GUYOT_DEPTH_PATTERN = re.compile(
    r"(?:summit|plateau|platform|top|break)\s+depths?(?:\s*\([^)]*\))?\s*(?:of|averaging|about|around|at)?\s*"
    r"(?:~|approximately|approx\.)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*m",
    re.IGNORECASE,
)
AGE_BP_PATTERN = re.compile(r"(\d{1,2},\d{3}|\d{3,5})\s*(?:yrs\s*B\.P\.|years\s*B\.P\.|B\.P\.)", re.IGNORECASE)
TITLE_FALLBACK_PATTERN = re.compile(r'(?m)^[A-Z][A-Za-z0-9.,;:\'"()\-–— ]{20,}$')


def clean_title(raw_title: str, text: str, source_path: Path) -> str:
    title = raw_title.strip()
    if not title or title.startswith("PII:") or title == source_path.stem or title.lower() == "terms of use":
        whale_title = re.search(
            r"Late Pleistocene and Holocene whale remains.*?trace element concentrations",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if whale_title:
            return " ".join(whale_title.group(0).split())[:180]
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        for line in lines[:20]:
            if TITLE_KEYWORDS.search(line) and len(line) > 20:
                return line[:160]
        match = TITLE_KEYWORDS.search(text)
        if match:
            start = max(0, match.start() - 10)
            end = min(len(text), match.end() + 110)
            snippet = " ".join(text[start:end].split())
            return snippet[:160]
        line_match = TITLE_FALLBACK_PATTERN.search(text)
        if line_match:
            return " ".join(line_match.group(0).split())[:160]
        return source_path.stem
    return " ".join(title.split())


def infer_location_name(title: str, text: str, source_path: Path) -> str:
    lowered_title = title.lower()
    lowered_text = text.lower()
    if "bermuda" in lowered_title or "bermuda" in lowered_text:
        return "Bermuda"
    if "wissant" in lowered_title or "wissant" in lowered_text:
        return "Wissant"
    if "ekofisk" in lowered_title or "ekofisk" in lowered_text:
        return "Ekofisk-Norwegian Trench"
    if "south australia" in lowered_title or "south australia" in lowered_text:
        return "Southeast South Australia"
    return source_path.stem


def extract_place_queries(title: str, text: str, source_path: Path) -> list[str]:
    queries: list[str] = []

    near_match = re.search(r"near\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)", title)
    if near_match:
        queries.append(near_match.group(1))

    off_match = re.search(r"off\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)", title)
    if off_match:
        queries.append(off_match.group(1))

    deduped: list[str] = []
    for query in queries:
        normalized = " ".join(query.split()).strip(" ,.;:")
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def extract_place_queries_from_text(text: str) -> list[str]:
    queries: list[str] = []
    patterns = [
        r"near\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)",
        r"off\s+([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)",
        r"between\s+([A-Z][A-Za-z\-]+)\s+and\s+([A-Z][A-Za-z\-]+)",
        r"(?:map|figure|fig\.?|plate)\s*(?:of|showing)?\s*([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)*)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if len(match.groups()) == 2:
                queries.extend([group for group in match.groups() if group])
            else:
                queries.append(match.group(1))

    deduped: list[str] = []
    for query in queries:
        normalized = " ".join(query.split()).strip(" ,.;:")
        if normalized and normalized not in deduped and normalized.lower() not in {"figure", "table", "plate", "map"}:
            deduped.append(normalized)
    return deduped[:5]


def extract_coordinates(text: str) -> tuple[float | None, float | None, str | None]:
    lat_match = LATITUDE_PATTERN.search(text)
    lon_match = LONGITUDE_PATTERN.search(text)
    if lat_match and lon_match:
        lat_deg, lat_min, lat_hemi = lat_match.groups()
        lon_deg, lon_min, lon_hemi = lon_match.groups()
        lat = float(lat_deg) + float(lat_min) / 60.0
        lon = float(lon_deg) + float(lon_min) / 60.0
        if lat_hemi.lower().startswith("s"):
            lat *= -1
        if lon_hemi.lower().startswith("w"):
            lon *= -1
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, "reported_dms"

    for lat_text, lon_text in DECIMAL_PAIR_PATTERN.findall(text[:20000]):
        lat = float(lat_text)
        lon = float(lon_text)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon, "reported_decimal"

    return None, None, None


def extract_depth_m(text: str) -> float | None:
    meter_match = METER_DEPTH_PATTERN.search(text)
    if meter_match:
        return float(meter_match.group(1))
    if GUYOT_KEYWORD_PATTERN.search(text):
        guyot_match = GUYOT_DEPTH_PATTERN.search(text)
        if guyot_match:
            return float(guyot_match.group(1).replace(",", ""))
    fathom_match = FATHOM_PATTERN.search(text)
    if fathom_match:
        fathoms = float(fathom_match.group(1).replace(",", ""))
        return round(fathoms * 1.8288, 1)
    return None


def extract_age_models(text: str) -> list[AgeModel]:
    models: list[AgeModel] = []
    for match in AGE_BP_PATTERN.finditer(text[:30000]):
        age_bp = float(match.group(1).replace(",", ""))
        models.append(
            AgeModel(
                method="radiocarbon" if "radiocarbon" in text.lower() else "other",
                relation="equal",
                age_ka=round(age_bp / 1000.0, 3),
                notes=f"Extracted from text span: {match.group(0)}",
            )
        )
        if len(models) >= 5:
            break
    return models


def infer_indicator_type(title: str, text: str) -> str:
    lowered = f"{title} {text[:4000]}".lower()
    if GUYOT_KEYWORD_PATTERN.search(lowered):
        return "guyot_or_drowned_platform"
    if "ancient beach" in lowered or "raised beach" in lowered:
        return "raised_beach"
    if "submerged beach" in lowered:
        return "submerged_beach"
    if "submerged forest" in lowered:
        return "terrestrial_over_marine_contact"
    if "marine terrace" in lowered:
        return "marine_terrace"
    return "submerged_beach"


def infer_indicator_types(title: str, text: str) -> list[str]:
    lowered_title = title.lower()
    lowered = f"{title} {text[:6000]}".lower()
    ordered: list[str] = []
    if "ancient beach" in lowered_title or "raised beach" in lowered_title:
        ordered.append("raised_beach")
    if "submerged beach" in lowered_title:
        ordered.append("submerged_beach")
    if "submerged forest" in lowered_title:
        ordered.append("terrestrial_over_marine_contact")
    if "marine terrace" in lowered_title:
        ordered.append("marine_terrace")
    if GUYOT_KEYWORD_PATTERN.search(lowered_title):
        ordered.append("guyot_or_drowned_platform")
    if ordered:
        deduped: list[str] = []
        for item in ordered:
            if item not in deduped:
                deduped.append(item)
        return deduped

    if "ancient beach" in lowered or "raised beach" in lowered:
        ordered.append("raised_beach")
    if "submerged beach" in lowered:
        ordered.append("submerged_beach")
    if "submerged forest" in lowered:
        ordered.append("terrestrial_over_marine_contact")
    if "marine terrace" in lowered:
        ordered.append("marine_terrace")
    if GUYOT_KEYWORD_PATTERN.search(lowered):
        ordered.append("guyot_or_drowned_platform")
    if not ordered:
        ordered.append("submerged_beach")
    deduped: list[str] = []
    for item in ordered:
        if item not in deduped:
            deduped.append(item)
    return deduped


def infer_record_class(indicator_type: str) -> str:
    if indicator_type == "terrestrial_over_marine_contact":
        return "terrestrial_limiting"
    return "sea_level_indicator"


def build_indicator_specific_quote(text: str, indicator_type: str) -> str:
    lowered = text.lower()
    phrase_map = {
        "raised_beach": ["ancient beach", "raised beach"],
        "submerged_beach": ["submerged beach"],
        "terrestrial_over_marine_contact": ["submerged forest"],
        "marine_terrace": ["marine terrace"],
        "guyot_or_drowned_platform": ["guyot", "flat-topped seamount", "flat topped seamount", "drowned carbonate platform"],
    }
    for phrase in phrase_map.get(indicator_type, []):
        index = lowered.find(phrase)
        if index != -1:
            start = max(0, index - 150)
            end = min(len(text), index + 450)
            return " ".join(text[start:end].split())[:800]
    return " ".join(text.split())[:800]


def build_heuristic_sample_points(
    source_id: str,
    source_path: Path,
    payload: DocumentPayload,
    runtime: PipelineRuntime | None = None,
) -> list[SamplePoint]:
    title = clean_title(payload.title, payload.text, source_path)
    lat, lon, coord_method = extract_coordinates(payload.text)
    geocode_result = None
    if lat is None or lon is None:
        for place_query in extract_place_queries(title, payload.text, source_path):
            geocode_result = geocode_place_query(source_path, place_query, title, payload.text, runtime=runtime)
            if geocode_result is not None:
                lat = geocode_result.latitude
                lon = geocode_result.longitude
                coord_method = "inferred_text"
                break
    if lat is None or lon is None:
        return []

    depth_m = extract_depth_m(payload.text)
    age_models = extract_age_models(payload.text)
    indicator_types = infer_indicator_types(title, payload.text)
    location_name = geocode_result.display_name if geocode_result is not None else infer_location_name(title, payload.text, source_path)
    points: list[SamplePoint] = []
    for index, indicator_type in enumerate(indicator_types, start=1):
        quote = build_indicator_specific_quote(payload.text, indicator_type)
        locator = SourceLocator(page="1-2" if payload.page_count else None, quote_or_paraphrase=quote)
        point = SamplePoint(
            id="",
            source_id=source_id,
            record_class=infer_record_class(indicator_type),
            site_name=location_name,
            sample_id=f"heuristic_feature_{index}",
            latitude=lat,
            longitude=lon,
            coordinate_source="reported" if coord_method and coord_method.startswith("reported") else "inferred_text",
            coordinate_uncertainty_m=100.0 if coord_method and coord_method.startswith("reported") else (geocode_result.uncertainty_m if geocode_result is not None else 5000.0),
            elevation_m=(-depth_m if depth_m is not None else None),
            elevation_reference="MSL",
            depth_source="reported" if depth_m is not None else "other",
            indicator_type=indicator_type,
            indicator_subtype="heuristically extracted from explicit text",
            indicative_range_m=None,
            age_ka=age_models[0].age_ka if age_models else None,
            dating_method=age_models[0].method if age_models else "other",
            description=title,
            location_name=location_name,
            bibliographic_reference=title,
            doi_or_url="",
            confidence_score=None,
            notes=(
                f"Extracted using heuristic parser from {payload.source_format} text after methods: {', '.join(payload.extraction_methods)}."
                + (f" Place name geocoded from query '{geocode_result.query}'." if geocode_result is not None else "")
            ),
            source_locator=locator,
            age_models=age_models or [AgeModel(method="other", relation="unknown", age_ka=None, notes="No direct age model extracted from text.")],
        )
        point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
        point.reported_observations.reported_depth_m = depth_m
        point.reported_observations.reported_datum = "MSL" if depth_m is not None else None
        points.append(point)
    return points


def build_page_block_sample_points(
    source_id: str,
    source_path: Path,
    payload: DocumentPayload,
    runtime: PipelineRuntime | None = None,
) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    title = clean_title(payload.title, payload.text, source_path)
    for block in payload.page_blocks:
        if not block.text.strip():
            continue
        lat, lon, coord_method = extract_coordinates(block.text)
        geocode_result = None
        if lat is None or lon is None:
            for place_query in extract_place_queries_from_text(block.text):
                geocode_result = geocode_place_query(source_path, place_query, title, block.text, runtime=runtime)
                if geocode_result is not None:
                    lat = geocode_result.latitude
                    lon = geocode_result.longitude
                    coord_method = "inferred_text"
                    break
        if lat is None or lon is None:
            continue

        indicator_types = infer_indicator_types(title, block.text)
        depth_m = extract_depth_m(block.text)
        age_models = extract_age_models(block.text)
        location_name = geocode_result.display_name if geocode_result is not None else infer_location_name(title, block.text, source_path)
        for index, indicator_type in enumerate(indicator_types, start=1):
            locator = SourceLocator(page=str(block.page_number), figure=block.cue if "fig" in block.cue.lower() or "plate" in block.cue.lower() or "map" in block.cue.lower() else None, table=block.cue if "table" in block.cue.lower() else None, quote_or_paraphrase=" ".join(block.text.split())[:800])
            point = SamplePoint(
                id="",
                source_id=source_id,
                record_class=infer_record_class(indicator_type),
                site_name=location_name,
                sample_id=f"page_{block.page_number}_feature_{index}",
                latitude=lat,
                longitude=lon,
                coordinate_source="reported" if coord_method and coord_method.startswith("reported") else "inferred_text",
                coordinate_uncertainty_m=100.0 if coord_method and coord_method.startswith("reported") else (geocode_result.uncertainty_m if geocode_result is not None else 5000.0),
                elevation_m=(-depth_m if depth_m is not None else None),
                elevation_reference="MSL",
                depth_source="reported" if depth_m is not None else "other",
                indicator_type=indicator_type,
                indicator_subtype="page OCR extracted from figure/table/map block",
                indicative_range_m=None,
                age_ka=age_models[0].age_ka if age_models else None,
                dating_method=age_models[0].method if age_models else "other",
                description=title,
                location_name=location_name,
                bibliographic_reference=title,
                doi_or_url="",
                confidence_score=None,
                notes=f"Extracted from page OCR block on page {block.page_number} with cue '{block.cue}' from source '{block.source}'." + (f" Place name geocoded from query '{geocode_result.query}'." if geocode_result is not None else ""),
                source_locator=locator,
                age_models=age_models or [AgeModel(method="other", relation="unknown", age_ka=None, notes="No direct age model extracted from page OCR block.")],
            )
            point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
            point.reported_observations.reported_depth_m = depth_m
            point.reported_observations.reported_datum = "MSL" if depth_m is not None else None
            points.append(point)
    return points


def summarize_payload(payload: DocumentPayload) -> list[str]:
    lines = [
        f"- Extraction methods: `{', '.join(payload.extraction_methods)}`",
        f"- Native text length: `{payload.native_text_length}`",
        f"- OCR text length: `{payload.ocr_text_length}`",
        f"- Text quality score: `{payload.text_quality_score}`",
        f"- Page count: `{payload.page_count}`",
        f"- Page OCR blocks: `{len(payload.page_blocks)}`",
    ]
    if "mineru_hybrid_auto_markdown" in payload.extraction_methods:
        lines.append(f"- MinerU Markdown: `{payload.metadata.get('MinerUMarkdown', '')}`")
        lines.append(f"- MinerU structured table/image/chart blocks: `{len(payload.page_blocks)}`")
    return lines


def llm_candidate_to_sample_point(
    source_id: str,
    source_path: Path,
    candidate: dict,
    title: str,
    runtime: PipelineRuntime | None = None,
) -> SamplePoint | None:
    try:
        latitude = candidate.get("latitude")
        longitude = candidate.get("longitude")
        coordinate_source = str(candidate.get("coordinate_source") or "inferred_text")
        coordinate_uncertainty_m = candidate.get("coordinate_uncertainty_m")
        place_query = candidate.get("place_query")
        geocode_result = None
        manual_table = load_manual_geocodes(source_id, source_path)
        manual_match = manual_table.match(
            [
                candidate.get("sample_id"),
                candidate.get("site_name"),
                candidate.get("location_name"),
                place_query,
            ]
        )
        manual_note = ""
        if manual_match is not None and manual_match.latitude is not None and manual_match.longitude is not None:
            latitude = manual_match.latitude
            longitude = manual_match.longitude
            coordinate_source = "inferred_map"
            coordinate_uncertainty_m = manual_match.coordinate_uncertainty_m or coordinate_uncertainty_m or 1000.0
            manual_note = manual_geocode_note(manual_table, manual_match)
        elif manual_table.suppresses_fuzzy_geocoding and coordinate_source != "reported":
            latitude = None
            longitude = None
            coordinate_source = "inferred_map"
            coordinate_uncertainty_m = None
            manual_note = manual_geocode_note(manual_table, manual_match)
        if (latitude is None or longitude is None) and not manual_table.suppresses_fuzzy_geocoding:
            geocode_queries = [
                str(value)
                for value in [
                    place_query,
                    candidate.get("location_name"),
                    candidate.get("site_name"),
                    candidate.get("description"),
                    title,
                ]
                if value
            ]
            geocode_result = geocode_contextual_location(
                source_path,
                geocode_queries,
                title,
                str(candidate.get("quote_or_paraphrase") or title),
                runtime=runtime,
            )
            if geocode_result is not None:
                latitude = geocode_result.latitude
                longitude = geocode_result.longitude
                coordinate_source = "inferred_text"
                coordinate_uncertainty_m = geocode_result.uncertainty_m
        latitude = float(latitude) if latitude is not None else None
        longitude = float(longitude) if longitude is not None else None
        if latitude is not None and not (-90 <= latitude <= 90):
            return None
        if longitude is not None and not (-180 <= longitude <= 180):
            return None
        if coordinate_source not in {"reported", "inferred_text", "inferred_map"}:
            coordinate_source = "inferred_text" if latitude is not None and longitude is not None else "inferred_map"
        if (latitude is None or longitude is None) and coordinate_source != "reported":
            coordinate_source = "inferred_map"
        coordinate_uncertainty_m = float(coordinate_uncertainty_m) if coordinate_uncertainty_m is not None else None
        indicator_type = str(candidate.get("indicator_type") or "submerged_beach")
        dating_method = str(candidate.get("dating_method") or "other")
        age_ka = normalize_candidate_age_ka(candidate.get("age_ka"), dating_method)
        location_name = str(candidate.get("location_name") or candidate.get("site_name") or (geocode_result.display_name if geocode_result is not None else source_path.stem))
        quote = str(candidate.get("quote_or_paraphrase") or "")[:800]
        locator = SourceLocator(page=str(candidate.get("page") or ""), figure=str(candidate.get("figure") or "") or None, table=str(candidate.get("table") or "") or None, quote_or_paraphrase=quote)
        point = SamplePoint(
            id="",
            source_id=source_id,
            record_class=str(candidate.get("record_class") or "sea_level_indicator"),
            site_name=str(candidate.get("site_name") or location_name),
            sample_id=str(candidate.get("sample_id") or "llm_feature_1"),
            latitude=latitude,
            longitude=longitude,
            coordinate_source=coordinate_source,
            coordinate_uncertainty_m=coordinate_uncertainty_m,
            elevation_m=candidate.get("elevation_m"),
            elevation_reference=str(candidate.get("elevation_reference") or "unknown"),
            depth_source=str(candidate.get("depth_source") or "other"),
            indicator_type=indicator_type,
            indicator_subtype=str(candidate.get("indicator_subtype") or "llm-extracted candidate"),
            indicative_range_m=candidate.get("indicative_range_m"),
            age_ka=age_ka,
            dating_method=dating_method,
            description=str(candidate.get("description") or title),
            location_name=location_name,
            bibliographic_reference=str(candidate.get("bibliographic_reference") or title),
            doi_or_url=str(candidate.get("doi_or_url") or ""),
            confidence_score=None,
            notes=str(candidate.get("notes") or "Optional Ollama interpretation output.")
            + manual_note
            + (f" Geocoded from contextual query '{geocode_result.query}'." if geocode_result is not None else " No contextual geocode result was found."),
            source_locator=locator,
            age_models=[AgeModel(method=dating_method, relation="unknown", age_ka=age_ka, notes="LLM-assisted extraction; raw BP-like numeric ages are normalized to ka.")],
        )
        point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
        return point
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def normalize_candidate_age_ka(value, dating_method: str = ""):
    method = dating_method.lower()
    if isinstance(value, list):
        return [normalize_candidate_age_ka(item, dating_method) if isinstance(item, (int, float)) else item for item in value]
    if not isinstance(value, (int, float)):
        return value
    numeric = float(value)
    if numeric <= 0:
        return numeric
    if numeric > 1000 or ("14c" in method or "radiocarbon" in method) and numeric > 300:
        return round(numeric / 1000.0, 3)
    return numeric
