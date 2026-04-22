from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from src.common.models import AgeModel, SamplePoint, SourceLocator, deterministic_sample_id
from src.extract.document_loader import DocumentPayload
from src.extract.geocode import geocode_contextual_location
from src.extract.heuristics import clean_title, infer_record_class


DEPTH_PATTERN = re.compile(r"~?\s*(\d+(?:\.\d+)?)\s*(?:[–-]\s*~?\s*(\d+(?:\.\d+)?))?\s*m?", re.IGNORECASE)
AGE_KA_PATTERN = re.compile(r"(?:MIS\s*\d+|Holocene|Pleistocene|Last Glacial Maximum|LGM|(\d+(?:\.\d+)?)\s*ka)", re.IGNORECASE)
FATHOM_TO_M = 1.8288


@dataclass(slots=True)
class MinerUInferenceResult:
    sample_points: list[SamplePoint]
    deterministic_records: int
    llm_contexts: list[tuple[str, str]]


def mine_mineru_outputs(source_id: str, source_path: Path, payload: DocumentPayload) -> MinerUInferenceResult:
    content_items = _load_content_items(payload)
    manual_points = _mine_manual_map_geocodes(source_id, source_path, payload)
    table_points = _mine_tables(source_id, source_path, payload, content_items)
    feature_points = _mine_feature_paragraphs(source_id, source_path, payload)
    contexts = _build_llm_contexts(payload, content_items)
    points = manual_points + table_points + feature_points
    return MinerUInferenceResult(sample_points=points, deterministic_records=len(points), llm_contexts=contexts)


def _load_content_items(payload: DocumentPayload) -> list[dict]:
    content_path = payload.metadata.get("MinerUContentList")
    if not content_path:
        return []
    path = Path(content_path)
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _mine_manual_map_geocodes(source_id: str, source_path: Path, payload: DocumentPayload) -> list[SamplePoint]:
    manual_path = source_path.parents[1] / "manual_geocodes" / f"{source_id}.csv"
    if not manual_path.exists():
        return []
    title = clean_title(payload.title, payload.text, source_path)
    points: list[SamplePoint] = []
    with manual_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            site_name = str(row.get("site_name") or row.get("sample_id") or f"manual_map_{row_index}").strip()
            sample_id = str(row.get("sample_id") or site_name).strip()
            latitude = _number(str(row.get("latitude") or ""))
            longitude = _number(str(row.get("longitude") or ""))
            depth_m = _number(str(row.get("depth_m") or ""))
            age_ka = _number(str(row.get("age_ka") or ""))
            coordinate_uncertainty_m = _number(str(row.get("coordinate_uncertainty_m") or "")) or 1000.0
            if latitude is None or longitude is None or depth_m is None:
                continue
            figure = str(row.get("figure") or "").strip() or None
            description = str(row.get("description") or f"Manual map geocode for {site_name}.").strip()
            locator = SourceLocator(
                figure=figure,
                quote_or_paraphrase=description[:800],
            )
            elevation_m = -depth_m
            point = SamplePoint(
                id="",
                source_id=source_id,
                record_class="sea_level_indicator",
                site_name=site_name,
                sample_id=sample_id,
                latitude=latitude,
                longitude=longitude,
                coordinate_source="inferred_map",
                coordinate_uncertainty_m=coordinate_uncertainty_m,
                elevation_m=elevation_m,
                elevation_reference="MSL",
                depth_source="reported",
                indicator_type="submerged_beach",
                indicator_subtype="manual map geocode: submerged beach",
                indicative_range_m=[round(elevation_m - 2.5, 2), round(elevation_m + 2.5, 2)],
                age_ka=age_ka,
                dating_method="stratigraphic_context" if age_ka is not None else "other",
                description=description,
                location_name="North Sea, Norwegian Trench",
                bibliographic_reference=title,
                doi_or_url="",
                confidence_score=None,
                notes=f"Coordinate manually geocoded from source map in {figure or 'map figure'}; depth from manual map geocode file.",
                source_locator=locator,
                age_models=[
                    AgeModel(
                        method="stratigraphic_context" if age_ka is not None else "other",
                        relation="unknown",
                        age_ka=age_ka,
                        notes="Age retained from manual map geocode file/source interpretation.",
                    )
                ],
            )
            point.reported_observations.reported_depth_m = depth_m
            point.reported_observations.reported_datum = "MSL"
            point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
            points.append(point)
    return points


def _mine_tables(source_id: str, source_path: Path, payload: DocumentPayload, content_items: list[dict]) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    title = clean_title(payload.title, payload.text, source_path)
    for item in content_items:
        if item.get("type") != "table":
            continue
        caption = _join_text(item.get("table_caption"))
        rows = _parse_table_rows(str(item.get("table_body") or ""))
        if not rows:
            continue
        header = [cell.lower() for cell in rows[0]]
        if _looks_like_palaeoshoreline_table(header):
            points.extend(_points_from_palaeoshoreline_table(source_id, source_path, title, item, caption, rows[1:]))
        elif _looks_like_cawthra_feature_table(header, caption):
            points.extend(_points_from_cawthra_feature_table(source_id, source_path, title, item, caption, rows[1:]))
        elif _looks_like_dated_altitude_sample_table(header):
            points.extend(_points_from_dated_altitude_sample_table(source_id, source_path, title, item, caption, rows))
        elif _looks_like_radiocarbon_depth_table(header):
            points.extend(_points_from_radiocarbon_depth_table(source_id, source_path, payload, title, item, caption, rows))
        elif _looks_like_dated_beach_name_table(header):
            points.extend(_points_from_dated_beach_name_table(source_id, source_path, title, item, caption, rows))
    return points


def _points_from_palaeoshoreline_table(
    source_id: str,
    source_path: Path,
    title: str,
    item: dict,
    caption: str,
    rows: list[list[str]],
) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    current_group = ""
    current_location = ""
    for row_index, row in enumerate(rows, start=1):
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) == 1:
            current_group = cells[0]
            continue
        if len(cells) == 5 and current_location:
            cells = [current_location] + cells
        if len(cells) < 3:
            continue
        location = cells[0]
        current_location = location
        morphology = cells[1] if len(cells) > 1 else ""
        depth_text = cells[2] if len(cells) > 2 else ""
        reference = cells[5] if len(cells) > 5 else title
        if not location or _depth_summary(depth_text, allow_bare_table_cell=True)[0] is None:
            continue
        indicator_type = _indicator_from_text(f"{current_group} {location} {morphology}")
        description = f"{current_group}: {morphology}".strip(": ")
        point = _make_point(
            source_id=source_id,
            source_path=source_path,
            title=title,
            site_name=location,
            sample_id=f"mineru_table_{item.get('page_idx', 0) + 1}_{row_index}",
            indicator_type=indicator_type,
            indicator_subtype=f"MinerU table row: {current_group or 'palaeoshoreline feature'}",
            description=description,
            depth_text=depth_text,
            bibliographic_reference=reference or title,
            locator=SourceLocator(
                page=str(int(item.get("page_idx", 0)) + 1) if isinstance(item.get("page_idx"), int) else None,
                table=caption or "MinerU table",
                quote_or_paraphrase=" | ".join(cells)[:800],
            ),
            notes=f"Extracted from MinerU table. Original depth text: {depth_text}.",
            allow_bare_depth_cell=True,
        )
        points.append(point)
    return points


def _points_from_cawthra_feature_table(
    source_id: str,
    source_path: Path,
    title: str,
    item: dict,
    caption: str,
    rows: list[list[str]],
) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    for row_index, row in enumerate(rows, start=1):
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) < 3:
            continue
        feature, facies, timing = cells[:3]
        if not feature or _depth_summary(feature)[0] is None:
            continue
        indicator_type = _indicator_from_text(feature)
        point = _make_point(
            source_id=source_id,
            source_path=source_path,
            title=title,
            site_name=_site_from_feature_text(feature, "Mossel Bay continental shelf"),
            sample_id=f"mineru_feature_table_{row_index}",
            indicator_type=indicator_type,
            indicator_subtype=f"MinerU interpreted feature table; facies={facies}",
            description=f"{feature}: {timing}",
            depth_text=feature,
            bibliographic_reference=title,
            locator=SourceLocator(
                page=str(int(item.get("page_idx", 0)) + 1) if isinstance(item.get("page_idx"), int) else None,
                table=caption or "MinerU table",
                quote_or_paraphrase=" | ".join(cells)[:800],
            ),
            notes="Extracted from MinerU interpreted geomorphic feature table.",
        )
        points.append(point)
    return points


def _points_from_dated_altitude_sample_table(
    source_id: str,
    source_path: Path,
    title: str,
    item: dict,
    caption: str,
    rows: list[list[str]],
) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    header = [_normalize_header(cell) for cell in rows[0]]
    group = ""
    for row_index, row in enumerate(rows[1:], start=1):
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) == 1:
            group = cells[0]
            continue
        if len(cells) < len(header):
            continue
        record = {header[index]: cells[index] for index in range(min(len(header), len(cells)))}
        sample_id = record.get("sample number") or record.get("sample no") or record.get("sample") or f"sample_{row_index}"
        material = record.get("material") or ""
        locality = record.get("locality") or record.get("location") or ""
        age_text = record.get("age year") or record.get("age years bp") or record.get("age") or ""
        altitude_text = record.get("altitude m") or record.get("elevation m") or record.get("height m") or ""
        age_ka, uncertainty_ka = _age_bp_to_ka(age_text)
        altitude_m = _number(altitude_text)
        if not locality or age_ka is None or altitude_m is None:
            continue
        indicator_type, record_class = _indicator_from_material_and_context(material, f"{group} {title}")
        point = _make_direct_point(
            source_id=source_id,
            source_path=source_path,
            title=title,
            site_name=locality,
            sample_id=sample_id,
            location_name=f"{locality}, {group}".strip(", "),
            indicator_type=indicator_type,
            record_class=record_class,
            indicator_subtype=f"MinerU dated altitude table; material={material or 'unknown'}",
            elevation_m=altitude_m,
            elevation_reference="MSL",
            indicative_range_m=_indicative_range_for_indicator(indicator_type, altitude_m),
            age_ka=age_ka,
            dating_method="radiocarbon",
            description=f"{material or 'Sample'} from {locality} dated to {age_text} at {altitude_text} m altitude.",
            bibliographic_reference=title,
            locator=SourceLocator(
                page=str(int(item.get("page_idx", 0)) + 1) if isinstance(item.get("page_idx"), int) else None,
                table=caption or "MinerU dated altitude table",
                quote_or_paraphrase=" | ".join(cells)[:800],
            ),
            notes="Extracted from MinerU sample table with age and altitude fields. Vertical datum is reported as altitude relative to present sea level when stated by source context.",
            reported_elevation_m=altitude_m,
            material=material or None,
            age_uncertainty_ka=uncertainty_ka,
        )
        points.append(point)
    return points


def _points_from_radiocarbon_depth_table(
    source_id: str,
    source_path: Path,
    payload: DocumentPayload,
    title: str,
    item: dict,
    caption: str,
    rows: list[list[str]],
) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    header = [_normalize_header(cell) for cell in rows[0]]
    beach_midpoint, beach_range = _reported_beach_depth_context(payload.text)
    for row_index, row in enumerate(rows[1:], start=1):
        cells = [cell.strip() for cell in row if cell.strip()]
        if len(cells) < len(header):
            continue
        record = {header[index]: cells[index] for index in range(min(len(header), len(cells)))}
        station = record.get("sample station") or record.get("station") or record.get("sample") or f"station_{row_index}"
        sample_depth_text = record.get("depth below sea bed m") or record.get("depth m") or ""
        age_text = record.get("radiocarbon age yrs bp") or record.get("radiocarbon age") or record.get("age") or ""
        lab_ref = record.get("laboratory ref number") or record.get("lab number") or ""
        age_ka, uncertainty_ka = _age_bp_to_ka(age_text)
        sample_depth_m = _number(sample_depth_text)
        if age_ka is None or sample_depth_m is None:
            continue
        elevation_m = -beach_midpoint if beach_midpoint is not None else None
        point = _make_direct_point(
            source_id=source_id,
            source_path=source_path,
            title=title,
            site_name=station,
            sample_id=lab_ref or f"radiocarbon_depth_{row_index}",
            location_name=_submerged_beach_location(payload.text) or station,
            indicator_type="submerged_beach",
            record_class="sea_level_indicator",
            indicator_subtype="MinerU radiocarbon shell depth table",
            elevation_m=elevation_m,
            elevation_reference="MSL" if elevation_m is not None else "unknown",
            indicative_range_m=[-beach_range[1], -beach_range[0]] if beach_range is not None else None,
            age_ka=age_ka,
            dating_method="radiocarbon",
            description=f"Radiocarbon-dated shell/sample material from {station}; sample depth below seabed {sample_depth_text} m.",
            bibliographic_reference=title,
            locator=SourceLocator(
                page=str(int(item.get("page_idx", 0)) + 1) if isinstance(item.get("page_idx"), int) else None,
                table=caption or "MinerU radiocarbon depth table",
                quote_or_paraphrase=" | ".join(cells)[:800],
            ),
            notes="Extracted from MinerU table of radiocarbon ages below seabed. Indicator elevation uses source-level submerged beach water-depth context when available; sample depth below seabed is retained as a reported observation note.",
            reported_depth_m=sample_depth_m,
            material="shell material",
            age_uncertainty_ka=uncertainty_ka,
        )
        points.append(point)
    return points


def _points_from_dated_beach_name_table(
    source_id: str,
    source_path: Path,
    title: str,
    item: dict,
    caption: str,
    rows: list[list[str]],
) -> list[SamplePoint]:
    points: list[SamplePoint] = []
    header = [_normalize_header(cell) for cell in rows[0]]
    current_group = ""
    last_age_ka: float | list[float | None] | str | None = None
    for row_index, row in enumerate(rows[1:], start=1):
        cells = [cell.strip() for cell in row if cell.strip()]
        if not cells:
            continue
        if len(cells) == 1:
            current_group = cells[0]
            continue
        record = _dated_beach_record(header, cells)
        beach_name = record.get("beach name") or record.get("name") or ""
        age_text = record.get("age years bp") or record.get("age") or ""
        if not beach_name and len(cells) >= 2:
            beach_name = cells[-2]
            age_text = cells[-1]
        if "modern sea level" in beach_name.lower():
            continue
        age_ka = _age_text_to_ka(age_text)
        if age_ka is not None:
            last_age_ka = age_ka
        depth_m = _depth_from_beach_name(beach_name)
        if depth_m is None:
            continue
        age_value = age_ka if age_ka is not None else last_age_ka
        indicator_type = "submerged_beach" if depth_m > 0 else "raised_beach"
        elevation_m = -depth_m
        point = _make_direct_point(
            source_id=source_id,
            source_path=source_path,
            title=title,
            site_name=_clean_beach_name(beach_name),
            sample_id=f"mineru_beach_depth_{row_index}",
            location_name="southeast South Australia",
            indicator_type=indicator_type,
            record_class=infer_record_class(indicator_type),
            indicator_subtype=f"MinerU dated beach table; group={current_group or 'unknown'}",
            elevation_m=round(elevation_m, 2),
            elevation_reference="MSL",
            indicative_range_m=[round(elevation_m - 2.5, 2), round(elevation_m + 2.5, 2)],
            age_ka=age_value,
            dating_method="stratigraphic_context" if age_value is not None else "other",
            description=f"{beach_name} from dated stranded/submerged beach table.",
            bibliographic_reference=title,
            locator=SourceLocator(
                page=str(int(item.get("page_idx", 0)) + 1) if isinstance(item.get("page_idx"), int) else None,
                table=caption or "MinerU dated beach table",
                quote_or_paraphrase=" | ".join(cells)[:800],
            ),
            notes="Extracted from MinerU beach-name table where the row itself reports a depth/fathom position. Rows without vertical context are intentionally left for review.",
            reported_depth_m=depth_m,
        )
        points.append(point)
    return points


def _mine_feature_paragraphs(source_id: str, source_path: Path, payload: DocumentPayload) -> list[SamplePoint]:
    title = clean_title(payload.title, payload.text, source_path)
    points: list[SamplePoint] = []
    for index, paragraph in enumerate(_feature_paragraphs(payload.text), start=1):
        if _depth_summary(paragraph)[0] is None:
            continue
        feature = paragraph.split(".", 1)[0].strip("# ").strip()
        indicator_type = _indicator_from_text(feature)
        site_name = _site_from_feature_text(paragraph, "Mossel Bay continental shelf")
        point = _make_point(
            source_id=source_id,
            source_path=source_path,
            title=title,
            site_name=site_name,
            sample_id=f"mineru_feature_text_{index}",
            indicator_type=indicator_type,
            indicator_subtype="MinerU text feature paragraph",
            description=feature,
            depth_text=paragraph,
            bibliographic_reference=title,
            locator=SourceLocator(
                section=feature,
                quote_or_paraphrase=" ".join(paragraph.split())[:800],
            ),
            notes="Extracted from MinerU Markdown feature paragraph.",
        )
        points.append(point)
    return points


def _make_point(
    source_id: str,
    source_path: Path,
    title: str,
    site_name: str,
    sample_id: str,
    indicator_type: str,
    indicator_subtype: str,
    description: str,
    depth_text: str,
    bibliographic_reference: str,
    locator: SourceLocator,
    notes: str,
    allow_bare_depth_cell: bool = False,
) -> SamplePoint:
    depth_midpoint, indicative_range = _depth_summary(depth_text, allow_bare_table_cell=allow_bare_depth_cell)
    geocode_result = geocode_contextual_location(
        source_path,
        [site_name, description, title],
        title,
        locator.quote_or_paraphrase,
    )
    latitude = geocode_result.latitude if geocode_result is not None else None
    longitude = geocode_result.longitude if geocode_result is not None else None
    coordinate_source = "inferred_text" if geocode_result is not None else "inferred_map"
    coordinate_uncertainty_m = geocode_result.uncertainty_m if geocode_result is not None else None
    location_name = geocode_result.display_name if geocode_result is not None else site_name
    elevation_m = -depth_midpoint if depth_midpoint is not None else None
    point = SamplePoint(
        id="",
        source_id=source_id,
        record_class=infer_record_class(indicator_type),
        site_name=site_name,
        sample_id=sample_id,
        latitude=latitude,
        longitude=longitude,
        coordinate_source=coordinate_source,
        coordinate_uncertainty_m=coordinate_uncertainty_m,
        elevation_m=elevation_m,
        elevation_reference="MSL",
        depth_source="reported" if depth_midpoint is not None else "other",
        indicator_type=indicator_type,
        indicator_subtype=indicator_subtype,
        indicative_range_m=indicative_range,
        age_ka=_age_summary(depth_text),
        dating_method="stratigraphic_context",
        description=description,
        location_name=location_name,
        bibliographic_reference=bibliographic_reference,
        doi_or_url="",
        confidence_score=None,
        notes=notes + (f" Geocoded from contextual query '{geocode_result.query}'." if geocode_result is not None else " No contextual geocode result was found."),
        source_locator=locator,
        age_models=[AgeModel(method="stratigraphic_context", relation="unknown", age_ka=_age_summary(depth_text), notes="Age context inferred from table/text when present.")],
    )
    point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
    point.reported_observations.reported_depth_m = depth_midpoint
    point.reported_observations.reported_datum = "MSL" if depth_midpoint is not None else None
    return point


def _make_direct_point(
    source_id: str,
    source_path: Path,
    title: str,
    site_name: str,
    sample_id: str,
    location_name: str,
    indicator_type: str,
    record_class: str,
    indicator_subtype: str,
    elevation_m: float | list[float | None] | str | None,
    elevation_reference: str,
    indicative_range_m: list[float | None] | None,
    age_ka: float | list[float | None] | str | None,
    dating_method: str,
    description: str,
    bibliographic_reference: str,
    locator: SourceLocator,
    notes: str,
    reported_depth_m: float | None = None,
    reported_elevation_m: float | None = None,
    material: str | None = None,
    age_uncertainty_ka: float | None = None,
) -> SamplePoint:
    latitude = None
    longitude = None
    coordinate_source = "inferred_map"
    coordinate_uncertainty_m = None
    if site_name or location_name:
        geocode_result = geocode_contextual_location(source_path, [site_name, location_name, title], title, locator.quote_or_paraphrase)
        if geocode_result is not None:
            latitude = geocode_result.latitude
            longitude = geocode_result.longitude
            coordinate_source = "inferred_text"
            coordinate_uncertainty_m = geocode_result.uncertainty_m
            location_name = geocode_result.display_name
            notes += f" Geocoded from contextual query '{geocode_result.query}'."
        else:
            notes += " No contextual geocode result was found."
    point = SamplePoint(
        id="",
        source_id=source_id,
        record_class=record_class,
        site_name=site_name,
        sample_id=sample_id,
        latitude=latitude,
        longitude=longitude,
        coordinate_source=coordinate_source,
        coordinate_uncertainty_m=coordinate_uncertainty_m,
        elevation_m=elevation_m,
        elevation_reference=elevation_reference,
        depth_source="reported" if reported_depth_m is not None or elevation_m is not None else "other",
        indicator_type=indicator_type,
        indicator_subtype=indicator_subtype,
        indicative_range_m=indicative_range_m,
        age_ka=age_ka,
        dating_method=dating_method,
        description=description,
        location_name=location_name or site_name,
        bibliographic_reference=bibliographic_reference,
        doi_or_url="",
        confidence_score=None,
        notes=notes,
        source_locator=locator,
        age_models=[AgeModel(method=dating_method, relation="unknown", age_ka=age_ka, uncertainty_ka=age_uncertainty_ka, material=material, notes="Age extracted from MinerU structured table.")],
    )
    point.reported_observations.reported_depth_m = reported_depth_m
    point.reported_observations.reported_elevation_m = reported_elevation_m
    point.reported_observations.reported_datum = elevation_reference if reported_elevation_m is not None else None
    point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
    return point


def _build_llm_contexts(payload: DocumentPayload, content_items: list[dict]) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []
    for item in content_items:
        item_type = str(item.get("type") or "")
        if item_type not in {"table", "image", "chart"}:
            continue
        caption = _join_text(item.get("table_caption") or item.get("image_caption") or item.get("chart_caption"))
        body = str(item.get("table_body") or item.get("text") or "")
        text = f"{caption}\n{_strip_html(body)}".strip()
        if not _is_relevant_context(text):
            continue
        page = int(item.get("page_idx", 0)) + 1 if isinstance(item.get("page_idx"), int) else "?"
        contexts.append((f"MinerU {item_type} page {page}: {caption[:80]}", text[:5000]))
    for block in payload.page_blocks:
        if _is_relevant_context(block.text):
            contexts.append((f"{block.source} page {block.page_number}: {block.cue[:80]}", block.text[:5000]))
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, text in contexts:
        key = text[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, text))
    return deduped


def _looks_like_palaeoshoreline_table(header: list[str]) -> bool:
    joined = " ".join(header)
    return "feature type" in joined and "depth range" in joined


def _looks_like_cawthra_feature_table(header: list[str], caption: str) -> bool:
    joined = " ".join(header)
    return "geomorphic feature" in joined or "interpreted timing" in caption.lower()


def _looks_like_dated_altitude_sample_table(header: list[str]) -> bool:
    normalized = " ".join(_normalize_header(cell) for cell in header)
    return "sample" in normalized and "material" in normalized and "locality" in normalized and "age" in normalized and "altitude" in normalized


def _looks_like_radiocarbon_depth_table(header: list[str]) -> bool:
    normalized = " ".join(_normalize_header(cell) for cell in header)
    return "radiocarbon age" in normalized and "depth below sea bed" in normalized


def _looks_like_dated_beach_name_table(header: list[str]) -> bool:
    normalized = " ".join(_normalize_header(cell) for cell in header)
    return "beach name" in normalized and "age" in normalized


def _feature_paragraphs(text: str) -> list[str]:
    names = ["Terraces", "Sea cliffs", "Shelf banks/shoals", "Low-relief ridges", "Incised valleys", "Seabed depressions", "Shelf sediments"]
    paragraphs = [" ".join(part.split()) for part in text.split("\n\n") if part.strip()]
    return [paragraph for paragraph in paragraphs if any(paragraph.startswith(f"{name}.") for name in names)]


def _indicator_from_text(text: str) -> str:
    lowered = text.lower()
    if "reef" in lowered or "coral" in lowered:
        return "coral_reef"
    if "bench" in lowered or "cliff" in lowered or "notch" in lowered:
        return "wave_cut_notch_or_bench"
    if "terrace" in lowered:
        return "marine_terrace"
    if "estuar" in lowered or "channel" in lowered or "valley" in lowered:
        return "estuarine_sediment"
    if "dune" in lowered:
        return "raised_beach"
    if "barrier" in lowered or "beach" in lowered or "shoreline" in lowered or "ridge" in lowered or "shoal" in lowered:
        return "submerged_beach"
    if "facies" in lowered or "sediment" in lowered:
        return "subtidal_facies"
    return "submerged_beach"


def _site_from_feature_text(text: str, fallback: str) -> str:
    for pattern in [
        r"offshore of ([A-Z][A-Za-z ]+?)(?:\s+extends|\s+and|,|\.|$)",
        r"off the ([A-Z][A-Za-z ]+?)(?:\s+River|\s+region|\s+in|,|\.|$)",
        r"vicinity of the ([A-Z][A-Za-z ]+?)(?:\s+indicates|\s+and|,|\.|$)",
        r"adjacent to the ([A-Z][A-Za-z ]+?)(?:\s+River|\s+and|,|\.|$)",
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return fallback


def _depth_summary(text: str, allow_bare_table_cell: bool = False) -> tuple[float | None, list[float | None] | None]:
    depths: list[float] = []
    normalized = text.replace("−", "-")
    patterns = [
        r"(?:water\s+)?depth(?:s| range)?[^.;:]{0,80}?(\d+(?:\.\d+)?)\s*(?:[–-]\s*(\d+(?:\.\d+)?))?\s*m",
        r"(\d+(?:\.\d+)?)\s*(?:[–-]\s*(\d+(?:\.\d+)?))?\s*m\s*bmsl",
        r"(\d+(?:\.\d+)?)\s*(?:[–-]\s*(\d+(?:\.\d+)?))?\s*m\s+below",
        r"at\s+(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s+and\s+~?\s*(\d+(?:\.\d+)?)",
        r"at\s+(?:a\s+)?depth\s+of\s+(\d+(?:\.\d+)?)\s*m",
        r"(\d+(?:\.\d+)?)\s*m\s+isobath",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            values = [float(value) for value in match.groups() if value is not None]
            depths.extend(_clean_depth_values(values, normalized[match.start() : match.end()]))
    if allow_bare_table_cell and not depths:
        for match in DEPTH_PATTERN.finditer(normalized):
            values = [float(value) for value in match.groups() if value is not None]
            depths.extend(_clean_depth_values(values, normalized))
    if not depths:
        return None, None
    low = min(depths)
    high = max(depths)
    midpoint = round((low + high) / 2.0, 2)
    return midpoint, [round(-high, 2), round(-low, 2)]


def _clean_depth_values(values: list[float], context: str) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if "isobath" in context.lower() and value >= 200:
            value -= 200
        if 0 <= value <= 150:
            cleaned.append(value)
    return cleaned


def _age_summary(text: str):
    match = AGE_KA_PATTERN.search(text)
    if not match:
        return None
    if match.group(1):
        return float(match.group(1))
    return match.group(0)


def _has_depth(text: str) -> bool:
    lowered = text.lower()
    return bool(DEPTH_PATTERN.search(text)) and any(token in lowered for token in ["depth", "bmsl", "below", "shoreline", "shelf", "reef", "barrier", "terrace", "shoal", "ridge"])


def _is_relevant_context(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["depth", "bmsl", "shoreline", "palaeoshoreline", "reef", "barrier", "terrace", "shoal", "cliff", "map"])


def _normalize_header(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized = normalized.replace("yrs b p", "yrs bp").replace("years b p", "years bp")
    if normalized == "age year":
        return "age year"
    return normalized


def _number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def _age_bp_to_ka(text: str) -> tuple[float | None, float | None]:
    numbers = [float(value.replace(",", "")) for value in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)]
    if not numbers:
        return None, None
    age_ka = numbers[0] / 1000.0
    uncertainty_ka = numbers[1] / 1000.0 if len(numbers) > 1 else None
    return round(age_ka, 3), round(uncertainty_ka, 3) if uncertainty_ka is not None else None


def _age_text_to_ka(text: str) -> float | list[float | None] | None:
    cleaned = text.replace(",", "")
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", cleaned)]
    if not numbers:
        return None
    if len(numbers) >= 2 and re.search(r"\d+\s*(?:-|–|to)\s*\d+", cleaned):
        low, high = numbers[0], numbers[1]
        if low < 1000 <= high:
            magnitude = 10 ** max(len(str(int(high))) - len(str(int(low))), 0)
            low *= magnitude
        return [round(low / 1000.0, 3), round(high / 1000.0, 3)]
    return round(numbers[0] / 1000.0, 3)


def _indicator_from_material_and_context(material: str, context: str) -> tuple[str, str]:
    lowered = f"{material} {context}".lower()
    if "coral" in lowered:
        return "coral_reef", "sea_level_indicator"
    if "mollusc" in lowered or "mollusca" in lowered or "ostrea" in lowered or "shell" in lowered:
        return "marine_shell_bed", "marine_limiting"
    if "wood" in lowered:
        return "marine_over_terrestrial_contact", "marine_limiting"
    return "raised_beach", "sea_level_indicator"


def _indicative_range_for_indicator(indicator_type: str, elevation_m: float) -> list[float | None]:
    if indicator_type == "coral_reef":
        return [round(elevation_m - 6.0, 2), round(elevation_m + 0.5, 2)]
    if indicator_type == "marine_shell_bed":
        return [round(elevation_m - 5.0, 2), round(elevation_m + 2.0, 2)]
    return [round(elevation_m - 4.0, 2), round(elevation_m + 4.0, 2)]


def _reported_beach_depth_context(text: str) -> tuple[float | None, tuple[float, float] | None]:
    for pattern in [
        r"waterdepths?\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*m",
        r"at\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*m\s+waterdepth",
        r"relative sea level[^.]{0,80}?about\s+(\d+(?:\.\d+)?)\s*m\s+below",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        values = [float(value) for value in match.groups()]
        if len(values) == 1:
            return values[0], (values[0], values[0])
        low, high = min(values), max(values)
        return round((low + high) / 2.0, 2), (low, high)
    return None, None


def _submerged_beach_location(text: str) -> str | None:
    for pattern in [
        r"submerged beach[^.]{0,120}?northern North Sea",
        r"northern part of the North Sea Plateau",
        r"Northern North Sea",
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return "Northern North Sea Plateau"
    return None


def _dated_beach_record(header: list[str], cells: list[str]) -> dict[str, str]:
    if len(cells) == len(header):
        return {header[index]: cells[index] for index in range(len(cells))}
    if len(cells) == 2 and len(header) >= 3:
        return {"beach name": cells[0], "age years bp": cells[1]}
    if len(cells) >= 3:
        return {"group": cells[0], "beach name": cells[1], "age years bp": cells[2]}
    return {}


def _depth_from_beach_name(name: str) -> float | None:
    text = name.replace("−", "-").replace("–", "-")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*fathoms?", text, re.IGNORECASE)
    if match:
        low, high = [float(value) * FATHOM_TO_M for value in match.groups()]
        return round((low + high) / 2.0, 2)
    match = re.search(r"(\d+(?:\.\d+)?)\s*fathoms?", text, re.IGNORECASE)
    if match:
        return round(float(match.group(1)) * FATHOM_TO_M, 2)
    match = re.search(r"toes?\s+at\s+(?:approx\.\s*)?(\d+(?:\.\d+)?)\s*m", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def _clean_beach_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", name).strip(" .;") or "Submerged beach"


def _join_text(value) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item).strip()
    return ""


def _strip_html(value: str) -> str:
    rows = _parse_table_rows(value)
    if rows:
        return "\n".join(" | ".join(row) for row in rows)
    return " ".join(value.split())


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_row is not None and self.current_cell is not None:
            self.current_row.append(" ".join(" ".join(self.current_cell).split()))
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if any(cell for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)


def _parse_table_rows(html: str) -> list[list[str]]:
    if not html:
        return []
    parser = _TableParser()
    parser.feed(html)
    return parser.rows
