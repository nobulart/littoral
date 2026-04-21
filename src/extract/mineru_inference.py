from __future__ import annotations

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


@dataclass(slots=True)
class MinerUInferenceResult:
    sample_points: list[SamplePoint]
    deterministic_records: int
    llm_contexts: list[tuple[str, str]]


def mine_mineru_outputs(source_id: str, source_path: Path, payload: DocumentPayload) -> MinerUInferenceResult:
    content_items = _load_content_items(payload)
    table_points = _mine_tables(source_id, source_path, payload, content_items)
    feature_points = _mine_feature_paragraphs(source_id, source_path, payload)
    contexts = _build_llm_contexts(payload, content_items)
    points = table_points + feature_points
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
