from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.common.models import AgeModel, SamplePoint, SourceLocator, deterministic_sample_id
from src.extract.document_loader import DocumentPayload
from src.extract.geocode import geocode_contextual_location
from src.extract.heuristics import clean_title, infer_record_class


FEET_TO_M = 0.3048
FATHOM_TO_M = 1.8288


@dataclass(slots=True)
class NarrativeFallbackResult:
    sample_points: list[SamplePoint]
    ledger_lines: list[str]
    evidence_count: int


@dataclass(slots=True)
class EvidenceCluster:
    site_name: str
    indicator_type: str
    indicator_subtype: str
    record_class: str
    elevation_m: float | None
    indicative_range_m: list[float | None] | None
    reported_depth_m: float | None
    reported_elevation_m: float | None
    elevation_reference: str
    depth_source: str
    age_ka: float | str | None
    dating_method: str
    description: str
    quote: str
    section: str | None
    notes: str
    confidence_hint: str


def build_narrative_fallback_sample_points(source_id: str, source_path: Path, payload: DocumentPayload) -> NarrativeFallbackResult:
    title = clean_title(payload.title, payload.text, source_path)
    clusters = _extract_evidence_clusters(payload.text)
    points: list[SamplePoint] = []
    ledger_lines: list[str] = []
    for index, cluster in enumerate(clusters, start=1):
        if _is_promotable_cluster(cluster):
            point = _cluster_to_sample_point(source_id, source_path, title, cluster, len(points) + 1)
            points.append(point)
            sample_id = point.sample_id
            promoted = "promoted"
        else:
            sample_id = f"ledger_only_{index}"
            promoted = "ledger-only"
        ledger_lines.append(
            f"- `{sample_id}` {cluster.site_name}: {cluster.indicator_type}, {promoted}, "
            f"elevation/depth `{cluster.elevation_m}`, confidence `{cluster.confidence_hint}`; "
            f"quote `{cluster.quote[:220]}`"
        )
    return NarrativeFallbackResult(sample_points=points, ledger_lines=ledger_lines, evidence_count=len(clusters))


def _extract_evidence_clusters(text: str) -> list[EvidenceCluster]:
    paragraphs = _sectioned_paragraphs(text)
    clusters: list[EvidenceCluster] = []
    for section, paragraph in paragraphs:
        lowered = paragraph.lower()
        if "raised beach" in lowered:
            clusters.extend(_raised_beach_clusters(section, paragraph))
        if "submerged forest" in lowered or ("stumps of trees" in lowered and "low-water" in lowered):
            clusters.extend(_submerged_forest_clusters(section, paragraph))
        if any(token in lowered for token in ["marine terrace", "terrace", "palaeoshoreline", "palaeoshore platform", "shoreline"]):
            clusters.extend(_submerged_landform_clusters(section, paragraph))
    return _deduplicate_clusters(clusters)


def _sectioned_paragraphs(text: str) -> list[tuple[str | None, str]]:
    section: str | None = None
    paragraphs: list[tuple[str | None, str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if current:
                paragraphs.append((section, " ".join(current)))
                current = []
            continue
        if line.startswith("#"):
            if current:
                paragraphs.append((section, " ".join(current)))
                current = []
            section = line.strip("# ").strip() or section
            continue
        if _looks_like_short_heading(line):
            if current:
                paragraphs.append((section, " ".join(current)))
                current = []
            section = line
            continue
        current.append(line)
    if current:
        paragraphs.append((section, " ".join(current)))
    return paragraphs


def _looks_like_short_heading(line: str) -> bool:
    if len(line) > 80 or len(line.split()) > 8:
        return False
    lowered = line.lower().strip(":")
    return lowered in {
        "nw gozo",
        "nw malta",
        "ne malta",
        "se malta",
        "sikka il-bajda",
        "western sector of the submerged palaeolandscape",
        "northeastern sector of the submerged palaeolandscape",
        "southeastern sector of the submerged palaeolandscape",
    }


def _raised_beach_clusters(section: str | None, paragraph: str) -> list[EvidenceCluster]:
    clusters: list[EvidenceCluster] = []
    site_name = _infer_site_name(section, paragraph)
    quote = _short_quote(paragraph, "raised beach")
    ranges = []
    for pattern in [
        r"(\d+(?:\.\d+)?)\s*(?:or|to|-|–)\s*(\d+(?:\.\d+)?)\s*feet\s+above\s+(?:the\s+)?present\s+high-water",
        r"(\d+(?:\.\d+)?)\s*feet\s+(?:over|above)\s+(?:the\s+)?highest\s+shingle",
    ]:
        for match in re.finditer(pattern, paragraph, re.IGNORECASE):
            values_ft = [float(value) for value in match.groups() if value is not None]
            ranges.append((min(values_ft) * FEET_TO_M, max(values_ft) * FEET_TO_M, "reported above high-water/highest shingle"))
    if not ranges and "raised beach" in paragraph.lower():
        ranges.append((None, None, "raised beach described without extractable elevation"))
    for low, high, relation in ranges:
        elevation = round((low + high) / 2.0, 2) if low is not None and high is not None else None
        indicative = [round(low, 2), round(high, 2)] if low is not None and high is not None else None
        clusters.append(
            EvidenceCluster(
                site_name=site_name,
                indicator_type="raised_beach",
                indicator_subtype="narrative fallback: raised beach",
                record_class="sea_level_indicator",
                elevation_m=elevation,
                indicative_range_m=indicative,
                reported_depth_m=None,
                reported_elevation_m=elevation,
                elevation_reference="unknown",
                depth_source="reported" if elevation is not None else "other",
                age_ka=None,
                dating_method="other",
                description=f"Raised beach evidence at {site_name}",
                quote=quote,
                section=section,
                notes=f"Narrative fallback cluster; vertical relation is {relation}. Datum is not MSL and should be reviewed.",
                confidence_hint="moderate" if elevation is not None else "low",
            )
        )
    return clusters


def _submerged_forest_clusters(section: str | None, paragraph: str) -> list[EvidenceCluster]:
    site_name = _infer_site_name(section, paragraph)
    quote = _short_quote(paragraph, "submerged forest")
    if quote == paragraph[:800]:
        quote = _short_quote(paragraph, "stumps of trees")
    elevation = 0.0 if "low-water" in paragraph.lower() else None
    indicative = [-3.0, 0.0] if elevation is not None else None
    return [
        EvidenceCluster(
            site_name=site_name,
            indicator_type="terrestrial_over_marine_contact",
            indicator_subtype="narrative fallback: submerged forest or peat with in-situ stumps",
            record_class="terrestrial_limiting",
            elevation_m=elevation,
            indicative_range_m=indicative,
            reported_depth_m=None,
            reported_elevation_m=elevation,
            elevation_reference="unknown",
            depth_source="other",
            age_ka=None,
            dating_method="stratigraphic_context",
            description=f"Submerged forest evidence at {site_name}",
            quote=quote,
            section=section,
            notes="Narrative fallback cluster; low-water exposure is treated as an approximate terrestrial limiting position with broad vertical uncertainty.",
            confidence_hint="moderate" if elevation is not None else "low",
        )
    ]


def _submerged_landform_clusters(section: str | None, paragraph: str) -> list[EvidenceCluster]:
    clusters: list[EvidenceCluster] = []
    lowered = paragraph.lower()
    if "terrace" not in lowered and "shoreline" not in lowered and "shore platform" not in lowered and "palaeoshore" not in lowered:
        return clusters
    site_name = _infer_site_name(section, paragraph)
    indicator_type = _indicator_from_narrative(paragraph)
    quote = _short_quote(paragraph, "terrace" if "terrace" in lowered else "shoreline")
    depths = _extract_depth_values(paragraph)
    age = _extract_age_context(paragraph)
    if not depths:
        return clusters
    for depth in depths[:6]:
        clusters.append(
            EvidenceCluster(
                site_name=site_name,
                indicator_type=indicator_type,
                indicator_subtype="narrative fallback: submerged geomorphic sea-level feature",
                record_class=infer_record_class(indicator_type),
                elevation_m=round(-depth, 2),
                indicative_range_m=[round(-depth - 2.5, 2), round(-depth + 2.5, 2)],
                reported_depth_m=round(depth, 2),
                reported_elevation_m=None,
                elevation_reference="MSL",
                depth_source="reported",
                age_ka=age,
                dating_method="stratigraphic_context" if age is not None else "other",
                description=f"{indicator_type.replace('_', ' ')} at {site_name}, reported around {depth:g} m depth",
                quote=quote,
                section=section,
                notes="Narrative fallback cluster from prose description of submerged palaeolandscape; depth uncertainty widened pending human review.",
                confidence_hint="moderate",
            )
        )
    return clusters


def _extract_depth_values(text: str) -> list[float]:
    normalized = text.replace("−", "-").replace("–", "-")
    values: list[float] = []
    patterns = [
        r"depth(?:s| range)?\s+(?:of\s+)?(?:around|approximately|between|ranging between|from)?\s*(\d+(?:\.\d+)?)\s*(?:-|and|to)\s*(\d+(?:\.\d+)?)\s*m",
        r"depth(?:s)?\s*(?:of|at|around|approximately)?\s*\(?((?:\d+(?:\.\d+)?(?:\s*,\s*|\s+and\s+|,\s*and\s*|\s+))*\d+(?:\.\d+)?)\s*m\)?",
        r"located at\s+(?:a\s+)?(?:mean\s+)?depth\s+of\s+(\d+(?:\.\d+)?)\s*m",
        r"(\d+(?:\.\d+)?)\s*m\s+isobath",
        r"sea level (?:was at|had dropped to)\s+2?(\d+(?:\.\d+)?)\s*m",
        r"located at\s+2?(\d+(?:\.\d+)?)\s*m",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            groups = [group for group in match.groups() if group]
            if len(groups) == 1 and re.search(r"[, ](?:and\s+)?\d", groups[0]):
                values.extend(_numbers_from_depth_list(groups[0]))
            else:
                numbers = [float(group) for group in groups]
                if len(numbers) == 2:
                    values.append(round((numbers[0] + numbers[1]) / 2.0, 2))
                else:
                    values.extend(numbers)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*fathoms", normalized, re.IGNORECASE):
        low, high = [float(value) * FATHOM_TO_M for value in match.groups()]
        values.append(round((low + high) / 2.0, 2))
    cleaned: list[float] = []
    for value in values:
        if 0 <= value <= 160 and value not in cleaned:
            cleaned.append(round(value, 2))
    return cleaned


def _numbers_from_depth_list(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]


def _extract_age_context(text: str) -> float | str | None:
    lowered = text.lower()
    ka_match = re.search(r"(\d+(?:\.\d+)?)\s*ka", text, re.IGNORECASE)
    if ka_match:
        return float(ka_match.group(1))
    if "last glacial maximum" in lowered or "lgm" in lowered:
        return "LGM"
    if "holocene" in lowered:
        return "Holocene"
    if "pleistocene" in lowered:
        return "Pleistocene"
    return None


def _indicator_from_narrative(text: str) -> str:
    lowered = text.lower()
    if "terrace" in lowered:
        return "marine_terrace"
    if "shore platform" in lowered or "palaeoshore platform" in lowered:
        return "wave_cut_notch_or_bench"
    if "shoreline" in lowered or "palaeoshoreline" in lowered:
        return "submerged_beach"
    return "submerged_beach"


def _infer_site_name(section: str | None, text: str) -> str:
    candidates: list[str] = []
    title_match = re.search(r"\bNEAR\s+([A-Z][A-Z-]+(?:-[A-Z]+)*)", text)
    if title_match:
        return _clean_site_name(title_match.group(1).title())
    if ("submerged forest" in text.lower() or "stumps of trees" in text.lower()) and "wissant" in text.lower():
        return "Wissant"
    for pattern in [
        r"offshore of ([A-Z][A-Za-z' -]+?)(?:,|\.|\sand|\sin|\sstretches|\sextends|\sis|\shas|\sranges)",
        r"off ([A-Z][A-Za-z' -]+?)(?:,|\.|\sand|\sin|\sstretches|\sextends|\sis|\shas|\sranges)",
        r"between ([A-Z][A-Za-z' -]+?) and ([A-Z][A-Za-z' -]+?)(?:,|\.|\swhen|\swhere)",
        r"vicinity of ([A-Z][A-Za-z' -]+?)(?:,|\.|\sand|\sin)",
        r"neighbourhood of ([A-Z][A-Za-z' -]+?)(?:,|\.|\sand|\sin)",
        r"near ([A-Z][A-Za-z' -]+?)(?:,|\.|\sand|\sin)",
    ]:
        match = re.search(pattern, text)
        if not match:
            continue
        groups = [group.strip(" -") for group in match.groups() if group]
        candidates.append(" and ".join(groups))
    if candidates:
        return _clean_site_name(candidates[0])
    if "direction of sangatte" in text.lower() or "near sangatte" in text.lower():
        return "Sangatte"
    if "wissant" in text.lower():
        return "Wissant"
    if "sangatte" in text.lower():
        return "Sangatte"
    if section:
        return _clean_site_name(section)
    for name in ["Wissant", "Sangatte", "Cap Blanc-nez", "NW Gozo", "NW Malta", "SE Malta", "NE Malta", "Sikka il-Bajda", "Malta"]:
        if name.lower() in text.lower():
            return name
    return "Narrative evidence locality"


def _clean_site_name(value: str) -> str:
    value = " ".join(value.split()).strip(" ,.;:")
    value = re.sub(r"\b(the|former|latter)\b$", "", value, flags=re.IGNORECASE).strip()
    return value[:120] or "Narrative evidence locality"


def _short_quote(text: str, cue: str) -> str:
    lowered = text.lower()
    index = lowered.find(cue.lower())
    if index == -1:
        return " ".join(text.split())[:800]
    start = max(0, index - 220)
    end = min(len(text), index + 580)
    return " ".join(text[start:end].split())[:800]


def _deduplicate_clusters(clusters: list[EvidenceCluster]) -> list[EvidenceCluster]:
    deduped: list[EvidenceCluster] = []
    seen: set[tuple[str, str, float | None, str | None]] = set()
    for cluster in clusters:
        key = (cluster.site_name.lower(), cluster.indicator_type, cluster.elevation_m, cluster.section)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cluster)
    return deduped[:16]


def _is_promotable_cluster(cluster: EvidenceCluster) -> bool:
    if cluster.elevation_m is not None or cluster.reported_depth_m is not None or cluster.reported_elevation_m is not None:
        return True
    return False


def _cluster_to_sample_point(source_id: str, source_path: Path, title: str, cluster: EvidenceCluster, index: int) -> SamplePoint:
    locator = SourceLocator(section=cluster.section, quote_or_paraphrase=cluster.quote)
    geocode_result = geocode_contextual_location(source_path, [cluster.site_name, cluster.description, title], title, cluster.quote)
    latitude = geocode_result.latitude if geocode_result is not None else None
    longitude = geocode_result.longitude if geocode_result is not None else None
    location_name = geocode_result.display_name if geocode_result is not None else cluster.site_name
    coordinate_uncertainty_m = geocode_result.uncertainty_m if geocode_result is not None else None
    point = SamplePoint(
        id="",
        source_id=source_id,
        record_class=cluster.record_class,
        site_name=cluster.site_name,
        sample_id=f"narrative_fallback_{index}",
        latitude=latitude,
        longitude=longitude,
        coordinate_source="inferred_text" if geocode_result is not None else "inferred_map",
        coordinate_uncertainty_m=coordinate_uncertainty_m,
        elevation_m=cluster.elevation_m,
        elevation_reference=cluster.elevation_reference,
        depth_source=cluster.depth_source,
        indicator_type=cluster.indicator_type,
        indicator_subtype=cluster.indicator_subtype,
        indicative_range_m=cluster.indicative_range_m,
        age_ka=cluster.age_ka,
        dating_method=cluster.dating_method,
        description=cluster.description,
        location_name=location_name,
        bibliographic_reference=title,
        doi_or_url="",
        confidence_score=None,
        notes=cluster.notes + (f" Geocoded from contextual query '{geocode_result.query}'." if geocode_result is not None else " No contextual geocode result was found."),
        source_locator=locator,
        age_models=[
            AgeModel(
                method=cluster.dating_method,
                relation="unknown",
                age_ka=cluster.age_ka,
                notes="Narrative fallback age context; review before analytical use.",
            )
        ],
    )
    point.reported_observations.reported_depth_m = cluster.reported_depth_m
    point.reported_observations.reported_elevation_m = cluster.reported_elevation_m
    point.reported_observations.reported_datum = cluster.elevation_reference
    point.id = deterministic_sample_id(source_id, point.site_name, point.sample_id, point.latitude, point.longitude, point.indicator_type, locator)
    return point
