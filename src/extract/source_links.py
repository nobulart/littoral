from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from src.extract.settings import source_workspace_root

if TYPE_CHECKING:
    from src.extract.document_loader import DocumentPayload


DOI_PREFIX_PATTERN = re.compile(r"10\.\d{4,9}\s*/\s*", re.IGNORECASE)
PII_PATTERN = re.compile(r"\bPII:\s*(S?\d{4}-\d{4}\(\d{2}\)\d{5}-[A-Z0-9])\b", re.IGNORECASE)
REFERENCE_MARKERS = ("# references", "\nreferences", "\nbibliography", "\nliterature cited")
SOURCE_CONTEXT_MARKERS = (
    "this article",
    "published online",
    "citation",
    "cite this article",
    "supplementary data associated with this article",
)


@dataclass(frozen=True, slots=True)
class SourceLink:
    doi: str
    url: str
    source: str
    score: int


@dataclass(slots=True)
class _Candidate:
    doi: str
    source: str
    score: int
    position: int | None


def determine_source_doi_or_url(source_path: Path, payload: DocumentPayload) -> str:
    link = determine_source_link(source_path, payload)
    return link.url if link is not None else ""


def determine_source_link(source_path: Path, payload: DocumentPayload) -> SourceLink | None:
    candidates: list[_Candidate] = []
    candidates.extend(_candidates_from_pdf_metadata(source_path))
    candidates.extend(_candidates_from_text(payload.title, "payload_title", base_score=80))
    candidates.extend(_candidates_from_text("\n".join(payload.metadata.values()), "payload_metadata", base_score=90))
    candidates.extend(_candidates_from_text(payload.text, "payload_text", base_score=0))
    candidates.extend(_candidates_from_mineru_artifacts(source_path, payload))

    if not candidates:
        return None

    occurrence_counts: dict[str, int] = {}
    for candidate in candidates:
        occurrence_counts[candidate.doi] = occurrence_counts.get(candidate.doi, 0) + 1

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.score + min(occurrence_counts[candidate.doi], 5) * 5,
            -(candidate.position if candidate.position is not None else 10**9),
        ),
        reverse=True,
    )
    best = ranked[0]
    score = best.score + min(occurrence_counts[best.doi], 5) * 5
    return SourceLink(doi=best.doi, url=f"https://doi.org/{best.doi}", source=best.source, score=score)


def _candidates_from_pdf_metadata(source_path: Path) -> list[_Candidate]:
    metadata_text = _pdfinfo_text(source_path)
    if not metadata_text:
        return []
    candidates = _candidates_from_text(metadata_text, "pdf_metadata", base_score=115)
    for pii in PII_PATTERN.findall(metadata_text):
        doi = _normalize_doi(f"10.1016/{pii.upper()}")
        if _valid_doi(doi):
            candidates.append(_Candidate(doi=doi, source="pdf_metadata_pii", score=105, position=0))
    return candidates


def _pdfinfo_text(source_path: Path) -> str:
    if source_path.suffix.lower() != ".pdf" or not source_path.exists():
        return ""
    try:
        result = subprocess.run(["pdfinfo", str(source_path)], check=True, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout


def _candidates_from_mineru_artifacts(source_path: Path, payload: DocumentPayload) -> list[_Candidate]:
    artifact_paths = _mineru_artifact_paths(source_path, payload)
    candidates: list[_Candidate] = []
    for artifact_path in artifact_paths:
        text = _artifact_text(artifact_path)
        if not text:
            continue
        candidates.extend(_candidates_from_text(text, f"mineru_artifact:{artifact_path.name}", base_score=20))
    return candidates


def _mineru_artifact_paths(source_path: Path, payload: DocumentPayload) -> list[Path]:
    paths: list[Path] = []
    for key in ("MinerUContentList", "MinerUMarkdown"):
        value = payload.metadata.get(key)
        if value:
            paths.append(Path(value))

    source_id = source_path.stem.replace(" ", "_").lower()
    stage_dir = source_workspace_root(source_path) / "data" / "staged" / source_id / "hybrid_auto"
    paths.extend(
        [
            stage_dir / f"{source_id}_content_list.json",
            stage_dir / f"{source_id}_content_list_v2.json",
            stage_dir / f"{source_id}_middle.json",
            stage_dir / f"{source_id}.md",
        ]
    )

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _artifact_text(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            return ""
        return "\n".join(_walk_strings(payload))
    return path.read_text(encoding="utf-8", errors="replace")


def _walk_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _candidates_from_text(text: str, source: str, base_score: int) -> list[_Candidate]:
    if not text:
        return []
    normalized_text = _normalize_text(text)
    candidates: list[_Candidate] = []
    for match in DOI_PREFIX_PATTERN.finditer(normalized_text):
        raw = _read_doi_at(normalized_text, match.start())
        doi = _normalize_doi(raw)
        if not _valid_doi(doi):
            continue
        candidates.append(
            _Candidate(
                doi=doi,
                source=source,
                score=base_score + _position_score(normalized_text, match.start()) + _context_score(normalized_text, match.start(), match.end()),
                position=match.start(),
            )
        )
    return candidates


def _read_doi_at(text: str, start: int) -> str:
    index = start
    output: list[str] = []
    while index < len(text):
        char = text[index]
        if char.isalnum() or char in "-._;()/:+":
            output.append(char)
            index += 1
            continue
        if char.isspace():
            next_index = index + 1
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            previous = output[-1] if output else ""
            next_char = text[next_index] if next_index < len(text) else ""
            if _should_join_doi_whitespace(previous, text, next_index):
                index = next_index
                continue
        break
    return "".join(output)


def _should_join_doi_whitespace(previous: str, text: str, next_index: int) -> bool:
    if next_index >= len(text):
        return False
    next_char = text[next_index]
    if not (next_char.isalnum() or next_char in "-._;()/:+"):
        return False
    if previous in {"/", "-", "_"}:
        return True
    if previous != ".":
        return False
    token = _next_doi_token(text, next_index)
    return "." in token or "/" in token or any(char.isdigit() for char in token)


def _next_doi_token(text: str, start: int) -> str:
    token: list[str] = []
    index = start
    while index < len(text):
        char = text[index]
        if char.isalnum() or char in "-._;()/:+":
            token.append(char)
            index += 1
            continue
        break
    return "".join(token)


def _position_score(text: str, position: int) -> int:
    if position < 5_000:
        return 90
    if position < 20_000:
        return 70
    if position < 40_000:
        return 45
    return 5


def _context_score(text: str, start: int, end: int) -> int:
    context = text[max(0, start - 180) : min(len(text), end + 180)].lower()
    score = 0
    if "doi" in context:
        score += 20
    if any(marker in context for marker in SOURCE_CONTEXT_MARKERS):
        score += 50
    references_index = _first_reference_marker(text)
    if references_index is not None and start > references_index and not any(marker in context for marker in SOURCE_CONTEXT_MARKERS):
        score -= 80
    return score


def _first_reference_marker(text: str) -> int | None:
    lowered = text.lower()
    positions = [lowered.find(marker) for marker in REFERENCE_MARKERS]
    positions = [position for position in positions if position != -1]
    return min(positions) if positions else None


def _normalize_text(text: str) -> str:
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\\_", "_")
        .replace("doi. org", "doi.org")
    )


def _normalize_doi(raw: str) -> str:
    doi = _normalize_text(raw)
    doi = re.sub(r"\s+", "", doi)
    doi = doi.rstrip(".,;:)]}")
    return doi


def _valid_doi(doi: str) -> bool:
    if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:+A-Z0-9]+", doi, flags=re.IGNORECASE):
        return False
    suffix = doi.split("/", 1)[1]
    if len(suffix) < 4:
        return False
    if suffix.lower() in {"j", "text"}:
        return False
    if suffix.lower().endswith(("/j", "/text")):
        return False
    return True
