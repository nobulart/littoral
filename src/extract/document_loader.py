from __future__ import annotations

import csv
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from src.extract.ollama_client import OllamaClient
from src.extract.settings import load_extraction_settings, source_workspace_root


@dataclass(slots=True)
class PageOCRBlock:
    page_number: int
    source: str
    cue: str
    text: str


@dataclass(slots=True)
class DocumentPayload:
    title: str
    text: str
    source_format: str
    extraction_methods: list[str] = field(default_factory=list)
    page_count: int | None = None
    native_text_length: int = 0
    ocr_text_length: int = 0
    text_quality_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)
    page_blocks: list[PageOCRBlock] = field(default_factory=list)


def load_document_payload(source_path: Path) -> DocumentPayload:
    suffix = source_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _load_text_payload(source_path)
    if suffix == ".csv":
        return _load_csv_payload(source_path)
    if suffix == ".pdf":
        return _load_pdf_payload(source_path)
    raise ValueError(f"Unsupported source format: {suffix}")


def _load_text_payload(source_path: Path) -> DocumentPayload:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    title = _first_nonempty_line(text) or source_path.stem
    return DocumentPayload(
        title=title,
        text=text,
        source_format="text",
        extraction_methods=["native_text"],
        native_text_length=len(text),
        text_quality_score=score_text_quality(text),
    )


def _load_csv_payload(source_path: Path) -> DocumentPayload:
    with source_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    text = "\n".join([",".join(row) for row in rows])
    title = rows[0][0] if rows and rows[0] else source_path.stem
    return DocumentPayload(
        title=title,
        text=text,
        source_format="csv",
        extraction_methods=["native_csv"],
        native_text_length=len(text),
        text_quality_score=score_text_quality(text),
    )


def _load_pdf_payload(source_path: Path) -> DocumentPayload:
    mineru_payload = _load_mineru_payload(source_path)
    if mineru_payload is not None:
        return mineru_payload

    settings = load_extraction_settings(source_path)
    pdf_settings = settings["pdf"]
    pdf_metadata = _pdfinfo_metadata(source_path)
    title = pdf_metadata.get("Title") or source_path.stem
    page_count = int(pdf_metadata["Pages"]) if pdf_metadata.get("Pages", "").isdigit() else None

    native_text = _run_command(["pdftotext", str(source_path), "-"])
    page_texts = split_pdf_pages(native_text)
    native_quality = score_text_quality(native_text)
    best_text = native_text
    best_quality = native_quality
    methods = ["pdftotext"]
    ocr_text = ""
    page_blocks: list[PageOCRBlock] = []

    should_ocr = (
        pdf_settings.get("ocr_enabled", True)
        and (len(native_text) < pdf_settings.get("native_text_min_length", 1500) or native_quality < pdf_settings.get("min_quality_score", 0.55))
    )
    if should_ocr:
        ocr_text = _ocr_pdf_text(source_path, int(pdf_settings.get("ocr_max_pages", 12)), int(pdf_settings.get("ocr_dpi", 200)))
        ocr_quality = score_text_quality(ocr_text)
        methods.append("tesseract_ocr")
        if ocr_quality > best_quality or (ocr_quality >= best_quality and len(ocr_text) > len(best_text)):
            best_text = ocr_text
            best_quality = ocr_quality

    if pdf_settings.get("page_ocr_enabled", True):
        page_blocks = _extract_page_ocr_blocks(source_path, page_texts, int(pdf_settings.get("page_ocr_max_pages", 6)), int(pdf_settings.get("ocr_dpi", 200)), str(pdf_settings.get("page_cue_pattern", "table|fig\\.|figure|plate|map|caption")))
        if page_blocks:
            methods.append("glm_ocr_page_blocks")

    return DocumentPayload(
        title=title,
        text=best_text,
        source_format="pdf",
        extraction_methods=methods,
        page_count=page_count,
        native_text_length=len(native_text),
        ocr_text_length=len(ocr_text),
        text_quality_score=best_quality,
        metadata=pdf_metadata,
        page_blocks=page_blocks,
    )


def _load_mineru_payload(source_path: Path) -> DocumentPayload | None:
    source_id = source_path.stem.replace(" ", "_").lower()
    workspace_root = source_workspace_root(source_path)
    stage_dir = workspace_root / "data" / "staged" / source_id / "hybrid_auto"
    markdown_path = stage_dir / f"{source_id}.md"
    content_path = stage_dir / f"{source_id}_content_list.json"
    if not markdown_path.exists():
        return None

    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    content_items = _load_mineru_content_items(content_path)
    title = _mineru_title(content_items, text, source_path)
    page_count = _mineru_page_count(content_items)
    page_blocks = _mineru_page_blocks(content_items)
    return DocumentPayload(
        title=title,
        text=text,
        source_format="pdf",
        extraction_methods=["mineru_hybrid_auto_markdown", "mineru_content_list"],
        page_count=page_count,
        native_text_length=len(text),
        ocr_text_length=0,
        text_quality_score=score_text_quality(text),
        metadata={
            "MinerUStageDir": str(stage_dir),
            "MinerUMarkdown": str(markdown_path),
            "MinerUContentList": str(content_path) if content_path.exists() else "",
        },
        page_blocks=page_blocks,
    )


def _load_mineru_content_items(content_path: Path) -> list[dict]:
    if not content_path.exists():
        return []
    try:
        payload = json.loads(content_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _mineru_title(content_items: list[dict], text: str, source_path: Path) -> str:
    for item in content_items:
        if item.get("type") == "text" and int(item.get("text_level") or 0) == 1:
            title = str(item.get("text") or "").strip()
            if title:
                return " ".join(title.split())
    return _first_nonempty_line(text).lstrip("# ").strip() or source_path.stem


def _mineru_page_count(content_items: list[dict]) -> int | None:
    page_indexes = [item.get("page_idx") for item in content_items if isinstance(item.get("page_idx"), int)]
    if not page_indexes:
        return None
    return max(page_indexes) + 1


def _mineru_page_blocks(content_items: list[dict]) -> list[PageOCRBlock]:
    blocks: list[PageOCRBlock] = []
    for item in content_items:
        item_type = str(item.get("type") or "")
        if item_type not in {"image", "table", "chart"}:
            continue
        text = _mineru_block_text(item)
        if not text:
            continue
        page_idx = item.get("page_idx")
        page_number = int(page_idx) + 1 if isinstance(page_idx, int) else 0
        cue = _mineru_block_cue(item_type, item)
        blocks.append(PageOCRBlock(page_number=page_number, source=f"mineru_{item_type}", cue=cue, text=text))
    return blocks


def _mineru_block_cue(item_type: str, item: dict) -> str:
    captions = item.get(f"{item_type}_caption") or item.get("table_caption") or item.get("image_caption") or []
    caption = _join_mineru_text(captions)
    if caption:
        return caption[:120]
    return item_type


def _mineru_block_text(item: dict) -> str:
    parts: list[str] = []
    for key in ("image_caption", "table_caption", "chart_caption", "image_footnote", "table_footnote"):
        value = _join_mineru_text(item.get(key))
        if value:
            parts.append(value)
    table_body = item.get("table_body")
    if isinstance(table_body, str):
        parts.append(_html_to_text(table_body))
    for key in ("text", "sub_type", "img_path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(part for part in parts if part).strip()


def _join_mineru_text(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
        return " ".join(" ".join(parts).split())
    return ""


class _TableHTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if cleaned:
            self.parts.append(cleaned)


def _html_to_text(html: str) -> str:
    parser = _TableHTMLTextParser()
    parser.feed(html)
    return " | ".join(parser.parts)


def _pdfinfo_metadata(source_path: Path) -> dict[str, str]:
    output = _run_command(["pdfinfo", str(source_path)])
    metadata: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _ocr_pdf_text(source_path: Path, max_pages: int, dpi: int) -> str:
    metadata = _pdfinfo_metadata(source_path)
    page_count = int(metadata["Pages"]) if metadata.get("Pages", "").isdigit() else 1
    last_page = min(page_count, max_pages)

    with tempfile.TemporaryDirectory(prefix="littoral-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        prefix = temp_path / "page"
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-f", "1", "-l", str(last_page), "-png", str(source_path), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
        page_texts: list[str] = []
        for image_path in sorted(temp_path.glob("page-*.png")):
            text = _run_command(["tesseract", str(image_path), "stdout", "--psm", "1"], allow_failure=True)
            if text:
                page_texts.append(text)
        return "\n".join(page_texts)


def _extract_page_ocr_blocks(source_path: Path, page_texts: list[str], max_pages: int, dpi: int, cue_pattern: str) -> list[PageOCRBlock]:
    cue_regex = re.compile(cue_pattern, re.IGNORECASE)
    candidate_pages: list[tuple[int, str]] = []
    for index, page_text in enumerate(page_texts, start=1):
        page_quality = score_text_quality(page_text)
        cue_match = cue_regex.search(page_text)
        if cue_match and (page_quality < 0.7 or len(page_text) < 1200):
            candidate_pages.append((index, cue_match.group(0)))
            continue
        if page_quality < 0.45:
            candidate_pages.append((index, "low_text_quality"))

    candidate_pages = candidate_pages[:max_pages]
    if not candidate_pages:
        return []

    ollama = OllamaClient(source_path)
    if not ollama.can_run_model("glm-ocr:latest"):
        return []

    blocks: list[PageOCRBlock] = []
    with tempfile.TemporaryDirectory(prefix="littoral-page-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        for page_number, cue in candidate_pages:
            image_path = _render_pdf_page_to_png(source_path, page_number, dpi, temp_path)
            if image_path is None:
                continue
            ocr_text = ollama.ocr_page_image(
                image_path,
                prompt=(
                    "Perform OCR on this literature page. Focus on tables, maps, figure labels, locality names, coordinates, depths, elevations, dates, captions, and sample identifiers. "
                    "Return plain text only, preserving labels where possible."
                ),
            )
            source = "glm_ocr"
            if not ocr_text:
                ocr_text = _run_command(["tesseract", str(image_path), "stdout", "--psm", "1"], allow_failure=True)
                source = "tesseract_page_fallback"
            if ocr_text:
                blocks.append(PageOCRBlock(page_number=page_number, source=source, cue=cue, text=ocr_text))
    return blocks


def _render_pdf_page_to_png(source_path: Path, page_number: int, dpi: int, temp_path: Path) -> Path | None:
    prefix = temp_path / f"page-{page_number}"
    try:
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-f", str(page_number), "-l", str(page_number), "-png", str(source_path), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    images = sorted(temp_path.glob(f"page-{page_number}-*.png"))
    return images[0] if images else None


def _run_command(command: list[str], allow_failure: bool = False) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        if allow_failure:
            return ""
        raise
    return result.stdout


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def score_text_quality(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    alpha = sum(1 for char in text if char.isalpha())
    whitespace = sum(1 for char in text if char.isspace())
    weird = sum(1 for char in text if ord(char) > 126 and char not in {"°", "–", "—", "©"})
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    average_line_length = sum(len(line) for line in lines[:200]) / max(min(len(lines), 200), 1)

    alpha_ratio = alpha / total
    whitespace_ratio = whitespace / total
    weird_ratio = weird / total
    line_score = 1.0 if 25 <= average_line_length <= 180 else 0.6
    score = (alpha_ratio * 0.45) + (min(whitespace_ratio, 0.25) * 1.2) + ((1 - min(weird_ratio, 0.2)) * 0.25) + (line_score * 0.1)
    return round(max(0.0, min(score, 1.0)), 3)


def split_pdf_pages(text: str) -> list[str]:
    pages = [page for page in text.split("\f")]
    if len(pages) == 1:
        return [text]
    return pages
