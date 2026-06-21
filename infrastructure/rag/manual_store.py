from __future__ import annotations

import json
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
SOURCE_DIR = Path(os.environ.get("SHERMAN_MANUAL_SOURCE_DIR", WORKSPACE_ROOT / "docs" / "source_pdfs"))
DATA_DIR = Path(os.environ.get("SHERMAN_MANUAL_DATA_DIR", REPO_ROOT / "data" / "manual_assistant"))
INDEX_PATH = DATA_DIR / "index" / "manual_pages.json"

MANUALS = [
    {
        "profile": "cell_operation",
        "manual_id": "Operator's_manual_NEW",
        "filename": "Operator's_manual_NEW.pdf",
    },
    {
        "profile": "software",
        "manual_id": "A933EN_Design_Modul_I_2D",
        "filename": "A933EN_Design_Modul_I_2D.pdf",
    },
    {
        "profile": "software",
        "manual_id": "A934EN_Design_Modul_II_3D",
        "filename": "A934EN_Design_Modul_II_3D.pdf",
    },
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "about",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "please",
    "the",
    "to",
    "tell",
    "what",
    "when",
    "where",
    "who",
    "with",
    "this",
    "that",
    "these",
    "those",
    "זה",
    "זו",
    "זאת",
    "הזה",
    "הזו",
    "этот",
    "эта",
    "это",
    "эти",
    "на",
    "как",
}

NORMALIZATION_TERMS = {
    "פתיחת": "open opening",
    "קובץ": "file",
    "קווי": "lines",
    "כיפוף": "bending bend",
    "כיפופים": "bending bend",
    "מייבאים": "import importing",
    "לייבא": "import importing",
    "איך": "how",
    "מצב": "mode",
    "אוטומטי": "automatic",
    "ידני": "manual",
    "מכונה": "machine",
    "חשמל": "electrical",
    "ארון": "cabinet",
    "עצירת": "stop",
    "חירום": "emergency",
    "מסך": "screen",
    "ממשק": "interface",
    "עגולה": "rounded",
    "פינה": "corner",
    "רדיוס": "radius",
    "קיצור": "shortcut",
    "מקשים": "keys",
    "ייבוא": "import importing",
    "תלת": "3d",
    "ממד": "3d",
    "линии": "lines",
    "гибки": "bending bend",
    "сгиба": "bending bend",
    "импортировать": "import importing",
    "импорт": "import importing",
    "как": "how",
    "файл": "file",
    "шкаф": "cabinet",
    "электрический": "electrical",
    "аварийная": "emergency",
    "аварийной": "emergency",
    "остановить": "stop emergency",
    "остановка": "stop",
    "ручной": "manual",
    "автоматический": "automatic",
    "интерфейс": "interface",
    "экран": "screen",
    "радиальное": "radial",
    "меню": "menu",
    "закругленный": "rounded",
    "угол": "corner",
}

QUERY_EXPANSIONS = {
    "authorized personnel": "trained specialist personnel setup maintenance",
    "automatic mode": "automatic operating mode enabling conditions operating mode key switch",
    "bend radius": "production radius inside radius tool radius die width material thickness",
    "bending lines": "bendings dxf import options drawing window",
    "double sheet": "double sheet recognition double sheet separation sensors c arm",
    "dxf": "dwg import file bending lines import options teczone design",
    "electrical cabinet": "electrician electrical voltage shock main switch discharge cooling",
    "emergency stop": "emergency malfunction stop bendmaster trubend control column foot switch enable key",
    "formed sections": "detect formed sections assign tools automatically",
    "hatching": "special processing hatch outer contour surface",
    "igs": "3d import homezone teczone design importing file",
    "interface": "user interface screen surface panel design window toolguide tools status bar",
    "junction": "seam types no overlap converting into sheet allocate junctions",
    "machining side": "upper machining side upper side processing film scratch-free engraving",
    "main switch": "electrical cabinet switch off secure switched back discharge",
    "manual mode": "manual operating mode enable key portable manual control unit",
    "memo box": "memo box variant flanges pull function sheet options projecting lines validating unfolding",
    "mouse": "working with mouse wheel right button left button selection",
    "no overlap": "seam types junction type converting into sheet allocate junctions",
    "pull function": "generating flanges u-sheet sheet metal area pull function",
    "radial menu": "quick access frequently used tools right mouse button selection mode",
    "referencing": "reference axes z a b c z1 z2 robot position reference marks",
    "rounded corner": "roundings create rounded corner corner notches radius notch",
    "screen surface": "resetting screen surface reset layout modified user interface options",
    "seam": "seam types no overlap junction type converting into sheet allocate junctions",
    "shortcut": "frequently used shortcuts esc ctrl del mouse wheel space bar",
    "step": "stp step 3d import converting sheet allocating junction types simplifying",
    "toolguide": "user interface teczone design panel open files status bar design window tools",
    "toolmaster": "tool changer quick automatic setup tools tool magazine transport slides press brake",
    "upper machining side": "upper side processing laser punching film scratch-free engraving",
    "u-sheet": "generating flanges pull function sheet metal area rectangle",
    "user interface": "screen surface panel design window toolguide tools status bar",
    "ארון חשמל": "electrical cabinet electrician voltage main switch",
    "עצירת חירום": "emergency stop emergency malfunction control column foot switch",
    "מצב אוטומטי": "automatic operating mode enabling conditions",
    "מצב ידני": "manual operating mode enable key",
    "ממשק": "user interface screen surface panel toolguide",
    "כפתור": "button",
    "לחצן": "button",
    "אייקון": "icon",
    "סמל": "icon",
    "מדפסת": "printer",
    "רשת": "network",
    "גיבוי": "backup",
    "ענן": "cloud",
    "חשבונית": "invoice",
    "חשבוניות": "invoices",
    "פינה עגולה": "rounded corner roundings radius",
    "קווי כיפוף": "bending lines bendings dxf import options",
    "ייבוא": "import importing file",
    "аварийная остановка": "emergency stop malfunction control column foot switch",
    "радиальное меню": "radial menu quick access frequently used tools",
    "электрический шкаф": "electrical cabinet electrician voltage main switch",
    "кнопка": "button",
    "значок": "icon",
    "иконка": "icon",
    "принтер": "printer",
    "сетевой": "network",
    "сетевому": "network",
    "сеть": "network",
    "резервное": "backup",
    "копирование": "backup",
    "облако": "cloud",
    "счет": "invoice",
    "счета": "invoices",
}

SUPPORT_SCORE_THRESHOLD = 5.0
SUPPORT_QUERY_COVERAGE_THRESHOLD = 0.45
_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_CACHE: tuple[int, int, list["PageRecord"]] | None = None


@dataclass(frozen=True)
class PageRecord:
    profile: str
    manual_id: str
    filename: str
    pdf_path: str
    page_number: int
    text: str
    token_count: int
    image_count: int
    visual_heavy: bool
    page_image_path: str | None = None
    crop_path: str | None = None
    crop_bbox: list[int] | None = None
    section_title: str = ""
    is_toc: bool = False


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    score: float
    page: PageRecord
    query_term_coverage: float
    matched_query_terms: tuple[str, ...]
    missing_query_terms: tuple[str, ...]
    excerpt: str
    rerank_features: dict[str, float | bool | str]


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_text(value: str, include_query_expansions: bool = False) -> str:
    additions = []
    lowered = value.lower()
    for source, target in NORMALIZATION_TERMS.items():
        if source.lower() in lowered:
            additions.append(target)
    if include_query_expansions:
        for source, target in QUERY_EXPANSIONS.items():
            if source.lower() in lowered:
                additions.append(target)
    if additions:
        value = value + " " + " ".join(additions)
    return value


def tokenize(value: str, include_query_expansions: bool = False) -> list[str]:
    normalized = normalize_text(value, include_query_expansions=include_query_expansions)
    tokens = re.findall(r"[\w][\w\-/.]*", normalized.lower(), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def is_matchable_query_term(term: str) -> bool:
    return bool(re.search(r"[a-z0-9]", term))


def infer_section_title(text: str) -> str:
    compact = compact_whitespace(text)
    if not compact:
        return ""
    compact = re.sub(r"^[A-Z]\d+EN(?:_[A-Z]+)?\s+", "", compact)
    compact = re.sub(r"^\d+(?:[.-]\d+)*\s+", "", compact)
    words = compact.split()
    return " ".join(words[:14])


def is_toc_page(text: str) -> bool:
    prefix = compact_whitespace(text[:800]).lower()
    return "table of contents" in prefix or prefix.startswith("contents ")


def term_coverage(terms: Iterable[str], text: str) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    unique_terms = tuple(sorted({term for term in terms if len(term) >= 3 and is_matchable_query_term(term)}))
    if not unique_terms:
        return 0.0, (), ()
    lowered = text.lower()
    matched = tuple(term for term in unique_terms if term in lowered)
    missing = tuple(term for term in unique_terms if term not in lowered)
    return len(matched) / len(unique_terms), matched, missing


def page_image_count(page) -> int:
    resources = page.get("/Resources") or {}
    xobjects = resources.get("/XObject")
    if not xobjects:
        return 0
    try:
        xobjects = xobjects.get_object()
    except Exception:
        return 0

    count = 0
    for obj_ref in xobjects.values():
        try:
            obj = obj_ref.get_object()
            if obj.get("/Subtype") == "/Image":
                count += 1
        except Exception:
            continue
    return count


def best_excerpt(text: str, query_tokens: list[str], width: int = 520) -> str:
    if not text:
        return ""
    lowered = text.lower()
    positions = [
        lowered.find(token.lower())
        for token in query_tokens
        if len(token) >= 3 and lowered.find(token.lower()) >= 0
    ]
    start = max(0, min(positions) - width // 3) if positions else 0
    excerpt = text[start : start + width].strip()
    if start > 0:
        excerpt = "..." + excerpt
    if start + width < len(text):
        excerpt = excerpt + "..."
    return excerpt


def _manual_pdf_path(manual: dict[str, str]) -> Path:
    return SOURCE_DIR / manual["filename"]


def _page_image_path(manual_id: str, page_number: int) -> Path:
    return DATA_DIR / "page_images" / manual_id / f"page_{page_number:04d}.png"


def _crop_path(manual_id: str, page_number: int) -> Path:
    return DATA_DIR / "crops" / manual_id / f"page_{page_number:04d}_main.png"


def _relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_pdf_pages(pdf_path: Path, manual_id: str, dpi: int = 180) -> int:
    target_dir = DATA_DIR / "page_images" / manual_id
    target_dir.mkdir(parents=True, exist_ok=True)
    first_page = _page_image_path(manual_id, 1)
    if first_page.exists():
        return len(list(target_dir.glob("page_*.png")))

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required to render manual page images.")

    prefix = target_dir / "render"
    subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    rendered = 0
    for rendered_file in sorted(target_dir.glob("render-*.png")):
        match = re.search(r"-(\d+)\.png$", rendered_file.name)
        if not match:
            continue
        page_number = int(match.group(1))
        rendered_file.rename(_page_image_path(manual_id, page_number))
        rendered += 1
    return rendered


def create_main_crop(manual_id: str, page_number: int) -> tuple[Path | None, list[int] | None]:
    image_path = _page_image_path(manual_id, page_number)
    if not image_path.exists():
        return None, None

    crop_path = _crop_path(manual_id, page_number)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if crop_path.exists():
        try:
            with Image.open(crop_path) as image:
                return crop_path, [0, 0, image.width, image.height]
        except Exception:
            return crop_path, None

    with Image.open(image_path) as image:
        width, height = image.size
        left = int(width * 0.05)
        top = int(height * 0.08)
        right = int(width * 0.95)
        bottom = int(height * 0.92)
        image.crop((left, top, right, bottom)).save(crop_path)
        return crop_path, [left, top, right, bottom]


def build_index(render_visuals: bool = True, force: bool = False) -> list[PageRecord]:
    if INDEX_PATH.exists() and not force:
        return load_index()

    records: list[PageRecord] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    for manual in MANUALS:
        pdf_path = _manual_pdf_path(manual)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        reader = PdfReader(str(pdf_path))
        if render_visuals:
            render_pdf_pages(pdf_path, manual["manual_id"])

        for idx, page in enumerate(reader.pages, start=1):
            text = compact_whitespace(page.extract_text() or "")
            tokens = tokenize(text)
            image_count = page_image_count(page)
            visual_heavy = image_count > 0 and len(text) < 1200
            page_image = _page_image_path(manual["manual_id"], idx)
            crop = None
            bbox = None
            if render_visuals and page_image.exists() and (visual_heavy or image_count > 0):
                crop, bbox = create_main_crop(manual["manual_id"], idx)

            records.append(
                PageRecord(
                    profile=manual["profile"],
                    manual_id=manual["manual_id"],
                    filename=manual["filename"],
                    pdf_path=str(pdf_path),
                    page_number=idx,
                    text=text,
                    token_count=len(tokens),
                    image_count=image_count,
                    visual_heavy=visual_heavy,
                    page_image_path=_relative(page_image if page_image.exists() else None),
                    crop_path=_relative(crop),
                    crop_bbox=bbox,
                    section_title=infer_section_title(text),
                    is_toc=is_toc_page(text),
                )
            )

    INDEX_PATH.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _set_index_cache(records)
    return records


def _set_index_cache(records: list[PageRecord]) -> None:
    global _INDEX_CACHE
    try:
        stat = INDEX_PATH.stat()
    except FileNotFoundError:
        return
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE = (stat.st_mtime_ns, stat.st_size, records)


def load_index() -> list[PageRecord]:
    try:
        stat = INDEX_PATH.stat()
    except FileNotFoundError:
        raise
    with _INDEX_CACHE_LOCK:
        if _INDEX_CACHE and _INDEX_CACHE[0] == stat.st_mtime_ns and _INDEX_CACHE[1] == stat.st_size:
            return _INDEX_CACHE[2]
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    allowed_fields = set(PageRecord.__dataclass_fields__)
    records = []
    for raw_record in data:
        record = {key: value for key, value in raw_record.items() if key in allowed_fields}
        text = record.get("text") or ""
        record.setdefault("section_title", infer_section_title(text))
        record.setdefault("is_toc", is_toc_page(text))
        records.append(PageRecord(**record))
    _set_index_cache(records)
    return records


def ensure_index(render_visuals: bool = True) -> list[PageRecord]:
    if INDEX_PATH.exists():
        return load_index()
    return build_index(render_visuals=render_visuals, force=False)


class ManualRetriever:
    def __init__(self, pages: Iterable[PageRecord]):
        self.pages = list(pages)
        self.tokens_by_page = [tokenize(page.text, include_query_expansions=True) for page in self.pages]
        self.term_counts = [Counter(tokens) for tokens in self.tokens_by_page]
        self.doc_lengths = [len(tokens) for tokens in self.tokens_by_page]
        self.avgdl = statistics.mean(self.doc_lengths) if self.doc_lengths else 0.0
        self.df: dict[str, int] = defaultdict(int)
        for tokens in self.tokens_by_page:
            for token in set(tokens):
                self.df[token] += 1
        self.n = len(self.pages)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score_page(self, query_tokens: list[str], idx: int) -> float:
        if not query_tokens or self.avgdl <= 0:
            return 0.0
        k1 = 1.5
        b = 0.75
        counts = self.term_counts[idx]
        dl = self.doc_lengths[idx]
        score = 0.0
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * dl / self.avgdl)
            score += self.idf(term) * (tf * (k1 + 1) / denom)
        return score

    def retrieve(self, query: str, profile: str, top_k: int = 5) -> list[RetrievalHit]:
        query_tokens = tokenize(query, include_query_expansions=True)
        coverage_tokens = tokenize(query, include_query_expansions=False)
        exact_terms = {token for token in query_tokens if len(token) >= 3}
        scored: list[tuple[float, int, PageRecord, dict[str, float | bool | str]]] = []
        visual_query = self._looks_visual_query(query_tokens, query)
        normalized_query = normalize_text(query, include_query_expansions=True).lower()

        for idx, page in enumerate(self.pages):
            if page.profile != profile:
                continue
            score = self.score_page(query_tokens, idx)
            text_lower = page.text.lower()
            section_lower = page.section_title.lower()
            combined = f"{page.section_title} {page.text}"
            core_coverage, _matched_core, _missing_core = term_coverage(coverage_tokens, combined)
            expanded_coverage, _matched_expanded, _missing_expanded = term_coverage(query_tokens, combined)
            heading_coverage, _matched_heading, _missing_heading = term_coverage(query_tokens, page.section_title)

            for term in exact_terms:
                if term in text_lower:
                    score += 0.25
                if term in section_lower:
                    score += 0.75

            score += core_coverage * 4.0
            score += expanded_coverage * 2.0
            score += heading_coverage * 5.0
            phrase_boost = self._phrase_boost(normalized_query, page)
            score += phrase_boost

            if visual_query and page.crop_path:
                score += 1.2
            if page.visual_heavy and visual_query:
                score += 0.6
            if page.is_toc and "contents" not in normalized_query:
                score *= 0.25

            if score > 0:
                scored.append(
                    (
                        score,
                        idx,
                        page,
                        {
                            "bm25_score": round(self.score_page(query_tokens, idx), 4),
                            "core_coverage": round(core_coverage, 3),
                            "expanded_coverage": round(expanded_coverage, 3),
                            "heading_coverage": round(heading_coverage, 3),
                            "phrase_boost": round(phrase_boost, 3),
                            "visual_boost": bool(visual_query and page.crop_path),
                            "toc_penalty": page.is_toc and "contents" not in normalized_query,
                        },
                    )
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            self._hit(rank, score, page, query_tokens, coverage_tokens, features)
            for rank, (score, _idx, page, features) in enumerate(scored[:top_k], start=1)
        ]

    def _looks_visual_query(self, query_tokens: list[str], query: str) -> bool:
        lowered = query.lower()
        visual_terms = {
            "button",
            "diagram",
            "drawing",
            "figure",
            "icon",
            "image",
            "interface",
            "layout",
            "panel",
            "screen",
            "screenshot",
            "ui",
            "visual",
            "window",
        }
        return any(term in query_tokens for term in visual_terms) or any(term in lowered for term in visual_terms)

    def _phrase_boost(self, normalized_query: str, page: PageRecord) -> float:
        text = page.text.lower()
        heading = page.section_title.lower()
        boost = 0.0
        for phrase in QUERY_EXPANSIONS:
            if phrase not in normalized_query:
                continue
            if phrase in heading:
                boost += 3.0
            elif phrase in text:
                boost += 1.5
        return boost

    def _hit(
        self,
        rank: int,
        score: float,
        page: PageRecord,
        query_tokens: list[str],
        coverage_tokens: list[str],
        features: dict[str, float | bool | str],
    ) -> RetrievalHit:
        coverage, matched, missing = term_coverage(coverage_tokens, f"{page.section_title} {page.text}")
        return RetrievalHit(
            rank=rank,
            score=round(score, 4),
            page=page,
            query_term_coverage=round(coverage, 3),
            matched_query_terms=matched,
            missing_query_terms=missing,
            excerpt=best_excerpt(page.text, query_tokens),
            rerank_features=features,
        )


def new_request_id() -> str:
    return str(uuid.uuid4())
