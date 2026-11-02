"""An artifact's text, read by its format, versioned and recorded (decisions: "An item's text is everything its submitter
filed ... a PDF page without a text layer is read by OCR ... each page's method recorded in the extraction"; spec §5: a
claim's span is an offset into one extraction of an artifact).

Each reader has a name and a version. An extraction's id names the reader, its version and the tools that produced the
text (pypdf, poppler's pdftoppm, tesseract), and a paged document's text heads each page with the method that read it,
so the committed text says how every part of it was read.

Every extraction is recorded as a cassette (provider "extraction"), keyed by the artifact and the reader's name and
version, never by the local tools' versions, which the recorded result names: a replay anywhere, a fresh case or a clean
clone, reads the recorded text and never re-runs OCR, whose output varies with the tool's version.

What a file's own bytes make unreadable (a corrupt file, a page poppler cannot render, a format no reader here reads) is
a result, recorded with its reason. What the machine makes unreadable (a tool missing, OCR that cannot start) is no fact
about the file: it raises and records nothing.

The XML of .docx, .xlsx and .pptx files is parsed by the standard library's ElementTree: it resolves no external entity,
and the expat it runs on (2.6 here) refuses entity-expansion bombs (expat >= 2.4.0 has billion-laughs protection).
"""

from __future__ import annotations

import html
import io
import json
import re
import shutil
import subprocess  # noqa: S404  (poppler and tesseract, fixed arguments, no shell)
import tempfile
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree as ET  # noqa: S405  (see the module docstring)

import pypdf

from requisite import cassettes

PROVIDER = "extraction"
EXTRACTOR = "requisite.evidence.extract"
# Each reader's version: a change to what a reader returns for the same bytes is a new version, so a new extraction.
VERSIONS: Mapping[str, int] = {
    "comment": 1,
    "pdf": 2,  # 2 (November 2): a text layer in a font with no Unicode map is checked against OCR
    "docx": 1,
    "xlsx": 1,
    "pptx": 1,
    "html": 1,
    "text": 1,
    "image": 1,
}
# A file format (as the API's fileFormats names it, or a file's extension) and the reader that reads it.
READERS: Mapping[str, str] = {
    "pdf": "pdf",
    "docx": "docx",
    "xlsx": "xlsx",
    "pptx": "pptx",
    "htm": "html",
    "html": "html",
    "txt": "text",
    "text": "text",
    "csv": "text",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "gif": "image",
    "tif": "image",
    "tiff": "image",
    "bmp": "image",
    "webp": "image",
}
OCR_DPI = 300
# Spreadsheet cells in a row are joined by this, so a row keeps its columns once whitespace is normalized (ours).
CELL_SEPARATOR = " | "
# A page with images whose text layer holds fewer letters and digits than this is read by OCR: a scanned page carries at
# most a stamp or a page number in its text layer, while a typed page carries hundreds (ours). OCR reads the rendered
# page, so whatever text layer it has is read with it.
SCANNED_BELOW = 200
# A text layer can carry glyph codes rather than characters: a font's encoding and Unicode map are the producer's word,
# and both can be wrong (seen November 2: TREAS-DO-2026-0232-0055, TrueType subsets declared WinAnsi with no ToUnicode,
# and Type0 fonts whose ToUnicode maps to punctuation, read as "!\"#$%&'()*"). A page whose layer is suspect is read by OCR
# too, an independent reading of the same page, and its text layer stands only where it carries at least
# CARRIES_AT_LEAST of the words OCR reads there (alphabetic words of three letters or more): a correct layer carries
# nearly all of them, OCR's own misreadings aside, and a glyph-coded one almost none (ours). Where OCR reads fewer than
# CHECK_WORDS words the layer stands, there being too little to check it against.
# Which pages are verified (OCR on every page would read F4's thousands of typed pages twice): those set in a subset
# font with no Unicode map (_unmapped), and those whose layer is more than SYMBOLS_ABOVE symbols that prose rarely holds
# (SYMBOLS, dots, commas and hyphens left out, so a table of contents is not one); measured on F4's 6,850 text-layer
# pages: the glyph-coded ones 0.63-0.65, the next 0.46, typed prose about 0.01. A page that passes the screen is taken
# from its text layer unverified: a garbling into plain letters would pass it (ours, November 2).
CARRIES_AT_LEAST = 0.5
CHECK_WORDS = 20
SYMBOLS = frozenset("!\"#$%&'()*+/:;<=>?@[\\]^_`{|}~")
SYMBOLS_ABOVE = 0.15
RENDER_TIMEOUT_S = 300
OCR_TIMEOUT_S = 600


class ExtractionError(RuntimeError):
    """The machine cannot read here what the file may well hold (a tool missing or failing on our own rendering)."""


@dataclass(frozen=True)
class Extraction:
    """What a reader made of one artifact. id: the reader, its version and the tools that produced the text (None when
    nothing could be read). pages: each page's method, for a paged format. unreadable: what could not be read, and why
    (the whole file, or pages of it)."""

    reader: str
    id: str | None
    text: str | None
    pages: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()

    def kept(self) -> dict[str, Any]:
        return {"reader": self.reader, "id": self.id, "text": self.text, "pages": list(self.pages),
                "unreadable": list(self.unreadable)}  # fmt: skip

    @staticmethod
    def from_kept(kept: Mapping[str, Any]) -> Extraction:
        return Extraction(kept["reader"], kept["id"], kept["text"], tuple(kept["pages"]), tuple(kept["unreadable"]))


def extract(reader: str, artifact: str, content: bytes, *, root: Path | None = None,
            budget: cassettes.Budget | None = None) -> Extraction:  # fmt: skip
    """The artifact's text by `reader`, replayed from its cassette or read once here and recorded."""
    if reader not in VERSIONS:
        raise ValueError(f"no reader {reader!r}")
    body = {"artifact": artifact, "extractor": f"{EXTRACTOR}.{reader}", "version": VERSIONS[reader]}
    recorded = cassettes.call(
        PROVIDER, reader, body, lambda: (read(reader, content).kept(), 0.0), budget=budget, estimate_usd=0.0, root=root
    )
    return Extraction.from_kept(recorded.response)


def read(reader: str, content: bytes) -> Extraction:
    """Read the bytes here, now (no cassette)."""
    return {
        "comment": _comment,
        "pdf": _pdf,
        "docx": _docx,
        "xlsx": _xlsx,
        "pptx": _pptx,
        "html": _html,
        "text": _text,
        "image": _image,
    }[reader](content)


def _id(reader: str, tools: str = "") -> str:
    return f"{reader}/{VERSIONS[reader]}" + (f" {tools}" if tools else "")


def _unreadable(reader: str, why: str) -> Extraction:
    return Extraction(reader, None, None, (), (why,))


# ---------------------------------------------------------------- text, normalized

_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_SPACE = re.compile(r"[^\S\n]+")


def normalize(text: str) -> str:
    """Whitespace normalized within lines, line and paragraph breaks kept: each run of spaces, tabs and the like is one
    space, lines are trimmed, and blank lines between paragraphs are kept as one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n")
    text = text.replace("\u2029", "\n\n").replace("\x0c", "\n\n")
    text = _CONTROL.sub("", text)
    lines = [_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


# The markup the API's comment field carries: line breaks, paragraph-level and inline tags. Only tags of these names are
# read as markup, so a "<" in a commenter's prose ("a < b", "costs <5%") stays text; entities are decoded after, so an
# escaped tag (&lt;p&gt;) is text too.
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK = re.compile(r"</?(?:p|div|blockquote|h[1-6]|ul|ol|table|tr)(?:\s[^<>]*)?/?>", re.IGNORECASE)
_LIST_ITEM = re.compile(r"<li(?:\s[^<>]*)?>", re.IGNORECASE)
_CELL = re.compile(r"</t[dh]\s*>", re.IGNORECASE)
_INLINE = re.compile(
    r"</?(?:span|b|i|u|s|em|strong|a|font|sup|sub|li|td|th|thead|tbody|small|big|code|pre|strike)(?:\s[^<>]*)?/?>",
    re.IGNORECASE,
)


def comment_text(raw: str) -> str:
    """A comment field's text: its markup read as line and paragraph breaks, its HTML entities decoded, its whitespace
    normalized within lines."""
    text = _BREAK.sub("\n", raw)
    text = _BLOCK.sub("\n\n", text)
    text = _LIST_ITEM.sub("\n- ", text)
    text = _CELL.sub(" ", text)
    text = _INLINE.sub("", text)
    return normalize(html.unescape(text))


def _comment(content: bytes) -> Extraction:
    """The comment field of a comment's API record (the record is the artifact; this is the text the submitter typed)."""
    attributes = (_json(content).get("data") or {}).get("attributes") or {}
    return Extraction("comment", _id("comment"), comment_text(attributes.get("comment") or ""))


def _json(content: bytes) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("a record is a JSON object")
    return value


def _decode(content: bytes) -> tuple[str, str]:
    """Text bytes as UTF-8 where they are, else as Windows-1252 where they are, else as Latin-1 (which reads any byte);
    the encoding read is named in the extraction id."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1"), "latin-1"


def _text(content: bytes) -> Extraction:
    text, encoding = _decode(content)
    return Extraction("text", _id("text", encoding), normalize(text))


class _HTMLText(HTMLParser):
    BLOCKS = frozenset({"p", "div", "br", "li", "tr", "table", "ul", "ol", "blockquote", "section", "article", "header",
                        "footer", "title", "h1", "h2", "h3", "h4", "h5", "h6", "hr"})  # fmt: skip
    SKIPPED = frozenset({"script", "style", "template", "noscript"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skipping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIPPED:
            self._skipping += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIPPED:
            self._skipping = max(self._skipping - 1, 0)
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def _html(content: bytes) -> Extraction:
    markup, encoding = _decode(content)
    parser = _HTMLText()
    parser.feed(markup)
    parser.close()
    return Extraction("html", _id("html", encoding), normalize("".join(parser.parts)))


# ---------------------------------------------------------------- OCR (poppler renders, tesseract reads)


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ExtractionError(f"{name} is not installed here; a scanned page cannot be read without it")
    return path


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout, check=False)  # noqa: S603
    except subprocess.TimeoutExpired as e:
        raise ExtractionError(f"{Path(args[0]).name} did not finish within {timeout}s") from e


@cache
def _versions() -> dict[str, str]:
    """The OCR tools' versions, and a check that tesseract reads English here at all (so a failure on a page is the
    page's, never a missing language)."""
    pdftoppm = _run([_tool("pdftoppm"), "-v"], 60)
    tesseract = _run([_tool("tesseract"), "--version"], 60)
    langs = _run([_tool("tesseract"), "--list-langs"], 60)
    if "eng" not in (langs.stdout + langs.stderr).decode(errors="replace").split():
        raise ExtractionError("tesseract has no English language data here")
    p = re.search(r"pdftoppm version (\S+)", (pdftoppm.stdout + pdftoppm.stderr).decode(errors="replace"))
    t = re.search(r"tesseract (\S+)", (tesseract.stdout + tesseract.stderr).decode(errors="replace"))
    if p is None or t is None:
        raise ExtractionError("cannot tell the versions of pdftoppm and tesseract")
    return {"pdftoppm": p.group(1), "tesseract": t.group(1)}


def _tesseract(image: Path) -> str:
    """OCR of an image file; a failure here is the image's (tesseract reads English here: _versions checked)."""
    done = _run([_tool("tesseract"), str(image), "stdout"], OCR_TIMEOUT_S)
    if done.returncode != 0:
        raise _Unreadable(f"tesseract could not read it: {done.stderr.decode(errors='replace').strip()[-300:]}")
    return done.stdout.decode("utf-8", errors="replace")


class _Unreadable(Exception):
    """The file's own bytes cannot be read this way (recorded, with the reason)."""


def _ocr_page(pdf: Path, number: int, work: Path) -> str:
    """Render one page with pdftoppm and read it with tesseract."""
    prefix = work / f"page-{number}"
    done = _run([_tool("pdftoppm"), "-r", str(OCR_DPI), "-f", str(number), "-l", str(number), "-png", "-singlefile",
                 str(pdf), str(prefix)], RENDER_TIMEOUT_S)  # fmt: skip
    image = prefix.with_suffix(".png")
    if done.returncode != 0 or not image.exists():
        raise _Unreadable(f"pdftoppm could not render it: {done.stderr.decode(errors='replace').strip()[-300:]}")
    try:
        return _tesseract(image)
    finally:
        image.unlink(missing_ok=True)


def _image(content: bytes) -> Extraction:
    versions = _versions()
    with tempfile.TemporaryDirectory(prefix="requisite-ocr-") as work:
        path = Path(work) / "image"
        path.write_bytes(content)
        try:
            text = _tesseract(path)
        except _Unreadable as e:
            return _unreadable("image", str(e))
    return Extraction("image", _id("image", f"OCR by tesseract {versions['tesseract']}"), normalize(text))


# ---------------------------------------------------------------- PDF


def _has_images(page: pypdf.PageObject) -> bool:
    try:
        return len(page.images) > 0
    except Exception:  # an image the page names but pypdf cannot enumerate: OCR reads the rendered page whole anyway
        return True


_SUBSET = re.compile(r"^/[A-Z]{6}\+")
_WORD = re.compile(r"[^\W\d_]{3,}")


def _unmapped(page: pypdf.PageObject) -> bool:
    """Whether the page sets text in a subset font with no Unicode map (SCANNED_BELOW's neighbour, CARRIES_AT_LEAST)."""
    try:
        fonts = cast(Any, page["/Resources"])["/Font"]
        for ref in fonts.values():
            font = ref.get_object()
            if "/ToUnicode" not in font and _SUBSET.match(str(font.get("/BaseFont", ""))):
                return True
    except Exception:  # resources pypdf cannot read: the page is checked by OCR, which reads the rendered page whole
        return True
    return False


def suspect(layer: str) -> bool:
    """Whether a text layer is more than SYMBOLS_ABOVE symbols prose rarely holds (CARRIES_AT_LEAST's screen)."""
    seen = [c for c in layer if not c.isspace()]
    return bool(seen) and sum(c in SYMBOLS for c in seen) / len(seen) > SYMBOLS_ABOVE


def carried(layer: str, ocr: str) -> float | None:
    """The share of the words OCR reads on a page that the text layer carries, or None where OCR reads fewer than
    CHECK_WORDS (CARRIES_AT_LEAST)."""
    seen = [w.lower() for w in _WORD.findall(ocr)]
    if len(seen) < CHECK_WORDS:
        return None
    have = {w.lower() for w in _WORD.findall(layer)}
    return sum(w in have for w in seen) / len(seen)


def _pdf(content: bytes) -> Extraction:
    """Each page by its text layer (pypdf), and by OCR where it has none or where its text layer, set in a font with no
    Unicode map, does not carry the page's words (module constants). A file pypdf cannot open is rendered and read page
    by page by poppler and tesseract; one neither can read is unreadable, with both reasons."""
    layers: list[tuple[str, bool, bool]] | None
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            reader.decrypt("")  # a file restricted only by an owner password opens with the empty user password
        layers = [(page.extract_text() or "", _has_images(page), _unmapped(page)) for page in reader.pages]
        failure = None
    except Exception as e:  # pypdf raises many types on malformed files; the reason is kept, and poppler tries
        layers, failure = None, f"pypdf could not read it ({type(e).__name__}: {e})"
    methods: list[str] = []
    texts: list[str] = []
    unreadable: list[str] = []
    ocr = False
    with tempfile.TemporaryDirectory(prefix="requisite-ocr-") as tmp:
        work = Path(tmp)
        pdf = work / "file.pdf"
        pdf.write_bytes(content)
        if layers is None:
            count = _page_count(pdf)
            if count is None:
                return _unreadable("pdf", f"{failure}; poppler could not read it either")
            layers = [("", True, False)] * count
        for number, (layer, images, unmapped) in enumerate(layers, start=1):
            letters = sum(ch.isalnum() for ch in layer)
            if letters and not (images and letters < SCANNED_BELOW):
                if unmapped or suspect(layer):
                    _versions()
                    try:
                        seen = _ocr_page(pdf, number, work)
                    except _Unreadable:
                        seen = ""  # nothing to check the layer against: it stands, as for a page OCR cannot read
                    share = carried(layer, seen)
                    if share is not None and share < CARRIES_AT_LEAST:
                        texts.append(seen)
                        methods.append(f"OCR (its text layer carries {share:.0%} of the words OCR reads)")
                        ocr = True
                        continue
                methods.append("text layer")
                texts.append(layer)
                continue
            _versions()
            try:
                texts.append(_ocr_page(pdf, number, work))
                methods.append("OCR")
                ocr = True
            except _Unreadable as e:
                methods.append("unreadable")
                texts.append("")
                unreadable.append(f"page {number}: {e}")
    total = len(layers)
    body = "\n\n".join(f"[page {n} of {total}: {m}]\n{normalize(t)}".rstrip("\n")
                       for n, (m, t) in enumerate(zip(methods, texts, strict=True), start=1))  # fmt: skip
    tools = [] if failure else [f"pypdf {pypdf.__version__}"]
    if ocr:
        v = _versions()
        tools.append(f"OCR by pdftoppm {v['pdftoppm']} and tesseract {v['tesseract']}")
    return Extraction("pdf", _id("pdf", "; ".join(tools)), body, tuple(methods), tuple(unreadable))


def _page_count(pdf: Path) -> int | None:
    done = _run([_tool("pdfinfo"), str(pdf)], RENDER_TIMEOUT_S)
    m = re.search(r"^Pages:\s+(\d+)", done.stdout.decode(errors="replace"), re.MULTILINE)
    return int(m.group(1)) if done.returncode == 0 and m else None


# ---------------------------------------------------------------- Office Open XML (.docx, .xlsx, .pptx)

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _xml(z: zipfile.ZipFile, name: str) -> ET.Element:
    return ET.fromstring(z.read(name))  # noqa: S314  (see the module docstring)


def _ooxml(reader: str, content: bytes, parse: Any) -> Extraction:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            return Extraction(reader, _id(reader), normalize(parse(z)))
    except (zipfile.BadZipFile, KeyError, ET.ParseError, ValueError, IndexError) as e:
        return _unreadable(reader, f"not a readable .{reader} file ({type(e).__name__}: {e})")


def _paragraphs(element: ET.Element, out: list[str], run: list[str] | None) -> None:
    """Word paragraphs in document order: the text of w:t, tabs and breaks; deleted text (w:delText) and field codes are
    not text. A paragraph nested in another (a text box) is its own."""
    for child in element:
        if child.tag == W + "p":
            mine: list[str] = []
            _paragraphs(child, out, mine)
            out.append("".join(mine))
        elif run is not None and child.tag == W + "t":
            run.append(child.text or "")
        elif run is not None and child.tag == W + "tab":
            run.append("\t")
        elif run is not None and child.tag in (W + "br", W + "cr"):
            run.append("\n")
        else:
            _paragraphs(child, out, run)


def _docx_text(z: zipfile.ZipFile) -> str:
    names = set(z.namelist())
    parts = []
    for name, heading in (("word/document.xml", None), ("word/footnotes.xml", "[footnotes]"),
                          ("word/endnotes.xml", "[endnotes]")):  # fmt: skip
        if name not in names:
            if heading is None:
                raise KeyError(name)
            continue
        paragraphs: list[str] = []
        _paragraphs(_xml(z, name), paragraphs, None)
        text = "\n".join(paragraphs)
        if text.strip():
            parts.append(text if heading is None else f"{heading}\n{text}")
    return "\n\n".join(parts)


def _docx(content: bytes) -> Extraction:
    return _ooxml("docx", content, _docx_text)


_DATE_FORMATS = frozenset({*range(14, 23), *range(27, 37), *range(45, 48), *range(50, 59)})


def _date_styles(z: zipfile.ZipFile) -> set[int]:
    """The cell styles (cellXfs indices) whose number format shows a date or a time."""
    if "xl/styles.xml" not in z.namelist():
        return set()
    styles = _xml(z, "xl/styles.xml")
    custom = {}
    for fmt in styles.iter(S + "numFmt"):
        code = re.sub(r'"[^"]*"|\[[^\]]*\]|\\.', "", fmt.get("formatCode", "")).lower()
        custom[int(fmt.get("numFmtId", "-1"))] = bool(re.search(r"[ydhms]", code)) and "general" not in code
    xfs = styles.find(S + "cellXfs")
    dates = set()
    for i, xf in enumerate(xfs if xfs is not None else []):
        fid = int(xf.get("numFmtId", "0"))
        if fid in _DATE_FORMATS or custom.get(fid, False):
            dates.add(i)
    return dates


def _column(ref: str | None, default: int) -> int:
    if not ref:
        return default
    letters = re.match(r"[A-Z]+", ref)
    if letters is None:
        return default
    n = 0
    for ch in letters.group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _cell(c: ET.Element, shared: list[str], dates: set[int], date1904: bool) -> str:
    kind = c.get("t")
    if kind == "inlineStr":
        return "".join(t.text or "" for t in c.iter(S + "t"))
    v = c.find(S + "v")
    value = v.text if v is not None and v.text is not None else ""
    if kind == "s":
        return shared[int(value)]
    if kind == "b":
        return "TRUE" if value == "1" else "FALSE"
    if kind in ("str", "e") or not value:
        return value
    if int(c.get("s", "0")) in dates:
        serial = float(value)
        at = (datetime(1904, 1, 1) if date1904 else datetime(1899, 12, 30)) + timedelta(days=serial)
        return at.date().isoformat() if serial == int(serial) else at.isoformat(timespec="seconds")
    return value


def _shared_text(si: ET.Element) -> Iterator[str]:
    for child in si:  # t, or rich-text runs r/t; phonetic hints (rPh) are not the cell's text
        if child.tag == S + "t":
            yield child.text or ""
        elif child.tag == S + "r":
            yield from (t.text or "" for t in child.iter(S + "t"))


def _xlsx_text(z: zipfile.ZipFile) -> str:
    names = set(z.namelist())
    shared = []
    if "xl/sharedStrings.xml" in names:
        shared = ["".join(_shared_text(si)) for si in _xml(z, "xl/sharedStrings.xml").iter(S + "si")]
    workbook = _xml(z, "xl/workbook.xml")
    pr = workbook.find(S + "workbookPr")
    date1904 = pr is not None and pr.get("date1904", "0") in ("1", "true")
    rels = {r.get("Id"): r.get("Target", "") for r in _xml(z, "xl/_rels/workbook.xml.rels").iter(P + "Relationship")}
    dates = _date_styles(z)
    sheets = []
    for sheet in workbook.iter(S + "sheet"):
        target = rels[sheet.get(R + "id")]
        path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
        rows = []
        for row in _xml(z, path).iter(S + "row"):
            cells: dict[int, str] = {}
            for i, c in enumerate(row.iter(S + "c")):
                cells[_column(c.get("r"), i)] = _cell(c, shared, dates, date1904)
            width = max(cells) + 1 if cells else 0
            rows.append(CELL_SEPARATOR.join(cells.get(i, "") for i in range(width)))
        sheets.append(f"[sheet {sheet.get('name')}]\n" + "\n".join(rows))
    return "\n\n".join(sheets)


def _xlsx(content: bytes) -> Extraction:
    return _ooxml("xlsx", content, _xlsx_text)


def _pptx_text(z: zipfile.ZipFile) -> str:
    slides = sorted((int(m.group(1)), name) for name in z.namelist()
                    if (m := re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)))  # fmt: skip
    if not slides:
        raise KeyError("ppt/slides")
    parts = []
    for number, name in slides:
        lines = ["".join(t.text or "" for t in p.iter(A + "t")) for p in _xml(z, name).iter(A + "p")]
        parts.append(f"[slide {number}]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _pptx(content: bytes) -> Extraction:
    return _ooxml("pptx", content, _pptx_text)
