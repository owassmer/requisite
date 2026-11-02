"""Reading an artifact's text: the fixture PDF by its text layer, a page with none by OCR, the comment field with its
entities decoded and whitespace normalized within lines, Office files with the standard library, and what cannot be read
recorded with its reason. Extractions are recorded, and a replay never re-runs OCR."""

import io
import json
import logging
import pathlib
import sys
import threading
import zipfile

import pypdf
import pytest
from fake_api import fixture_pdf, load
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from requisite import cassettes
from requisite.core.evidence import artifact_id
from requisite.evidence import extract


def test_the_fixture_pdf_is_read_by_its_text_layer():
    x = extract.read("pdf", fixture_pdf())
    assert x.pages == ("text layer",) and x.unreadable == ()
    assert x.id == f"pdf/{extract.VERSIONS['pdf']} pypdf {pypdf.__version__}"
    assert x.text.startswith("[page 1 of 1: text layer]\nTechnical Operations\n13th Floor – WHQEG")
    assert "United Airlines (UA) has reviewed Ref /A/ and /B/" in x.text


def two_pages(first: bytes, second: bytes) -> bytes:
    writer = pypdf.PdfWriter()
    for pdf in (first, second):
        for page in pypdf.PdfReader(io.BytesIO(pdf)).pages:
            writer.add_page(page)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_a_page_without_a_text_layer_is_read_by_ocr_and_each_pages_method_is_recorded(scan):
    assert pypdf.PdfReader(io.BytesIO(scan)).pages[0].extract_text().strip() == ""  # the scan has no text layer
    x = extract.read("pdf", two_pages(fixture_pdf(), scan))
    assert x.pages == ("text layer", "OCR") and x.unreadable == ()
    versions = extract._versions()
    assert x.id == (f"pdf/{extract.VERSIONS['pdf']} pypdf {pypdf.__version__}; OCR by pdftoppm {versions['pdftoppm']} and tesseract "
                    f"{versions['tesseract']}")  # fmt: skip
    page2 = x.text.split("[page 2 of 2: OCR]\n", 1)[1]
    for words in ("United Airlines", "Wacker", "Aviation Safety Engineer", "Seattle"):
        assert words in page2


def test_comment_text_has_its_entities_decoded_its_markup_read_and_whitespace_normalized_within_lines():
    raw = load("comment_0.json")["body"]["data"]["attributes"]["comment"]
    assert "&ndash;" in raw and ".  United" in raw
    assert extract.comment_text(raw) == ("Docket No. FAA–2026–2730; Project Identifier AD–2025–01579–T. United Airlines "
                                         "(UA) has reviewed and provides comments in attached letter. Thanks.")  # fmt: skip
    typed = "Costs  rise &amp; fall:<br/>labor\t&gt; 5%<br><br/><br/>\n\n\n<p>a < b, &lt;p&gt; stays</p>  "
    assert extract.comment_text(typed) == "Costs rise & fall:\nlabor > 5%\n\na < b, <p> stays"


def zipped(files: dict[str, str]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return out.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def test_a_docx_is_read_paragraph_by_paragraph_without_its_deleted_text():
    body = (f'<w:document {W}><w:body><w:p><w:r><w:t>Our labor costs</w:t></w:r><w:r><w:tab/><w:t>rise.</w:t></w:r>'
            '<w:del><w:r><w:delText>struck</w:delText></w:r></w:del></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell'
            '</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')  # fmt: skip
    notes = f'<w:footnotes {W}><w:footnote><w:p><w:r><w:t>Source: our payroll.</w:t></w:r></w:p></w:footnote></w:footnotes>'
    x = extract.read("docx", zipped({"word/document.xml": body, "word/footnotes.xml": notes}))
    assert x.text == "Our labor costs rise.\nCell\n\n[footnotes]\nSource: our payroll."


def test_an_xlsx_is_read_sheet_by_sheet_with_shared_strings_and_dates():
    S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    files = {
        "xl/workbook.xml": f'<workbook {S} {R}><sheets><sheet name="Costs" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": f"<sst {S}><si><t>Item</t></si><si><r><t>Wa</t></r><r><t>ges</t></r></si></sst>",
        "xl/styles.xml": f'<styleSheet {S}><cellXfs><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>',
        "xl/worksheets/sheet1.xml": f'<worksheet {S}><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
        '<c r="C1" t="inlineStr"><is><t>From</t></is></c></row><row r="2"><c r="A2" t="s"><v>1</v></c>'
        '<c r="B2"><v>1250.5</v></c><c r="C2" s="1"><v>46023</v></c></row></sheetData></worksheet>',
    }
    assert extract.read("xlsx", zipped(files)).text == "[sheet Costs]\nItem | | From\nWages | 1250.5 | 2026-01-01"


def test_what_cannot_be_read_is_recorded_with_its_reason():
    x = extract.read("pdf", b"%PDF-1.4\nthis is no pdf at all")
    assert x.text is None and x.id is None
    assert len(x.unreadable) == 1 and x.unreadable[0].startswith("pypdf could not read it")
    assert "poppler could not read it either" in x.unreadable[0]
    docx = extract.read("docx", b"PK\x03\x04 truncated")
    assert docx.text is None and docx.unreadable[0].startswith("not a readable .docx file (BadZipFile")


def test_an_extraction_is_recorded_and_a_replay_never_reruns_ocr(live, scan, monkeypatch):
    # The fixture's budget, not an import of conftest by name: two test directories' conftests share that module name.
    spend = cassettes.Budget(run_usd=0.0, run_calls=sys.maxsize, day_usd=float("inf"), log=live / "spend.jsonl")
    art = artifact_id(scan)
    first = extract.extract("pdf", art, scan, root=live / "cassettes", budget=spend)
    kept = json.loads(next((live / "cassettes" / "extraction").glob("*.json")).read_text())
    assert kept["request"] == {"artifact": art, "extractor": "requisite.evidence.extract.pdf",
                               "version": extract.VERSIONS["pdf"]}
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")

    def no_ocr(*args, **kwargs):
        raise AssertionError("a replay ran a local tool")

    monkeypatch.setattr(extract, "_run", no_ocr)
    monkeypatch.setattr(extract, "read", no_ocr)
    again = extract.extract("pdf", art, scan, root=live / "cassettes")
    assert again == first and again.pages == ("OCR",)
    with pytest.raises(Exception, match="no cassette"):
        extract.extract("pdf", artifact_id(b"other"), b"other", root=live / "cassettes")


UNMAPPED = pathlib.Path(__file__).resolve().parents[2] / "fixtures/regulations_gov/TREAS-DO-2026-0232/0055_attachment_1.pdf"


def test_a_text_layer_in_a_font_with_no_unicode_map_that_does_not_carry_the_pages_words_is_read_by_ocr():
    # TREAS-DO-2026-0232-0055's attachment sets its text in TrueType subsets declared WinAnsi with no ToUnicode map, and
    # its codes are glyph indices: pypdf reads them as punctuation (November 2). Each such page is read by OCR too, and
    # OCR's text is used where the layer carries less than CARRIES_AT_LEAST of OCR's words, said in the page's method.
    content = UNMAPPED.read_bytes()
    reader = pypdf.PdfReader(io.BytesIO(content))
    layer = reader.pages[0].extract_text()
    assert extract._unmapped(reader.pages[0])  # page 1: its font has no Unicode map
    assert not extract._unmapped(reader.pages[1]) and extract.suspect(reader.pages[1].extract_text())  # page 2: a wrong map
    x = extract.read("pdf", content)
    assert x.pages and all(m.startswith("OCR (its text layer carries") for m in x.pages)
    first = x.text.split("[page 2 of", 1)[0]
    share = extract.carried(layer, first)
    assert share is not None and share < extract.CARRIES_AT_LEAST  # the rule, read from the page's own two readings
    assert extract._WORD.findall(first) and "OCR by pdftoppm" in str(x.id) and x.id.startswith(f"pdf/{extract.VERSIONS['pdf']} ")


def form_page(visible: str, hidden: str, padding: int) -> bytes:
    """One page that sets `visible` in its own content and draws a form XObject that sets `hidden` after `padding` path
    operators, so the form's stream can be past a parsing budget the page's own stream is within."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(612, 792)
    font = writer._add_object(DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                                NameObject("/Subtype"): NameObject("/Type1"),
                                                NameObject("/BaseFont"): NameObject("/Helvetica")}))  # fmt: skip
    fonts = DictionaryObject({NameObject("/F1"): font})
    form = DecodedStreamObject()
    form.set_data(("0 0 m\n" * padding + f"BT /F1 28 Tf 72 300 Td ({hidden}) Tj ET").encode())
    form.update({NameObject("/Type"): NameObject("/XObject"), NameObject("/Subtype"): NameObject("/Form"),
                 NameObject("/BBox"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)]),
                 NameObject("/Resources"): DictionaryObject({NameObject("/Font"): fonts})})  # fmt: skip
    content = DecodedStreamObject()
    content.set_data(f"BT /F1 28 Tf 72 600 Td ({visible}) Tj ET q /Fx0 Do Q".encode())
    page[NameObject("/Contents")] = writer._add_object(content)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): fonts, NameObject("/XObject"): DictionaryObject({NameObject("/Fx0"): writer._add_object(form)})})
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# A budget between the page's own stream (a few tokens) and its form's (thousands): pypdf parses the page and gives up on
# the form, as it gave up on F4's /Group_93 past its default budget (November 2).
BETWEEN = 3000


def test_a_page_pypdf_cannot_wholly_decode_is_read_whole_by_ocr():
    content = form_page("Visible page heading", "Hidden form caption", padding=4000)
    whole = extract.read("pdf", content)
    assert whole.pages == ("text layer",) and "Hidden form caption" in whole.text  # within its budget, read whole
    with pypdf.apply_configuration(stream_parsing_work_maximum_count=BETWEEN):
        layer, undecoded = extract._layer(pypdf.PdfReader(io.BytesIO(content)).pages[0])
        assert "Visible page heading" in layer and "Hidden" not in layer  # pypdf left the form out, and said so
        assert len(undecoded) == 1 and "Reached maximum number of tokens per stream" in undecoded[0]
        x = extract.read("pdf", content)
    assert x.pages == (f"OCR (pypdf could not decode part of its text layer: {undecoded[0]})",) and x.unreadable == ()
    for words in ("Visible page heading", "Hidden form caption"):
        assert words in x.text


def test_pypdfs_report_is_read_from_the_thread_reading_the_page():
    handler = extract._Undecoded()
    log = logging.getLogger("pypdf._page")
    log.addHandler(handler)
    try:
        other = threading.Thread(target=lambda: log.warning(extract._UNDECODED, {"operand": "/A", "exception": "x"}))
        other.start()
        other.join()
        assert handler.seen == []  # another thread's page
        log.warning(extract._UNDECODED, {"operand": "/B", "exception": "y"})
        log.warning("Ignoring wrong pointing object %(n)s", {"n": 6})  # a repair pypdf made, no text left out
    finally:
        log.removeHandler(handler)
    assert handler.seen == ["Impossible to decode XFormObject /B: y"]


def test_pdf_3_takes_pdf_2s_recorded_reading_where_pypdf_decodes_every_part_and_reads_afresh_where_not(live):
    spend = cassettes.Budget(run_usd=0.0, run_calls=sys.maxsize, day_usd=float("inf"), log=live / "spend.jsonl")
    root = live / "cassettes"
    assert extract.VERSIONS["pdf"] == extract.SAME_AS_PDF_2 + 1

    def recorded_by_pdf_2(content: bytes) -> extract.Extraction:
        # A reading pdf/2 recorded, told apart from anything read here by its text, so taking it is seen.
        art = artifact_id(content)
        kept = extract.Extraction("pdf", "pdf/2 pypdf 0.0; OCR by pdftoppm 0 and tesseract 0", f"read by pdf/2: {art}",
                                  ("OCR",))  # fmt: skip
        body = extract._body("pdf", art, extract.SAME_AS_PDF_2)
        cassettes.call("extraction", "pdf", body, lambda: (kept.kept(), 0.0), budget=spend, estimate_usd=0.0, root=root)
        return kept

    content = form_page("Visible page heading", "Hidden form caption", padding=4000)
    earlier = recorded_by_pdf_2(content)
    x = extract.extract("pdf", artifact_id(content), content, root=root, budget=spend)
    assert x.text == earlier.text and x.pages == earlier.pages
    assert x.id == f"pdf/{extract.VERSIONS['pdf']} pypdf 0.0; OCR by pdftoppm 0 and tesseract 0"
    other = form_page("Another page heading", "Another form caption", padding=4000)
    recorded_by_pdf_2(other)
    with pypdf.apply_configuration(stream_parsing_work_maximum_count=BETWEEN):
        y = extract.extract("pdf", artifact_id(other), other, root=root, budget=spend)
    assert y.pages[0].startswith("OCR (pypdf could not decode part of its text layer") and "Another form caption" in y.text


def test_typed_prose_and_a_table_of_contents_are_not_suspect():
    assert not extract.suspect(extract.read("pdf", fixture_pdf()).text)
    assert not extract.suspect("Introduction ........ 1\nScope ........ 3\nCosts, by line ........ 7")


def test_a_layer_carries_the_words_ocr_reads_or_says_too_few_were_read():
    words = " ".join(f"word{chr(97 + i % 26)}{'x' * (i // 26)}" for i in range(40))
    assert extract.carried(words, words) == 1.0
    assert extract.carried("!\"#$%&'()*", words) == 0.0
    assert extract.carried(words, "too few words here") is None
