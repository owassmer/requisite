"""Reading an artifact's text: the fixture PDF by its text layer, a page with none by OCR, the comment field with its
entities decoded and whitespace normalized within lines, Office files with the standard library, and what cannot be read
recorded with its reason. Extractions are recorded, and a replay never re-runs OCR."""

import io
import json
import zipfile

import pypdf
import pytest
from fake_api import fixture_pdf, load

from requisite.core.evidence import artifact_id
from requisite.evidence import extract


def test_the_fixture_pdf_is_read_by_its_text_layer():
    x = extract.read("pdf", fixture_pdf())
    assert x.pages == ("text layer",) and x.unreadable == ()
    assert x.id == f"pdf/1 pypdf {pypdf.__version__}"
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
    assert x.id == (f"pdf/1 pypdf {pypdf.__version__}; OCR by pdftoppm {versions['pdftoppm']} and tesseract "
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
    from conftest import budget

    art = artifact_id(scan)
    first = extract.extract("pdf", art, scan, root=live / "cassettes", budget=budget(live))
    kept = json.loads(next((live / "cassettes" / "extraction").glob("*.json")).read_text())
    assert kept["request"] == {"artifact": art, "extractor": "requisite.evidence.extract.pdf", "version": 1}
    monkeypatch.setenv("REQUISITE_REPLAY_ONLY", "1")

    def no_ocr(*args, **kwargs):
        raise AssertionError("a replay ran a local tool")

    monkeypatch.setattr(extract, "_run", no_ocr)
    monkeypatch.setattr(extract, "read", no_ocr)
    again = extract.extract("pdf", art, scan, root=live / "cassettes")
    assert again == first and again.pages == ("OCR",)
    with pytest.raises(Exception, match="no cassette"):
        extract.extract("pdf", artifact_id(b"other"), b"other", root=live / "cassettes")
