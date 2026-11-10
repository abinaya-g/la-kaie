"""Write paper_eswa/highlights.txt as a minimal Word file (highlights.docx), one bullet per paragraph.

Standard library only (no python-docx). Usage: python scripts/eswa_highlights_docx.py <out.docx> [highlights.txt]
"""
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper_eswa" / "highlights.txt"
lines = [l.lstrip("- ").strip() for l in SRC.read_text(encoding="utf-8").splitlines()
         if l.strip()]
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
paras = "".join(f'<w:p><w:r><w:t xml:space="preserve">• {escape(l)}</w:t></w:r></w:p>' for l in lines)
doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W}><w:body>' \
      f'<w:p><w:r><w:rPr><w:b/></w:rPr><w:t>Highlights</w:t></w:r></w:p>{paras}</w:body></w:document>'
types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
         '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
         '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
         '<Default Extension="xml" ContentType="application/xml"/>'
         '<Override PartName="/word/document.xml" '
         'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>')
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", types)
    z.writestr("_rels/.rels", rels)
    z.writestr("word/document.xml", doc)
print(len(lines), "highlights ->", sys.argv[1])
