import hashlib
import json
import re
from functools import partial
from pathlib import Path
from xml.sax.saxutils import escape

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
fonts = Path("/System/Library/Fonts/Supplemental")
for name, file in [
    ("TNR", "Times New Roman.ttf"),
    ("TNR-Bold", "Times New Roman Bold.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
pdfmetrics.registerFontFamily(
    "TNR", normal="TNR", bold="TNR-Bold", italic="TNR", boldItalic="TNR-Bold"
)
M = 72 * 2.5 / 2.54
W, H = A4
styles = {
    "body": ParagraphStyle(
        "body", fontName="TNR", fontSize=12, leading=12, spaceAfter=5
    ),
    "heading": ParagraphStyle(
        "heading",
        fontName="TNR-Bold",
        fontSize=12,
        leading=12,
        spaceBefore=4,
        spaceAfter=3,
    ),
    "title": ParagraphStyle(
        "title",
        fontName="TNR-Bold",
        fontSize=12,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=5,
    ),
    "author": ParagraphStyle(
        "author",
        fontName="TNR",
        fontSize=12,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=2,
    ),
    "equation": ParagraphStyle(
        "equation",
        fontName="TNR",
        fontSize=12,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=5,
    ),
    "cell": ParagraphStyle("cell", fontName="TNR", fontSize=12, leading=12),
}


def sub(x):
    return '<sub size="12" rise="1.5">' + x + "</sub>"


def sup(x):
    return '<super size="12" rise="3">' + x + "</super>"


mapping = {
    r"A_l,F_l": "A" + sub("l") + ", F" + sub("l"),
    r"s=0.02": "s = 0.02",
    r"d_l,f_l": "d" + sub("l") + ", f" + sub("l"),
    r"Q\ge0": "Q ≥ 0",
    r"\beta<0": "β &lt; 0",
    r"\beta>0": "β &gt; 0",
    r"\beta=0": "β = 0",
    r"0<w<2": "0 &lt; w &lt; 2",
    r"w(0;\beta)=1": "w(0; β) = 1",
    r"q_l": "q" + sub("l"),
    r"Q_l": "Q" + sub("l"),
    r"\beta\in\{0,\pm0.1,\pm1,\pm10,\pm100,\pm1000\}": "β from {0, ±0.1, ±1, ±10, ±100, ±1000}",
    r"|\beta|": "|β|",
    r"\rho": "ρ",
    r"\tau_b": "τ" + sub("b"),
    r"k=1,3": "k = 1, 3",
}


def inline(t):
    parts = re.split(r"(\$[^$]+\$)", t)
    return "".join(mapping[p[1:-1]] if p.startswith("$") else escape(p) for p in parts)


src = (ROOT / "docs/ProjectProposal-en.md").read_text()
src = re.sub(r"<!--.*?-->", "", src, flags=re.S).strip()
blocks = src.split("\n\n")
story = []
for block in blocks:
    if block.startswith("# "):
        story.append(Paragraph(escape(block[2:]), styles["title"]))
    elif block.startswith("## "):
        story.append(Paragraph(escape(block[3:]), styles["heading"]))
    elif block.startswith("Jingzhe"):
        for line in block.splitlines():
            story.append(Paragraph(escape(line), styles["author"]))
    elif block.startswith("$$"):
        if "q_l=" in block:
            eq = (
                "q"
                + sub("l")
                + " = ½ d"
                + sub("l")
                + " f"
                + sub("l")
                + " s"
                + sup("4")
                + ",   Q"
                + sub("l")
                + " = ∑"
                + sub("j&lt;l")
                + " q"
                + sub("j")
                + ",   w(Q; β) = 1 + βQ/(1 + |β|Q),"
            )
        else:
            eq = (
                "S(β) = ∑"
                + sub("l")
                + " [A"
                + sub("l")
                + " + w(Q"
                + sub("l")
                + "; β) F"
                + sub("l")
                + "]."
            )
        story.append(Paragraph(eq, styles["equation"]))
    elif block.startswith("|"):
        rows = []
        for i, line in enumerate(block.splitlines()):
            if i == 1:
                continue
            row = [c.strip() for c in line.strip("|").split("|")]
            rows.append(
                [
                    Paragraph(
                        ("<b>" + inline(c) + "</b>") if i == 0 or j == 0 else inline(c),
                        styles["cell"],
                    )
                    for j, c in enumerate(row)
                ]
            )
        table = Table(rows, colWidths=[83, W - 2 * M - 83], hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                ]
            )
        )
        story.append(table)
    else:
        story.append(Paragraph(inline(block.replace("\n", " ")), styles["body"]))
# Zero frame padding makes the 2.5 cm margins exact.

out = ROOT / "docs/ProjectProposal-en.pdf"
doc = BaseDocTemplate(
    str(out),
    pagesize=A4,
    leftMargin=M,
    rightMargin=M,
    topMargin=M,
    bottomMargin=M,
    title="Signed FFN Layer-Order Context for Neural Spectral Capacity",
    author="Jingzhe Feng, Jin Xing, Zheng Zhang",
)
doc.addPageTemplates(
    PageTemplate(
        id="A4",
        frames=[
            Frame(
                M,
                M,
                W - 2 * M,
                H - 2 * M,
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
            )
        ],
    )
)


doc.build(
    story,
    canvasmaker=partial(
        Canvas, initialFontName="TNR", initialFontSize=12, initialLeading=12
    ),
)
# Replace ReportLab's unused default font selection and remove its resource.

r = PdfReader(out)
writer = PdfWriter(clone_from=r)
for page in writer.pages:
    resources = page["/Resources"]["/Font"]
    regular = next(
        k
        for k, v in resources.items()
        if "TimesNewRomanPSMT" in str(v.get_object()["/BaseFont"])
    )
    unused = [
        k
        for k, v in resources.items()
        if str(v.get_object()["/BaseFont"]) == "/Helvetica"
    ]
    stream = ContentStream(page.get_contents(), writer)
    for operands, operator in stream.operations:
        if operator == b"Tf" and operands[0] in unused:
            operands[0] = NameObject(regular)
    page.replace_contents(stream)
    for k in unused:
        del resources[k]
with out.open("wb") as f:
    writer.write(f)
r = PdfReader(out)
report = {
    "pages": len(r.pages),
    "page_size_pt": list(r.pages[0].mediabox),
    "specified_margin_pt": M,
    "body_font_pt": 12,
    "body_leading_pt": 12,
    "fonts": [],
    "source_sha256": hashlib.sha256(
        (ROOT / "docs/ProjectProposal-en.md").read_bytes()
    ).hexdigest(),
}
for page in r.pages:
    for ref in page["/Resources"]["/Font"].values():
        font = ref.get_object()
        desc = font.get("/FontDescriptor")
        report["fonts"].append(
            {
                "name": str(font["/BaseFont"]),
                "embedded": bool(
                    desc
                    and (
                        "/FontFile2" in desc.get_object()
                        or "/FontFile3" in desc.get_object()
                    )
                ),
            }
        )
with pdfplumber.open(out) as pdf:
    report["actual_font_sizes"] = sorted(
        {round(c["size"], 4) for page in pdf.pages for c in page.chars}
    )
    report["actual_fonts"] = sorted(
        {c["fontname"] for page in pdf.pages for c in page.chars}
    )
    report["text_bounds"] = [
        {
            "left": min(c["x0"] for c in p.chars),
            "right": max(c["x1"] for c in p.chars),
            "top": min(c["top"] for c in p.chars),
            "bottom": max(c["bottom"] for c in p.chars),
        }
        for p in pdf.pages
    ]
print(json.dumps(report, indent=2))
verification = ROOT / "work/proposal/verification.json"
verification.parent.mkdir(parents=True, exist_ok=True)
verification.write_text(json.dumps(report, indent=2))
assert len(r.pages) == 1, "Must fit one page without shrinking type or margins"
assert report["actual_font_sizes"] == [12.0]
assert all(f["embedded"] and "TimesNewRoman" in f["name"] for f in report["fonts"])
assert all("TimesNewRoman" in n for n in report["actual_fonts"])
for b in report["text_bounds"]:
    assert (
        b["left"] >= M - 0.01
        and b["right"] <= W - M + 0.01
        and b["top"] >= M - 0.01
        and b["bottom"] <= H - M + 0.01
    )
