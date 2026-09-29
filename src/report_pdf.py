"""Optional PDF rendering for the report.

``src.report`` always writes ``docs/ieee_report.md`` and ``docs/ieee_report.tex``.
This module renders a printable PDF when a toolchain is available, in this
order:

1. ``pdflatex`` (or ``xelatex``) on the ``IEEEtran`` source -- the real thing.
2. A self-contained, two-column IEEE-styled HTML build printed by headless
   Chrome or Edge.

Both paths are optional. When neither exists the Markdown and ``.tex`` sources
are still written, the in-app **Report** page still renders the whole document,
and :func:`build_pdf` returns ``None`` so the caller can say so rather than
pretend. Nothing about the content changes between the three renderings: they
are all generated from the same section builder.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.config import IMAGE_DIR, REPORT_LATEX, REPORT_MARKDOWN

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

#: Two-column conference layout, sized for US Letter.
PDF_CSS = """
@page { size: Letter; margin: 0.75in 0.6in; }
* { box-sizing: border-box; }
body {
  font-family: "Times New Roman", Times, serif;
  font-size: 9.6pt;
  line-height: 1.28;
  color: #111;
  margin: 0;
  column-count: 2;
  column-gap: 0.32in;
  column-fill: auto;
  text-align: justify;
  hyphens: auto;
}
h1 { column-span: all; font-size: 19pt; text-align: center; margin: 0 0 4pt; }
h2 { column-span: all; font-size: 10.5pt; text-transform: uppercase;
     letter-spacing: 0.05em; border-bottom: 0.6pt solid #111;
     margin: 11pt 0 5pt; padding-bottom: 1.5pt; break-after: avoid; }
h3 { font-size: 9.6pt; font-style: italic; margin: 7pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt; }
em { font-style: italic; }
strong { font-style: italic; }
code { font-family: Consolas, monospace; font-size: 8.4pt; background: #f2f2f4; }
a { color: inherit; text-decoration: none; }
ul { margin: 0 0 5pt; padding-left: 12pt; }
li { margin-bottom: 2pt; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt;
        margin: 4pt 0 7pt; break-inside: avoid; }
th, td { border-top: 0.5pt solid #666; border-bottom: 0.5pt solid #666;
         padding: 1.6pt 3pt; text-align: left; }
th { border-top: 1pt solid #111; font-weight: bold; }
tbody tr:nth-child(even) { background: #f6f6f8; }
img { width: 100%; margin: 4pt 0 2pt; break-inside: avoid; }
figure { margin: 6pt 0; break-inside: avoid; }
figcaption { font-size: 8pt; font-style: italic; text-align: center; }
blockquote { margin: 4pt 0; padding-left: 6pt; border-left: 1.5pt solid #999;
             font-style: italic; }
hr { border: 0; border-top: 0.5pt solid #bbb; margin: 8pt 0; }
.abstract { column-span: all; background: #f4f4f7; padding: 7pt 9pt;
            margin: 0 0 6pt; font-size: 9.2pt; }
.abstract .label { font-weight: bold; font-style: normal; }
.abstract .keywords { margin-top: 5pt; }
.frontmatter { column-span: all; text-align: center; margin-bottom: 6pt; }
.frontmatter .authors { font-size: 10pt; }
.frontmatter .stamp { font-size: 7.6pt; color: #555; margin-top: 3pt; }
"""


# --------------------------------------------------------------------------
# Markdown -> HTML
# --------------------------------------------------------------------------

def _inline(text: str) -> str:
    """Convert the small Markdown subset the report uses to HTML."""
    out = html.escape(text)
    out = out.replace("`", "<code>").replace("</code> ", " ")
    while "**" in out:
        out = out.replace("**", "<strong>", 1)
        head, _, tail = out.partition("</strong>")
        out = head + tail if head is not out else out
    return out


def _to_html(markdown: str) -> str:
    """Render the report Markdown into a two-column HTML document."""
    lines = markdown.splitlines()
    body: list[str] = []
    index = 0
    in_list = False
    in_table = False
    in_frontmatter = False
    in_abstract = False

    def close_blocks() -> None:
        nonlocal in_list, in_table
        if in_list:
            body.append("</ul>")
            in_list = False
        if in_table:
            body.append("</tbody></table>")
            in_table = False

    while index < len(lines):
        line = lines[index].rstrip()

        if not line:
            close_blocks()
            index += 1
            continue

        if line.startswith("# "):
            close_blocks()
            body.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            close_blocks()
            body.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            close_blocks()
            body.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("*") and line.endswith("*") and not line.startswith("**"):
            close_blocks()
            body.append(f"<p class='caption'>{_inline(line.strip('*'))}</p>")
        elif line.startswith("> "):
            close_blocks()
            body.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.startswith("!["):
            close_blocks()
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                alt, src = match.groups()
                body.append(
                    f"<figure><img src='{src}' alt='{html.escape(alt)}'>"
                    f"<figcaption>{_inline(alt)}</figcaption></figure>"
                )
        elif line.startswith("|") and set(line.replace("|", "").strip()) <= set("-: "):
            # separator row: already inside a table
            index += 1
            continue
        elif line.startswith("|"):
            close_blocks()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not in_table:
                body.append("<table><thead><tr>")
                body.extend(f"<th>{_inline(cell)}</th>" for cell in cells)
                body.append("</tr></thead><tbody>")
                in_table = True
            else:
                body.append(
                    "<tr>" + "".join(f"<td>{_inline(cell) or '&nbsp;'}</td>" for cell in cells) + "</tr>"
                )
        elif re.match(r"^-\s+", line):
            if not in_list:
                close_blocks()
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{_inline(line[2:])}</li>")
        elif line.startswith("```"):
            close_blocks()
            index += 1
            block: list[str] = []
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            body.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
        else:
            close_blocks()
            body.append(f"<p>{_inline(line)}</p>")

        index += 1

    close_blocks()
    return "\n".join(body)


def build_html(markdown_path: Path = REPORT_MARKDOWN, output_path: Path | None = None) -> Path:
    """Write a self-contained, printable HTML rendering of the report."""
    markdown = markdown_path.read_text(encoding="utf-8")
    content = _to_html(markdown)

    # Lift the abstract into a full-width band.
    content = re.sub(
        r"<h2>Abstract</h2>\s*(<p>.*?</p>)\s*(<p><strong>Index Terms</strong>.*?</p>)",
        r"<div class='abstract'>\1\2</div>",
        content,
        flags=re.S,
    )
    content = re.sub(
        r"<p><em>Generated (.*?) by <code>python -m src.report</code> from the artifacts in "
        r"<code>results/</code>.</em></p>",
        r"<p class='stamp'>Generated \1 by <code>python -m src.report</code> from the "
        r"artifacts in <code>results/</code>.</p>",
        content,
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI-Powered Fake News Detection Using Text Classification</title>
<style>{PDF_CSS}</style>
</head>
<body>
{content}
</body>
</html>
"""
    output = output_path or markdown_path.with_suffix(".html")
    output.write_text(document, encoding="utf-8")
    return output


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def find_browser() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    for name in ("chrome", "msedge", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    return None


def build_pdf(html_path: Path, output_path: Path) -> bool:
    """Print an HTML file to PDF with a headless Chromium browser."""
    browser = find_browser()
    if browser is None:
        return False

    with tempfile.TemporaryDirectory() as profile:
        try:
            subprocess.run(
                [
                    browser,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    "--no-pdf-header-footer",
                    f"--user-data-dir={profile}",
                    f"--print-to-pdf={output_path}",
                    html_path.resolve().as_uri(),
                ],
                capture_output=True,
                timeout=180,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return output_path.exists() and output_path.stat().st_size > 0


def compile_pdf(tex_path: Path = REPORT_LATEX) -> Path | None:
    """Compile the LaTeX source when a TeX distribution is installed."""
    for engine in ("pdflatex", "xelatex"):
        if shutil.which(engine) is None:
            continue
        with tempfile.TemporaryDirectory() as scratch:
            for _ in range(2):
                subprocess.run(
                    [engine, "-interaction=nonstopmode", "-output-directory", scratch, tex_path.name],
                    cwd=tex_path.parent,
                    capture_output=True,
                    check=False,
                )
        produced = Path(scratch) / tex_path.with_suffix(".pdf").name
        if produced.exists():
            target = tex_path.with_suffix(".pdf")
            shutil.copyfile(produced, target)
            return target
    return None
