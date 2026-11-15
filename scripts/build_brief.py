"""Render docs/TECHNICAL_BRIEF.md -> docs/TECHNICAL_BRIEF.pdf using headless Edge/Chrome.

References start on a new page so the 3-page body limit (excluding references) is easy to verify.
Usage: python scripts/build_brief.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "TECHNICAL_BRIEF.md"
HTML = ROOT / "docs" / "TECHNICAL_BRIEF.html"
PDF = ROOT / "docs" / "TECHNICAL_BRIEF.pdf"

CSS = """
@page { size: Letter; margin: 0.55in 0.6in; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 9.4pt; line-height: 1.28; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h1 + p { margin-top: 0; color: #444; }
h2 { font-size: 11pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #bbb; padding-bottom: 1pt; color: #1a3d5c;
     break-after: avoid; page-break-after: avoid; }
table, tr { break-inside: avoid; }
p { margin: 3pt 0; text-align: justify; }
ul { margin: 2pt 0 2pt 0; padding-left: 15pt; }
li { margin: 1.5pt 0; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 1.5pt 5pt; }
th { background: #eef2f6; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
code { font-size: 8.6pt; background: #f3f3f3; padding: 0 2px; }
.refs { page-break-before: always; }
.refs li { margin: 3pt 0; }
"""


def browser() -> str:
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              shutil.which("chromium") or "", shutil.which("google-chrome") or "",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]:
        if p and Path(p).exists():
            return p
    sys.exit("No Edge/Chrome found for PDF rendering.")


def main():
    md = SRC.read_text(encoding="utf-8")
    body, refs = md.split("## References", 1)
    html_body = markdown.markdown(body, extensions=["tables"])
    html_refs = markdown.markdown("## References" + refs, extensions=["tables"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>BRD Arrival Decision Aid - "
                    f"Technical Brief</title><style>{CSS}</style></head><body>{html_body}"
                    f"<div class='refs'>{html_refs}</div></body></html>", encoding="utf-8")
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
