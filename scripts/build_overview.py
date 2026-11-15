"""Render SUBMISSION_OVERVIEW.md -> SUBMISSION_OVERVIEW.pdf (headless Edge/Chrome), same style as the brief.

Usage: python scripts/build_overview.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import markdown

from scripts.build_brief import CSS, browser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "SUBMISSION_OVERVIEW.md"
HTML = ROOT / "docs" / "SUBMISSION_OVERVIEW.html"
PDF = ROOT / "SUBMISSION_OVERVIEW.pdf"

EXTRA = "td, th { text-align: left !important; } blockquote { margin: 4pt 0; padding: 3pt 8pt; background: #fff6e0; border-left: 3px solid #d9a400; }"


def main():
    html = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>BRD Arrival Decision Aid - "
                    f"Overview</title><style>{CSS}{EXTRA}</style></head><body>{html}</body></html>", encoding="utf-8")
    subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=120)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
