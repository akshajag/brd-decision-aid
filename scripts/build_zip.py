"""Build the submission ZIP (dist/BRD_Decision_Aid_submission.zip) from the working tree.

Excludes virtual environments, caches, local session notes and build intermediates.
Usage: python -m scripts.build_zip
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "BRD_Decision_Aid_submission.zip"
TOP = "BRD_Decision_Aid"
EXCLUDE_DIRS = {".cache", ".venv", "venv", "__pycache__", ".pytest_cache", ".remember", ".git", "dist", ".claude",
                ".streamlit_cache", "node_modules", "test-results"}
EXCLUDE_SUFFIX = {".pyc", ".log"}
EXCLUDE_NAMES = {"TECHNICAL_BRIEF.html", "SUBMISSION_OVERVIEW.html", "CLAUDE.md"}


def files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.suffix in EXCLUDE_SUFFIX or p.name in EXCLUDE_NAMES:
            continue
        yield p, rel


def main():
    OUT.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p, rel in files():
            z.write(p, f"{TOP}/{rel.as_posix()}")
            n += 1
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
