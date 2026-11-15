"""Real-browser smoke test of the dashboard (Playwright).

Skipped automatically if Playwright or a browser is unavailable. Uses an installed Edge/Chrome
(channel) so no browser download is needed:  pip install playwright
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def app_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "app.py"),
                             "--server.headless", "true", "--server.port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=2).read() == b"ok":
                break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        pytest.skip("Streamlit did not start")
    yield url
    proc.kill()


@pytest.fixture(scope="module")
def page(app_url):
    with pw.sync_playwright() as p:
        browser = None
        for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(headless=True, **kw)
                break
            except Exception:
                continue
        if browser is None:
            pytest.skip("No Chromium-based browser available")
        pg = browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        pg.goto(app_url)
        # Tab panels other than the first are hidden, so wait for attachment, not visibility.
        pg.get_by_text("Recommended plan vs current approach", exact=False).first.wait_for(
            state="attached", timeout=120_000)
        pg.wait_for_function("() => document.querySelectorAll('.js-plotly-plot').length >= 4",
                             timeout=120_000)
        yield pg
        browser.close()


def _no_exceptions(pg):
    assert pg.locator('[data-testid="stException"]').count() == 0


def test_loads_with_banner_tabs_and_charts(page):
    _no_exceptions(page)
    assert page.get_by_text("Medication-related recommendations require veterinary review").count() >= 1
    for t in ["Start here", "Lots & data", "Risk priorities", "Plan vs current approach", "What-if", "Decision summary",
              "Days 1-14 check-in", "Model & evidence"]:
        assert page.get_by_role("tab", name=t, exact=False).count() == 1
    assert page.locator(".js-plotly-plot").count() >= 4


def test_capacity_change_clears_shortfall(page):
    page.get_by_role("tab", name="Plan vs current approach", exact=False).click()
    shortfall = page.get_by_text("Capacity shortfall -", exact=False)
    shortfall.first.wait_for(state="visible", timeout=30_000)
    page.get_by_text("Operating constraints").click()
    box = page.get_by_label("B: hospital head")
    box.fill("30")
    box.press("Enter")
    shortfall.first.wait_for(state="detached", timeout=120_000)
    _no_exceptions(page)


def test_summary_downloads(page):
    page.get_by_role("tab", name="Decision summary", exact=False).click()
    with page.expect_download(timeout=60_000) as dl:
        page.get_by_role("button", name="Download summary", exact=False).click()
    text = Path(dl.value.path()).read_text(encoding="utf-8")
    assert "BRD arrival decision summary" in text
    assert "veterinary review" in text.lower()
