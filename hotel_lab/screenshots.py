"""Screenshot mode: serve the site locally and capture the pages with Playwright (Chromium).

Setup once:  python -m playwright install chromium
Adapted from ACES (agent/src/environment.py), which drives Selenium on a product grid.
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

VIEWPORT = {"width": 1280, "height": 900}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalSite:
    """Run the StayCompare site in a background thread for the duration of a `with` block."""

    def __init__(self, experiment: str, port: int | None = None, show_category_scores: bool = False):
        self.experiment = experiment
        self.show_category_scores = show_category_scores
        self.port = port or _free_port()
        self.server = None
        self.thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "LocalSite":
        import uvicorn
        from hotel_lab.site.app import create_app
        config = uvicorn.Config(create_app(self.experiment, self.show_category_scores), host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("Local site did not start")

    def __exit__(self, *exc) -> None:
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)


class Screenshotter:
    def __init__(self, site: LocalSite):
        self.site = site
        self._pw = None
        self._browser = None

    def __enter__(self) -> "Screenshotter":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError("Screenshot mode needs Playwright: pip install playwright && "
                               "python -m playwright install chromium") from e
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        return self

    def __exit__(self, *exc) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def capture_trial(self, trial: dict[str, Any], out_dir: Path) -> list[Path]:
        """Results page + one full-page screenshot per room, in display order."""
        out_dir.mkdir(parents=True, exist_ok=True)
        page = self._browser.new_page(viewport=VIEWPORT)
        paths = []
        urls = [f"/trial/{trial['trial_id']}"] + [f"/trial/{trial['trial_id']}/room/{rid}" for rid in trial["room_order"]]
        for i, url in enumerate(urls):
            page.goto(self.site.base_url + url, wait_until="networkidle")
            p = out_dir / f"{i}.png"
            page.screenshot(path=str(p), full_page=True)
            paths.append(p)
        page.close()
        return paths
