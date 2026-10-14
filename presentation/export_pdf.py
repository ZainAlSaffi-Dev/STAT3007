"""Export the rendered deck to a PDF, one page per slide.

The course asks for the slides as a .ppt or a .pdf file, so this PDF is what we
upload. It uses the print mode of reveal.js and a Chrome-based browser in
headless mode. It needs no Python packages.

Render the deck first, then run this from anywhere:

    quarto render presentation/slides.qmd
    python presentation/export_pdf.py                 # writes presentation/exports/slides.pdf
    python presentation/export_pdf.py talk.pdf        # or name the file

Set the CHROME environment variable to the browser if the script cannot find
one. A slide that shows a clip prints its still, so render the stills as well
as the clips before the final export.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "slides.html"

CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    str(Path.home() / ".local/share/quarto/chrome-headless-shell/chrome-headless-shell-linux64/chrome-headless-shell"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
NAMES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"]


def find_browser():
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    for path in CANDIDATES:
        if Path(path).exists():
            return path
    for name in NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def main():
    if not DECK.exists():
        sys.exit("slides.html does not exist. Run: quarto render presentation/slides.qmd")
    browser = find_browser()
    if not browser:
        sys.exit("No Chrome, Chromium or Edge found. Set CHROME to the browser, or run "
                 "quarto install chrome-headless-shell.")
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "exports" / "slides.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    command = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--print-to-pdf-no-header",
        "--virtual-time-budget=20000",
        f"--print-to-pdf={target}",
        DECK.as_uri() + "?print-pdf",
    ]
    if platform.system() == "Linux":
        command.insert(1, "--no-sandbox")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists():
        sys.exit(f"The browser did not write the PDF.\n{result.stderr}")
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
