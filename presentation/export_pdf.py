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

Chrome's own --print-to-pdf flag printed this deck unreliably. From a file://
address it printed a blank page, and from a web address it sometimes printed
before MathJax had set the formulas, which left some of them overlapping or
missing. So the script serves the repository on a free local port, drives the
browser through its DevTools protocol, waits until the print layout exists,
MathJax has finished and every font has loaded, and only then prints.
"""

import base64
import functools
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "slides.html"
TIMEOUT = 120

CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    str(Path.home() / ".local/share/quarto/chrome-headless-shell/chrome-headless-shell-linux64/chrome-headless-shell"),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
NAMES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"]

# True once reveal.js has built the print pages, MathJax has emptied its queue and
# every font of the page has loaded.
READY = """(() => {
  const pages = document.querySelectorAll('.pdf-page').length;
  const hub = window.MathJax && window.MathJax.Hub;
  const mathDone = !hub || (hub.queue.pending === 0 && !hub.queue.running);
  return document.readyState === 'complete' && pages > 0 && mathDone && document.fonts.status === 'loaded';
})()"""


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


class QuietHandler(SimpleHTTPRequestHandler):
    """Serves files without printing a line for every request."""

    def log_message(self, *args):
        pass


def serve(root):
    """Serve the folder root on a free local port in a background thread."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class DevTools:
    """A minimal client of Chrome's DevTools protocol over one WebSocket, with the standard library only.

    It sends text frames, masked as a client must, and reads the browser's
    unmasked replies. Messages without an id are events, which it skips.
    """

    def __init__(self, url):
        parts = urllib.parse.urlparse(url)
        self.sock = socket.create_connection((parts.hostname, parts.port), timeout=TIMEOUT)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {parts.path} HTTP/1.1\r\nHost: {parts.hostname}:{parts.port}\r\n"
                           f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
        reply = b""
        while b"\r\n\r\n" not in reply:
            reply += self.sock.recv(4096)
        if b" 101 " not in reply.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"The browser refused the WebSocket: {reply[:80]!r}")
        self.last_id = 0

    def _exactly(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(min(n - len(data), 1 << 20))
            if not chunk:
                raise ConnectionError("The browser closed the connection.")
            data += chunk
        return bytes(data)

    def _send(self, text):
        data = text.encode()
        header = bytearray([0x81])
        if len(data) < 126:
            header.append(0x80 | len(data))
        elif len(data) < 1 << 16:
            header += bytes([0x80 | 126]) + len(data).to_bytes(2, "big")
        else:
            header += bytes([0x80 | 127]) + len(data).to_bytes(8, "big")
        mask = os.urandom(4)
        self.sock.sendall(bytes(header) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _receive(self):
        message = bytearray()
        while True:
            first, second = self._exactly(2)
            length = second & 0x7F
            if length == 126:
                length = int.from_bytes(self._exactly(2), "big")
            elif length == 127:
                length = int.from_bytes(self._exactly(8), "big")
            message += self._exactly(length)
            if first & 0x80:
                return json.loads(message)

    def call(self, method, **params):
        self.last_id += 1
        self._send(json.dumps({"id": self.last_id, "method": method, "params": params}))
        while True:
            reply = self._receive()
            if reply.get("id") == self.last_id:
                if "error" in reply:
                    raise RuntimeError(f"{method} failed: {reply['error']}")
                return reply["result"]


def page_socket(profile, deadline):
    """The DevTools address of the browser's first page, read once the browser has started."""
    port_file = Path(profile) / "DevToolsActivePort"
    while not port_file.exists():
        if time.time() > deadline:
            sys.exit("The browser did not start its DevTools server.")
        time.sleep(0.2)
    while not port_file.read_text().strip():
        time.sleep(0.1)
    port = int(port_file.read_text().split()[0])
    targets = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10).read())
    return next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")


def main():
    if not DECK.exists():
        sys.exit("slides.html does not exist. Run: quarto render presentation/slides.qmd")
    browser = find_browser()
    if not browser:
        sys.exit("No Chrome, Chromium or Edge found. Set CHROME to the browser, or run "
                 "quarto install chrome-headless-shell.")
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "exports" / "slides.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)

    # The deck loads its clips and stills from ../animations, so serve the whole repository.
    server = serve(HERE.parent)
    url = f"http://127.0.0.1:{server.server_port}/{DECK.relative_to(HERE.parent).as_posix()}?print-pdf"
    profile = tempfile.mkdtemp(prefix="g2-deck-chrome-")
    command = [browser, "--headless", "--disable-gpu", "--remote-debugging-port=0", f"--user-data-dir={profile}",
               "--no-first-run", "--no-default-browser-check", "about:blank"]
    if sys.platform.startswith("linux"):
        command.insert(1, "--no-sandbox")
    chrome = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + TIMEOUT
    try:
        tools = DevTools(page_socket(profile, deadline))
        tools.call("Page.navigate", url=url)
        while not tools.call("Runtime.evaluate", expression=READY, returnByValue=True)["result"].get("value"):
            if time.time() > deadline:
                sys.exit(f"The deck was not ready to print within {TIMEOUT} s.")
            time.sleep(0.5)
        time.sleep(1.5)  # one more layout pass after the last formula
        pdf = tools.call("Page.printToPDF", printBackground=True, preferCSSPageSize=True, marginTop=0,
                         marginBottom=0, marginLeft=0, marginRight=0)
        target.write_bytes(base64.b64decode(pdf["data"]))
    finally:
        chrome.terminate()
        chrome.wait(timeout=10)
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)
    print(f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
