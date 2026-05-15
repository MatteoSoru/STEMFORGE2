#!/usr/bin/env python3
"""
StemForge launcher — avvia il backend FastAPI e apre il mixer nel browser.
Questo è il punto di ingresso dell'app bundle macOS.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import webbrowser
from pathlib import Path

PORT = 58432
RESOURCES = Path(sys.executable).parent.parent / "Resources"


def first_run_setup():
    """Se i modelli non sono presenti, lancia setup_models.py con dialog nativa."""
    cache = Path.home() / ".cache" / "stemforge"
    marker = cache / ".models_ready"
    if not marker.exists():
        import subprocess
        subprocess.run(
            [sys.executable, str(RESOURCES / "setup_models.py")],
            check=True,
        )


def wait_for_server(timeout=30):
    """Aspetta che il server sia pronto prima di aprire il browser."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def open_browser():
    """Apre il mixer nel browser di default dopo che il server è pronto."""
    if wait_for_server():
        webbrowser.open(f"http://localhost:{PORT}")


def main():
    # Primo avvio: scarica modelli se necessario
    first_run_setup()

    # Avvia FastAPI in background
    server = subprocess.Popen(
        [
            sys.executable,
            "-m", "uvicorn",
            "api_server_v2:app",
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--log-level", "warning",
        ],
        cwd=str(RESOURCES),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Apri browser in un thread separato (aspetta che il server sia su)
    threading.Thread(target=open_browser, daemon=True).start()

    def shutdown(sig, frame):
        server.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()


if __name__ == "__main__":
    main()
