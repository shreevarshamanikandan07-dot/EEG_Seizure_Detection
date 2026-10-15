"""Render deployment launcher for NeuroGuard.

Starts the FastAPI REST backend on 127.0.0.1:8000 (for internal health checks and APIs)
and launches the interactive Streamlit clinical dashboard on $PORT (0.0.0.0) for external web traffic.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request


def main() -> None:
    port = os.environ.get("PORT", "10000")
    print(f"=== Starting NeuroGuard on Render (Port: {port}) ===")

    # 1. Start FastAPI backend on 127.0.0.1:8000
    print("[1/2] Launching internal FastAPI REST service on 127.0.0.1:8000...")
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "warning",
    ]
    api_proc = subprocess.Popen(api_cmd)

    # 2. Wait briefly for FastAPI to initialize
    for _ in range(15):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
                if resp.status == 200:
                    print("✓ FastAPI backend ready and healthy.")
                    break
        except Exception:
            time.sleep(0.5)

    # 3. Launch Streamlit frontend on the public Render $PORT
    print(f"[2/2] Launching Streamlit Clinical Dashboard on 0.0.0.0:{port}...")
    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false",
        "--server.maxMessageSize=200",
    ]

    try:
        streamlit_proc = subprocess.Popen(streamlit_cmd)

        def sig_handler(sig, frame):
            print("Received termination signal, shutting down services...")
            streamlit_proc.terminate()
            api_proc.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)

        # Wait for Streamlit process to complete
        exit_code = streamlit_proc.wait()
        api_proc.terminate()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error launching services: {e}", file=sys.stderr)
        api_proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
