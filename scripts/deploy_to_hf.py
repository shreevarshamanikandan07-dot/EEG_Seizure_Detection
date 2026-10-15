"""NeuroGuard — Hugging Face Spaces Deployment Script.

Automates deploying models, frontend, and evaluation bundles to Hugging Face Spaces
using the official huggingface_hub Python client.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "Shreevarsha/neuroguard"
DEFAULT_TOKEN = os.getenv("HF_TOKEN")


def deploy(repo_id: str = DEFAULT_REPO_ID, token: str = DEFAULT_TOKEN) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    print(f"[*] Deploying NeuroGuard to Hugging Face Space: {repo_id}")
    print(f"[*] Workspace Root: {root_dir}")

    api = HfApi(token=token)

    # 1. Verify Authentication & Space
    try:
        user = api.whoami()
        print(f"[+] Authenticated as: {user.get('name')}")
    except Exception as e:
        print(f"[!] Authentication error: {e}")
        sys.exit(1)

    # 2. Upload complete folder to Hugging Face Spaces
    print("[*] Uploading application files, pre-trained models, and assets...")
    t0 = time.time()
    api.upload_folder(
        folder_path=str(root_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy NeuroGuard Clinical EEG Seizure Intelligence System",
        ignore_patterns=[
            ".git/**",
            ".git",
            "data/raw/**",
            "data/interim/**",
            "**/mlruns/**",
            "artifacts/mlruns/**",
            "**/.pytest_cache/**",
            "**/__pycache__/**",
            "*.pyc",
            ".env*",
            "*.log",
            ".python-version",
            "Dockerfile.test",
        ],
    )
    elapsed = time.time() - t0
    print(f"[+] Upload completed successfully in {elapsed:.1f} seconds!")

    # 3. Fetch Space Status
    info = api.space_info(repo_id)
    print(f"[+] Space SDK: {info.sdk}")
    print(f"[+] Space URL: https://huggingface.co/spaces/{repo_id}")
    if info.runtime:
        print(f"[+] Current Stage: {info.runtime.stage}")


if __name__ == "__main__":
    deploy()
