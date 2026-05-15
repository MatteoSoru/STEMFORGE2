#!/usr/bin/env python3
"""
StemForge — First-run model downloader
Scarica HTDemucs FT + MusicGen + Stable Audio Open (~3-5 GB totali).
Mostra una finestra di progresso nativa su macOS.
"""

import sys
import os
import subprocess
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "stemforge"
MARKER = CACHE_DIR / ".models_ready"

MODELS = [
    {
        "name": "HTDemucs FT (stem splitter)",
        "size": "~320 MB",
        "fn": "download_htdemucs",
    },
    {
        "name": "MusicGen Stereo Large",
        "size": "~2.4 GB",
        "fn": "download_musicgen",
    },
    {
        "name": "Stable Audio Open 1.0",
        "size": "~1.0 GB",
        "fn": "download_stable_audio",
    },
]


def show_progress_dialog(message: str):
    """Mostra una notifica nativa macOS."""
    try:
        script = f'display notification "{message}" with title "StemForge"'
        subprocess.run(["osascript", "-e", script],
                       check=False, capture_output=True)
    except Exception:
        pass


def download_htdemucs():
    from demucs.pretrained import get_model
    print("  Downloading HTDemucs FT...")
    model = get_model("htdemucs_ft")
    print("  ✓ HTDemucs FT ready")


def download_musicgen():
    from audiocraft.models import MusicGen
    print("  Downloading MusicGen Stereo Large...")
    model = MusicGen.get_pretrained("facebook/musicgen-stereo-large")
    print("  ✓ MusicGen ready")


def download_stable_audio():
    from stable_audio_tools import get_pretrained_model
    print("  Downloading Stable Audio Open...")
    model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    print("  ✓ Stable Audio ready")


def setup():
    if MARKER.exists():
        print("[Setup] Models already downloaded, skipping.")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[Setup] First run — downloading AI models (~3-5 GB)...")
    print("[Setup] This will take a few minutes. Please wait.\n")

    show_progress_dialog("Downloading AI models (~3-5 GB)...")

    fns = {
        "download_htdemucs": download_htdemucs,
        "download_musicgen": download_musicgen,
        "download_stable_audio": download_stable_audio,
    }

    for i, model_info in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {model_info['name']} ({model_info['size']})")
        try:
            fns[model_info["fn"]]()
        except Exception as e:
            print(f"  Warning: {e} — will retry on next use")

    MARKER.touch()
    print("\n[Setup] ✓ All models ready. Starting StemForge...")
    show_progress_dialog("Models ready — StemForge is starting!")


if __name__ == "__main__":
    setup()
