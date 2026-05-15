#!/usr/bin/env python3
"""
StemForge — Stem Splitter
Pipeline: HTDemucs FT (shifts=2, overlap=0.25) → Wiener mask → de-click 10ms
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

warnings.filterwarnings("ignore")

STEMS = ["drums", "bass", "other", "vocals"]
SAMPLE_RATE = 44100
FADE_MS = 10  # de-click fade in milliseconds


def load_audio(path: Path, target_sr: int = SAMPLE_RATE):
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    return wav  # (2, T)


def save_audio(wav: torch.Tensor, path: Path, sr: int = SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.cpu(), sr, bits_per_sample=32)


def declick(wav: torch.Tensor, sr: int = SAMPLE_RATE, ms: int = FADE_MS):
    """Apply fade-in/out to remove click artefacts at segment boundaries."""
    n = int(sr * ms / 1000)
    fade = torch.linspace(0, 1, n, device=wav.device)
    wav[:, :n] *= fade
    wav[:, -n:] *= fade.flip(0)
    return wav


def wiener_filter(stems: dict[str, torch.Tensor], eps: float = 1e-8):
    """
    Soft Wiener mask applied in the spectrogram domain.
    Reduces bleeding between stems (e.g. kick in vocal track).
    """
    n_fft = 4096
    hop = 1024
    window = torch.hann_window(n_fft)

    specs = {}
    for name, wav in stems.items():
        mono = wav.mean(0)
        spec = torch.stft(mono, n_fft=n_fft, hop_length=hop,
                          window=window, return_complex=True)
        specs[name] = spec.abs() ** 2  # power spectrogram

    total_power = sum(specs.values()) + eps

    filtered = {}
    for name, wav in stems.items():
        mask = specs[name] / total_power  # soft mask [0,1]
        # Apply mask channel-wise
        out_channels = []
        for ch in range(wav.shape[0]):
            spec = torch.stft(wav[ch], n_fft=n_fft, hop_length=hop,
                              window=window, return_complex=True)
            spec_masked = spec * mask
            audio = torch.istft(spec_masked, n_fft=n_fft, hop_length=hop,
                                window=window, length=wav.shape[1])
            out_channels.append(audio)
        filtered[name] = torch.stack(out_channels)

    return filtered


def separate(input_path: Path, output_dir: Path, device: str = "auto"):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"[StemForge] Loading HTDemucs FT on {device}...")
    model = get_model("htdemucs_ft")
    model.to(device)
    model.eval()

    wav = load_audio(input_path)
    wav = wav.unsqueeze(0).to(device)  # (1, 2, T)

    print("[StemForge] Separating stems (shifts=2, overlap=0.25)...")
    with torch.no_grad():
        sources = apply_model(
            model, wav,
            shifts=2,
            overlap=0.25,
            progress=True,
        )  # (1, 4, 2, T)

    raw_stems = {
        name: sources[0, i].cpu()
        for i, name in enumerate(model.sources)
    }

    print("[StemForge] Applying Wiener filtering...")
    filtered_stems = wiener_filter(raw_stems)

    print("[StemForge] De-clicking and saving...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, wav_stem in filtered_stems.items():
        wav_stem = declick(wav_stem)
        out_path = output_dir / f"{name}.wav"
        save_audio(wav_stem, out_path)
        print(f"  ✓ {out_path}")

    print(f"[StemForge] Done → {output_dir}")
    return {name: output_dir / f"{name}.wav" for name in filtered_stems}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StemForge stem splitter")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("stems"), help="Output directory")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    separate(args.input, args.output, args.device)
