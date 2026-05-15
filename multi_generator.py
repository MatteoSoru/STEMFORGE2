#!/usr/bin/env python3
"""
StemForge — Multi-backend Music Generator
Backends: musicgen | stable-audio | audioldm2 | ensemble (media spettrale)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 44100


def save_audio(wav: np.ndarray, path: Path, sr: int = SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(wav)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).repeat(2, 1)
    elif tensor.shape[0] == 1:
        tensor = tensor.repeat(2, 1)
    torchaudio.save(str(path), tensor, sr, bits_per_sample=32)


# ── MusicGen ────────────────────────────────────────────────────────────────

def generate_musicgen(prompt: str, duration: int, checkpoint: str = None,
                      melody_path: Path = None) -> np.ndarray:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write

    model_name = "facebook/musicgen-stereo-large"
    print(f"[MusicGen] Loading {model_name}...")
    model = MusicGen.get_pretrained(model_name)

    if checkpoint:
        print(f"[MusicGen] Loading LoRA checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        model.lm.load_state_dict(state, strict=False)

    model.set_generation_params(duration=min(duration, 30))

    if duration > 30:
        print(f"[MusicGen] Long generation: chunking {duration}s with crossfade...")
        return _musicgen_long(model, prompt, duration)

    if melody_path:
        melody, sr = torchaudio.load(str(melody_path))
        wav = model.generate_with_chroma([prompt], melody[None], sr)
    else:
        wav = model.generate([prompt])

    return wav[0].cpu().numpy()


def _musicgen_long(model, prompt: str, duration: int,
                   chunk_s: int = 30, overlap_s: int = 3) -> np.ndarray:
    sr = SAMPLE_RATE
    overlap = overlap_s * sr
    chunks = []
    generated = 0

    while generated < duration:
        remaining = duration - generated
        seg_dur = min(chunk_s, remaining + overlap_s)
        wav = model.generate([prompt])[0].cpu().numpy()
        if chunks:
            # Linear crossfade
            fade_in = np.linspace(0, 1, overlap)
            fade_out = np.linspace(1, 0, overlap)
            chunks[-1][..., -overlap:] *= fade_out
            wav[..., :overlap] *= fade_in
            chunks[-1] = np.concatenate([chunks[-1], wav], axis=-1)
        else:
            chunks.append(wav)
        generated += seg_dur - overlap_s

    return chunks[0][..., :duration * sr]


# ── Stable Audio Open ────────────────────────────────────────────────────────

def generate_stable_audio(prompt: str, duration: int, steps: int = 100) -> np.ndarray:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    import json

    print("[StableAudio] Loading model...")
    model, config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    sample_rate = config["sample_rate"]
    sample_size = int(sample_rate * duration)

    print(f"[StableAudio] Generating {duration}s on {device} ({steps} steps)...")
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration,
    }]

    with torch.no_grad():
        output = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device,
        )

    wav = output[0].cpu().numpy()
    # Resample to 44100 if needed
    if sample_rate != SAMPLE_RATE:
        wav_t = torch.from_numpy(wav)
        wav_t = T.Resample(sample_rate, SAMPLE_RATE)(wav_t)
        wav = wav_t.numpy()

    return wav


# ── AudioLDM2 ────────────────────────────────────────────────────────────────

def generate_audioldm2(prompt: str, duration: int) -> np.ndarray:
    from diffusers import AudioLDM2Pipeline

    print("[AudioLDM2] Loading pipeline...")
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm2-music",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    print(f"[AudioLDM2] Generating {duration}s...")
    output = pipe(
        prompt,
        num_inference_steps=200,
        audio_length_in_s=float(duration),
        num_waveforms_per_prompt=1,
    )
    wav = output.audios[0]  # (T,) at 16kHz

    # Upsample to 44100
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    wav_t = T.Resample(16000, SAMPLE_RATE)(wav_t)
    return wav_t.squeeze(0).numpy()


# ── Ensemble ────────────────────────────────────────────────────────────────

def generate_ensemble(prompt: str, duration: int) -> np.ndarray:
    """
    Genera con tutti e 3 i backend e fa la media spettrale.
    Qualità massima, richiede ~16 GB VRAM.
    """
    print("[Ensemble] Running all backends...")
    wavs = []
    for fn in [generate_musicgen, generate_stable_audio, generate_audioldm2]:
        try:
            w = fn(prompt, duration)
            # Normalize length
            target = duration * SAMPLE_RATE
            if w.shape[-1] > target:
                w = w[..., :target]
            elif w.shape[-1] < target:
                pad = target - w.shape[-1]
                w = np.pad(w, ((0, 0), (0, pad)) if w.ndim == 2 else (0, pad))
            wavs.append(w)
        except Exception as e:
            print(f"  [Ensemble] Warning: {fn.__name__} failed: {e}")

    if not wavs:
        raise RuntimeError("All backends failed in ensemble mode.")

    # Spectral averaging
    n_fft = 4096
    result_specs = []
    for w in wavs:
        if w.ndim == 1:
            w = np.stack([w, w])
        spec = np.fft.rfft(w, n=n_fft, axis=-1)
        result_specs.append(spec)

    avg_spec = np.mean(result_specs, axis=0)
    mixed = np.fft.irfft(avg_spec, n=n_fft, axis=-1)

    # Trim/pad to target length
    target = duration * SAMPLE_RATE
    if mixed.shape[-1] > target:
        mixed = mixed[..., :target]

    return mixed


# ── Main ─────────────────────────────────────────────────────────────────────

BACKENDS = {
    "musicgen": generate_musicgen,
    "stable-audio": generate_stable_audio,
    "audioldm2": generate_audioldm2,
    "ensemble": generate_ensemble,
}


def generate(prompt: str, duration: int, backend: str = "stable-audio",
             output_dir: Path = Path("output"), split_stems: bool = False,
             checkpoint: str = None, melody: Path = None, steps: int = 100):

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[StemForge] Backend: {backend} | Duration: {duration}s")
    print(f"[StemForge] Prompt: {prompt}\n")

    t0 = time.time()

    if backend == "musicgen":
        wav = generate_musicgen(prompt, duration, checkpoint, melody)
    elif backend == "stable-audio":
        wav = generate_stable_audio(prompt, duration, steps)
    elif backend == "audioldm2":
        wav = generate_audioldm2(prompt, duration)
    elif backend == "ensemble":
        wav = generate_ensemble(prompt, duration)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    mix_path = output_dir / "full_mix.wav"
    save_audio(wav, mix_path)
    print(f"\n[StemForge] Mix saved → {mix_path} ({time.time()-t0:.1f}s)")

    if split_stems:
        from stem_splitter import separate
        stems_dir = output_dir / "stems"
        print("[StemForge] Splitting stems...")
        separate(mix_path, stems_dir)

    return mix_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StemForge music generator")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--duration", "-d", type=int, default=30)
    parser.add_argument("--backend", "-b", default="stable-audio",
                        choices=list(BACKENDS.keys()))
    parser.add_argument("--output", "-o", type=Path, default=Path("output"))
    parser.add_argument("--split-stems", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA fine-tuned checkpoint (.pt)")
    parser.add_argument("--melody", type=Path, default=None,
                        help="Reference melody WAV (MusicGen only)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Diffusion steps (Stable Audio only)")
    args = parser.parse_args()

    generate(
        prompt=args.prompt,
        duration=args.duration,
        backend=args.backend,
        output_dir=args.output,
        split_stems=args.split_stems,
        checkpoint=args.checkpoint,
        melody=args.melody,
        steps=args.steps,
    )
