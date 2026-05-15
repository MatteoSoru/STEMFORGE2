#!/usr/bin/env python3
"""
StemForge — LoRA Fine-tuner
Addestra MusicGen su dataset custom (cartella di file WAV + prompt .txt).

Struttura dataset attesa:
  dataset/
    brano1.wav
    brano1.txt   ← prompt testuale associato
    brano2.wav
    brano2.txt
    ...
"""

import json
import time
import random
from pathlib import Path
from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader


@dataclass
class TrainConfig:
    dataset_dir: Path
    genre: str = "custom"
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 4
    max_duration: int = 30
    sample_rate: int = 32000  # MusicGen native SR
    output_dir: Path = Path("checkpoints/custom")
    lora_rank: int = 8
    lora_alpha: float = 16.0
    warmup_steps: int = 50
    grad_clip: float = 1.0
    save_every: int = 2  # save checkpoint every N epochs
    device: str = "auto"


class AudioTextDataset(Dataset):
    def __init__(self, data_dir: Path, sr: int = 32000, max_dur: int = 30):
        self.sr = sr
        self.max_samples = sr * max_dur
        self.pairs = []

        for wav_path in sorted(data_dir.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            prompt = txt_path.read_text().strip() if txt_path.exists() else wav_path.stem
            self.pairs.append((wav_path, prompt))

        if not self.pairs:
            raise ValueError(f"No .wav files found in {data_dir}")
        print(f"[Dataset] Found {len(self.pairs)} audio/prompt pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wav_path, prompt = self.pairs[idx]
        wav, sr = torchaudio.load(str(wav_path))

        # Resample if needed
        if sr != self.sr:
            wav = T.Resample(sr, self.sr)(wav)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        # Trim or pad to max_samples
        if wav.shape[1] > self.max_samples:
            start = random.randint(0, wav.shape[1] - self.max_samples)
            wav = wav[:, start:start + self.max_samples]
        else:
            pad = self.max_samples - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad))

        return wav.squeeze(0), prompt


def inject_lora(model, rank: int = 8, alpha: float = 16.0):
    """Inject LoRA adapters into the transformer attention layers."""
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, original: nn.Linear, r: int, alpha: float):
            super().__init__()
            self.original = original
            self.r = r
            self.scale = alpha / r
            d_in, d_out = original.in_features, original.out_features
            self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        def forward(self, x):
            return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale

    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in
                                                   ["q_proj", "k_proj", "v_proj", "out_proj"]):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, rank, alpha))
            replaced += 1

    print(f"[LoRA] Injected into {replaced} attention layers (rank={rank}, alpha={alpha})")
    return model


def get_lora_params(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def train_lora(
    dataset_dir: Path,
    genre: str = "custom",
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 4,
    output_dir: Path = Path("checkpoints/custom"),
    lora_rank: int = 8,
) -> Path:

    config = TrainConfig(
        dataset_dir=dataset_dir,
        genre=genre,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        output_dir=output_dir,
        lora_rank=lora_rank,
    )

    if config.device == "auto":
        if torch.backends.mps.is_available():
            config.device = "mps"
        elif torch.cuda.is_available():
            config.device = "cuda"
        else:
            config.device = "cpu"

    print(f"[Trainer] Device: {config.device}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from audiocraft.models import MusicGen
    print("[Trainer] Loading MusicGen medium...")
    model = MusicGen.get_pretrained("facebook/musicgen-medium")
    model = inject_lora(model.lm, rank=config.lora_rank, alpha=config.lora_alpha)
    model = model.to(config.device)

    # Freeze everything except LoRA
    for p in model.parameters():
        p.requires_grad = False
    lora_params = get_lora_params(model)
    for p in lora_params:
        p.requires_grad = True

    total_params = sum(p.numel() for p in lora_params)
    print(f"[Trainer] Trainable LoRA params: {total_params:,}")

    # Dataset
    dataset = AudioTextDataset(config.dataset_dir, config.sample_rate, config.max_duration)
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(lora_params, lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(loader)
    )

    best_loss = float("inf")
    best_path = None
    log = []

    print(f"\n[Trainer] Starting training: {config.epochs} epochs, "
          f"{len(dataset)} samples, batch={config.batch_size}\n")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (wav, prompts) in enumerate(loader):
            wav = wav.to(config.device)

            # Encode audio to tokens via EnCodec
            with torch.no_grad():
                codes, _ = model.compression_model.encode(wav.unsqueeze(1))

            # Get text conditioning
            attributes, _ = model._prepare_tokens_and_attributes(prompts, None)

            # Forward pass through LM
            lm_output = model.lm.compute_predictions(codes, [], attributes)
            loss = lm_output.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"  Epoch {epoch}/{config.epochs} | "
                      f"Step {step}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"\n→ Epoch {epoch} done | avg_loss={avg_loss:.4f} | {elapsed:.1f}s\n")

        log.append({"epoch": epoch, "loss": avg_loss, "time": elapsed})

        # Save checkpoint
        if epoch % config.save_every == 0 or epoch == config.epochs:
            ckpt_path = output_dir / f"{genre}_epoch{epoch:03d}.pt"
            lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(lora_state, ckpt_path)
            print(f"[Trainer] Saved checkpoint → {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = output_dir / f"{genre}_best.pt"
                torch.save(lora_state, best_path)
                print(f"[Trainer] ✓ New best model saved → {best_path}")

    # Save training log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({"genre": genre, "epochs": config.epochs,
                   "final_loss": log[-1]["loss"], "history": log}, f, indent=2)
    print(f"\n[Trainer] Training complete. Best loss: {best_loss:.4f}")
    print(f"[Trainer] Best checkpoint → {best_path}")

    return best_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="StemForge LoRA trainer")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--genre", default="custom")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/custom"))
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()

    train_lora(
        dataset_dir=args.dataset_dir,
        genre=args.genre,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
    )
