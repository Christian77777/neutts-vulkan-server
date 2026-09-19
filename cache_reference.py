#!/usr/bin/env python3
"""
Voice Reference Pre-Encoder Utility for NeuTTS

Pre-encodes a WAV/MP3 reference voice file into a cached PyTorch token file (.pt)
to avoid the ~26-second CPU re-encoding overhead on every inference request.

Usage:
    python cache_reference.py --audio voice.wav --text "Exact transcript." --name my_voice
"""

import argparse
import os
import re
from pathlib import Path
import torch
from neutts import NeuTTS


def main():
    parser = argparse.ArgumentParser(description="Pre-encode voice audio for NeuTTS")
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to reference audio file (.wav or .mp3)")
    parser.add_argument("--text", "-t", type=str, required=True, help="Exact transcript of the reference audio, or path to a .txt file")
    parser.add_argument("--name", "-n", type=str, default="", help="Voice name for output files (e.g., 'aelita')")
    parser.add_argument("--out-dir", "-o", type=str, default="./voices", help="Output directory to save .pt and .txt files")
    parser.add_argument("--model", type=str, default="neuphonic/neutts-air-q8-gguf", help="Backbone model repo")
    parser.add_argument("--device", type=str, default="cpu", help="Device for encoder (cpu)")

    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Resolve reference text
    text_content = args.text
    if Path(args.text).exists():
        with open(args.text, "r", encoding="utf-8") as f:
            text_content = f.read().strip()

    raw_name = args.name or audio_path.stem
    voice_name = re.sub(r'[^a-z0-9_\-]', '', raw_name.strip().lower().replace(" ", "_"))
    if not voice_name:
        raise ValueError(f"Invalid voice name derived from '{raw_name}'. Only alphanumeric, underscores, and dashes allowed.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pt = out_dir / f"{voice_name}.pt"
    out_txt = out_dir / f"{voice_name}.txt"

    print(f"Loading NeuTTS codec to encode '{audio_path}'...")
    tts = NeuTTS(
        backbone_repo=args.model,
        backbone_device=args.device,
        codec_repo="neuphonic/neucodec",
        codec_device="cpu"
    )

    print(f"Encoding audio: {audio_path}")
    ref_codes = tts.encode_reference(str(audio_path))

    torch.save(ref_codes, out_pt)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text_content)

    print("Encoding complete!")
    print(f"  Token file:  {out_pt} ({os.path.getsize(out_pt)} bytes)")
    print(f"  Text file:   {out_txt}")
    print(f"  Voice ID:    '{voice_name}'")


if __name__ == "__main__":
    main()
