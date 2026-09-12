"""
TTS Module — generate Bangla audio per debate turn.
==============================================

Phase 4 of the BD Political Debater pipeline.

Two providers:
1. z-ai CLI (default, no API key needed, works for Bangla)
2. Facebook MMS-TTS-ben (local, best open Bangla TTS, requires PyTorch)

Each turn of the debate is rendered as a separate WAV file:
    /home/z/my-project/data/audio/<debate_id>/turn_<i>_<agent>.wav

This is later consumed by Phase 5 (SadTalker) to drive talking-head animation.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path("/home/z/my-project")
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TTSResult:
    turn_index: int
    agent: str
    audio_path: str
    duration_seconds: float = 0.0
    provider: str = ""
    text_chars: int = 0


def _clean_text_for_tts(text: str) -> str:
    """Remove citation markers [1] [2] etc. for TTS reading.

    Keeping them would make the TTS say "one" "two" awkwardly.
    """
    # Remove [n] markers
    text = re.sub(r"\[(\d+)\]", "", text)
    # Remove markdown emphasis
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences_bn(text: str, max_chars: int = 200) -> list[str]:
    """Split Bangla/English text into sentence-sized chunks.

    Bangla sentence terminator: । (DAARI, U+0964) and ॥ (double daari).
    Also splits on `.`, `!`, `?` for English segments.
    """
    # First, split on Bangla daari (।) — keep the daari with the preceding sentence
    parts = re.split(r"(?<=[।॥.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip() if current else p
        else:
            if current:
                chunks.append(current)
            # If the single sentence itself is too long, hard-split it
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def _zai_tts(text: str, out_path: Path, voice: str = "tongtong") -> bool:
    """Use z-ai CLI for TTS. Returns True on success.

    z-ai API rejects text >~250 chars, so we chunk on sentence boundaries
    and concatenate audio pieces with a small 200ms gap.
    """
    chunks = _split_sentences_bn(text, max_chars=200)
    if not chunks:
        return False

    # Synthesise each chunk to a temp file, then concat
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="zai_tts_"))
    piece_paths: list[Path] = []

    try:
        for i, chunk in enumerate(chunks):
            piece_path = tmpdir / f"piece_{i:03d}.wav"
            cmd = [
                "z-ai", "tts",
                "--input", chunk,
                "--output", str(piece_path),
                "--voice", voice,
            ]
            # Retry with backoff — z-ai API sometimes rate-limits
            ok = False
            for attempt in range(3):
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if proc.returncode == 0 and piece_path.exists():
                        ok = True
                        break
                    print(f"[tts] piece {i} attempt {attempt+1}: rc={proc.returncode} err={proc.stderr[:150]}", file=sys.stderr)
                except Exception as e:
                    print(f"[tts] piece {i} attempt {attempt+1}: {e}", file=sys.stderr)
                # Backoff
                import time as _t
                _t.sleep(2 ** attempt)
            if not ok:
                print(f"[tts] piece {i} failed after 3 retries", file=sys.stderr)
                return False
            piece_paths.append(piece_path)
            # Polite delay between calls
            import time as _t
            _t.sleep(0.5)

        # Concatenate all WAV pieces with a small 200ms gap
        if not piece_paths:
            return False

        import soundfile as sf
        import numpy as np

        all_audio: list[np.ndarray] = []
        sample_rate = None
        for pp in piece_paths:
            audio, sr = sf.read(str(pp))
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if mismatched (rare)
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * sample_rate / sr)),
                    np.arange(len(audio)),
                    audio,
                )
            all_audio.append(audio)
            # Add 200ms of silence
            silence = np.zeros(int(sample_rate * 0.2), dtype=audio.dtype)
            all_audio.append(silence)

        if not all_audio:
            return False
        full_audio = np.concatenate(all_audio)
        sf.write(str(out_path), full_audio, samplerate=sample_rate)
        return True
    finally:
        # Cleanup tmpdir
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _mms_tts(text: str, out_path: Path) -> bool:
    """Use Facebook MMS-TTS-ben for TTS. Returns True on success.

    Note: MMS-TTS for Bangla is a fine-tuned VITS model. It works best on
    short sentences — long debate turns should be split into ~200-char chunks
    and concatenated as audio.
    """
    try:
        import torch
        from transformers import VitsModel, AutoTokenizer
    except ImportError as e:
        print(f"[tts] MMS-TTS needs torch+transformers: {e}", file=sys.stderr)
        return False

    # Load model (cached after first run)
    if not hasattr(_mms_tts, "_model"):
        model_name = "facebook/mms-tts-ben"
        _mms_tts._model = VitsModel.from_pretrained(model_name)
        _mms_tts._tokenizer = AutoTokenizer.from_pretrained(model_name)
        _mms_tts._model.eval()
    model = _mms_tts._model
    tokenizer = _mms_tts._tokenizer

    # Chunk the text if very long
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    audio_pieces = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        audio_pieces.append(output[0].numpy())

    if not audio_pieces:
        return False

    import numpy as np
    import soundfile as sf
    full_audio = np.concatenate(audio_pieces)
    sf.write(str(out_path), full_audio, samplerate=model.config.sampling_rate)
    return True


def synth_turn_audio(
    turn_text: str,
    turn_index: int,
    agent: str,
    out_dir: Path,
    provider: str = "z-ai",
) -> TTSResult | None:
    """Synthesise audio for a single debate turn.

    provider: "z-ai" (default, dev) | "mms" (local Bangla, prod)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_text = _clean_text_for_tts(turn_text)
    out_path = out_dir / f"turn_{turn_index:02d}_{agent}.wav"

    print(f"[tts] turn {turn_index+1} ({agent}, {len(clean_text)} chars) → {out_path.name}")
    t0 = time.time()

    if provider == "z-ai":
        ok = _zai_tts(clean_text, out_path)
    elif provider == "mms":
        ok = _mms_tts(clean_text, out_path)
    else:
        print(f"[tts] unknown provider: {provider}", file=sys.stderr)
        return None

    if not ok:
        print(f"[tts] failed for turn {turn_index+1}", file=sys.stderr)
        return None

    # Probe duration with soundfile if available
    duration = 0.0
    try:
        import soundfile as sf
        info = sf.info(str(out_path))
        duration = info.frames / info.samplerate
    except Exception:
        pass

    return TTSResult(
        turn_index=turn_index,
        agent=agent,
        audio_path=str(out_path),
        duration_seconds=duration,
        provider=provider,
        text_chars=len(clean_text),
    )


def synth_debate_audio(
    debate_path: Path,
    out_dir: Path | None = None,
    provider: str = "z-ai",
) -> list[TTSResult]:
    """Synthesise audio for every turn in a saved debate transcript."""
    with open(debate_path, "r", encoding="utf-8") as f:
        debate = json.load(f)

    if out_dir is None:
        out_dir = debate_path.parent / (debate_path.stem + "_audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[TTSResult] = []
    for turn in debate["turns"]:
        r = synth_turn_audio(
            turn_text=turn["text"],
            turn_index=turn["turn_index"],
            agent=turn["agent"],
            out_dir=out_dir,
            provider=provider,
        )
        if r:
            results.append(r)

    # Save audio manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[tts] {len(results)} turns synthesised → {out_dir}")
    print(f"[tts] manifest: {manifest_path}")
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debate", type=str, required=True, help="Path to debate transcript JSON")
    ap.add_argument("--out", type=str, default=None, help="Output dir for audio")
    ap.add_argument("--provider", type=str, default="z-ai", choices=["z-ai", "mms"])
    args = ap.parse_args()

    debate_path = Path(args.debate)
    out_dir = Path(args.out) if args.out else None
    synth_debate_audio(debate_path, out_dir, args.provider)


if __name__ == "__main__":
    main()
