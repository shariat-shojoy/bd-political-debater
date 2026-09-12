"""
SadTalker Animation Module (Phase 5)
======================================

Drives per-turn audio WAV through SadTalker to produce a talking-head video
per debate turn.

SadTalker is a heavy model (audio2face + 3DMM + face render). It needs a GPU
for usable speed. The code below documents the integration; you should run
it on your GPU machine (16GB+ VRAM).

Setup on a GPU machine
----------------------

1. Clone SadTalker:
    git clone https://github.com/OpenTalker/SadTalker.git /home/z/my-project/SadTalker
    cd /home/z/my-project/SadTalker
    pip install -r requirements.txt

2. Download checkpoints:
    bash scripts/download_models.sh

3. Prepare two static portrait images:
    /home/z/my-project/assets/analyst_avatar.png   # 512x512, frontal face, neutral expression
    /home/z/my-project/assets/journalist_avatar.png # same

4. Run per-turn animation:
    python -m app.animation.animate_turn \
        --debate /home/z/my-project/download/debate_1975_coup.json \
        --audio-dir /home/z/my-project/data/audio/debate_1975_coup \
        --out-dir /home/z/my-project/data/animations/debate_1975_coup

Output:
    /home/z/my-project/data/animations/debate_1975_coup/turn_00_analyst.mp4
    /home/z/my-project/data/animations/debate_1975_coup/turn_01_journalist.mp4
    ...

If SadTalker isn't installed (e.g., on a CPU-only dev box), this module
returns a fallback: a static image + audio, rendered as an MP4 by ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app._paths import BASE_DIR, CONFIG_PATH, ASSETS_DIR
SADTALKER_DIR = BASE_DIR / "SadTalker"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_sadtalker_installed() -> bool:
    return SADTALKER_DIR.exists() and (SADTALKER_DIR / "inference.py").exists()


def animate_with_sadtalker(image_path: Path, audio_path: Path, out_path: Path) -> bool:
    """Run SadTalker inference.py on a single (image, audio) pair."""
    if not is_sadtalker_installed():
        return False
    cmd = [
        "python", str(SADTALKER_DIR / "inference.py"),
        "--driven_audio", str(audio_path),
        "--source_image", str(image_path),
        "--result_dir", str(out_path.parent),
        "--enhancer", "gfpgan",
        "--still",
        "--preprocess", "full",
    ]
    print(f"[anim] SadTalker: {audio_path.name} + {image_path.name}")
    try:
        proc = subprocess.run(cmd, cwd=str(SADTALKER_DIR), capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(f"[anim] SadTalker failed: {proc.stderr[-500:]}", file=sys.stderr)
            return False
        # SadTalker writes to result_dir with a timestamped name; rename
        # Find the newest mp4 in result_dir
        out_dir = out_path.parent
        mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4s and mp4s[0] != out_path:
            mp4s[0].rename(out_path)
        return out_path.exists()
    except Exception as e:
        print(f"[anim] SadTalker exception: {e}", file=sys.stderr)
        return False


def animate_with_fallback(image_path: Path, audio_path: Path, out_path: Path) -> bool:
    """Fallback: combine static image + audio into an MP4 via ffmpeg.

    No actual talking-head motion — just an image with the audio playing.
    Useful for development testing when SadTalker isn't available.
    """
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(image_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"[anim] ffmpeg failed: {proc.stderr[-500:]}", file=sys.stderr)
            return False
        return out_path.exists()
    except FileNotFoundError:
        print("[anim] ffmpeg not installed", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[anim] ffmpeg exception: {e}", file=sys.stderr)
        return False


@dataclass
class AnimResult:
    turn_index: int
    agent: str
    video_path: str
    duration_seconds: float = 0.0
    provider: str = ""  # "sadtalker" or "fallback"


def animate_debate(
    debate_path: Path,
    audio_dir: Path,
    out_dir: Path | None = None,
    use_sadtalker: bool = True,
) -> list[AnimResult]:
    """Animate every turn of a saved debate."""
    cfg = load_config()
    if out_dir is None:
        out_dir = debate_path.parent / (debate_path.stem + "_anim")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Avatar mapping
    avatar_map = {
        "analyst": BASE_DIR / "assets" / "analyst_avatar.png",
        "journalist": BASE_DIR / "assets" / "journalist_avatar.png",
    }
    # Make placeholder avatar images if missing (just a colored square)
    for agent, p in avatar_map.items():
        if not p.exists():
            _make_placeholder_avatar(p, agent)

    with open(debate_path, "r", encoding="utf-8") as f:
        debate = json.load(f)

    results: list[AnimResult] = []
    use_real = use_sadtalker and is_sadtalker_installed()
    provider = "sadtalker" if use_real else "fallback"

    for turn in debate["turns"]:
        agent = turn["agent"]
        idx = turn["turn_index"]
        audio_path = audio_dir / f"turn_{idx:02d}_{agent}.wav"
        if not audio_path.exists():
            print(f"[anim] missing audio for turn {idx+1}: {audio_path}", file=sys.stderr)
            continue
        video_path = out_dir / f"turn_{idx:02d}_{agent}.mp4"
        image_path = avatar_map[agent]

        ok = False
        if use_real:
            ok = animate_with_sadtalker(image_path, audio_path, video_path)
        if not ok:
            ok = animate_with_fallback(image_path, audio_path, video_path)
            provider = "fallback"

        if ok:
            results.append(AnimResult(
                turn_index=idx,
                agent=agent,
                video_path=str(video_path),
                duration_seconds=turn.get("elapsed_seconds", 0.0),
                provider=provider,
            ))
            print(f"[anim] turn {idx+1} ({agent}) → {video_path.name} [{provider}]")

    # Save manifest
    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[anim] {len(results)} turns animated → {out_dir}")
    return results


def _make_placeholder_avatar(path: Path, agent: str) -> None:
    """Create a simple colored PNG avatar placeholder using Pillow."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # If no Pillow, just touch the file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # 512x512 colored square with agent initials
    color = (25, 118, 210) if agent == "analyst" else (245, 124, 0)
    img = Image.new("RGB", (512, 512), color=color)
    d = ImageDraw.Draw(img)
    # Draw a simple face: circle for head, two dots for eyes
    d.ellipse([156, 100, 356, 300], fill=(255, 220, 180))  # face
    d.ellipse([206, 160, 226, 180], fill=(0, 0, 0))        # left eye
    d.ellipse([286, 160, 306, 180], fill=(0, 0, 0))        # right eye
    d.arc([206, 200, 306, 240], 0, 180, fill=(0, 0, 0), width=3)  # smile
    img.save(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debate", type=str, required=True)
    ap.add_argument("--audio-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--no-sadtalker", action="store_true", help="Use ffmpeg fallback only")
    args = ap.parse_args()
    animate_debate(
        debate_path=Path(args.debate),
        audio_dir=Path(args.audio_dir),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        use_sadtalker=not args.no_sadtalker,
    )


if __name__ == "__main__":
    main()
