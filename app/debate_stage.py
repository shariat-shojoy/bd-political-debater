"""
Debate Stage — 2-character live animation viewer.
==============================================

Renders an interactive HTML/CSS/JS "debate stage" inside Streamlit with:
  • Two SVG cartoon characters (Analyst বিশ্লেষক on left, Journalist সাংবাদিক on right)
  • Speech bubbles showing the Bangla text with [n] citation markers highlighted
  • Audio-amplitude-driven mouth animation synced to each turn's WAV
  • Auto-advance through the 4 turns (opening → rebuttal → counter → closing)
  • Speaker highlighted (glow) + listener dimmed
  • Play / Pause / Next-turn controls

If SadTalker MP4s exist for each turn (data/animations/<debate>/turn_NN_agent.mp4),
the stage will overlay the realistic talking-head video on top of the SVG character.

Usage from app/ui.py:
    from app.debate_stage import render_debate_stage
    render_debate_stage(debate_path, audio_dir)
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import streamlit.components.v1 as components
import streamlit as st

# Make project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app._paths import BASE_DIR


def _encode_audio(path: Path | None) -> str:
    """Return base64-encoded audio data URI string (or empty string if missing)."""
    if not path or not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _encode_video(path: Path | None) -> str:
    """Return base64-encoded video data URI (or empty string)."""
    if not path or not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _build_stage_html(debate: dict, audio_paths: dict[int, Path], video_paths: dict[int, Path] | None = None) -> str:
    """Build the full HTML/CSS/JS for the 2-character debate stage."""

    # Prepare turn payload
    turns_payload = []
    for turn in debate["turns"]:
        idx = turn["turn_index"]
        agent = turn["agent"]
        audio_uri = _encode_audio(audio_paths.get(idx))
        video_uri = ""
        if video_paths and video_paths.get(idx):
            video_uri = _encode_video(video_paths.get(idx))
        citations = turn.get("citations", [])
        # Normalize citations (they may be dataclass objects or dicts)
        cites_clean = []
        for c in citations:
            if hasattr(c, "__dict__"):
                cites_clean.append(c.__dict__)
            else:
                cites_clean.append(c)

        turns_payload.append({
            "turn_index": idx,
            "agent": agent,
            "phase": turn["phase"],
            "agent_name": turn.get("agent_name", agent),
            "text": turn["text"],
            "citations": cites_clean,
            "audio_uri": audio_uri,
            "video_uri": video_uri,
            "duration_seconds": turn.get("elapsed_seconds", 0.0),
        })

    payload = {
        "topic": debate.get("topic", ""),
        "turns": turns_payload,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    # The HTML template — keep as one big string. Inline CSS + JS.
    html = """<!DOCTYPE html>
<html lang="bn">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    font-family: 'Noto Sans Bengali', 'Hind Siliguri', sans-serif;
    color: #fff; overflow: hidden;
  }
  .stage { display: flex; flex-direction: column; height: 100vh; padding: 16px; }
  .topic-bar {
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,217,61,0.3);
    border-radius: 8px; padding: 12px 20px; margin-bottom: 15px;
    font-size: 14px; text-align: center; line-height: 1.5;
  }
  .topic-bar strong { color: #FFD93D; }
  .characters {
    display: flex; justify-content: space-between; flex: 1;
    align-items: flex-end; padding: 0 40px 60px; position: relative;
    min-height: 480px;
  }
  .character {
    width: 42%; position: relative;
    display: flex; flex-direction: column; align-items: center;
    transition: opacity 0.5s, transform 0.5s;
  }
  .character.speaking { opacity: 1; transform: translateY(0); }
  .character.listening { opacity: 0.55; transform: translateY(5px); }
  .avatar-wrap {
    width: 240px; height: 240px; position: relative;
    border-radius: 50%; border: 5px solid;
    background: white;
    box-shadow: 0 0 30px rgba(0,0,0,0.5);
    transition: box-shadow 0.3s;
    overflow: visible;
  }
  .character.speaking .avatar-wrap {
    box-shadow: 0 0 60px var(--glow-color), 0 0 100px var(--glow-color);
  }
  .analyst { --glow-color: #1976D2; }
  .analyst .avatar-wrap { border-color: #1976D2; }
  .journalist { --glow-color: #F57C00; }
  .journalist .avatar-wrap { border-color: #F57C00; }
  .avatar-svg, .avatar-video {
    position: absolute; inset: 0; width: 100%; height: 100%;
    border-radius: 50%; object-fit: cover;
  }
  .avatar-video { display: none; background: #000; }
  .character.has-video .avatar-svg { display: none; }
  .character.has-video .avatar-video { display: block; }
  .name { margin-top: 14px; font-size: 18px; font-weight: bold; }
  .analyst .name { color: #64B5F6; }
  .journalist .name { color: #FFB74D; }
  .role-badge {
    font-size: 11px; opacity: 0.7; margin-top: 2px;
    text-transform: uppercase; letter-spacing: 1px;
  }
  /* Speech bubble */
  .bubble {
    position: absolute; top: -240px; left: 50%; transform: translateX(-50%);
    width: 380px; max-height: 220px;
    background: white; color: #1a1a2e; padding: 14px 18px;
    border-radius: 14px; font-size: 13px; line-height: 1.7;
    overflow-y: auto; display: none;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
    white-space: pre-wrap; word-break: break-word;
  }
  .character.speaking .bubble { display: block; }
  .bubble:after {
    content: ''; position: absolute; bottom: -12px; left: 50%;
    transform: translateX(-50%);
    border: 12px solid transparent; border-top-color: white;
  }
  .bubble sup { color: #1976D2; font-weight: bold; margin: 0 1px; }
  /* Scrollbar styling */
  .bubble::-webkit-scrollbar { width: 6px; }
  .bubble::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.3); border-radius: 3px; }
  /* Controls */
  .controls {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 12px; align-items: center;
    background: rgba(0,0,0,0.7); padding: 12px 24px;
    border-radius: 32px; border: 1px solid rgba(255,217,61,0.3);
  }
  .controls button {
    background: #FFD93D; color: #1a1a2e; border: none;
    padding: 10px 22px; border-radius: 22px; cursor: pointer;
    font-weight: bold; font-size: 14px; transition: background 0.2s;
  }
  .controls button:hover { background: #FFC107; }
  .controls button:disabled { opacity: 0.4; cursor: not-allowed; }
  .progress { color: #FFD93D; padding: 8px 16px; font-size: 13px; min-width: 200px; text-align: center; }
  .turn-info { font-size: 11px; opacity: 0.7; margin-top: 4px; }
  /* Stage loaded indicator */
  .loaded-badge {
    position: fixed; top: 12px; right: 12px;
    background: #4CAF50; color: white; padding: 4px 12px;
    border-radius: 12px; font-size: 11px; font-weight: bold;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  /* Debug console */
  .debug-console {
    position: fixed; bottom: 80px; right: 12px;
    background: rgba(0,0,0,0.9); color: #0F0; padding: 10px 14px;
    border-radius: 8px; font-family: monospace; font-size: 11px;
    max-width: 400px; max-height: 180px; overflow-y: auto;
    display: none;
  }
  .debug-console.show { display: block; }
  .debug-console .err { color: #F44; }
  .debug-console .warn { color: #FC0; }
  /* Bottom source panel */
  .sources {
    position: fixed; bottom: 90px; left: 50%; transform: translateX(-50%);
    background: rgba(255,255,255,0.95); color: #1a1a2e;
    padding: 12px 20px; border-radius: 8px; max-width: 600px;
    max-height: 200px; overflow-y: auto;
    font-size: 12px; line-height: 1.6; display: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }
  .sources.show { display: block; }
  .sources h4 { margin: 0 0 8px 0; font-size: 13px; color: #1976D2; }
  .sources .src-item { margin: 6px 0; padding: 4px 8px; background: #f5f5f5; border-left: 3px solid #1976D2; }
  .sources .src-item .src-title { font-weight: bold; }
  .sources .src-item a { color: #1976D2; text-decoration: none; word-break: break-all; }
  .sources .src-item .src-excerpt { color: #555; font-size: 11px; margin-top: 4px; }
</style>
</head>
<body>
<div class="stage">
  <div class="topic-bar">
    📣 <strong>বিষয়:</strong> <span id="topic"></span>
  </div>
  <div class="characters">
    <div class="character analyst" id="analyst">
      <div class="bubble" id="bubble-analyst"></div>
      <div class="avatar-wrap">
        <svg class="avatar-svg" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
          <!-- Face -->
          <ellipse cx="120" cy="120" rx="90" ry="100" fill="#FFD3B5"/>
          <!-- Hair -->
          <path d="M 30 100 Q 60 30 120 30 Q 180 30 210 100 L 200 80 Q 180 50 120 50 Q 60 50 40 80 Z" fill="#2C1810"/>
          <!-- Glasses (analyst = studious) -->
          <circle cx="95" cy="115" r="22" fill="rgba(255,255,255,0.2)" stroke="#333" stroke-width="3"/>
          <circle cx="145" cy="115" r="22" fill="rgba(255,255,255,0.2)" stroke="#333" stroke-width="3"/>
          <line x1="117" y1="115" x2="123" y2="115" stroke="#333" stroke-width="3"/>
          <!-- Eyes behind glasses -->
          <circle cx="95" cy="115" r="5" fill="#000"/>
          <circle cx="145" cy="115" r="5" fill="#000"/>
          <!-- Eyebrows (slight, neutral) -->
          <line x1="78" y1="88" x2="112" y2="92" stroke="#2C1810" stroke-width="3"/>
          <line x1="128" y1="92" x2="162" y2="88" stroke="#2C1810" stroke-width="3"/>
          <!-- Nose -->
          <path d="M 115 140 L 120 158 L 125 140" stroke="#5D4037" stroke-width="2" fill="none"/>
          <!-- Mouth (animated) -->
          <ellipse class="mouth" id="analyst-mouth" cx="120" cy="180" rx="15" ry="3" fill="#8B0000"/>
          <!-- Suit collar (blue) -->
          <path d="M 50 225 L 95 235 L 120 220 L 145 235 L 190 225 L 190 240 L 50 240 Z" fill="#1976D2"/>
          <path d="M 110 235 L 120 220 L 130 235 L 120 245 Z" fill="white"/>
          <!-- Tie -->
          <path d="M 115 220 L 125 220 L 128 245 L 112 245 Z" fill="#B71C1C"/>
        </svg>
        <video class="avatar-video" id="video-analyst" muted playsinline></video>
      </div>
      <div class="name">বিশ্লেষক</div>
      <div class="role-badge">ANALYST</div>
    </div>

    <div class="character journalist" id="journalist">
      <div class="bubble" id="bubble-journalist"></div>
      <div class="avatar-wrap">
        <svg class="avatar-svg" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">
          <ellipse cx="120" cy="120" rx="90" ry="100" fill="#FFE0BD"/>
          <!-- Hair (different style) -->
          <path d="M 30 90 Q 60 20 120 25 Q 180 30 210 100 L 200 70 Q 180 40 120 45 Q 60 50 40 80 Z" fill="#6D4C2F"/>
          <path d="M 25 100 Q 50 75 80 80 L 75 95 Q 50 90 30 110 Z" fill="#6D4C2F"/>
          <path d="M 215 100 Q 190 75 160 80 L 165 95 Q 190 90 210 110 Z" fill="#6D4C2F"/>
          <!-- Eyes (no glasses) -->
          <circle cx="95" cy="115" r="6" fill="#000"/>
          <circle cx="145" cy="115" r="6" fill="#000"/>
          <!-- Eyebrows (slightly raised, skeptical look) -->
          <path d="M 78 88 Q 95 84 112 88" stroke="#6D4C2F" stroke-width="3" fill="none"/>
          <path d="M 128 88 Q 145 84 162 88" stroke="#6D4C2F" stroke-width="3" fill="none"/>
          <!-- Nose -->
          <path d="M 115 140 L 120 158 L 125 140" stroke="#5D4037" stroke-width="2" fill="none"/>
          <!-- Mouth (animated) -->
          <ellipse class="mouth" id="journalist-mouth" cx="120" cy="180" rx="15" ry="3" fill="#8B0000"/>
          <!-- Casual collar (orange sweater) -->
          <path d="M 50 225 L 95 235 L 120 220 L 145 235 L 190 225 L 190 240 L 50 240 Z" fill="#F57C00"/>
          <!-- Notepad hint -->
          <rect x="100" y="240" width="40" height="4" fill="#FFEB3B"/>
        </svg>
        <video class="avatar-video" id="video-journalist" muted playsinline></video>
      </div>
      <div class="name">সাংবাদিক</div>
      <div class="role-badge">JOURNALIST</div>
    </div>
  </div>

  <!-- Source panel -->
  <div class="sources" id="sources-panel">
    <h4>📚 সূত্রসমূহ</h4>
    <div id="sources-list"></div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <button id="play-btn">▶ শুরু করুন</button>
    <button id="prev-btn">⏮</button>
    <button id="next-btn">⏭</button>
    <span class="progress" id="progress">তৈরি হচ্ছে…</span>
    <button id="src-btn">📚 সূত্র দেখুন</button>
    <button id="dbg-btn">🐞 Debug</button>
  </div>
</div>

<!-- Loaded indicator -->
<div class="loaded-badge" id="loaded-badge" style="display:none;">✓ STAGE LOADED</div>

<!-- Debug console -->
<div class="debug-console" id="debug-console"></div>

<script>
const PAYLOAD = """ + payload_json + """;

// Debug logger
function dbg(msg, level) {
  const console_ = document.getElementById('debug-console');
  if (!console_) return;
  const line = document.createElement('div');
  if (level) line.className = level;
  const ts = new Date().toLocaleTimeString();
  line.textContent = '[' + ts + '] ' + msg;
  console_.appendChild(line);
  console_.scrollTop = console_.scrollHeight;
  // Also log to browser console
  if (level === 'err') console.error(msg);
  else if (level === 'warn') console.warn(msg);
  else console.log(msg);
}

// Show loaded badge + initial debug
window.addEventListener('load', function() {
  document.getElementById('loaded-badge').style.display = 'block';
  dbg('Stage DOM loaded');
  dbg('Payload turns: ' + PAYLOAD.turns.length);
  PAYLOAD.turns.forEach(function(turn, i) {
    dbg('Turn ' + (i+1) + ' [' + turn.agent + '/' + turn.phase + ']: ' +
        'audio=' + (turn.audio_uri ? 'YES (' + Math.round(turn.audio_uri.length / 1024) + 'KB)' : 'NO') +
        ', video=' + (turn.video_uri ? 'YES' : 'NO') +
        ', text=' + turn.text.length + ' chars');
  });
});

// Toggle debug console
document.getElementById('dbg-btn').addEventListener('click', function() {
  document.getElementById('debug-console').classList.toggle('show');
});

let currentTurnIdx = 0;
let isPlaying = false;
let audioCtx = null;
let analyser = null;
let mouthAnimFrame = null;
let audioElements = {};
let videoElements = {};
let connectedAudios = new Set();  // tracks which audios are already wired to analyser

// Pre-create audio + video elements
PAYLOAD.turns.forEach((turn, i) => {
  if (turn.audio_uri) {
    const audio = new Audio(turn.audio_uri);
    audioElements[i] = audio;
  }
  if (turn.video_uri) {
    // Two video elements (one per character) that can play the same source
    const vAnalyst = document.getElementById('video-analyst');
    const vJournalist = document.getElementById('video-journalist');
    // Determine which character this video belongs to
    if (turn.agent === 'analyst') {
      vAnalyst.src = turn.video_uri;
    } else {
      vJournalist.src = turn.video_uri;
    }
  }
});

// Init topic
document.getElementById('topic').textContent = PAYLOAD.topic;

function formatText(text) {
  // Convert [n] markers to styled <sup>
  let escaped = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  escaped = escaped.replace(/\\[(\\d+)\\]/g, '<sup>[$1]</sup>');
  return escaped;
}

function setSpeaking(turnIdx) {
  const turn = PAYLOAD.turns[turnIdx];
  if (!turn) return;

  // Reset both characters
  ['analyst', 'journalist'].forEach(key => {
    document.getElementById(key).classList.remove('speaking', 'listening', 'has-video');
  });

  // Set speaking / listening
  const speakerKey = turn.agent;
  const listenerKey = speakerKey === 'analyst' ? 'journalist' : 'analyst';
  const speakerEl = document.getElementById(speakerKey);
  const listenerEl = document.getElementById(listenerKey);
  speakerEl.classList.add('speaking');
  listenerEl.classList.add('listening');

  // If this turn has a video, mark the speaker as having video
  if (turn.video_uri) {
    speakerEl.classList.add('has-video');
  }

  // Update speech bubbles
  document.getElementById('bubble-' + speakerKey).innerHTML = formatText(turn.text);
  document.getElementById('bubble-' + listenerKey).innerHTML = '';

  // Update progress
  const phaseLabels = {
    'opening': 'উদ্বোধন',
    'rebuttal': 'প্রতিবাদ',
    'counter_rebuttal': 'প্রত্যুত্তর',
    'closing': 'সমাপ্তি',
  };
  const phaseLabel = phaseLabels[turn.phase] || turn.phase;
  document.getElementById('progress').innerHTML =
    `<strong>টার্ন ${turnIdx + 1} / ${PAYLOAD.turns.length}</strong><br>` +
    `<span class="turn-info">${turn.agent_name} · ${phaseLabel}</span>`;

  // Reset mouths
  document.getElementById('analyst-mouth').setAttribute('ry', '3');
  document.getElementById('journalist-mouth').setAttribute('ry', '3');

  // Update sources panel
  const sourcesList = document.getElementById('sources-list');
  sourcesList.innerHTML = '';
  if (turn.citations && turn.citations.length > 0) {
    turn.citations.forEach((c, i) => {
      const div = document.createElement('div');
      div.className = 'src-item';
      div.innerHTML = `
        <div class="src-title">[${c.ref_index}] ${c.source_title}
          <span style="font-weight:normal;color:#666;"> · ${c.lang.toUpperCase()} · ${c.period_label}</span>
        </div>
        <a href="${c.source_url}" target="_blank">${c.source_url}</a>
        <div class="src-excerpt">${(c.excerpt || '').substring(0, 200)}…</div>
      `;
      sourcesList.appendChild(div);
    });
  } else {
    sourcesList.innerHTML = '<p style="color:#888;font-style:italic;">এই বাক্যে কোনো উদ্ধৃতি নেই।</p>';
  }
}

function ensureAudioGraph(audioEl) {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (!analyser) {
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.6;
    analyser.connect(audioCtx.destination);
  }
  if (!connectedAudios.has(audioEl)) {
    try {
      const source = audioCtx.createMediaElementSource(audioEl);
      source.connect(analyser);
      connectedAudios.add(audioEl);
    } catch (e) {
      // already connected — ignore
      connectedAudios.add(audioEl);
    }
  }
}

function startMouthAnimation(speakerKey) {
  const mouthEl = document.getElementById(speakerKey + '-mouth');
  if (!analyser) return;

  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  function animate() {
    analyser.getByteTimeDomainData(dataArray);
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const v = (dataArray[i] - 128) / 128;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / dataArray.length);
    // Map RMS (0..~0.5) to mouth openness (3..18)
    const open = 3 + Math.min(rms * 60, 15);
    mouthEl.setAttribute('ry', open.toFixed(1));
    mouthAnimFrame = requestAnimationFrame(animate);
  }
  animate();
}

function stopMouthAnimation() {
  if (mouthAnimFrame) {
    cancelAnimationFrame(mouthAnimFrame);
    mouthAnimFrame = null;
  }
  document.getElementById('analyst-mouth').setAttribute('ry', '3');
  document.getElementById('journalist-mouth').setAttribute('ry', '3');
}

async function playTurn(idx) {
  if (idx >= PAYLOAD.turns.length) {
    stopMouthAnimation();
    isPlaying = false;
    document.getElementById('play-btn').textContent = '↻ আবার চালান';
    document.getElementById('progress').innerHTML = '<strong>🎉 বিতর্ক শেষ</strong>';
    dbg('Debate ended.');
    return;
  }
  currentTurnIdx = idx;
  setSpeaking(idx);
  dbg('Playing turn ' + (idx+1) + ' (' + PAYLOAD.turns[idx].agent + '/' + PAYLOAD.turns[idx].phase + ')');

  const turn = PAYLOAD.turns[idx];
  const audio = audioElements[idx];

  // If video exists, play it too
  const speakerVideo = document.getElementById('video-' + turn.agent);
  if (turn.video_uri && speakerVideo) {
    speakerVideo.currentTime = 0;
    speakerVideo.muted = true;
    try { await speakerVideo.play(); dbg('  Video playing for ' + turn.agent); } catch (e) { dbg('  Video play failed: ' + e.message, 'err'); }
  } else {
    dbg('  No video for this turn (will use SVG + audio)');
  }

  if (!audio) {
    dbg('  NO AUDIO for this turn - skipping after 3s', 'warn');
    setTimeout(() => { if (isPlaying) playTurn(idx + 1); }, 3000);
    return;
  }

  ensureAudioGraph(audio);
  if (audioCtx.state === 'suspended') {
    dbg('  AudioContext suspended - resuming');
    await audioCtx.resume();
  }

  audio.currentTime = 0;
  audio.onended = () => {
    dbg('  Audio ended for turn ' + (idx+1));
    stopMouthAnimation();
    if (turn.video_uri && speakerVideo) {
      speakerVideo.pause();
    }
    if (isPlaying) {
      setTimeout(() => playTurn(idx + 1), 400);
    }
  };

  startMouthAnimation(turn.agent);
  dbg('  Starting mouth animation');

  try {
    await audio.play();
    dbg('  Audio playing OK');
  } catch (e) {
    dbg('  Audio play FAILED: ' + e.message, 'err');
    stopMouthAnimation();
  }
}

document.getElementById('play-btn').addEventListener('click', async () => {
  const btn = document.getElementById('play-btn');
  if (isPlaying) {
    isPlaying = false;
    const audio = audioElements[currentTurnIdx];
    if (audio) audio.pause();
    const turn = PAYLOAD.turns[currentTurnIdx];
    if (turn && turn.video_uri) {
      const v = document.getElementById('video-' + turn.agent);
      if (v) v.pause();
    }
    stopMouthAnimation();
    btn.textContent = '▶ চালিয়ে যান';
  } else {
    isPlaying = true;
    btn.textContent = '⏸ বিরতি';
    await playTurn(currentTurnIdx);
  }
});

document.getElementById('next-btn').addEventListener('click', () => {
  const audio = audioElements[currentTurnIdx];
  if (audio) audio.pause();
  const turn = PAYLOAD.turns[currentTurnIdx];
  if (turn && turn.video_uri) {
    const v = document.getElementById('video-' + turn.agent);
    if (v) v.pause();
  }
  stopMouthAnimation();
  const next = (currentTurnIdx + 1) % PAYLOAD.turns.length;
  currentTurnIdx = next;
  if (isPlaying) {
    playTurn(next);
  } else {
    setSpeaking(next);
  }
});

document.getElementById('prev-btn').addEventListener('click', () => {
  const audio = audioElements[currentTurnIdx];
  if (audio) audio.pause();
  stopMouthAnimation();
  const prev = (currentTurnIdx - 1 + PAYLOAD.turns.length) % PAYLOAD.turns.length;
  currentTurnIdx = prev;
  if (isPlaying) {
    playTurn(prev);
  } else {
    setSpeaking(prev);
  }
});

document.getElementById('src-btn').addEventListener('click', () => {
  const panel = document.getElementById('sources-panel');
  panel.classList.toggle('show');
});

// Initialize: show first turn as preview
setSpeaking(0);
document.getElementById('progress').innerHTML =
  '<strong>প্রস্তুত</strong><br><span class="turn-info">▶ বোতাম চাপুন</span>';
</script>
</body>
</html>
"""
    return html


def render_debate_stage(debate_path: str | Path, audio_dir: str | Path | None = None, height: int = 820) -> None:
    """Render the live 2-character debate stage in Streamlit.

    Args:
        debate_path: Path to the debate JSON transcript
        audio_dir: Directory containing per-turn WAV files (default: <debate_stem>_audio)
        height: Pixel height of the embedded stage
    """
    debate_path = Path(debate_path)
    if not debate_path.exists():
        st.error(f"Debate file not found: {debate_path}")
        return

    with open(debate_path, "r", encoding="utf-8") as f:
        debate = json.load(f)

    # Resolve audio dir
    if audio_dir is None:
        audio_dir = debate_path.parent / (debate_path.stem + "_audio")
    audio_dir = Path(audio_dir)

    # Pre-flight check: count available audio + video files
    audio_files_found = []
    audio_files_missing = []
    for turn in debate["turns"]:
        idx = turn["turn_index"]
        agent = turn["agent"]
        p = audio_dir / f"turn_{idx:02d}_{agent}.wav"
        if p.exists():
            audio_files_found.append(p.name)
        else:
            audio_files_missing.append(p.name)

    anim_dir = debate_path.parent / (debate_path.stem + "_anim")
    video_files_found = []
    if anim_dir.exists():
        for turn in debate["turns"]:
            idx = turn["turn_index"]
            agent = turn["agent"]
            p = anim_dir / f"turn_{idx:02d}_{agent}.mp4"
            if p.exists():
                video_files_found.append(p.name)

    # ---------- Diagnostic banner ----------
    st.markdown("### 🎬 Live debate stage")
    st.caption(f"Debate: `{debate_path.name}`")

    col1, col2 = st.columns(2)
    with col1:
        if audio_files_found:
            st.success(f"✅ Audio: {len(audio_files_found)}/{len(debate['turns'])} turns")
        else:
            st.error(f"❌ Audio: 0/{len(debate['turns'])} turns")
    with col2:
        if video_files_found:
            st.success(f"✅ SadTalker MP4s: {len(video_files_found)}/{len(debate['turns'])} turns")
        else:
            st.info("ℹ️ SadTalker MP4s: none (will use SVG cartoon)")

    # If no audio at all, show a clear actionable error
    if not audio_files_found:
        st.error(
            f"❌ **Cannot play stage — no audio files found.**\n\n"
            f"Expected at: `{audio_dir}`\n\n"
            f"Missing files: `{', '.join(audio_files_missing)}`\n\n"
            f"**Fix — run this in PowerShell:**\n"
            f"```\n"
            f"python app/tts/synth.py "
            f"--debate {debate_path} "
            f"--out {audio_dir} "
            f"--provider mms\n"
            f"```"
        )
        if st.button("🔄 I've generated audio — reload stage", type="primary"):
            st.rerun()
        return

    # Build per-turn audio path map
    audio_paths: dict[int, Path] = {}
    for turn in debate["turns"]:
        idx = turn["turn_index"]
        agent = turn["agent"]
        p = audio_dir / f"turn_{idx:02d}_{agent}.wav"
        audio_paths[idx] = p if p.exists() else None

    # Look for SadTalker videos (optional - Phase 5 upgrade)
    video_paths: dict[int, Path] = {}
    has_videos = False
    if anim_dir.exists():
        for turn in debate["turns"]:
            idx = turn["turn_index"]
            agent = turn["agent"]
            p = anim_dir / f"turn_{idx:02d}_{agent}.mp4"
            if p.exists():
                video_paths[idx] = p
                has_videos = True

    # Total payload size estimate (so user knows it's loading)
    total_bytes = sum(p.stat().st_size for p in audio_paths.values() if p)
    if has_videos:
        total_bytes += sum(p.stat().st_size for p in video_paths.values() if p)
    total_mb = total_bytes / (1024 * 1024)
    st.caption(f"📦 Payload: {total_mb:.1f} MB (audio + videos) - may take a few seconds to load")

    # Build the HTML payload
    html = _build_stage_html(debate, audio_paths, video_paths if has_videos else None)

    # Embed in Streamlit
    components.html(html, height=height, scrolling=False)
