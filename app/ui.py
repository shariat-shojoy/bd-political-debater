"""
Streamlit UI — single-page debate viewer
=========================================

Phase 6 of the BD Political Debater pipeline.

Run with:
    streamlit run /home/z/my-project/app/ui.py --server.port 8501

Features:
- Topic picker (preset topics or free-form input)
- "Start debate" button → triggers the 4-turn debate engine
- Each turn rendered as a card with:
  * Agent name + role badge (Analyst/Journalist)
  * Full Bangla text with [n] citation markers
  * Source panel showing the cited Wikipedia chunks (title, URL, excerpt)
  * Inline audio player (MMS-TTS-generated WAV)
  * Turn duration
- After all turns: full transcript + audio download link
- Sidebar: corpus stats (article count, chunk count, period coverage)

Phase-5 upgrade: replace the static audio player with a talking-head video
per turn (SadTalker driven by the WAV file). The UI is wired so each turn
slot has an `<audio>` element that can be swapped for `<video>` later.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app._paths import BASE_DIR, CONFIG_PATH

import streamlit as st


# ---- Helpers ----

def load_config() -> dict[str, Any]:
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def corpus_stats() -> dict[str, Any]:
    stats_path = BASE_DIR / "data" / "processed" / "corpus_stats.json"
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text(encoding="utf-8"))


def list_existing_debates() -> list[Path]:
    """Return saved debate transcripts in /home/z/my-project/download/."""
    out = []
    dl_dir = BASE_DIR / "download"
    if not dl_dir.exists():
        return out
    for p in sorted(dl_dir.glob("debate_*.json")):
        out.append(p)
    return out


def render_turn(turn: dict, audio_dir: Path | None = None) -> None:
    """Render a single debate turn card."""
    agent = turn.get("agent", "")
    phase = turn.get("phase", "")
    text = turn.get("text", "")
    cites = turn.get("citations", [])
    elapsed = turn.get("elapsed_seconds", 0.0)

    # Color-code by agent
    if agent == "analyst":
        bg = "#E8F4FD"
        badge_color = "#1976D2"
        role_label = "বিশ্লেষক (Analyst)"
    else:
        bg = "#FFF4E5"
        badge_color = "#F57C00"
        role_label = "সাংবাদিক (Journalist)"

    # Title with phase + agent name
    phase_bn = {
        "opening": "উদ্বোধন",
        "rebuttal": "প্রতিবাদ",
        "counter_rebuttal": "প্রত্যুত্তর",
        "closing": "সমাপ্তি",
    }.get(phase, phase)

    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            border-left: 5px solid {badge_color};
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="color: {badge_color}; margin: 0;">{role_label}</h3>
                <span style="background-color: {badge_color}; color: white; padding: 4px 12px;
                             border-radius: 12px; font-size: 0.85rem;">
                    {phase_bn} · {elapsed:.1f}s
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Turn text with proper Bangla rendering
    st.markdown(f"""
        <div style='
            font-family: "Noto Sans Bengali", "Hind Siliguri", sans-serif;
            font-size: 1.05rem;
            line-height: 1.8;
            padding: 0 16px 16px 16px;
            white-space: pre-wrap;
        '>{_format_text_with_citations(text, cites)}</div>
    """, unsafe_allow_html=True)

    # Audio player (if WAV exists)
    if audio_dir:
        audio_path = audio_dir / f"turn_{turn['turn_index']:02d}_{agent}.wav"
        if audio_path.exists():
            st.audio(str(audio_path), format="audio/wav")

    # Source panel
    if cites:
        with st.expander(f"📚 সূত্রসমূহ ({len(cites)} টি)", expanded=False):
            for c in cites:
                title = c.get("source_title", "")
                url = c.get("source_url", "")
                lang = c.get("lang", "")
                period = c.get("period_label", "")
                excerpt = c.get("excerpt", "")

                st.markdown(
                    f"""
                    <div style='
                        background: #f7f7f7;
                        border-left: 3px solid #888;
                        padding: 8px 12px;
                        margin: 6px 0;
                        font-family: "Noto Sans Bengali", sans-serif;
                    '>
                        <div style='font-weight: bold; color: #333;'>
                            [{c['ref_index']}] {title}
                            <span style='font-weight: normal; color: #666;'>
                                · {lang.upper()} · {period}
                            </span>
                        </div>
                        <div style='font-size: 0.85rem; color: #555; margin: 4px 0;'>
                            <a href='{url}' target='_blank'>{url}</a>
                        </div>
                        <div style='font-size: 0.9rem; color: #444; line-height: 1.5;'>
                            {excerpt[:300]}{'…' if len(excerpt) > 300 else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("এই বাক্যে কোনো উদ্ধৃতি নেই।")


def _format_text_with_citations(text: str, cites: list[dict]) -> str:
    """HTML-escape and convert [n] markers to superscript badges."""
    import html
    text = html.escape(text)
    # Convert [n] to a styled superscript
    import re
    def repl(m):
        n = int(m.group(1))
        return f"<sup style='color: #1976D2; font-weight: bold;'>[{n}]</sup>"
    return re.sub(r"\[(\d+)\]", repl, text)


# ---- Main Streamlit app ----

def main():
    st.set_page_config(
        page_title="Bangladesh Political Debater",
        page_icon="🇧🇩",
        layout="wide",
    )

    st.title("🇧🇩 Bangladesh Political Debater")
    st.markdown("*Bangla RAG-grounded multi-agent debate (1971 – today)*")
    st.divider()

    # Sidebar: corpus stats
    with st.sidebar:
        st.header("📚 Knowledge Base")
        stats = corpus_stats()
        if stats:
            st.metric("Articles", stats.get("total_articles", 0))
            st.metric("Chunks", stats.get("total_chunks", 0))
            st.metric("Bangla chunks", stats.get("by_lang", {}).get("bn", 0))
            st.metric("English chunks", stats.get("by_lang", {}).get("en", 0))
            st.metric("Avg chunk chars", int(stats.get("avg_chunk_chars", 0)))

            with st.expander("Coverage by period"):
                for k, v in sorted(stats.get("by_period", {}).items()):
                    st.text(f"{k}: {v}")

        st.divider()
        st.header("⚙️ Settings")
        try:
            from app.agents.llm import is_groq_available, _ollama_available
            import os
            provider_pref = os.environ.get("LLM_PROVIDER", "").lower()
            ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
            if provider_pref == "ollama" or (provider_pref == "" and _ollama_available(ollama_model)):
                st.markdown(f"**LLM**: 🏠 Ollama local (`{ollama_model}`)")
                st.caption("✓ 100% offline — no network calls")
            elif is_groq_available():
                st.markdown("**LLM**: ☁ Groq GPT-OSS-120B")
                st.caption("Best Bangla + reasoning (external API)")
            else:
                st.markdown("**LLM**: ☁ z-ai GLM-4-Plus (dev)")
                st.caption("Set `LLM_PROVIDER=ollama` for 100% local runtime")
        except Exception as e:
            st.error(f"LLM check failed: {e}")

        st.divider()
        st.header("🗂 Saved debates")
        for p in list_existing_debates():
            st.text(p.name)
            if st.button(f"Open {p.name}", key=f"open_{p.name}"):
                st.session_state["view_debate_path"] = str(p)

    # Main area: topic input + debate starter OR viewer
    if "view_debate_path" in st.session_state:
        _show_saved_debate(st.session_state["view_debate_path"])
        if st.button("← Back to start"):
            del st.session_state["view_debate_path"]
            st.rerun()
        return

    if "stage_debate_path" in st.session_state:
        _show_live_stage(st.session_state["stage_debate_path"])
        return

    tab_start, tab_open = st.tabs(["Start new debate", "Open saved"])

    with tab_start:
        st.subheader("Pick a topic")

        presets = [
            "১৯৭৫ সালের ১৫ আগস্ট অভ্যুত্থান কি বাংলাদেশের রাজনীতিতে একটি মোড় ঘুরিয়ে দিয়েছিল?",
            "১৯৭১ সালের মুক্তিযুদ্ধ কি একটি জাতিগত নিপীড়নের বিরুদ্ধে যুদ্ধ ছিল, নাকি ভূ-রাজনৈতিক দ্বন্দ্ব?",
            "একাত্তরের সরকার (1/11) কি দেশের জন্য কল্যাণকর ছিল নাকি ক্ষতিকর?",
            "আন্তর্জাতিক অপরাধ ট্রাইব্যুনাল কি ন্যায়বিচার প্রতিষ্ঠা করেছে, নাকি রাজনৈতিক প্রতিহিংসা?",
            "২০২৪ সালের কোটা আন্দোলন কি গণঅভ্যুত্থান ছিল, নাকি ষড়যন্ত্র?",
            "বঙ্গবন্ধু শেখ মুজিবুর রহমানের বাকশাল ব্যবস্থা কি স্বৈরতন্ত্র ছিল, নাকি জাতীয় ঐক্যের প্রয়োজন ছিল?",
        ]

        choice = st.selectbox("Preset topics (Bangla)", ["(free-form)"] + presets)
        if choice == "(free-form)":
            topic = st.text_area(
                "Type your own topic (Bangla or English):",
                value="",
                height=100,
            )
        else:
            topic = choice
            st.text_area("Selected topic", value=topic, height=100, disabled=True)

        col1, col2 = st.columns(2)
        with col1:
            run_debate = st.button("▶ Start debate", type="primary", disabled=not topic.strip())
        with col2:
            synth_audio = st.checkbox("Also synthesise Bangla audio per turn", value=True)

        if run_debate and topic.strip():
            _run_debate(topic.strip(), synth_audio)

    with tab_open:
        st.subheader("Saved debates")
        debates = list_existing_debates()
        if not debates:
            st.info("No saved debates yet.")
        for p in debates:
            col_a, col_b = st.columns([4, 1])
            col_a.text(p.name)
            if col_b.button("Open", key=f"btn_{p.name}"):
                st.session_state["view_debate_path"] = str(p)
                st.rerun()


def _run_debate(topic: str, synth: bool) -> None:
    """Trigger the full debate pipeline."""
    from app.agents.debate import DebateEngine

    with st.status("Running debate…", expanded=True) as status:
        st.write("Initialising RAG retriever + LLM…")
        engine = DebateEngine()

        # We'll render turns live as they complete
        # Hack: monkey-patch the engine's `print` to capture live updates
        save_path = BASE_DIR / "download" / f"debate_{int(time.time())}.json"
        st.write(f"Topic: {topic}")
        st.write(f"Save path: {save_path.name}")

        # Use a progress placeholder per turn
        turn_placeholders = [st.empty() for _ in range(4)]
        # Override DebateEngine.run_debate to render turns incrementally
        # Simpler: just run the whole thing, then render.
        try:
            debate = engine.run_debate(topic, save_path=save_path)
        except Exception as e:
            st.error(f"Debate failed: {e}")
            return

        status.update(label="Debate complete!", state="complete")

    # ─── Watch as live animation ───
    st.divider()
    st.markdown("### 🎬 Watch as live animation")
    st.caption("Open the 2-character debate stage: two cartoon avatars, audio-synced mouth movement, speech bubbles with citations.")
    if st.button("▶ Open live debate stage", type="primary", key="open_stage_after_debate"):
        st.session_state["stage_debate_path"] = str(save_path)
        st.rerun()

    # Render all turns
    st.divider()
    st.subheader("Debate transcript")

    audio_dir = None
    if synth:
        with st.status("Synthesising Bangla audio…"):
            from app.tts.synth import synth_debate_audio
            audio_dir = Path(save_path.parent / (save_path.stem + "_audio"))
            try:
                synth_debate_audio(save_path, audio_dir, provider="mms")
            except Exception as e:
                st.warning(f"TTS failed: {e}")
                audio_dir = None

    for turn in debate.turns:
        render_turn(
            turn={
                "turn_index": turn.turn_index,
                "phase": turn.phase,
                "agent": turn.agent,
                "agent_name": turn.agent_name,
                "text": turn.text,
                "citations": [c.__dict__ if hasattr(c, '__dict__') else c for c in turn.citations],
                "elapsed_seconds": turn.elapsed_seconds,
            },
            audio_dir=audio_dir,
        )

    # Download buttons
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        with open(save_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📄 Download transcript JSON",
                data=f.read(),
                file_name=save_path.name,
                mime="application/json",
            )
    with col_b:
        if audio_dir and audio_dir.exists():
            # Zip up the audio dir
            import zipfile
            import io
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for wav in audio_dir.glob("*.wav"):
                    zf.write(wav, arcname=wav.name)
                manifest = audio_dir / "manifest.json"
                if manifest.exists():
                    zf.write(manifest, arcname="manifest.json")
            st.download_button(
                label="🔊 Download all audio (zip)",
                data=zip_buffer.getvalue(),
                file_name=f"{audio_dir.name}.zip",
                mime="application/zip",
            )


def _show_saved_debate(path_str: str) -> None:
    p = Path(path_str)
    if not p.exists():
        st.error(f"File not found: {p}")
        return
    with open(p, "r", encoding="utf-8") as f:
        debate = json.load(f)

    st.title(f"🇧🇩 Saved debate")
    st.write(f"**Topic**: {debate.get('topic','')}")
    st.write(f"**Created**: {debate.get('created_at','')}")

    # Look for sibling audio dir
    audio_dir = p.parent / (p.stem + "_audio")
    if not audio_dir.exists():
        audio_dir = None

    # ─── Live animation button ───
    st.divider()
    st.markdown("### 🎬 Live debate animation")
    st.caption("Two cartoon characters seated across a stage, with audio-synced mouth animation and speech bubbles. Auto-advances through all 4 turns.")
    col_a, col_b = st.columns([1, 4])
    if col_a.button("▶ Open live stage", type="primary"):
        st.session_state["stage_debate_path"] = str(p)
        st.rerun()
    if col_b.button("▶ Generate audio first (if missing)"):
        from app.tts.synth import synth_debate_audio
        with st.status("Synthesising Bangla audio…"):
            try:
                synth_debate_audio(p, audio_dir or (p.parent / (p.stem + "_audio")), provider="mms")
                st.success("Audio synthesised!")
            except Exception as e:
                st.error(f"TTS failed: {e}")
        st.rerun()

    st.divider()
    st.markdown("### 📄 Turn-by-turn transcript")
    for turn in debate.get("turns", []):
        render_turn(turn, audio_dir=audio_dir)


def _show_live_stage(path_str: str) -> None:
    """Full-screen live 2-character debate stage."""
    from app.debate_stage import render_debate_stage
    p = Path(path_str)
    audio_dir = p.parent / (p.stem + "_audio")
    render_debate_stage(p, audio_dir)
    st.divider()
    if st.button("← Back to transcript"):
        del st.session_state["stage_debate_path"]
        st.rerun()


if __name__ == "__main__":
    main()
