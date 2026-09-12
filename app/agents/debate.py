"""
Debate Engine — RAG-grounded multi-turn debate between Analyst and Journalist.
============================================================================

Phase 3 of the BD Political Debater pipeline.

Structure (per config.yaml `debate.structure`):
  1. opening           — Agent A opens with thesis
  2. rebuttal          — Agent B refutes + counter-thesis
  3. counter_rebuttal  — Agent A responds to B's rebuttal + reinforces thesis
  4. closing           — Agent B closes with summary + final position

Each turn:
- Runs a RAG retrieve against the corpus for the topic + recent opponent turn
- Injects retrieved chunks as grounded evidence into the LLM prompt
- Produces Bangla output with explicit source citations (numbered [1], [2], …)
- Records the cited chunks per turn for UI source panel display

Debate feels "natural" because:
- The opponent's previous turn text is in the prompt context
- Agents are instructed to reference specific opponent claims by quote
- Persona prompts instruct for turn-appropriate moves (concede, refute, etc.)
- Source citations are mandatory and must match retrieved chunks
"""
from __future__ import annotations

import json
import re
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


# ============================================================
# Data classes
# ============================================================

@dataclass
class CitationRef:
    """A reference to a retrieved chunk — embedded in a turn's output."""
    ref_index: int  # the [1], [2] label in the text
    chunk_id: str
    source_title: str
    source_url: str
    lang: str
    period_label: str
    excerpt: str  # the snippet of text the agent drew from


@dataclass
class DebateTurn:
    turn_index: int
    phase: str  # opening / rebuttal / counter_rebuttal / closing
    agent: str  # "analyst" or "journalist"
    agent_name: str
    text: str
    citations: list[CitationRef] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    raw_retrieval_count: int = 0


@dataclass
class Debate:
    topic: str
    turns: list[DebateTurn] = field(default_factory=list)
    created_at: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Persona + phase prompts (Bangla)
# ============================================================

# Turn-level instruction: how each agent should approach this phase.
# These are deliberately rich — the user wants a *natural* debate, not a
# canned pro/con template.

PHASE_INSTRUCTIONS_BN = {
    "analyst": {
        "opening": (
            "এটি আপনার উদ্বোধন বাক্য। "
            "একটি স্পষ্ট মূল যুক্তি (thesis) দাও — সরকার বা প্রাতিষ্ঠানিক পক্ষের সমর্থনে। "
            "তিনটি প্রধান বিন্দু দাও, প্রতিটিতে সূত্র-উদ্ঘৃতি [1] [2] ইত্যাদি ব্যবহার করো। "
            "আবেগের চেয়ে তথ্য ও প্রমাণকে অগ্রাধিকার দাও।"
        ),
        "rebuttal": (
            "প্রতিপক্ষের উদ্বোধন যুক্তি উদ্ঘৃত করে প্রত্যাখ্যান করো। "
            "তাদের দুর্বল স্থান চিহ্নিত করো — উৎসের অভাব, প্রসঙ্গ-বিচ্যুতি, বা অতি-সাধারণীকরণ। "
            "তারপর আপনার নিজের যুক্তিকে শক্তিশালী করে তোলো নতুন সূত্র দিয়ে।"
        ),
        "counter_rebuttal": (
            "প্রতিপক্ষের প্রতিবাদের উত্তর দাও। "
            "যেখানে সত্য সেখানে স্বীকার করো (concession) — এটি সততা প্রদর্শন করে। "
            "তারপর যেখানে তারা ভুল সেখানে দৃঢ়ভাবে প্রত্যাখ্যান করো, সূত্রসহ।"
        ),
        "closing": (
            "একটি শক্তিশালী সমাপ্তি দাও। "
            "মূল যুক্তি পুনর্ব্যক্ত করো — কিন্তু নতুন শব্দে। "
            "প্রতিপক্ষের সবচেয়ে দুর্বল দাবি একটি বাক্যে উড়িয়ে দাও। "
            "শেষ বাক্যটি স্মরণীয় হবে — কোনো নতুন সূত্র নয়, কেবল সমাপ্তি।"
        ),
    },
    "journalist": {
        "opening": (
            "এটি আপনার উদ্বোধন বাক্য। "
            "একটি স্পষ্ট মূল যুক্তি (thesis) দাও — সমালোচক বা প্রান্তিক পক্ষের প্রতিনিধিত্ব করে। "
            "মানবাধিকার, দুর্নীতি, ক্ষমতার অপব্যবহার বিষয়ে জোর দাও। "
            "তিনটি প্রধান বিন্দু দাও, প্রতিটিতে সূত্র-উদ্ঘৃতি [1] [2] ইত্যাদি ব্যবহার করো।"
        ),
        "rebuttal": (
            "প্রতিপক্ষের (বিশ্লেষকের) উদ্বোধন যুক্তি উদ্ঘৃত করে প্রত্যাখ্যান করো। "
            "তাদের সরকারপন্থী দৃষ্টিভঙ্গির সীমাবদ্ধতা চিহ্নিত করো। "
            "নিপীড়িত বা প্রান্তিক মানুষের অভিজ্ঞতা সামনে আনো, সূত্রসহ।"
        ),
        "counter_rebuttal": (
            "বিশ্লেষকের প্রতিবাদের উত্তর দাও। "
            "তারা যদি দাবি করে যে সরকার কিছু ভালো করেছে — সেটি অস্বীকার না করে, "
            "দেখাও যে সেটি প্রান্তিক বা সময়োপযুক্ত ছিল না। "
            "বিকল্প সূত্র দিয়ে তাদের প্রতিটি দাবিকে প্রশ্ন করো।"
        ),
        "closing": (
            "শক্তিশালী সমাপ্তি দাও। "
            "সমালোচক অবস্থান পুনর্ব্যক্ত করো — নতুন শব্দে। "
            "বিশ্লেষকের সবচেয়ে বড় অমীমাংসিত সমস্যা একটি বাক্যে তুলে ধাও। "
            "শেষ বাক্যে পাঠককে একটি চ্যালেঞ্জ বা প্রশ্ন ছাড়ো।"
        ),
    },
}


def build_turn_prompt(
    agent_key: str,
    agent_persona: str,
    phase: str,
    phase_instruction: str,
    topic: str,
    opponent_last_turn: str | None,
    retrieved_chunks: list[dict],
) -> list:
    """Build the full chat-message list for a single debate turn.

    Returns list[ChatMessage] for app.agents.llm.chat().
    """
    from app.agents.llm import ChatMessage

    # Build the evidence block with citation indices
    evidence_lines = []
    for i, c in enumerate(retrieved_chunks, 1):
        excerpt = c["text"].strip()
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        evidence_lines.append(
            f"[{i}] উৎস: {c['source_title']} ({c['lang']}, {c.get('period_label','')})\n"
            f"    URL: {c.get('source_url','')}\n"
            f"    উদ্ধৃতি: {excerpt}"
        )
    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "(কোনো সম্পর্কিত সূত্র পাওয়া যায়নি।)"

    system_prompt = (
        f"{agent_persona}\n\n"
        f"তুমি একটি রাজনৈতিক বিতর্কে অংশ নিচ্ছ। "
        f"তোমার প্রতিটি দাবি অবশ্যই নিচের সূত্রগুলির একটি বা একাধিকের উপর ভিত্তি করে হবে। "
        f"উদ্ধৃতি করার সময় [1], [2] ইত্যাদি ব্যবহার করো — যেখানে সংখ্যাটি নিচের সূত্রের ক্রমাঙ্ক। "
        f"কোনো সূত্রে না থাকা দাবি কোনোভাবেই করবে না। "
        f"বাংলা ব্যাকরণ অনুসরণ করো — কোনো ইংরেজি শব্দ বাংলা অক্ষরে লিখবে না, "
        f"সঠিক বাংলা পরিভাষা ব্যবহার করো।"
    )

    user_prompt = (
        f"বিতর্কের বিষয়: {topic}\n\n"
        f"তোমার ভূমিকা: {phase} (turn phase)\n"
        f"নির্দেশ: {phase_instruction}\n\n"
    )
    if opponent_last_turn:
        user_prompt += (
            f"প্রতিপক্ষের সর্বশেষ বক্তব্য:\n"
            f"---\n{opponent_last_turn}\n---\n\n"
            f"তোমার উত্তরে এই বক্তব্যের কোনো না কোনো অংশ সরাসরি উদ্ঘৃত করো।\n\n"
        )

    user_prompt += (
        f"সম্পর্কিত সূত্রসমূহ (retrieved):\n"
        f"===\n{evidence_block}\n===\n\n"
        f"এখন তোমার বাংলা বক্তব্য লেখো (১৫০-২৫০ শব্দে)। "
        f"মনে রাখবে: প্রতিটি প্রধান দাবিতে অন্তত একটি [n] উদ্ধৃতি থাকবে।"
    )

    return [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]


# ============================================================
# Citation extraction
# ============================================================

CITE_PATTERN = re.compile(r"\[(\d+)\]")


def extract_citations(text: str, retrieved_chunks: list[dict]) -> list[CitationRef]:
    """Find all [n] refs in the text and map them to chunks."""
    refs = []
    seen_indices = set()
    for match in CITE_PATTERN.finditer(text):
        n = int(match.group(1))
        if n in seen_indices:
            continue
        seen_indices.add(n)
        if 1 <= n <= len(retrieved_chunks):
            c = retrieved_chunks[n - 1]
            # Take a short excerpt from the chunk
            excerpt = c["text"].strip()[:300] + ("…" if len(c["text"]) > 300 else "")
            refs.append(CitationRef(
                ref_index=n,
                chunk_id=c["chunk_id"],
                source_title=c["source_title"],
                source_url=c.get("source_url", ""),
                lang=c.get("lang", ""),
                period_label=c.get("period_label", ""),
                excerpt=excerpt,
            ))
    return refs


# ============================================================
# Debate runner
# ============================================================

class DebateEngine:
    def __init__(self):
        cfg = load_config()
        self.cfg = cfg
        from app.rag.embeddings import HybridRetriever
        self.retriever = HybridRetriever()
        self.agents = cfg["debate"]
        self.structure = cfg["debate"]["structure"]
        self.top_k = cfg["rag"]["rerank_top_k"]  # use rerank count for turn evidence

    def _retrieve_for_turn(self, topic: str, opponent_last_turn: str | None) -> list[dict]:
        """Retrieve chunks relevant to the topic + opponent's last turn."""
        # Query 1: topic alone
        results_topic = self.retriever.retrieve(topic, top_k=self.top_k)
        # Query 2: opponent's last turn (if any) — focus on what they said
        results_opp = []
        if opponent_last_turn:
            # Take first 200 chars of opponent turn as a focused query
            opp_query = opponent_last_turn[:300]
            results_opp = self.retriever.retrieve(opp_query, top_k=3)

        # Merge dedupe by chunk_id, keep topic results first
        seen = set()
        merged = []
        for c in results_topic + results_opp:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                merged.append(c)
        return merged

    def run_debate(self, topic: str, save_path: Path | None = None) -> Debate:
        """Execute the full 4-turn debate."""
        from app.agents.llm import chat

        print(f"\n=== Debate started ===")
        print(f"Topic: {topic}\n")

        # Determine turn order:
        # 1. opening       -> agent_a (analyst)
        # 2. rebuttal      -> agent_b (journalist)
        # 3. counter_rebuttal -> agent_a
        # 4. closing       -> agent_b
        turn_order = [
            ("analyst", "opening"),
            ("journalist", "rebuttal"),
            ("analyst", "counter_rebuttal"),
            ("journalist", "closing"),
        ]

        debate = Debate(
            topic=topic,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            config_snapshot={
                "llm_model": self.cfg["debate"]["llm_model"],
                "temperature": self.cfg["debate"]["temperature"],
                "embedding_model": self.cfg["rag"]["embedding_model"],
                "max_tokens_per_turn": self.cfg["debate"]["max_tokens_per_turn"],
            },
        )

        last_analyst_turn = None
        last_journalist_turn = None

        for turn_idx, (agent_key, phase) in enumerate(turn_order):
            # Opponent last turn = whichever agent's prior output is most recent
            if agent_key == "analyst":
                opponent_last = last_journalist_turn
            else:
                opponent_last = last_analyst_turn

            # RAG retrieve
            retrieved = self._retrieve_for_turn(topic, opponent_last)
            print(f"[turn {turn_idx+1}] {agent_key} / {phase}: retrieved {len(retrieved)} chunks")

            agent_cfg = self.cfg["debate"][f"agent_{agent_key[-1]}"] if f"agent_{agent_key[-1]}" in self.cfg["debate"] else None
            # Actually the config uses agent_a / agent_b keys
            agent_cfg_key = "agent_a" if agent_key == "analyst" else "agent_b"
            agent_cfg = self.cfg["debate"][agent_cfg_key]

            phase_instruction = PHASE_INSTRUCTIONS_BN[agent_key][phase]

            # Build prompt
            msgs = build_turn_prompt(
                agent_key=agent_key,
                agent_persona=agent_cfg["persona"],
                phase=phase,
                phase_instruction=phase_instruction,
                topic=topic,
                opponent_last_turn=opponent_last,
                retrieved_chunks=retrieved,
            )

            # Call LLM
            try:
                resp = chat(
                    msgs,
                    temperature=self.cfg["debate"]["temperature"],
                    max_tokens=self.cfg["debate"]["max_tokens_per_turn"],
                )
                text = resp.text.strip()
            except Exception as e:
                print(f"[error] LLM call failed: {e}", file=sys.stderr)
                text = f"(এই বাক্যের জন্য ত্রুটি: {e})"
                resp = None

            # Extract citations
            cites = extract_citations(text, retrieved)

            turn = DebateTurn(
                turn_index=turn_idx,
                phase=phase,
                agent=agent_key,
                agent_name=agent_cfg["name"],
                text=text,
                citations=cites,
                elapsed_seconds=resp.elapsed_seconds if resp else 0.0,
                raw_retrieval_count=len(retrieved),
            )
            debate.turns.append(turn)

            # Track for next turn
            if agent_key == "analyst":
                last_analyst_turn = text
            else:
                last_journalist_turn = text

            print(f"  → {turn.agent_name} ({len(text)} chars, {len(cites)} cites, {turn.elapsed_seconds:.1f}s)")
            print(f"  Preview: {text[:200]}…\n")

        # Save full debate transcript
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(
                json.dumps(_serialize_debate(debate), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"\nDebate transcript saved: {save_path}")

        return debate


def _serialize_debate(d: Debate) -> dict:
    return {
        "topic": d.topic,
        "created_at": d.created_at,
        "config_snapshot": d.config_snapshot,
        "turns": [
            {
                "turn_index": t.turn_index,
                "phase": t.phase,
                "agent": t.agent,
                "agent_name": t.agent_name,
                "text": t.text,
                "elapsed_seconds": t.elapsed_seconds,
                "raw_retrieval_count": t.raw_retrieval_count,
                "citations": [asdict(c) for c in t.citations],
            }
            for t in d.turns
        ],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", type=str, required=True, help="Debate topic (Bangla or English)")
    ap.add_argument("--save", type=str, default=None, help="Path to save transcript JSON")
    args = ap.parse_args()

    engine = DebateEngine()
    save_path = Path(args.save) if args.save else None
    debate = engine.run_debate(args.topic, save_path=save_path)

    print("\n=== Final Debate ===")
    print(f"Topic: {debate.topic}")
    for t in debate.turns:
        print(f"\n--- Turn {t.turn_index+1}: {t.agent_name} ({t.phase}) ---")
        print(t.text)


if __name__ == "__main__":
    main()
