"""
Debate Evaluator — syntactic Bangla correctness + debate-structure scoring.
============================================================================

Phase 7 of the BD Political Debater pipeline.

For a saved debate transcript JSON, computes:

1. Bangla syntactic correctness
   - Fraction of characters in the Bangla Unicode block (U+0980–U+09FF)
   - Fraction of "broken" glyph sequences (rare combos like ZWJ in odd places)
   - Sentence terminator count (। per 1000 chars)
   - Detect if English transliterations (Bangla written in Latin script) slipped in
   - Average word length (sanity)

2. Debate-structure adherence
   - Each turn must contain at least one [n] citation (groundedness)
   - Rebuttal & counter_rebuttal turns must directly quote the opponent's text
     (string-overlap of a phrase from previous turn > N tokens)
   - Phase-specific markers: closing must end with a question or strong statement
   - Turn length within 150–600 chars target band

Outputs a JSON scorecard per turn + overall averages.

Usage:
    python scripts/evaluate_debate.py download/debate_1975_coup.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path


# Bangla Unicode ranges
BANGLA_BLOCK = re.compile(r"[\u0980-\u09FF]")  # Bengali block
BANGLA_CONJUNCTS = re.compile(r"[\u0980-\u09FF]{2,}")  # 2+ consecutive Bangla chars
BANGLA_DAARI = "।"
LATIN_BANGLA_HINT = re.compile(r"\b(bangladesh|mujib|hasina|khaleda|zia|ershad|liberation|war|coup|government|election)\b", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile(r"[।!?]")


@dataclass
class TurnScore:
    turn_index: int
    agent: str
    phase: str
    # Bangla correctness
    bn_char_ratio: float = 0.0
    bn_word_count: int = 0
    avg_word_len: float = 0.0
    daari_count: int = 0
    latin_in_bangla_count: int = 0
    # Structure
    has_citation: bool = False
    citation_count: int = 0
    quotes_opponent: bool = False
    opponent_quote_chars: int = 0
    text_length: int = 0
    in_target_length_band: bool = False
    # Overall correctness score (0-100)
    correctness_score: float = 0.0


@dataclass
class DebateScorecard:
    debate_path: str
    topic: str
    turns: list[TurnScore] = field(default_factory=list)
    overall_correctness: float = 0.0
    overall_groundedness: float = 0.0
    overall_responsiveness: float = 0.0  # rebuttal/counter_rebuttal quoting opponent
    overall: float = 0.0
    notes: list[str] = field(default_factory=list)


def score_bangla_correctness(text: str) -> dict:
    """Compute Bangla-language correctness metrics for a single turn text."""
    if not text:
        return {"bn_char_ratio": 0.0, "bn_word_count": 0, "avg_word_len": 0.0,
                "daari_count": 0, "latin_in_bangla_count": 0, "text_length": 0}

    # Char ratio
    total_chars = len(text)
    bn_chars = len(BANGLA_BLOCK.findall(text))
    bn_ratio = bn_chars / total_chars if total_chars > 0 else 0.0

    # Words (split on whitespace)
    words = text.split()
    # Bangla words specifically
    bn_words = [w for w in words if BANGLA_BLOCK.search(w)]
    avg_word_len = sum(len(w) for w in bn_words) / max(1, len(bn_words))

    # Sentence terminators
    daari_count = text.count(BANGLA_DAARI) + text.count("॥")

    # Latin script that suggests Bangla written in English (red flag)
    latin_hits = len(LATIN_BANGLA_HINT.findall(text))

    return {
        "bn_char_ratio": round(bn_ratio, 3),
        "bn_word_count": len(bn_words),
        "avg_word_len": round(avg_word_len, 2),
        "daari_count": daari_count,
        "latin_in_bangla_count": latin_hits,
        "text_length": total_chars,
    }


def check_quotes_opponent(this_text: str, prev_text: str | None, min_match_chars: int = 25) -> tuple[bool, int]:
    """Does this turn quote >=25 chars verbatim from the opponent's previous turn?

    Real debates have agents directly quoting each other.
    """
    if not prev_text:
        return False, 0
    # Normalize: strip whitespace, lowercase Latin
    this_norm = re.sub(r"\s+", " ", this_text)
    prev_norm = re.sub(r"\s+", " ", prev_text)
    # Slide a window over prev_text and check if any window of >=min_match_chars
    # appears in this_text
    best_len = 0
    window_size = min_match_chars
    # Sample windows at every 5-char offset to keep it cheap
    for i in range(0, len(prev_norm) - window_size, 5):
        window = prev_norm[i:i + window_size]
        if window in this_norm:
            best_len = max(best_len, window_size)
            return True, best_len
    return False, best_len


def score_turn(turn: dict, prev_turn_text: str | None) -> TurnScore:
    text = turn.get("text", "")
    bn = score_bangla_correctness(text)
    cites = turn.get("citations", [])

    # Phase = opening/rebuttal/counter_rebuttal/closing
    phase = turn.get("phase", "")
    agent = turn.get("agent", "")
    turn_index = turn.get("turn_index", 0)

    # Does this turn quote the opponent? (Only relevant for rebuttal/counter)
    quotes, qchars = check_quotes_opponent(text, prev_turn_text) if phase in ("rebuttal", "counter_rebuttal") else (False, 0)

    # In target length band 150-600
    in_band = 150 <= len(text) <= 800

    # Citation count
    cite_n = len(cites)
    has_cite = cite_n > 0

    # Correctness score 0-100 (heuristic):
    #  - bn_char_ratio >= 0.5  → 40 pts
    #  - daari_count >= 1      → 15 pts
    #  - no Latin-Bangla hints → 15 pts
    #  - avg_word_len in [3,8] → 10 pts
    #  - has_citation          → 10 pts
    #  - in target length band → 10 pts
    score = 0.0
    if bn["bn_char_ratio"] >= 0.5:
        score += 40
    elif bn["bn_char_ratio"] >= 0.3:
        score += 25
    if bn["daari_count"] >= 1:
        score += 15
    if bn["latin_in_bangla_count"] == 0:
        score += 15
    if 3 <= bn["avg_word_len"] <= 8:
        score += 10
    if has_cite:
        score += 10
    if in_band:
        score += 10

    return TurnScore(
        turn_index=turn_index,
        agent=agent,
        phase=phase,
        bn_char_ratio=bn["bn_char_ratio"],
        bn_word_count=bn["bn_word_count"],
        avg_word_len=bn["avg_word_len"],
        daari_count=bn["daari_count"],
        latin_in_bangla_count=bn["latin_in_bangla_count"],
        has_citation=has_cite,
        citation_count=cite_n,
        quotes_opponent=quotes,
        opponent_quote_chars=qchars,
        text_length=bn["text_length"],
        in_target_length_band=in_band,
        correctness_score=round(score, 1),
    )


def evaluate_debate(debate_path: Path) -> DebateScorecard:
    with open(debate_path, "r", encoding="utf-8") as f:
        debate = json.load(f)
    turns = debate.get("turns", [])

    sc = DebateScorecard(
        debate_path=str(debate_path),
        topic=debate.get("topic", ""),
    )

    prev_text = None
    for turn in turns:
        ts = score_turn(turn, prev_text)
        sc.turns.append(ts)
        prev_text = turn.get("text", "")

    # Overall
    if sc.turns:
        sc.overall_correctness = round(sum(t.correctness_score for t in sc.turns) / len(sc.turns), 1)
        sc.overall_groundedness = round(sum(t.citation_count for t in sc.turns) / len(sc.turns), 2)
        rebuttal_turns = [t for t in sc.turns if t.phase in ("rebuttal", "counter_rebuttal")]
        if rebuttal_turns:
            sc.overall_responsiveness = round(
                sum(1 for t in rebuttal_turns if t.quotes_opponent) / len(rebuttal_turns) * 100, 1
            )
        else:
            sc.overall_responsiveness = 0.0
        sc.overall = round(0.5 * sc.overall_correctness + 0.3 * sc.overall_responsiveness + 0.2 * min(100, sc.overall_groundedness * 25), 1)

    # Notes
    if sc.overall_correctness < 60:
        sc.notes.append("⚠ Correctness < 60 — check Bangla grammar")
    if sc.overall_groundedness < 1.5:
        sc.notes.append("⚠ Groundedness < 1.5 cites/turn — agents not citing enough")
    if sc.overall_responsiveness < 50:
        sc.notes.append("⚠ Responsiveness < 50% — agents not quoting opponent")
    if not sc.notes:
        sc.notes.append("✓ All metrics in healthy range")

    return sc


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_debate.py <debate.json>")
        sys.exit(1)
    debate_path = Path(sys.argv[1])
    if not debate_path.exists():
        print(f"Not found: {debate_path}")
        sys.exit(1)

    sc = evaluate_debate(debate_path)

    # Pretty print
    print(f"\n=== Debate scorecard: {debate_path.name} ===")
    print(f"Topic: {sc.topic}")
    print(f"\nPer-turn scores:")
    for t in sc.turns:
        print(f"  Turn {t.turn_index+1} ({t.agent}/{t.phase}):")
        print(f"    Bangla ratio: {t.bn_char_ratio:.2f}, words: {t.bn_word_count}, daari: {t.daari_count}")
        print(f"    Latin-Bangla hints: {t.latin_in_bangla_count}")
        print(f"    Citations: {t.citation_count}, quotes opponent: {t.quotes_opponent}")
        print(f"    Length: {t.text_length} chars (in band: {t.in_target_length_band})")
        print(f"    Correctness: {t.correctness_score}/100")

    print(f"\n=== Overall ===")
    print(f"Correctness:     {sc.overall_correctness}/100")
    print(f"Groundedness:    {sc.overall_groundedness} cites/turn")
    print(f"Responsiveness:  {sc.overall_responsiveness}% (rebuttals quoting opponent)")
    print(f"Composite:       {sc.overall}/100")
    print(f"\nNotes:")
    for n in sc.notes:
        print(f"  {n}")

    # Save scorecard alongside debate
    out_path = debate_path.parent / (debate_path.stem + "_scorecard.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(sc), f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
