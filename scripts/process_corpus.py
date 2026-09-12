"""
Corpus Processor — turns raw Wikipedia JSON dumps into RAG-ready JSONL chunks.

Phase 1, Step 4 of the BD Political Debater pipeline.

Reads:  /home/z/my-project/data/raw/<period>/<lang>/<slug>.json   (one per article)
Writes: /home/z/my-project/data/processed/chunks.jsonl           (one chunk per line)
        /home/z/my-project/data/processed/corpus_stats.json      (statistics)
        /home/z/my-project/data/processed/manifest.json          (chunk → source map)

Chunking strategy:
- Split plain_text on `\n\n` boundaries (paragraphs in the Wikipedia extract).
- Group consecutive short paragraphs (<=120 chars) so chunks meet target size.
- Each chunk: {chunk_id, period_key, lang, source_title, source_url,
   year_range, position_in_article, text, categories}
- De-duplicate identical chunks across the corpus (same hash → kept once, with merged sources).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

BASE_DIR = Path("/home/z/my-project")
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_articles(raw_dir: Path) -> list[dict]:
    """Walk the raw corpus dir, return list of article dicts."""
    articles = []
    for p in sorted(raw_dir.rglob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                art = json.load(f)
            articles.append(art)
        except Exception as e:
            print(f"[ERR] failed to load {p}: {e}", file=sys.stderr)
    return articles


def make_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[tuple[str, int]]:
    """Split a long article into chunks.

    Returns list of (chunk_text, position_index).
    Uses paragraph boundaries where possible; otherwise falls back to
    char-based sliding window.
    """
    if not text or not text.strip():
        return []

    # Try paragraph-based first
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[tuple[str, int]] = []
    current = ""
    pos = 0
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append((current, pos))
                pos += 1
                # Use overlap from tail of current
                if chunk_overlap > 0 and len(current) > chunk_overlap:
                    tail = current[-chunk_overlap:]
                    current = tail + "\n\n" + para
                else:
                    current = para
            else:
                # Single paragraph > chunk_size; char-window it
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunk = para[i : i + chunk_size]
                    chunks.append((chunk, pos))
                    pos += 1
                    if i + chunk_size >= len(para):
                        break
                current = ""
    if current:
        chunks.append((current, pos))
    return chunks


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def build_corpus(raw_dir: Path, processed_dir: Path, cfg: dict) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    articles = load_articles(raw_dir)
    print(f"Loaded {len(articles)} articles from {raw_dir}")

    chunk_size = cfg["rag"]["chunk_size"]
    chunk_overlap = cfg["rag"]["chunk_overlap"]
    periods_cfg = cfg["corpus"]["event_periods"]

    out_chunks_path = processed_dir / "chunks.jsonl"
    out_manifest_path = processed_dir / "manifest.json"
    out_stats_path = processed_dir / "corpus_stats.json"

    seen_hashes: dict[str, dict] = {}  # dedup
    all_chunks: list[dict] = []
    manifest: list[dict] = []

    for art in tqdm(articles, desc="chunking"):
        pkey = art.get("period_key", "unknown")
        lang = art.get("lang", "unknown")
        title = art.get("title", "Untitled")
        url = art.get("canonical_url", "")
        period_meta = periods_cfg.get(pkey, {})
        year_range = period_meta.get("year_range", [None, None])
        period_label = period_meta.get("label", pkey)
        categories = art.get("categories", [])

        if not art.get("plain_text"):
            continue

        chunks = make_chunks(art["plain_text"], chunk_size, chunk_overlap)
        for chunk_text, position in chunks:
            if len(chunk_text) < 50:
                continue
            h = hash_text(chunk_text)
            chunk_id = f"{pkey}_{lang}_{art.get('pageid', 'np')}_{position:04d}_{h}"
            if h in seen_hashes:
                # Already seen this exact text; merge source refs
                seen_hashes[h]["source_titles"].add(title)
                continue
            chunk_record = {
                "chunk_id": chunk_id,
                "hash": h,
                "period_key": pkey,
                "period_label": period_label,
                "year_range": year_range,
                "lang": lang,
                "source_title": title,
                "source_url": url,
                "position_in_article": position,
                "categories": categories[:20],  # cap
                "text": chunk_text,
            }
            seen_hashes[h] = {
                "record": chunk_record,
                "source_titles": {title},
            }
            all_chunks.append(chunk_record)

    # Write chunks.jsonl
    with open(out_chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Build manifest
    for c in all_chunks:
        manifest.append({
            "chunk_id": c["chunk_id"],
            "hash": c["hash"],
            "source_title": c["source_title"],
            "source_url": c["source_url"],
            "period_key": c["period_key"],
            "lang": c["lang"],
            "position": c["position_in_article"],
        })

    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Stats
    stats = {
        "total_articles": len(articles),
        "total_chunks": len(all_chunks),
        "unique_chunks": len(seen_hashes),
        "by_period": {},
        "by_lang": {"bn": 0, "en": 0},
        "by_period_lang": {},
        "avg_chunk_chars": sum(len(c["text"]) for c in all_chunks) / max(1, len(all_chunks)),
    }
    for c in all_chunks:
        stats["by_period"][c["period_key"]] = stats["by_period"].get(c["period_key"], 0) + 1
        stats["by_lang"][c["lang"]] = stats["by_lang"].get(c["lang"], 0) + 1
        key = f"{c['period_key']}_{c['lang']}"
        stats["by_period_lang"][key] = stats["by_period_lang"].get(key, 0) + 1

    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Corpus stats ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote: {out_chunks_path}")
    print(f"Wrote: {out_manifest_path}")
    print(f"Wrote: {out_stats_path}")


def main():
    cfg = load_config()
    raw_dir = BASE_DIR / cfg["project"]["raw_dir"]
    processed_dir = BASE_DIR / cfg["project"]["processed_dir"]
    build_corpus(raw_dir, processed_dir, cfg)


if __name__ == "__main__":
    main()
