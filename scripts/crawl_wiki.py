"""
Bangladesh Political Knowledge Base — Wikipedia Crawler (v2, Action-API-only)
=============================================================================

Phase 1, Step 1 of the BD Political Debater pipeline.

Per Wikipedia's User-Agent policy and rate-limit policy, this crawler:
- Uses a single Action API call per article (combines extracts + info + links
  + categories + revisions into one request) — minimum request footprint.
- Sends a properly-formatted User-Agent header.
- Sleeps 2s between requests with jitter, retries on 429 with exponential
  backoff.
- Stores raw fetched data per-article under:
      /home/z/my-project/data/raw/<period_key>/<lang>/<article_slug>.json

Usage:
    python /home/z/my-project/scripts/crawl_wiki.py
    python /home/z/my-project/scripts/crawl_wiki.py --periods liberation_1971
    python /home/z/my-project/scripts/crawl_wiki.py --langs bn --periods quota_2024_now
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import random
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import requests
import yaml
from tqdm import tqdm


BASE_DIR = Path("/home/z/my-project")
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

WIKI_API = "https://{lang}.wikipedia.org/w/api.php"


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r"\s+", "_", title.strip())
    title = re.sub(r"[^\w\u0800-\u0FFF\u2000-\u2BFF\u3000-\u303F.-]", "", title)
    return title[:120] or "untitled"


@dataclass
class WikiArticle:
    title: str
    lang: str
    period_key: str
    pageid: int | None = None
    revision_id: int | None = None
    last_modified: str | None = None
    plain_text: str = ""
    extract_intro: str = ""
    categories: list[str] = field(default_factory=list)
    outgoing_links: list[str] = field(default_factory=list)
    canonical_url: str = ""
    fetch_errors: list[str] = field(default_factory=list)


class WikiSession:
    """Wikipedia Action-API client with polite UA + retry + backoff."""

    def __init__(self, lang: str, user_agent: str, delay: float = 2.0):
        self.lang = lang
        self.base_url = WIKI_API.format(lang=lang)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay

    def _sleep(self):
        time.sleep(self.delay + random.random() * 0.5)

    def _get(self, params: dict) -> dict:
        last_exc = None
        for attempt in range(5):
            try:
                r = self.session.get(self.base_url, params=params, timeout=60)
                if r.status_code == 429 or r.status_code == 503:
                    wait = 2 ** attempt * 5
                    print(f"    [rate-limit] {r.status_code}, backing off {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/5] {type(e).__name__}: {e} -- wait {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"Wikipedia API failed after 5 retries: {last_exc}")

    def fetch_article(self, title: str) -> dict:
        """Single Action-API call to fetch everything we need for one article."""
        params = {
            "action": "query",
            "prop": "extracts|info|categories|links|revisions",
            "titles": title,
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "cllimit": 100,
            "pllimit": 200,
            "rvprop": "ids|timestamp",
            "rvlimit": 1,
            "format": "json",
            "redirects": 1,
        }
        data = self._get(params)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return {}
        page = next(iter(pages.values()))
        if "missing" in page:
            # Fallback: search Wikipedia for the closest matching title
            return self._search_fallback(title)
        return page

    def _search_fallback(self, title: str) -> dict:
        """When exact-title fetch returns missing, do a wiki search and
        try to fetch the top hit."""
        print(f"    [search] trying fallback search for: {title}")
        params = {
            "action": "query",
            "list": "search",
            "srsearch": title,
            "srlimit": 3,
            "format": "json",
        }
        data = self._get(params)
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return {"missing": True, "original_title": title}
        best = search_results[0]["title"]
        print(f"    [search] found: {best}")
        return self.fetch_article(best)


def parse_article_from_api(page: dict, title: str, lang: str, period_key: str) -> WikiArticle:
    art = WikiArticle(title=title, lang=lang, period_key=period_key)
    art.pageid = page.get("pageid")
    art.canonical_url = page.get("fullurl", "")
    if "revisions" in page:
        rev = page["revisions"][0]
        art.revision_id = rev.get("revid")
        art.last_modified = rev.get("timestamp")
    if "extract" in page:
        full = page["extract"]
        # First paragraph as intro
        intro = full.split("\n\n", 1)[0] if "\n\n" in full else full[:500]
        art.plain_text = full
        art.extract_intro = intro.strip()
    if "categories" in page:
        art.categories = [c["title"].replace("Category:", "") for c in page["categories"]]
    if "links" in page:
        seen = set()
        for link in page["links"]:
            t = link["title"]
            if t not in seen and ":" not in t:
                seen.add(t)
                art.outgoing_links.append(t)
    return art


def save_article(art: WikiArticle, raw_dir: Path) -> Path:
    out_dir = raw_dir / art.period_key / art.lang
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(art.title)
    out_path = out_dir / f"{slug}.json"
    out_path.write_text(
        json.dumps(asdict(art), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


def run_crawler(periods_filter: list[str] | None, langs_filter: list[str] | None) -> None:
    cfg = load_config()
    raw_dir = BASE_DIR / cfg["project"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Properly-formatted Wikipedia User-Agent per https://meta.wikimedia.org/wiki/User-Agent_policy
    # Format: name/version (URL; contact) library/version
    user_agent = (
        "BangladeshPoliticalDebater/0.1 "
        "(https://github.com/local/bd-political-debater; researcher@local) "
        "Python-requests/" + requests.__version__
    )

    delay = 2.0  # hard floor — Wikipedia requires polite rate
    periods = cfg["corpus"]["event_periods"]
    selected_periods = periods_filter or list(periods.keys())
    selected_langs = langs_filter or cfg["corpus"]["wikipedia_languages"]

    total_articles = 0
    skipped = 0
    failed = 0
    sessions = {lang: WikiSession(lang, user_agent, delay) for lang in selected_langs}

    for pkey in selected_periods:
        if pkey not in periods:
            print(f"[WARN] Unknown period: {pkey}")
            continue
        period = periods[pkey]
        print(f"\n=== Period: {period['label']} ({pkey}) ===")

        for lang in selected_langs:
            seeds_key = f"seed_articles_{lang}"
            seeds = period.get(seeds_key, [])
            if not seeds:
                print(f"  [{lang}] no seed articles for this period")
                continue

            print(f"  [{lang}] {len(seeds)} seed articles")
            for title in tqdm(seeds, desc=f"{pkey} [{lang}]", leave=False):
                slug = slugify(title)
                out_path = raw_dir / pkey / lang / f"{slug}.json"
                if out_path.exists():
                    skipped += 1
                    continue
                try:
                    page = sessions[lang].fetch_article(title)
                    if not page:
                        failed += 1
                        print(f"  [MISS] {title} (no page returned)")
                        continue
                    if page.get("missing"):
                        failed += 1
                        print(f"  [MISS] {title} (article does not exist on {lang}.wikipedia.org)")
                        continue
                    art = parse_article_from_api(page, title, lang, pkey)
                    save_article(art, raw_dir)
                    total_articles += 1
                    # If the article had errors during fetch, report
                    if art.fetch_errors:
                        print(f"  [WARN] {title}: {art.fetch_errors}")
                except Exception as e:
                    failed += 1
                    print(f"  [EXC] {title}: {type(e).__name__}: {e}")
                sessions[lang]._sleep()

    print(f"\n=== Done ===")
    print(f"  Fetched (new): {total_articles}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output: {raw_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", nargs="*", help="Subset of period keys (default: all)")
    ap.add_argument("--langs", nargs="*", choices=["bn", "en"], help="Subset of languages (default: both)")
    args = ap.parse_args()
    run_crawler(args.periods, args.langs)


if __name__ == "__main__":
    main()
