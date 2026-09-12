"""
Zenodo Snapshot Uploader
========================

Phase 1, Step 5 of the BD Political Debater pipeline.

Packages the corpus (raw + processed chunks) into a single tar.gz snapshot
and uploads it to Zenodo as a new deposit. Requires a Zenodo personal
access token (create one at https://zenodo.org/account/settings/applications/tokens/new/).

Usage:
    # 1. Set your Zenodo token
    export ZENODO_TOKEN="your-token-here"

    # 2. (Optional) edit metadata in this file or in config/config.yaml
    # 3. Run
    python /home/z/my-project/scripts/zenodo_upload.py

The script will:
- Create a new deposit on Zenodo (draft state)
- Upload the tarball as a single file
- Publish the deposit (toggle this off with --no-publish)
- Print the DOI + HTML URL of the published dataset

You can re-run this any time to publish a new snapshot.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml

BASE_DIR = Path("/home/z/my-project")
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

# Use the SANDBOX for testing, PROD for real uploads.
ZENODO_API = {
    "sandbox": "https://sandbox.zenodo.org/api/deposit/depositions",
    "prod": "https://zenodo.org/api/deposit/depositions",
}


@dataclass
class SnapshotMetadata:
    title: str
    creators: list[dict]
    description: str
    keywords: list[str]
    license: str = "CC-BY-4.0"
    upload_type: str = "dataset"
    publication_type: str = ""
    version: str = "0.1.0"


def default_metadata() -> SnapshotMetadata:
    return SnapshotMetadata(
        title="Bangladesh Political History Corpus (1971–2026) — Wikipedia Snapshot for RAG",
        creators=[
            {
                "name": "Local, Researcher",
                "affiliation": "Independent",
                "orcid": "",
            }
        ],
        description=(
            "A bilingual (Bangla + English) text corpus of Wikipedia articles covering "
            "Bangladesh's political history from the 1971 Liberation War through the "
            "2024 Quota Movement and the interim Yunus government. \n\n"
            "Topics span: Liberation War (1971), 1975 coups, Zia era (1977–81), "
            "Ershad era (1982–90), 1990s democracy, 2000s + 1/11/2007, AL rule (2010s), "
            "2024 Quota Movement and aftermath. \n\n"
            "Each article is stored as JSON with plain text, extract, categories, "
            "outgoing links, revision ID, and last-modified timestamp. "
            "Processed chunks (500-char, 80-overlap, deduplicated) are also included "
            "as a single JSONL file ready for RAG ingestion. \n\n"
            "Source: Wikipedia (Bangla + English editions), fetched politely via the "
            "Action API with a proper User-Agent per Wikimedia UA policy."
        ),
        keywords=[
            "Bangladesh", "politics", "Wikipedia", "corpus",
            "RAG", "Bangla", "Bengali", "history",
            "1971", "Liberation War", "Mujib", "Zia", "Ershad",
            "Hasina", "Khaleda Zia", "Quota Movement", "2024",
            "interim government", "Yunus",
        ],
    )


def build_tarball(out_path: Path) -> Path:
    """Tar+gzip the raw + processed corpus into a single archive."""
    if out_path.exists():
        out_path.unlink()
    print(f"[tar] building {out_path} …")
    with tarfile.open(out_path, "w:gz") as tar:
        # Add raw + processed + manifest
        raw_dir = BASE_DIR / "data" / "raw"
        proc_dir = BASE_DIR / "data" / "processed"
        if raw_dir.exists():
            tar.add(raw_dir, arcname="raw", filter=lambda ti: ti)
        if proc_dir.exists():
            tar.add(proc_dir, arcname="processed", filter=lambda ti: ti)
        # Add config snapshot
        cfg = BASE_DIR / "config" / "config.yaml"
        if cfg.exists():
            tar.add(cfg, arcname="config.yaml")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[tar] done: {out_path} ({size_mb:.2f} MB)")
    return out_path


def create_deposition(api_url: str, token: str, meta: SnapshotMetadata) -> dict:
    """Create a new draft deposition on Zenodo and return its metadata."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "metadata": {
            "title": meta.title,
            "upload_type": meta.upload_type,
            "publication_type": meta.publication_type or None,
            "description": meta.description,
            "creators": meta.creators,
            "keywords": meta.keywords,
            "access_right": "open",
            "license": meta.license,
            "version": meta.version,
        }
    }
    # Strip None values — Zenodo rejects them
    payload["metadata"] = {k: v for k, v in payload["metadata"].items() if v is not None}

    r = requests.post(
        api_url,
        params={"access_token": token},
        json=payload,
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def upload_file(
    api_url: str,
    token: str,
    deposition_id: str,
    file_path: Path,
) -> dict:
    """Upload a single file to an existing draft deposition."""
    upload_url = f"{api_url}/{deposition_id}/files"
    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "application/gzip")}
        data = {"name": file_path.name}
        r = requests.post(
            upload_url,
            params={"access_token": token},
            data=data,
            files=files,
            timeout=600,
        )
    r.raise_for_status()
    return r.json()


def publish_deposition(api_url: str, token: str, deposition_id: str) -> dict:
    """Publish the deposition (makes it public + assigns a DOI)."""
    r = requests.post(
        f"{api_url}/{deposition_id}/actions/publish",
        params={"access_token": token},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org instead of zenodo.org")
    ap.add_argument("--no-publish", action="store_true", help="Keep deposition in draft state")
    ap.add_argument("--no-tar", action="store_true", help="Skip tarball step (assume already built)")
    args = ap.parse_args()

    cfg = load_config()
    token = os.environ.get("ZENODO_TOKEN") or cfg.get("zenodo", {}).get("access_token", "")
    if not token:
        print("ERROR: No Zenodo token found.")
        print("Set one of:")
        print("  export ZENODO_TOKEN='your-token'")
        print("  or edit config/config.yaml -> zenodo.access_token")
        sys.exit(1)

    env_key = "sandbox" if args.sandbox else "prod"
    api_url = ZENODO_API[env_key]
    print(f"[zenodo] using {env_key} API: {api_url}")

    # 1. Build tarball
    tar_path = BASE_DIR / "download" / "bd_political_corpus_snapshot.tar.gz"
    if not args.no_tar:
        build_tarball(tar_path)
    elif not tar_path.exists():
        print(f"[error] {tar_path} not found; run without --no-tar first")
        sys.exit(1)

    # 2. Create deposition
    meta = default_metadata()
    print(f"[zenodo] creating draft deposition: {meta.title}")
    dep = create_deposition(api_url, token, meta)
    dep_id = dep["id"]
    print(f"[zenodo] draft created: id={dep_id}")

    # 3. Upload file
    print(f"[zenodo] uploading {tar_path.name} ({tar_path.stat().st_size / 1024 / 1024:.1f} MB)…")
    upload_result = upload_file(api_url, token, str(dep_id), tar_path)
    print(f"[zenodo] uploaded file id={upload_result.get('id')}")

    # 4. Publish (optional)
    if not args.no_publish:
        print(f"[zenodo] publishing deposition {dep_id}…")
        published = publish_deposition(api_url, token, str(dep_id))
        doi = published.get("doi", "(unknown)")
        html_url = published.get("links", {}).get("html", "(unknown)")
        print()
        print("=============================================")
        print("Zenodo snapshot published successfully!")
        print(f"  DOI:  https://doi.org/{doi}")
        print(f"  URL:  {html_url}")
        print("=============================================")
        # Save record locally
        (BASE_DIR / "logs" / "zenodo_publish.json").write_text(
            json.dumps(published, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print()
        print("=============================================")
        print("Draft saved (NOT published).")
        print(f"  Edit URL: {dep['links']['html']}")
        print(f"  Publish via UI or rerun without --no-publish")
        print("=============================================")


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
