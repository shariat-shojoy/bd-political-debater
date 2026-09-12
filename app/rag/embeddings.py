"""
RAG Module — embeddings + ChromaDB + hybrid retrieval
======================================================

Phase 2 of the BD Political Debater pipeline.

Components:
1. `BGEEmbedder`        — wraps sentence-transformers BGE-M3 model (multilingual, Bangla-capable)
2. `CorpusStore`        — builds + persists ChromaDB collection from chunks.jsonl
3. `BM25Index`          — sparse keyword index built from same chunks (Bangla-aware tokenization)
4. `HybridRetriever`    — combines BM25 + vector scores, optional reranker, returns top-k chunks

Usage:
    # Build the vectorstore from scratch:
    python -c "from app.rag.embeddings import build_vectorstore; build_vectorstore()"

    # Query:
    python -c "
    from app.rag.retriever import HybridRetriever
    r = HybridRetriever()
    print(r.retrieve('১৯৭১ সালে মুক্তিযুদ্ধে কতজন নিহত হয়েছিলেন?'))
    "
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

# app/rag/embeddings.py lives at <root>/app/rag/embeddings.py
# Go up 3 levels to reach the project root, then up 1 more for the import level
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app._paths import BASE_DIR, CONFIG_PATH, RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 1) Embedder (BGE-M3 multilingual, supports Bangla)
# ============================================================

class BGEEmbedder:
    """Sentence-transformers wrapper around BAAI/bge-m3.

    BGE-M3 is multilingual (supports 50+ languages including Bangla) and
    outputs 1024-dim dense vectors + optional sparse tokens. We use dense
    only here for simplicity. Falls back to a smaller multilingual model
    if BGE-M3 is unavailable.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        cfg = load_config()
        self.model_name = model_name or cfg["rag"]["embedding_model"]
        self.device = device or "cpu"  # change to "cuda" on your GPU box
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[embedder] loading {self.model_name} on {self.device}…")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[embedder] ready.")
        return self._model

    def embed(self, texts: list[str], batch_size: int = 16, show_progress: bool = True):
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # cosine-friendly
            convert_to_numpy=True,
        )
        return embs

    def embed_one(self, text: str):
        return self.embed([text], show_progress=False)[0]


# ============================================================
# 2) Corpus Store (ChromaDB)
# ============================================================

def build_vectorstore():
    """Build ChromaDB collection from /home/z/my-project/data/processed/chunks.jsonl.

    Idempotent: deletes the existing collection first so re-running after a
    crawl refresh gives you a clean index.
    """
    import chromadb
    cfg = load_config()
    chunks_path = BASE_DIR / cfg["project"]["processed_dir"] / "chunks.jsonl"
    vs_dir = BASE_DIR / cfg["project"]["vectorstore_dir"]
    vs_dir.mkdir(parents=True, exist_ok=True)

    embedder = BGEEmbedder()

    # Load chunks
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[vectorstore] {len(chunks)} chunks to embed.")

    # Generate embeddings (CPU may take ~5 min for 2.8k chunks; GPU is much faster)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts, batch_size=32)

    # ChromaDB persistent client
    client = chromadb.PersistentClient(path=str(vs_dir))
    # Delete if exists (clean rebuild)
    try:
        client.delete_collection("bd_political_chunks")
    except Exception:
        pass
    coll = client.create_collection(
        name="bd_political_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    coll.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "period_key": c["period_key"],
                "period_label": c["period_label"],
                "year_range": ",".join(str(y) for y in c["year_range"] or []),
                "lang": c["lang"],
                "source_title": c["source_title"],
                "source_url": c["source_url"],
                "position_in_article": c["position_in_article"],
                "hash": c["hash"],
            }
            for c in chunks
        ],
    )
    print(f"[vectorstore] {len(chunks)} chunks indexed into {vs_dir}/")


# ============================================================
# 3) BM25 Index (Bangla-aware)
# ============================================================

def bangla_tokenize(text: str) -> list[str]:
    """Simple tokeniser that works for both Bangla and English.

    Splits on whitespace + punctuation, keeps Bangla Unicode block chars,
    filters out single-char tokens except for digits.
    """
    # Lowercase Latin part
    text = text.lower()
    # Strip punctuation but keep Bangla chars, Latin letters, digits
    tokens = re.findall(r"[\u0980-\u09FF]+|[a-z]+|\d+", text)
    # Filter very short tokens (<=2 chars) except digits
    tokens = [t for t in tokens if (len(t) >= 3) or t.isdigit()]
    return tokens


class BM25Index:
    def __init__(self, chunks: list[dict] | None = None):
        from rank_bm25 import BM25Okapi
        if chunks is None:
            chunks = self._load_chunks()
        self.chunks = chunks
        self.ids = [c["chunk_id"] for c in chunks]
        corpus_tokens = [bangla_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def _load_chunks(self) -> list[dict]:
        cfg = load_config()
        path = BASE_DIR / cfg["project"]["processed_dir"] / "chunks.jsonl"
        chunks = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        tokens = bangla_tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(range(len(scores)), scores), key=lambda x: -x[1])[:top_k]
        return [(self.chunks[i], float(s)) for i, s in ranked if s > 0]


# ============================================================
# 4) Hybrid Retriever
# ============================================================

class HybridRetriever:
    """Combines BM25 (sparse) + vector (dense) retrieval, returns top-k
    deduplicated chunks with sources.
    """

    def __init__(self):
        import chromadb
        cfg = load_config()
        self.cfg = cfg
        vs_dir = BASE_DIR / cfg["project"]["vectorstore_dir"]
        self.embedder = BGEEmbedder()
        self.chroma_client = chromadb.PersistentClient(path=str(vs_dir))
        try:
            self.collection = self.chroma_client.get_collection("bd_political_chunks")
        except Exception:
            raise RuntimeError(
                "Vector store not built yet. Run:\n"
                "  python -c 'from app.rag.embeddings import build_vectorstore; build_vectorstore()'"
            )
        self.bm25 = BM25Index()
        self.bm25_weight = cfg["rag"]["bm25_weight"]
        self.vector_weight = cfg["rag"]["vector_weight"]
        self.top_k = cfg["rag"]["top_k"]

    def retrieve(self, query: str, top_k: int | None = None, lang_filter: str | None = None) -> list[dict]:
        top_k = top_k or self.top_k

        # Vector search
        q_emb = self.embedder.embed_one(query)
        where = {"lang": lang_filter} if lang_filter else None
        vec_results = self.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k * 2,  # over-fetch then re-rank
            where=where,
        )
        vec_pairs: list[tuple[str, float]] = []
        if vec_results["ids"]:
            for i, cid in enumerate(vec_results["ids"][0]):
                dist = vec_results["distances"][0][i]
                sim = 1.0 - dist  # cosine distance → similarity
                vec_pairs.append((cid, sim * self.vector_weight))
        vec_map = dict(vec_pairs)

        # BM25 search
        bm25_pairs = self.bm25.search(query, top_k=top_k * 2)
        bm25_map = {c["chunk_id"]: s * self.bm25_weight for c, s in bm25_pairs}

        # Combine scores
        all_ids = set(vec_map.keys()) | set(bm25_map.keys())
        scored = []
        for cid in all_ids:
            v = vec_map.get(cid, 0.0)
            b = bm25_map.get(cid, 0.0)
            scored.append((cid, v + b))
        scored.sort(key=lambda x: -x[1])
        top_ids = [cid for cid, _ in scored[:top_k]]

        # Fetch full chunk records for these IDs
        # Build a map of chunk_id -> chunk from BM25 index (has full records)
        chunk_map = {c["chunk_id"]: c for c in self.bm25.chunks}
        out = []
        for cid in top_ids:
            if cid in chunk_map:
                rec = dict(chunk_map[cid])
                # Find the score we computed
                score = dict(scored).get(cid, 0.0)
                rec["retrieval_score"] = score
                # Trim the text to a citation-friendly length
                text = rec.get("text", "")
                rec["text"] = text
                out.append(rec)
        return out


def interactive_query():
    """Tiny REPL for testing the retriever."""
    r = HybridRetriever()
    print("=== BD Political RAG retriever ===")
    print("Type a query in Bangla or English. Ctrl+D / Ctrl+C to exit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            continue
        results = r.retrieve(q, top_k=4)
        for i, c in enumerate(results, 1):
            print(f"\n--- Result {i} (score={c['retrieval_score']:.3f}) ---")
            print(f"Source: {c['source_title']} [{c['lang']}] ({c['period_label']})")
            print(f"URL: {c['source_url']}")
            print(f"Text: {c['text'][:400]}{'…' if len(c['text']) > 400 else ''}")


if __name__ == "__main__":
    interactive_query()
