# Bangladesh Political Debater — Setup & Run

A RAG-grounded, multi-agent **Bangla** debate system covering Bangladesh political history 1971–today. Two agents (বিশ্লেষক / সাংবাদিক) debate any historical-political question, with every claim grounded in retrieved Wikipedia passages and synthesized to Bangla speech + talking-head animation.

## What's already built

| Phase | Status | Notes |
|------|--------|-------|
| 1. Corpus | ✅ Done | 108 Wikipedia articles (Bangla + English) across 8 periods (1971 Liberation → 2024 Quota Movement). 2,785 unique chunks. |
| 1. Zenodo snapshot | ⏸ Needs token | Local tarball ready at `download/bd_political_corpus_snapshot.tar.gz`. Run `python scripts/zenodo_upload.py` after `export ZENODO_TOKEN=...`. |
| 2. RAG | ✅ Done | paraphrase-multilingual-MiniLM embeddings + ChromaDB + BM25 hybrid retriever. |
| 3. Debate engine | ✅ Done | 4-turn Toulmin-style (opening → rebuttal → counter_rebuttal → closing). Analyst vs Journalist personas. RAG-grounded with [n] citations. |
| 4. Bangla TTS | ✅ Done | Facebook MMS-TTS-ben (local, no rate limits). Fallback: z-ai CLI. |
| 5. Talking-head animation | ⚠ Stub | SadTalker integration code present. ffmpeg fallback produces static-image + audio MP4. **On your GPU machine**: clone SadTalker → install → real talking-head videos per turn. |
| 6. UI | ✅ Done | Streamlit single page. Topic picker → live debate → per-turn cards with text + audio + sources. |
| 7. Evaluation | ✅ Done | Scorecard script for Bangla correctness + groundedness + responsiveness. |

## Run locally

### Dev environment (this machine — CPU only)

Everything runs end-to-end with no external API keys. The LLM is GLM-4-Plus via z-ai CLI; Bangla TTS uses local MMS-TTS; animation uses ffmpeg fallback (static avatar + audio).

```bash
# 1. Run the Streamlit UI
cd /home/z/my-project
streamlit run app/ui.py --server.port 8501 --server.headless true

# 2. Pick a topic (or write your own) → click "Start debate"
# 3. Wait ~60s for the 4-turn debate
# 4. Audio synthesises automatically after the debate
# 5. Each turn card shows Bangla text + inline audio player + cited sources
```

Or run the debate from CLI:

```bash
# Debate
python app/agents/debate.py \
    --topic "১৯৭৫ সালের ১৫ আগস্ট অভ্যুত্থান কি বাংলাদেশের রাজনীতিতে একটি মোড় ঘুরিয়ে দিয়েছিল?" \
    --save download/debate_1975_coup.json

# TTS per turn (local MMS Bangla)
python app/tts/synth.py \
    --debate download/debate_1975_coup.json \
    --out data/audio/debate_1975_coup \
    --provider mms

# Animation (ffmpeg fallback — no SadTalker installed)
python -m app.animation.animate_turn \
    --debate download/debate_1975_coup.json \
    --audio-dir data/audio/debate_1975_coup \
    --out-dir data/animations/debate_1975_coup \
    --no-sadtalker

# Evaluate
python scripts/evaluate_debate.py download/debate_1975_coup.json
```

### Production environment (your 16GB GPU machine)

1. **Switch LLM to Groq GPT-OSS-120B** (much better Bangla reasoning than GLM):
    ```bash
    export GROQ_API_KEY="gsk_..."  # from https://console.groq.com/keys
    ```
    The `app/agents/llm.py` wrapper auto-detects this env var and routes via Groq. z-ai CLI is the fallback if unset.

2. **Switch embeddings to BGE-M3** (1024-dim, better retrieval):
    ```bash
    # Edit config/config.yaml
    rag:
      embedding_model: "BAAI/bge-m3"
      embedding_dim: 1024
    ```
    Then rebuild the vectorstore:
    ```bash
    rm -rf data/vectorstore
    python -c "from app.rag.embeddings import build_vectorstore; build_vectorstore()"
    ```
    On GPU this takes ~30 seconds for 2,785 chunks.

3. **Install SadTalker for real talking-head videos**:
    ```bash
    git clone https://github.com/OpenTalker/SadTalker.git
    cd SadTalker && pip install -r requirements.txt
    bash scripts/download_models.sh   # ~1.5 GB checkpoints
    cd ..
    ```
    Replace the placeholder avatar images at `assets/analyst_avatar.png` and `assets/journalist_avatar.png` with two real frontal-face portraits (512×512, neutral expression, looking at camera).
    
    Then drop `--no-sadtalker` from the animate command. The animation module auto-detects SadTalker's presence and uses it; otherwise falls back to ffmpeg.

4. **Push corpus to Zenodo** (open-science publication + DOI):
    ```bash
    export ZENODO_TOKEN="your-token-from-zenodo.org"
    python scripts/zenodo_upload.py            # real upload
    python scripts/zenodo_upload.py --sandbox  # test in sandbox first
    ```

## Architecture

```
+-------------------+    +-------------------+    +-------------------+
|  Wikipedia crawl  | →  |  Chunk processor  | →  |  ChromaDB index   |
|  scripts/crawl_   |    |  scripts/process_ |    |  + BM25 index     |
|  wiki.py           |    |  corpus.py        |    |  data/vectorstore|
|  data/raw/         |    |  data/processed/  |    |                   |
+-------------------+    +-------------------+    +---------+---------+
                                                            |
                                                            v
+-------------------+    +-------------------+    +-------------------+
|  User picks topic| →  |  Debate engine    | →  |  Per-turn text +  |
|  in Streamlit UI  |    |  app/agents/      |    |  cited chunks     |
|                   |    |  debate.py        |    |                   |
+-------------------+    +---------+---------+    +---------+---------+
                                   |                       |
                                   v                       v
                         +-------------------+    +-------------------+
                         |  LLM via Groq or  |    |  Bangla TTS       |
                         |  z-ai (fallback)  |    |  app/tts/synth.py |
                         |  app/agents/llm.py|    |  (MMS-TTS-ben)    |
                         +-------------------+    +---------+---------+
                                                            |
                                                            v
                                                  +-------------------+
                                                  |  SadTalker anim  |
                                                  |  app/animation/   |
                                                  |  animate_turn.py  |
                                                  +-------------------+
```

## File layout

```
/home/z/my-project/
├── config/
│   └── config.yaml             ← all knobs (LLM, RAG, personas, TTS, animation, UI)
├── data/
│   ├── raw/                    ← 108 Wikipedia articles, 1 JSON per article
│   ├── processed/              ← chunks.jsonl + manifest.json + corpus_stats.json
│   ├── vectorstore/            ← ChromaDB persistent collection
│   ├── audio/                  ← per-turn WAV files
│   └── animations/             ← per-turn MP4 videos
├── assets/
│   ├── analyst_avatar.png      ← (placeholder) frontal portrait
│   └── journalist_avatar.png   ← (placeholder) frontal portrait
├── app/
│   ├── ui.py                   ← Streamlit single-page UI
│   ├── rag/
│   │   └── embeddings.py       ← BGEEmbedder + CorpusStore + BM25Index + HybridRetriever
│   ├── agents/
│   │   ├── llm.py              ← LLM provider (Groq primary, z-ai fallback)
│   │   └── debate.py           ← 4-turn Toulmin-style debate engine
│   ├── tts/
│   │   └── synth.py            ← MMS-TTS-ben + z-ai CLI fallback
│   └── animation/
│       └── animate_turn.py     ← SadTalker integration + ffmpeg fallback
├── scripts/
│   ├── crawl_wiki.py           ← polite Wikipedia crawler with search fallback
│   ├── process_corpus.py       ← chunking + dedup + stats
│   ├── zenodo_upload.py        ← publish corpus snapshot to zenodo.org
│   └── evaluate_debate.py      ← Bangla correctness + structure scorecard
├── download/                   ← user-facing deliverables
│   ├── bd_political_corpus_snapshot.tar.gz   (2.3 MB)
│   ├── debate_1975_coup.json                  (sample debate)
│   └── debate_1975_coup_scorecard.json
├── logs/
├── requirements.txt
└── worklog.md
```

## Sample debate output

See `download/debate_1975_coup.json` for a real 4-turn Bangla debate on the 1975 August coup, with 9 citations across 4 turns. The scorecard reports:
- Overall correctness: 92.5/100
- Groundedness: 2.25 cites per turn
- Responsiveness: 50% of rebuttal turns directly quote opponent

## Known limitations

- The dev embedding model (paraphrase-multilingual-MiniLM, 384-dim) is OK but BGE-M3 (1024-dim) is the proper production model. Switch via config.
- z-ai TTS has aggressive rate limits (429 after ~5 calls). Local MMS-TTS-ben is the right call for batch generation.
- SadTalker is the heaviest dependency. ffmpeg fallback gives you a static-image MP4 with audio, no actual lip-sync.
- Bangla Wikipedia coverage is uneven — some seed titles in Bangla don't exist (e.g., 1991 election). The crawler falls back to search and finds related articles.
