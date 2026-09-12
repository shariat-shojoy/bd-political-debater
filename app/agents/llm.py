"""
LLM Provider Abstraction
=========================

Phase 3 of the BD Political Debater pipeline.

Provides a single `chat()` function that routes to (in priority order):
1. Ollama (local, 100% offline)   — used when LLM_PROVIDER=ollama (or auto-detected if `ollama` is running locally and a model is pulled)
2. Groq API (GPT-OSS-120B)        — used when GROQ_API_KEY env var is set
3. z-ai CLI (GLM-4-Plus)          — fallback for dev/test in this environment

All three produce Bangla output for our prompts. The Ollama path is the
only one that is truly 100% local — no network calls at runtime.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Make project root importable when this file is run as a script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app._paths import BASE_DIR, CONFIG_PATH


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens: dict[str, int]
    elapsed_seconds: float


# ============================================================
# Groq provider
# ============================================================

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        try:
            from groq import Groq
        except ImportError as e:
            raise RuntimeError(
                "groq package not installed. Run: pip install groq"
            ) from e
        api_key = os.environ.get("GROQ_API_KEY") or load_config().get("groq", {}).get("api_key", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY env var not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _groq_chat(messages: list[ChatMessage], model: str, temperature: float, max_tokens: int) -> LLMResponse:
    client = _get_groq_client()
    payload = [{"role": m.role, "content": m.content} for m in messages]
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=payload,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0
    return LLMResponse(
        text=resp.choices[0].message.content,
        provider="groq",
        model=model,
        tokens={
            "prompt": resp.usage.prompt_tokens,
            "completion": resp.usage.completion_tokens,
            "total": resp.usage.total_tokens,
        },
        elapsed_seconds=elapsed,
    )


# ============================================================
# z-ai CLI provider (GLM-4-Plus) — dev fallback
# ============================================================

def _zai_chat(messages: list[ChatMessage], temperature: float, max_tokens: int) -> LLMResponse:
    """Invoke the z-ai CLI as a subprocess for a single chat completion."""
    system_prompt = ""
    user_prompt = ""
    # We can only pass one system + one user prompt to z-ai CLI, so we collapse
    # multi-turn history into a single user prompt with role labels.
    if len(messages) == 1:
        if messages[0].role == "system":
            system_prompt = messages[0].content
        else:
            user_prompt = messages[0].content
    elif len(messages) == 2:
        if messages[0].role == "system":
            system_prompt = messages[0].content
            user_prompt = messages[1].content
        else:
            user_prompt = "\n\n".join(f"[{m.role}]: {m.content}" for m in messages)
    else:
        # Multi-turn — collapse into one user prompt with role labels
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                user_prompt += f"\n\n[{m.role}]: {m.content}"
        user_prompt = user_prompt.strip()

    # Write the user prompt to a temp file because it can be long
    tmp_path = Path("/tmp") / f"zai_user_{os.getpid()}_{int(time.time()*1000)}.txt"
    tmp_path.write_text(user_prompt, encoding="utf-8")

    out_path = Path("/tmp") / f"zai_resp_{os.getpid()}_{int(time.time()*1000)}.json"
    cmd = [
        "z-ai", "chat",
        "--prompt", user_prompt,
        "--output", str(out_path),
    ]
    if system_prompt:
        cmd.extend(["--system", system_prompt])

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(f"z-ai CLI failed: {proc.stderr}")
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        elapsed = time.time() - t0
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            provider="z-ai",
            model=data.get("model", "glm-4-plus"),
            tokens={
                "prompt": usage.get("prompt_tokens", 0),
                "completion": usage.get("completion_tokens", 0),
                "total": usage.get("total_tokens", 0),
            },
            elapsed_seconds=elapsed,
        )
    finally:
        for p in [tmp_path, out_path]:
            try:
                p.unlink()
            except Exception:
                pass


# ============================================================
# Ollama provider (100% local, no network)
# ============================================================

def _ollama_chat(messages: list[ChatMessage], model: str, temperature: float, max_tokens: int) -> LLMResponse:
    """Call a local Ollama server running on http://localhost:11434.

    Requires:
    1. Ollama installed: https://ollama.com/download
    2. A model pulled:    ollama pull qwen2.5:14b-instruct-q4_K_M
       (or set OLLAMA_MODEL env var to a different model)
    3. Ollama running:    `ollama serve` (or it auto-starts on first call)
    """
    import ollama
    t0 = time.time()
    payload = [{"role": m.role, "content": m.content} for m in messages]
    resp = ollama.chat(
        model=model,
        messages=payload,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )
    elapsed = time.time() - t0
    text = resp.get("message", {}).get("content", "")
    # Ollama doesn't return exact token counts in all versions; estimate
    eval_count = resp.get("eval_count", 0)
    prompt_eval_count = resp.get("prompt_eval_count", 0)
    return LLMResponse(
        text=text,
        provider="ollama",
        model=model,
        tokens={
            "prompt": prompt_eval_count,
            "completion": eval_count,
            "total": prompt_eval_count + eval_count,
        },
        elapsed_seconds=elapsed,
    )


def _ollama_available(model: str) -> bool:
    """Check if Ollama is running locally AND has the requested model pulled."""
    try:
        import ollama
        # List local models
        models = ollama.list()
        # Ollama v0.4+ returns ListResponse with .models list
        if hasattr(models, "models"):
            available = [m.model for m in models.models]
        else:
            available = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
        # Match by prefix (model tag may have :tag suffix)
        return any(av.split(":")[0] == model.split(":")[0] for av in available if av)
    except Exception:
        return False


# ============================================================
# Unified chat() function
# ============================================================

def chat(messages: list[ChatMessage], **kwargs) -> LLMResponse:
    """Single entry point.

    Provider priority (first match wins):
    1. `LLM_PROVIDER=ollama` env var → use local Ollama with `OLLAMA_MODEL` (default: qwen2.5:14b-instruct)
    2. `GROQ_API_KEY` env var set     → use Groq with GPT-OSS-120B
    3. Fallback                       → z-ai CLI (GLM-4-Plus, needs network)

    For 100% local runtime, set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=qwen2.5:14b-instruct-q4_K_M`.
    """
    cfg = load_config()
    temperature = kwargs.get("temperature", cfg["debate"]["temperature"])
    max_tokens = kwargs.get("max_tokens", cfg["debate"]["max_tokens_per_turn"])

    # 1) Ollama (100% local)
    provider_pref = os.environ.get("LLM_PROVIDER", "").lower()
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    if provider_pref == "ollama" or (provider_pref == "" and _ollama_available(ollama_model)):
        try:
            return _ollama_chat(messages, ollama_model, temperature, max_tokens)
        except Exception as e:
            if provider_pref == "ollama":
                raise RuntimeError(f"LLM_PROVIDER=ollama but call failed: {e}") from e
            print(f"[warn] Ollama failed ({e}); falling back", file=sys.stderr)

    # 2) Groq
    api_key = os.environ.get("GROQ_API_KEY") or cfg.get("groq", {}).get("api_key", "")
    if api_key:
        try:
            model = kwargs.get("model", cfg["debate"]["llm_model"])
            return _groq_chat(messages, model, temperature, max_tokens)
        except Exception as e:
            print(f"[warn] Groq failed ({e}); falling back to z-ai CLI", file=sys.stderr)

    return _zai_chat(messages, temperature, max_tokens)


def is_groq_available() -> bool:
    api_key = os.environ.get("GROQ_API_KEY") or load_config().get("groq", {}).get("api_key", "")
    return bool(api_key)


if __name__ == "__main__":
    # Quick self-test
    print("=== LLM provider test ===")
    print(f"Groq available: {is_groq_available()}")
    print("\n--- Bangla test ---")
    msgs = [
        ChatMessage(role="system", content="তুমি একজন রাজনৈতিক বিশ্লেষক। বাংলায় উত্তর দাও।"),
        ChatMessage(role="user", content="১৯৭১ সালের মুক্তিযুদ্ধ সম্পর্কে এক বাক্যে বলো।"),
    ]
    resp = chat(msgs)
    print(f"Provider: {resp.provider} ({resp.model})")
    print(f"Tokens: {resp.tokens}")
    print(f"Elapsed: {resp.elapsed_seconds:.1f}s")
    print(f"Response:\n{resp.text}")
