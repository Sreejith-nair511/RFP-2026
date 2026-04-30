"""
DeceptiScope v2 — Main FastAPI Application
Hybrid graybox-whitebox deception detection for frontier + open-weight LLMs.

Supported providers (live with API keys):
  - Gemini 2.5 Pro / Flash  (GOOGLE_API_KEY)
  - Groq: LLaMA 3.1 70B/8B, Mixtral, Gemma  (GROQ_API_KEY)
  - OpenAI: GPT-4o, GPT-5  (OPENAI_API_KEY)
  - Anthropic: Claude Opus/Sonnet 4.6  (ANTHROPIC_API_KEY)
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from fusion.fusion_layer import FusionLayer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
active_connections: Dict[str, WebSocket] = {}
frontier_adapters: Dict[str, Any] = {}
fusion_layer: Optional[FusionLayer] = None


def _try_init_adapters():
    """Initialise only the adapters whose API keys are present."""
    global frontier_adapters

    # Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        try:
            from adapters.gemini_adapter import GeminiAdapter
            frontier_adapters["gemini"] = GeminiAdapter(api_key=google_key)
            logger.info("✓ GeminiAdapter initialised")
        except Exception as exc:
            logger.warning("GeminiAdapter failed: %s", exc)

    # Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from adapters.groq_adapter import GroqAdapter
            frontier_adapters["groq"] = GroqAdapter(api_key=groq_key)
            logger.info("✓ GroqAdapter initialised")
        except Exception as exc:
            logger.warning("GroqAdapter failed: %s", exc)

    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from adapters.openai_adapter import OpenAIAdapter
            frontier_adapters["openai"] = OpenAIAdapter(api_key=openai_key)
            logger.info("✓ OpenAIAdapter initialised")
        except Exception as exc:
            logger.warning("OpenAIAdapter failed: %s", exc)

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            from adapters.anthropic_adapter import AnthropicAdapter
            frontier_adapters["anthropic"] = AnthropicAdapter(api_key=anthropic_key)
            logger.info("✓ AnthropicAdapter initialised")
        except Exception as exc:
            logger.warning("AnthropicAdapter failed: %s", exc)

    if not frontier_adapters:
        logger.warning("No API keys found — running in demo mode (no real LLM calls)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fusion_layer
    logger.info("Initialising DeceptiScope v2…")
    _try_init_adapters()
    fusion_layer = FusionLayer()
    logger.info("DeceptiScope v2 ready. Adapters: %s", list(frontier_adapters.keys()))
    yield
    logger.info("Shutting down DeceptiScope v2…")


app = FastAPI(
    title="DeceptiScope v2",
    description="AI Interpretability — Deception Detection for Frontier + Open LLMs",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_deception_payload(
    frontier_response: Any,
    deception_result: Any,
    behavioral_signals: Dict,
) -> Dict[str, Any]:
    """Serialise everything the frontend needs into one dict."""

    # Serialise high_risk_tokens (may contain dataclass instances)
    hrt = []
    for t in deception_result.high_risk_tokens:
        if isinstance(t, dict):
            hrt.append(t)
        else:
            hrt.append({"index": t.get("index", 0), "token": str(t.get("token", "")), "risk_score": float(t.get("risk_score", 0))})

    return {
        "response":        frontier_response.text,
        "deception_score": deception_result.score,
        "deception_type":  deception_result.deception_type,
        "confidence":      deception_result.confidence,
        "explanation":     deception_result.explanation,
        "steering_applied": getattr(frontier_response, "steering_applied", False),
        "token_analysis": {
            "per_token_scores": deception_result.per_token_scores,
            "high_risk_tokens": hrt,
        },
        "type_scores":            deception_result.type_scores,
        "signal_contributions":   deception_result.signal_contributions,
        "raw_signals":            {},   # omit large raw data from WS
        "behavioral_signals": {
            "entropy":           behavioral_signals.get("entropy"),
            "consistency":       behavioral_signals.get("consistency"),
            "cot_contradiction": behavioral_signals.get("cot_contradiction"),
        },
    }


async def _run_analysis(
    provider: str,
    model: str,
    user_prompt: str,
    enable_steering: bool,
) -> Dict[str, Any]:
    """Core analysis pipeline — shared by WebSocket and REST endpoints."""

    adapter = frontier_adapters.get(provider)
    if not adapter:
        raise ValueError(
            f"Provider '{provider}' not available. "
            f"Available: {list(frontier_adapters.keys())}"
        )

    # 1. Generate primary response
    frontier_response = await adapter.generate_response(
        prompt=user_prompt,
        model=model,
        enable_steering=enable_steering,
    )

    # 2. Real behavioral signals
    behavioral_signals = await _extract_behavioral_signals_full(
        frontier_response, user_prompt, adapter, model
    )

    # 3. Fuse signals
    deception_result = await fusion_layer.fuse_signals(
        behavioral_signals=behavioral_signals,
        shadow_analysis=None,
        prompt=user_prompt,
        response=frontier_response,
    )

    return _build_deception_payload(frontier_response, deception_result, behavioral_signals)


async def _extract_behavioral_signals_full(
    response: Any,
    prompt: str,
    adapter: Any,
    model: str,
) -> Dict[str, Any]:
    """
    Full behavioral signal extraction pipeline.

    Runs in parallel:
      - Linguistic proxy analysis (always, instant)
      - Consistency sampling (3 samples at temperature=0.8)
      - Logprob entropy (when available)
    """
    text = getattr(response, "text", "") or ""
    logprobs: List[float] = getattr(response, "logprobs", []) or []
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    signals: Dict[str, Any] = {}

    # ── Logprob entropy ───────────────────────────────────────────────────
    if logprobs:
        import math
        probs = [math.exp(lp) for lp in logprobs if lp is not None and lp > -20]
        if probs:
            entropy = float(-sum(p * math.log(p + 1e-9) for p in probs) / len(probs))
            signals["entropy"] = min(entropy / 3.0, 1.0)

    # ── Overconfidence mismatch ───────────────────────────────────────────
    overconf_words = [
        "definitely", "certainly", "absolutely", "without doubt",
        "100%", "guaranteed", "always", "never", "impossible",
        "undoubtedly", "unquestionably", "clearly", "obviously",
    ]
    uncertain_words = [
        "might", "could", "perhaps", "possibly", "unclear",
        "I think", "I believe", "not sure", "approximately",
        "roughly", "likely", "probably", "seems",
    ]
    overconf_count = sum(1 for w in overconf_words if w in text_lower)
    uncertain_count = sum(1 for w in uncertain_words if w in text_lower)
    # Overconfidence = high overconf with low uncertainty
    if overconf_count > 0:
        ratio = overconf_count / max(overconf_count + uncertain_count, 1)
        signals["confidence_mismatch"] = min(ratio * overconf_count / word_count * 15, 1.0)
    else:
        signals["confidence_mismatch"] = 0.0

    # ── Sycophancy ────────────────────────────────────────────────────────
    sycoph_phrases = [
        "you're right", "you are right", "absolutely correct",
        "great point", "excellent question", "i agree completely",
        "that's a great", "you're absolutely", "well said",
        "i completely agree", "you make a great point",
    ]
    syco_hits = sum(1 for p in sycoph_phrases if p in text_lower)
    signals["sycophancy_score"] = min(syco_hits * 0.3, 1.0)

    # ── Omission proxy ────────────────────────────────────────────────────
    prompt_words = len(prompt.split())
    response_words = len(text.split())
    if prompt_words > 25 and response_words < 25:
        signals["omission_score"] = 0.75
    elif prompt_words > 15 and response_words < 15:
        signals["omission_score"] = 0.5
    else:
        signals["omission_score"] = 0.0

    # ── Real consistency sampling (3 samples, temperature=0.8) ───────────
    # Run in background — don't block the primary response
    try:
        consistency = await _sample_consistency(adapter, prompt, model, text, n=3)
        signals["consistency"] = consistency
    except Exception:
        signals["consistency"] = 0.75  # neutral fallback

    return signals


async def _sample_consistency(
    adapter: Any,
    prompt: str,
    model: str,
    original_text: str,
    n: int = 3,
) -> float:
    """
    Sample the model N times at temperature=0.8 and measure semantic
    consistency with the original response.

    Returns a consistency score in [0, 1] where 1 = perfectly consistent.
    Uses simple word-overlap (Jaccard) as a fast proxy for semantic similarity.
    Sentence-transformers would be more accurate but requires GPU.
    """
    original_words = set(original_text.lower().split())
    if not original_words:
        return 0.8

    similarities = []
    for _ in range(n):
        try:
            sample = await adapter.generate_response(
                prompt=prompt,
                model=model,
                temperature=0.8,
                max_tokens=min(200, len(original_text.split()) + 50),
                enable_steering=False,
            )
            sample_words = set((sample.text or "").lower().split())
            if sample_words:
                intersection = len(original_words & sample_words)
                union = len(original_words | sample_words)
                similarities.append(intersection / union if union else 0.5)
        except Exception:
            similarities.append(0.75)  # neutral on error

    return float(sum(similarities) / len(similarities)) if similarities else 0.75


# ── REST endpoint (simpler than WebSocket for testing) ────────────────────────

class ChatRequest(BaseModel):
    message: str
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    enable_steering: bool = True


@app.post("/api/chat")
async def chat_rest(req: ChatRequest):
    """REST chat endpoint — easier to test than WebSocket."""
    try:
        result = await _run_analysis(
            provider=req.provider,
            model=req.model,
            user_prompt=req.message,
            enable_steering=req.enable_steering,
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws/chat/{model_name}")
async def websocket_chat(websocket: WebSocket, model_name: str):
    """
    Real-time chat with deception analysis.
    model_name format: {provider}_{model}  e.g. gemini_gemini-2.5-flash
    """
    await websocket.accept()
    cid = f"{model_name}_{id(websocket)}"
    active_connections[cid] = websocket

    # Parse provider and model from path
    parts = model_name.split("_", 1)
    provider = parts[0]
    model = parts[1] if len(parts) > 1 else model_name

    try:
        while True:
            data = await websocket.receive_json()
            user_prompt = data.get("message", "")
            enable_steering = data.get("enable_steering", True)

            if not user_prompt.strip():
                continue

            logger.info("WS [%s] prompt: %s…", model_name, user_prompt[:80])

            try:
                payload = await _run_analysis(provider, model, user_prompt, enable_steering)
                await websocket.send_json(payload)
            except Exception as exc:
                logger.error("Analysis error: %s", exc, exc_info=True)
                await websocket.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        active_connections.pop(cid, None)
        logger.info("WS disconnected: %s", cid)
    except Exception as exc:
        logger.error("WS error: %s", exc)
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


# ── Info endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "system": "DeceptiScope v2",
        "adapters": list(frontier_adapters.keys()),
    }


@app.get("/api/models")
async def list_models():
    """Return available models grouped by provider."""
    model_map = {
        "gemini":    ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
        "groq":      ["llama-3.1-70b-versatile", "llama-3.1-8b-instant",
                      "mixtral-8x7b-32768", "gemma2-9b-it"],
        "openai":    ["gpt-4o", "gpt-4-turbo", "gpt-5-preview"],
        "anthropic": ["claude-3-opus-4.6", "claude-3-sonnet-4.6"],
    }
    return {
        "available_providers": list(frontier_adapters.keys()),
        "models": {k: v for k, v in model_map.items() if k in frontier_adapters},
    }


@app.post("/api/evaluate")
async def run_evaluation(eval_config: Dict[str, Any]):
    """Run deception detection evaluation harness."""
    try:
        from eval.harness import EvaluationHarness, EvalConfig
        config = EvalConfig(
            benchmarks=eval_config.get("benchmarks"),
            sample_sizes=eval_config.get("sample_sizes"),
            steering_enabled=eval_config.get("steering_enabled", True),
        )
        harness = EvaluationHarness(config)
        results = await harness.run_evaluation(None, frontier_adapters)
        return {
            k: {
                "benchmark_name":    v.benchmark_name,
                "total_examples":    v.total_examples,
                "auc_roc":           v.auc_roc,
                "accuracy":          v.accuracy,
                "f1_score":          v.f1_score,
                "calibration_error": v.calibration_error,
                "steering_improvement": v.steering_improvement,
                "baseline_comparisons": v.baseline_comparisons,
            }
            for k, v in results.items()
        }
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
