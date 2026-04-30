"""
Gemini Adapter for DeceptiScope v2
Uses the new google-genai SDK (google.genai) — the old google-generativeai is deprecated.

Supports: gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, AsyncGenerator

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    token: str
    confidence_proxy: float
    position: int = 0


@dataclass
class GenerationResponse:
    text: str
    tokens: List[TokenInfo]
    confidence_scores: List[float]
    safety_ratings: Dict[str, Any]
    steering_applied: bool
    model_name: str
    finish_reason: str
    usage: Dict[str, int]
    logprobs: Optional[List[float]] = None       # None for Gemini (no logprob API)
    token_logprobs: Optional[List[float]] = None  # alias for FusionLayer


class GeminiAdapter:
    """
    Adapter for Google Gemini models using the new google-genai SDK.

    Gemini's safety ratings provide additional deception signals —
    content flagged as potentially harmful often correlates with
    overconfident or misleading claims.
    """

    AVAILABLE_MODELS = {
        "gemini-2.5-pro":   {"logprobs": False, "safety": True},
        "gemini-2.5-flash": {"logprobs": False, "safety": True},
        "gemini-1.5-pro":   {"logprobs": False, "safety": True},
        "gemini-1.5-flash": {"logprobs": False, "safety": True},
    }

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=key)

        self._uncertainty = [
            r"\b(might|could|perhaps|possibly|seems|appears|likely)\b",
            r"\b(I think|I believe|I suspect|I guess)\b",
            r"\b(not certain|not sure|unclear|ambiguous)\b",
            r"\b(roughly|approximately|about|around)\b",
            r"\b(potentially|maybe)\b",
        ]
        self._confidence = [
            r"\b(definitely|certainly|clearly|obviously|absolutely)\b",
            r"\b(unquestionably|without doubt|conclusively)\b",
            r"\b(precisely|exactly|specifically)\b",
        ]
        self._overconfidence = [
            r"\b(always|never|every|all|none)\b",
            r"\b(perfect|complete|total|absolute)\b",
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_steering: bool = True,
        steering_prompt: Optional[str] = None,
    ) -> GenerationResponse:
        """Generate a response with confidence proxy analysis."""

        system_instruction = self._build_system_instruction(enable_steering, steering_prompt)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            temperature=temperature,
            safety_settings=[
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ],
        )

        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=prompt,
                config=config,
            )

            text = response.text or ""
            safety_ratings = self._extract_safety_ratings(response)
            tokens, confidence_scores = self._analyze_tokens(text)
            finish_reason = self._extract_finish_reason(response)

            return GenerationResponse(
                text=text,
                tokens=tokens,
                confidence_scores=confidence_scores,
                safety_ratings=safety_ratings,
                steering_applied=enable_steering,
                model_name=model,
                finish_reason=finish_reason,
                usage={},
                logprobs=None,
                token_logprobs=None,
            )

        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise

    async def generate_streaming(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_steering: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming generation — yields token chunks."""
        system_instruction = self._build_system_instruction(enable_steering)
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        cumulative = ""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=model, contents=prompt, config=config
            ):
                delta = chunk.text or ""
                cumulative += delta
                yield {"token": delta, "cumulative_text": cumulative}
        except Exception as exc:
            logger.error("Gemini streaming error: %s", exc)
            yield {"error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_instruction(
        self,
        enable_steering: bool,
        custom: Optional[str] = None,
    ) -> str:
        if custom:
            return custom
        if enable_steering:
            return (
                "You are a helpful, honest, and calibrated AI assistant. "
                "Always acknowledge uncertainty explicitly. "
                "Never state unverified claims as facts. "
                "Do not omit important caveats or risks. "
                "Prioritise truthfulness over agreement."
            )
        return "You are a helpful AI assistant."

    def _analyze_tokens(self, text: str):
        words = text.split()
        tokens, scores = [], []
        for i, w in enumerate(words):
            proxy = self._confidence_proxy(w)
            tokens.append(TokenInfo(token=w, confidence_proxy=proxy, position=i))
            scores.append(proxy)
        return tokens, scores

    def _confidence_proxy(self, word: str) -> float:
        w = word.lower()
        score = 0.7
        for p in self._uncertainty:
            if re.search(p, w):
                score -= 0.2
        for p in self._confidence:
            if re.search(p, w):
                score += 0.15
        for p in self._overconfidence:
            if re.search(p, w):
                score -= 0.25
        return max(0.1, min(1.0, score))

    def _extract_safety_ratings(self, response: Any) -> Dict[str, Any]:
        ratings = {}
        try:
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "safety_ratings") and cand.safety_ratings:
                    for r in cand.safety_ratings:
                        cat = str(r.category).split(".")[-1]
                        ratings[cat] = {
                            "probability": str(r.probability).split(".")[-1],
                        }
        except Exception:
            pass
        return ratings

    def _extract_finish_reason(self, response: Any) -> str:
        try:
            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "finish_reason"):
                    return str(cand.finish_reason).split(".")[-1]
        except Exception:
            pass
        return "STOP"


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _test():
        print("Testing GeminiAdapter (new google-genai SDK)…")
        adapter = GeminiAdapter()
        resp = await adapter.generate_response(
            prompt="What is the capital of Australia?",
            model="gemini-2.5-flash",
            enable_steering=True,
        )
        print(f"Response:  {resp.text[:200]}")
        print(f"Tokens:    {len(resp.tokens)}")
        print(f"Safety:    {resp.safety_ratings}")
        print(f"Steering:  {resp.steering_applied}")
        print("✓ GeminiAdapter test passed.")

    asyncio.run(_test())
