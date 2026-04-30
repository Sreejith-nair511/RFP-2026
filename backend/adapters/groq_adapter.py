"""
Groq Adapter for DeceptiScope v2
Interfaces with Groq's ultra-fast inference API (LLaMA 3.1, Mixtral, Gemma)

Key Features:
- Sub-second latency via Groq's LPU hardware
- Logprob extraction (Groq supports logprobs like OpenAI)
- Streaming token analysis
- Steering prompt injection

Groq is ideal for DeceptiScope's consistency analysis (N=5 samples)
because its speed makes multi-sample probing practical in real time.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from groq import AsyncGroq

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    token: str
    logprob: float
    position: int = 0


@dataclass
class GenerationResponse:
    text: str
    tokens: List[TokenInfo]
    logprobs: List[float]
    reasoning_tokens: List[str]
    steering_applied: bool
    model_name: str
    finish_reason: str
    usage: Dict[str, int]
    token_logprobs: Optional[List[float]] = None  # for FusionLayer heatmap


class GroqAdapter:
    """
    Adapter for Groq-hosted open models.

    Groq's LPU delivers ~500 tok/s — fast enough to run 5 consistency
    samples per user turn in under 2 seconds, making behavioral probing
    practical for real-time deception detection.
    """

    AVAILABLE_MODELS = {
        "llama-3.3-70b-versatile":              {"logprobs": False, "context": 131_072},
        "llama-3.1-8b-instant":                 {"logprobs": False, "context": 131_072},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"logprobs": False, "context": 131_072},
        "qwen/qwen3-32b":                       {"logprobs": False, "context": 131_072},
        "openai/gpt-oss-120b":                  {"logprobs": False, "context": 131_072},
        "openai/gpt-oss-20b":                   {"logprobs": False, "context": 131_072},
    }

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set")
        self.client = AsyncGroq(api_key=key)

        # Uncertainty / confidence patterns for proxy scoring
        self._uncertainty = [
            "might", "could", "perhaps", "possibly", "seems", "appears",
            "I think", "I believe", "not certain", "not sure", "unclear",
            "roughly", "approximately", "maybe", "potentially",
        ]
        self._confidence = [
            "definitely", "certainly", "clearly", "obviously",
            "without doubt", "precisely", "exactly", "absolutely",
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_response(
        self,
        prompt: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_steering: bool = True,
        steering_prompt: Optional[str] = None,
    ) -> GenerationResponse:
        """Generate a response with token-level analysis."""

        system_msg = self._build_system_message(enable_steering, steering_prompt)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]

        supports_logprobs = self.AVAILABLE_MODELS.get(model, {}).get("logprobs", False)

        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            if supports_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 5

            response = await self.client.chat.completions.create(**kwargs)

            text = response.choices[0].message.content or ""
            tokens, logprobs = self._extract_token_data(response, supports_logprobs)

            return GenerationResponse(
                text=text,
                tokens=tokens,
                logprobs=logprobs,
                reasoning_tokens=[],
                steering_applied=enable_steering,
                model_name=model,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=response.usage.model_dump() if response.usage else {},
                token_logprobs=logprobs if logprobs else None,
            )

        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            raise

    async def generate_streaming(
        self,
        prompt: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        enable_steering: bool = True,
    ):
        """Streaming generation — yields token chunks."""
        system_msg = self._build_system_message(enable_steering)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        cumulative = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            cumulative += delta
            yield {"token": delta, "cumulative_text": cumulative}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_message(
        self,
        enable_steering: bool,
        custom_prompt: Optional[str] = None,
    ) -> str:
        if custom_prompt:
            return custom_prompt
        if enable_steering:
            return (
                "You are a helpful, honest, and calibrated AI assistant. "
                "Always be accurate and acknowledge uncertainty explicitly. "
                "Never state unverified claims as facts. "
                "Do not omit important caveats or risks. "
                "Prioritise truthfulness over agreement."
            )
        return "You are a helpful AI assistant."

    def _extract_token_data(
        self, response: Any, supports_logprobs: bool
    ):
        tokens: List[TokenInfo] = []
        logprobs: List[float] = []

        text = response.choices[0].message.content or ""

        if supports_logprobs and response.choices[0].logprobs:
            lp_content = response.choices[0].logprobs.content or []
            for i, lp in enumerate(lp_content):
                tok = TokenInfo(
                    token=lp.token,
                    logprob=lp.logprob,
                    position=i,
                )
                tokens.append(tok)
                logprobs.append(lp.logprob)
        else:
            # Proxy: split text into words, assign confidence proxy
            words = text.split()
            for i, w in enumerate(words):
                proxy = self._word_confidence_proxy(w)
                tokens.append(TokenInfo(token=w, logprob=proxy, position=i))
                logprobs.append(proxy)

        return tokens, logprobs

    def _word_confidence_proxy(self, word: str) -> float:
        """Estimate token confidence from linguistic markers."""
        w = word.lower().strip(".,!?;:")
        if any(u in w for u in self._uncertainty):
            return -2.5   # uncertain → higher risk
        if any(c in w for c in self._confidence):
            return -0.3   # overconfident → moderate risk
        return -1.0       # neutral


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _test():
        print("Testing GroqAdapter...")
        adapter = GroqAdapter()

        resp = await adapter.generate_response(
            prompt="What is the capital of Australia?",
            model="llama-3.1-70b-versatile",
            enable_steering=True,
        )
        print(f"Response: {resp.text[:200]}")
        print(f"Tokens:   {len(resp.tokens)}")
        print(f"Logprobs: {resp.logprobs[:5]}")
        print(f"Steering: {resp.steering_applied}")
        print("✓ GroqAdapter test passed.")

    asyncio.run(_test())
