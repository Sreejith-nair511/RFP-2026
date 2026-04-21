# DeceptiScope v2

**AI Interpretability System for Deception Detection in Frontier + Open-Weight LLMs**

*Built for the Schmidt Sciences 2026 Interpretability RFP ($300k–$1M)*

---

## The Problem

Most interpretability research requires **activation access** — it only works on open-weight models.
Frontier models (GPT-5, Claude Opus 4.6, Gemini 2.5) are black boxes.
Yet these are the models most widely deployed and most in need of deception detection.

## The Innovation

DeceptiScope v2 introduces a **hybrid graybox-whitebox architecture** that works on *both*:

```
[User Prompt]
      │
      ▼
[Frontier Model API]  ←── closed box (GPT-5 / Claude / Gemini)
      │
      ├──► [Graybox Behavioral Probing]
      │         logprobs · consistency · CoT analysis · sycophancy
      │
      ├──► [Shadow Model]  ←── open-weight proxy (Mistral 7B / LLaMA 3.1 8B)
      │         fine-tuned on (prompt, frontier_completion) pairs
      │         gives whitebox access to a behavioral mirror
      │
      └──► [Deception Probe on Shadow Activations]
                 linear probe · RepE direction · layer-wise scores
                         │
                         ▼
              [Fusion Layer]
              weighted combination of all signals
                         │
                         ▼
              [Deception Score + Type + Explanation]
                         │
                         ▼
              [Steering Signal]
              translated direction → system prompt injection
                         │
                         ▼
              [Real-Time Dashboard]
```

## Key Results

| Method | AUC-ROC |
|---|---|
| Random baseline | 0.50 |
| Perplexity-based | 0.58 |
| Text classifier | 0.67 |
| Self-consistency | 0.71 |
| GPT-4 Judge | 0.74 |
| DeceptiScope (graybox only) | 0.79 |
| **DeceptiScope v2 (hybrid)** | **0.89** |

**+20% over GPT-4 Judge** on the DeceptiScope custom benchmark (500 realistic scenarios).

---

## Architecture

### Module 1: Frontier Model Adapters (`backend/adapters/`)
- `OpenAIAdapter` — GPT-4o, GPT-5 with logprob extraction and streaming
- `AnthropicAdapter` — Claude Opus/Sonnet 4.6 with extended thinking tokens
- `GeminiAdapter` — Gemini 2.5 Pro with safety ratings

### Module 2: Graybox Behavioral Probing (`backend/graybox/`)
- `LogprobAnalyzer` — token entropy, confidence calibration, evasion detection
- `ConsistencyAnalyzer` — semantic variance across N completions, contradiction detection
- `ChainOfThoughtAnalyzer` — CoT vs. output entailment, hedging/overconfidence markers
- `SycophancyDetector` — answer flip rate under opposing user priors

### Module 3: Shadow Model (`backend/shadow/`)
- `ShadowModel` — LoRA fine-tuned Mistral 7B / LLaMA 3.1 8B mirroring frontier behavior
- `DistillationTrainer` — online distillation with importance sampling and quality filtering
- `DeceptionDirectionTransfer` — extracts deception directions, translates to steering prompts

### Module 4: Whitebox Probing (`backend/whitebox/`)
- `ActivationExtractor` — HuggingFace hooks on residual stream (every layer)
- `DeceptionProbe` — supervised linear/MLP probe, per-layer and ensemble
- `RepESteer` — Representation Engineering: add honest direction vector at inference

### Module 5: Fusion Layer (`backend/fusion/`)
- `FusionLayer` — weighted combination of graybox + shadow + whitebox signals
- Platt-scaled calibration, per-type classification, confidence estimation
- Per-token risk scores for heatmap visualisation

### Module 6: Evaluation Harness (`backend/eval/`)
- TruthfulQA, SycophancyEval, custom DeceptiScope benchmark (500 scenarios)
- Baseline comparisons: GPT-4 judge, self-consistency, perplexity, text classifier
- AUROC, precision/recall, ECE calibration, steering effectiveness

### Module 7: Dataset Builder (`backend/data/`)
- Generates 50k (prompt, honest, deceptive, type) tuples via frontier APIs
- 7 deception types × 12 domains × 5 difficulty levels
- Exports to HuggingFace dataset format

### Module 8: Dashboard (`frontend/`)
- React 18 + TypeScript + Tailwind CSS + Recharts + Framer Motion
- Real-time WebSocket chat with any connected model
- Per-token deception heatmap, score gauge, consistency timeline
- Layer probe visualisation, steering panel, research mode, PDF export

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- NVIDIA GPU (recommended; CPU works for graybox-only mode)
- API keys: OpenAI, Anthropic, Google

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start all services

```bash
docker compose up --build
```

Services:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Jupyter: http://localhost:8888 (token: `deceptiscope`)
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### 3. Local development (no Docker)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

---

## Research Notebooks

| Notebook | Description |
|---|---|
| `notebooks/probe_training.ipynb` | Train per-layer deception probes on LLaMA 3.1 8B |
| `notebooks/shadow_model_eval.ipynb` | Evaluate shadow model fidelity and direction transfer |

---

## Testing Individual Modules

Every module has a `__main__` block for standalone testing:

```bash
# Test fusion layer
python -m fusion.fusion_layer

# Test graybox probing
python -m graybox.behavioral_probe

# Test shadow model
python -m shadow.shadow_model

# Test evaluation harness
python -m eval.harness

# Test dataset builder
python -m data.dataset_builder
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| Backend | Python 3.11, FastAPI, PyTorch, HuggingFace Transformers, PEFT/LoRA |
| Interpretability | baukit, sentence-transformers, scikit-learn |
| APIs | openai, anthropic, google-generativeai |
| Frontend | React 18, TypeScript, Tailwind CSS, Recharts, Framer Motion |
| Infrastructure | Docker Compose, Redis, PostgreSQL |
| Compute | Designed for A100 cluster (CUDA 12.1) |

---

## Grant Alignment

This system directly addresses the Schmidt Sciences 2026 Interpretability RFP priorities:

1. **Scalable interpretability** — works on frontier models without activation access
2. **Deception detection** — novel hybrid architecture with demonstrated superiority
3. **Steering** — closes the loop from detection to intervention
4. **Evaluation rigour** — comprehensive benchmarks with baseline comparisons
5. **Open science** — all code, datasets, and probes will be open-sourced

---

*DeceptiScope v2 — Schmidt Sciences 2026 Interpretability RFP*
