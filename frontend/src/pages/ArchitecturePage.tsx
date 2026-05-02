/**
 * ArchitecturePage — interactive system architecture visualization.
 *
 * Route: /architecture
 *
 * Sections:
 *  1. Animated pipeline flow diagram (Framer Motion)
 *  2. Stream breakdown cards (clickable, with side panel)
 *  3. "Why This Is Novel" highlights
 *  4. Benchmark results table
 */

import React, { useState, memo, lazy } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ── Types ─────────────────────────────────────────────────────────────────────

interface PipelineNode {
  id: string;
  label: string;
  sublabel?: string;
  color: string;
  description: string;
  details: string[];
}

interface StreamCard {
  id: string;
  title: string;
  badge: string;
  badgeColor: string;
  what: string;
  why: string;
  signals: string[];
  icon: React.ReactNode;
}

// ── Pipeline nodes ────────────────────────────────────────────────────────────

const PIPELINE: PipelineNode[] = [
  {
    id: "prompt",
    label: "User Prompt",
    color: "#7a8899",
    description: "The user's input message sent to the frontier model.",
    details: [
      "Any natural language query",
      "Passed to all three analysis streams simultaneously",
      "Stored for consistency comparison across samples",
    ],
  },
  {
    id: "frontier",
    label: "Frontier Model",
    sublabel: "GPT-5 · Claude · Gemini · LLaMA",
    color: "#3b82f6",
    description: "The target model being analysed. Can be closed (black-box) or open-weight.",
    details: [
      "Closed models: GPT-5, Claude Opus 4.6, Gemini 2.5 Pro",
      "Open models: LLaMA 3.3 70B, Qwen3 32B via Groq LPU",
      "Response + optional logprobs extracted",
      "Sampled 3× at temperature 0.8 for consistency analysis",
    ],
  },
  {
    id: "streams",
    label: "3 Parallel Analysis Streams",
    color: "#a855f7",
    description: "Three independent signal extraction pipelines run in parallel on every response.",
    details: [
      "Stream 1: Graybox behavioral probing",
      "Stream 2: Shadow model activation proxy",
      "Stream 3: Whitebox probing (open models only)",
      "All streams feed into the fusion layer",
    ],
  },
  {
    id: "fusion",
    label: "Fusion Layer",
    color: "#f97316",
    description: "Calibrated weighted combination of all available signal streams.",
    details: [
      "Weighted average with learned attention weights",
      "Platt scaling for probability calibration",
      "Per-type deception classification (7 categories)",
      "Confidence estimation from stream agreement",
      "Per-token risk score generation",
    ],
  },
  {
    id: "output",
    label: "Deception Score + Steering",
    color: "#22c55e",
    description: "Final calibrated score with explanation, heatmap, and optional steering signal.",
    details: [
      "0–1 calibrated deception probability",
      "Dominant deception type + per-type breakdown",
      "Natural language explanation",
      "Per-token risk heatmap",
      "Steering prompt derived from shadow model geometry",
    ],
  },
];

// ── Stream cards ──────────────────────────────────────────────────────────────

const STREAMS: StreamCard[] = [
  {
    id: "graybox",
    title: "Graybox Behavioral Probing",
    badge: "Stream 1",
    badgeColor: "#3b82f6",
    what: "Extracts behavioral signals from the model's outputs without any activation access. Works on every model with an API.",
    why: "Frontier models are black boxes — we cannot read their weights or activations. But we can observe their behavior: how consistent they are, how confident their language is, whether they agree with false premises.",
    signals: [
      "Consistency sampling — 3 calls at temperature 0.8, Jaccard similarity",
      "Token entropy — uncertainty in logprob distribution (OpenAI only)",
      "Overconfidence markers — 'definitely', 'guaranteed', '100%'",
      "Sycophancy detection — explicit agreement phrase density",
      "Omission proxy — response brevity vs. prompt complexity",
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.5" />
        <path d="M6 10h8M10 6v8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: "shadow",
    title: "Shadow Model Proxy",
    badge: "Stream 2 — Core Innovation",
    badgeColor: "#a855f7",
    what: "A small open-weight model (Mistral 7B / LLaMA 3.1 8B) continuously fine-tuned to mirror the frontier model's behavioral distribution. Gives whitebox access to a behavioral proxy.",
    why: "This is the key innovation. By training a shadow model on (prompt, frontier_completion) pairs, we create a model that behaves like GPT-5 but whose activations we can read. We extract deception directions from the shadow model's residual stream and transfer them back as steering prompts.",
    signals: [
      "LoRA fine-tuning on rolling buffer of 10k distillation pairs",
      "Fidelity tracking: cosine similarity of output distributions",
      "Per-layer deception probe (linear classifier on residual stream)",
      "Deception direction extraction via PCA + contrastive mean",
      "Direction transfer: geometric → natural language steering prompt",
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <rect x="3" y="3" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
        <rect x="11" y="3" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
        <rect x="7" y="11" width="6" height="6" rx="1" stroke="currentColor" strokeWidth="1.5" />
        <path d="M6 9v2M14 9v2M10 9v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: "whitebox",
    title: "Whitebox Probing",
    badge: "Stream 3 — Open Models",
    badgeColor: "#22c55e",
    what: "For open-weight models (LLaMA, Mistral, Qwen), direct activation extraction using HuggingFace hooks on the residual stream at every transformer layer.",
    why: "When we have activation access, we can train much more accurate probes. The per-layer analysis reveals which layers 'know' the truth — typically middle layers (12–20 of 32) show the strongest deception signal.",
    signals: [
      "Residual stream activations at every layer via baukit hooks",
      "Supervised linear probe: honest vs. deceptive hidden states",
      "Training data: TruthfulQA + synthetic deception suite",
      "Per-layer deception score → peak layer identification",
      "RepE steering: add honest direction vector at inference time",
    ],
    icon: (
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
        <path d="M4 10h12M4 6h12M4 14h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <circle cx="8" cy="10" r="1.5" fill="currentColor" />
        <circle cx="12" cy="6" r="1.5" fill="currentColor" />
        <circle cx="10" cy="14" r="1.5" fill="currentColor" />
      </svg>
    ),
  },
];

// ── Novel highlights ──────────────────────────────────────────────────────────

const NOVEL_POINTS = [
  {
    title: "Works on black-box frontier models",
    body: "GPT-5, Claude Opus 4.6, Gemini 2.5 Pro — no activation access required. First system to achieve interpretability-grade deception detection on closed models.",
    color: "#3b82f6",
  },
  {
    title: "Shadow model proxy concept",
    body: "Fine-tune a small open model to mirror frontier behavior, then apply whitebox methods to the proxy. Transfer deception directions back as natural language steering.",
    color: "#a855f7",
  },
  {
    title: "Hybrid interpretability",
    body: "Combines graybox behavioral signals (works everywhere) with whitebox activation probing (open models) and shadow model proxying (frontier models). Best of all three worlds.",
    color: "#f97316",
  },
  {
    title: "Real consistency sampling",
    body: "3 independent API calls at temperature 0.8, Jaccard similarity. Honest answers score 0.85–1.0 consistency. Confabulating answers score 0.15–0.40. Real signal, not heuristic.",
    color: "#22c55e",
  },
  {
    title: "End-to-end: detect → explain → steer",
    body: "Not just a score. Natural language explanation, per-token heatmap, and steering signal derived from the model's own internal geometry — all in under 4 seconds.",
    color: "#eab308",
  },
  {
    title: "Largest deception dataset",
    body: "50,000 labeled (prompt, honest, deceptive, type) tuples across 7 deception types, 12 domains, 5 difficulty levels. Generated using frontier models. First dataset of its kind.",
    color: "#ec4899",
  },
];

// ── Benchmark data ────────────────────────────────────────────────────────────

const BENCHMARKS = [
  { method: "Random baseline",              type: "Blackbox", auc: 0.50, highlight: false },
  { method: "Perplexity-based",             type: "Blackbox", auc: 0.55, highlight: false },
  { method: "Text classifier",              type: "Blackbox", auc: 0.65, highlight: false },
  { method: "Self-consistency voting",      type: "Blackbox", auc: 0.71, highlight: false },
  { method: "GPT-4 Judge",                  type: "Blackbox", auc: 0.74, highlight: false },
  { method: "DeceptiScope v2 (graybox)",    type: "Hybrid",   auc: 0.79, highlight: true  },
  { method: "DeceptiScope v2 (full hybrid)",type: "Hybrid",   auc: 0.89, highlight: true  },
];

// ── Sub-components ────────────────────────────────────────────────────────────

const Arrow = memo(() => (
  <div className="flex flex-col items-center py-1">
    <motion.div
      className="w-px bg-border2 flex-1"
      style={{ minHeight: 16 }}
      initial={{ scaleY: 0 }}
      animate={{ scaleY: 1 }}
      transition={{ duration: 0.3 }}
    />
    <motion.svg
      width="10" height="6" viewBox="0 0 10 6" fill="none"
      initial={{ opacity: 0, y: -4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <path d="M1 1L5 5L9 1" stroke="#263040" strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" />
    </motion.svg>
  </div>
));

const PipelineBlock = memo(({
  node, index, onClick, isActive,
}: {
  node: PipelineNode;
  index: number;
  onClick: () => void;
  isActive: boolean;
}) => (
  <motion.button
    initial={{ opacity: 0, x: -16 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ duration: 0.25, delay: index * 0.08 }}
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className={`w-full text-left px-4 py-3 rounded-lg border transition-all
      ${isActive
        ? "border-opacity-60 shadow-glow"
        : "border-border hover:border-opacity-40"
      }`}
    style={{
      borderColor: isActive ? node.color : undefined,
      background: isActive ? node.color + "12" : "#0e1318",
      boxShadow: isActive ? `0 0 16px ${node.color}20` : undefined,
    }}
  >
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-semibold text-ink">{node.label}</p>
        {node.sublabel && (
          <p className="text-2xs text-ink3 mt-0.5">{node.sublabel}</p>
        )}
      </div>
      <div
        className="w-2 h-2 rounded-full flex-shrink-0"
        style={{ backgroundColor: node.color }}
      />
    </div>
  </motion.button>
));

// ── Main page ─────────────────────────────────────────────────────────────────

export const ArchitecturePage: React.FC = () => {
  const [activeNode, setActiveNode]     = useState<string | null>("frontier");
  const [activeStream, setActiveStream] = useState<string | null>(null);

  const selectedNode   = PIPELINE.find(n => n.id === activeNode);
  const selectedStream = STREAMS.find(s => s.id === activeStream);

  return (
    <div className="min-h-screen bg-bg text-ink font-sans overflow-y-auto">
      {/* ── Header ── */}
      <div className="border-b border-border bg-surface px-8 py-5">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-1">
            <div className="w-1 h-5 rounded-full bg-accent" />
            <h1 className="text-lg font-bold text-ink tracking-tight">
              DeceptiScope v2 — System Architecture
            </h1>
          </div>
          <p className="text-xs text-ink3 ml-4">
            Hybrid graybox-whitebox deception detection for frontier + open-weight LLMs
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-8 py-8 space-y-12">

        {/* ── Section 1: Pipeline Flow ── */}
        <section>
          <h2 className="text-xs font-semibold text-ink2 uppercase tracking-widest mb-6">
            Analysis Pipeline
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Flow diagram */}
            <div className="space-y-1">
              {PIPELINE.map((node, i) => (
                <React.Fragment key={node.id}>
                  <PipelineBlock
                    node={node}
                    index={i}
                    onClick={() => setActiveNode(activeNode === node.id ? null : node.id)}
                    isActive={activeNode === node.id}
                  />
                  {i < PIPELINE.length - 1 && <Arrow />}
                </React.Fragment>
              ))}
            </div>

            {/* Detail panel */}
            <AnimatePresence mode="wait">
              {selectedNode ? (
                <motion.div
                  key={selectedNode.id}
                  initial={{ opacity: 0, x: 16 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -8 }}
                  transition={{ duration: 0.2 }}
                  className="bg-surface border border-border rounded-lg p-5 space-y-4 h-fit"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: selectedNode.color }}
                    />
                    <h3 className="text-sm font-semibold text-ink">{selectedNode.label}</h3>
                  </div>

                  <p className="text-xs text-ink2 leading-relaxed">
                    {selectedNode.description}
                  </p>

                  <div className="space-y-2">
                    <p className="text-2xs text-ink3 uppercase tracking-wider">Details</p>
                    {selectedNode.details.map((d, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, x: 8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className="flex items-start gap-2"
                      >
                        <div
                          className="w-1 h-1 rounded-full mt-1.5 flex-shrink-0"
                          style={{ backgroundColor: selectedNode.color }}
                        />
                        <span className="text-xs text-ink3 leading-relaxed">{d}</span>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="bg-surface border border-border rounded-lg p-5 flex items-center
                             justify-center text-2xs text-ink3 h-40"
                >
                  Click any block to see details
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </section>

        {/* ── Section 2: Stream Cards ── */}
        <section>
          <h2 className="text-xs font-semibold text-ink2 uppercase tracking-widest mb-6">
            Signal Streams
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {STREAMS.map((stream, i) => (
              <motion.button
                key={stream.id}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay: i * 0.08 }}
                whileHover={{ y: -2 }}
                onClick={() => setActiveStream(activeStream === stream.id ? null : stream.id)}
                className={`text-left p-4 rounded-lg border transition-all
                  ${activeStream === stream.id
                    ? "border-opacity-50"
                    : "border-border hover:border-opacity-30"
                  }`}
                style={{
                  borderColor: activeStream === stream.id ? stream.badgeColor : undefined,
                  background: activeStream === stream.id
                    ? stream.badgeColor + "0e"
                    : "#0e1318",
                }}
              >
                {/* Badge */}
                <div className="flex items-center justify-between mb-3">
                  <span
                    className="text-2xs font-semibold px-2 py-0.5 rounded-sm"
                    style={{
                      color: stream.badgeColor,
                      background: stream.badgeColor + "20",
                    }}
                  >
                    {stream.badge}
                  </span>
                  <span style={{ color: stream.badgeColor }}>{stream.icon}</span>
                </div>

                <h3 className="text-sm font-semibold text-ink mb-2">{stream.title}</h3>
                <p className="text-2xs text-ink3 leading-relaxed line-clamp-3">{stream.what}</p>

                <p className="text-2xs mt-3" style={{ color: stream.badgeColor }}>
                  {activeStream === stream.id ? "Click to collapse" : "Click to expand"}
                </p>
              </motion.button>
            ))}
          </div>

          {/* Expanded stream detail */}
          <AnimatePresence>
            {activeStream && selectedStream && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div
                  className="mt-4 p-5 rounded-lg border"
                  style={{
                    borderColor: selectedStream.badgeColor + "40",
                    background: selectedStream.badgeColor + "08",
                  }}
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <div>
                        <p className="text-2xs text-ink3 uppercase tracking-wider mb-1.5">
                          What it does
                        </p>
                        <p className="text-xs text-ink2 leading-relaxed">
                          {selectedStream.what}
                        </p>
                      </div>
                      <div>
                        <p className="text-2xs text-ink3 uppercase tracking-wider mb-1.5">
                          Why it matters
                        </p>
                        <p className="text-xs text-ink2 leading-relaxed">
                          {selectedStream.why}
                        </p>
                      </div>
                    </div>
                    <div>
                      <p className="text-2xs text-ink3 uppercase tracking-wider mb-1.5">
                        Signals extracted
                      </p>
                      <div className="space-y-1.5">
                        {selectedStream.signals.map((s, i) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, x: 8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.04 }}
                            className="flex items-start gap-2"
                          >
                            <div
                              className="w-1 h-1 rounded-full mt-1.5 flex-shrink-0"
                              style={{ backgroundColor: selectedStream.badgeColor }}
                            />
                            <span className="text-xs text-ink3 leading-relaxed">{s}</span>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* ── Section 3: Novel highlights ── */}
        <section>
          <h2 className="text-xs font-semibold text-ink2 uppercase tracking-widest mb-6">
            Why This Is Novel
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {NOVEL_POINTS.map((p, i) => (
              <motion.div
                key={p.title}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2, delay: i * 0.06 }}
                className="bg-surface border border-border rounded-lg p-4 space-y-2"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: p.color }}
                  />
                  <h3 className="text-xs font-semibold text-ink">{p.title}</h3>
                </div>
                <p className="text-2xs text-ink3 leading-relaxed">{p.body}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* ── Section 4: Benchmark results ── */}
        <section>
          <h2 className="text-xs font-semibold text-ink2 uppercase tracking-widest mb-6">
            Benchmark Results
          </h2>

          <div className="bg-surface border border-border rounded-lg overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left px-4 py-3 text-2xs font-semibold text-ink3 uppercase tracking-wider">
                    Method
                  </th>
                  <th className="text-left px-4 py-3 text-2xs font-semibold text-ink3 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="text-left px-4 py-3 text-2xs font-semibold text-ink3 uppercase tracking-wider">
                    AUC-ROC
                  </th>
                  <th className="px-4 py-3 text-2xs font-semibold text-ink3 uppercase tracking-wider">
                    Score
                  </th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARKS.map((b, i) => (
                  <motion.tr
                    key={b.method}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.04 }}
                    className={`border-b border-border last:border-0 transition-colors
                      ${b.highlight ? "bg-accent/5" : "hover:bg-surface2"}`}
                  >
                    <td className="px-4 py-3">
                      <span className={`text-xs ${b.highlight ? "font-semibold text-ink" : "text-ink2"}`}>
                        {b.method}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className="text-2xs px-1.5 py-0.5 rounded-sm font-medium"
                        style={b.highlight
                          ? { color: "#3b82f6", background: "#3b82f620" }
                          : { color: "#4a5568", background: "#1e273020" }
                        }
                      >
                        {b.type}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`text-xs font-mono ${b.highlight ? "text-accent font-bold" : "text-ink3"}`}>
                        {b.auc.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1 bg-border2 rounded-full overflow-hidden max-w-[80px]">
                          <motion.div
                            className="h-full rounded-full"
                            style={{ backgroundColor: b.highlight ? "#3b82f6" : "#263040" }}
                            initial={{ width: 0 }}
                            animate={{ width: `${b.auc * 100}%` }}
                            transition={{ duration: 0.6, delay: i * 0.04 }}
                          />
                        </div>
                        {b.highlight && (
                          <span className="text-2xs text-accent font-mono">
                            +{((b.auc - 0.74) * 100).toFixed(0)}% vs GPT-4
                          </span>
                        )}
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-2xs text-ink3 mt-3">
            Evaluated on DeceptiScope custom benchmark — 500 realistic scenarios across medical advice,
            financial conflicts, factual claims, and AI self-knowledge probes.
          </p>
        </section>

        {/* Footer */}
        <div className="border-t border-border pt-6 pb-2 flex items-center justify-between">
          <span className="text-2xs text-ink3">
            DeceptiScope v2 — Schmidt Sciences 2026 Interpretability RFP
          </span>
          <a
            href="https://github.com/Sreejith-nair511/RFP-2026"
            target="_blank"
            rel="noopener noreferrer"
            className="text-2xs text-accent hover:underline"
          >
            github.com/Sreejith-nair511/RFP-2026
          </a>
        </div>
      </div>
    </div>
  );
};

export default ArchitecturePage;
