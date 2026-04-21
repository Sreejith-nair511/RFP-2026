/**
 * Shared TypeScript types for DeceptiScope v2 dashboard.
 * Mirrors the Python dataclasses returned by the backend WebSocket.
 */

export type DeceptionType =
  | "factual_error"
  | "omission"
  | "overconfidence"
  | "sycophancy"
  | "evasion"
  | "contradiction"
  | "false_expertise"
  | "none";

export interface TokenRisk {
  index: number;
  token: string;
  risk_score: number;
}

export interface DeceptionResult {
  score: number;                          // 0–1 calibrated probability
  deception_type: DeceptionType;
  confidence: number;                     // epistemic confidence in score
  explanation: string;
  per_token_scores: number[];
  high_risk_tokens: TokenRisk[];
  type_scores: Record<DeceptionType, number>;
  signal_contributions: Record<string, number>;
  raw_signals: Record<string, unknown>;
}

export interface BehavioralSignals {
  entropy?: number;
  consistency?: number;
  cot_contradiction?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  deception?: DeceptionResult;
  tokens?: string[];
}

export type ModelProvider = "openai" | "anthropic" | "gemini";

export interface ModelOption {
  provider: ModelProvider;
  model: string;
  label: string;
  supportsLogprobs: boolean;
  supportsCoT: boolean;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { provider: "openai",    model: "gpt-4o",            label: "GPT-4o",            supportsLogprobs: true,  supportsCoT: false },
  { provider: "openai",    model: "gpt-5-preview",     label: "GPT-5 Preview",     supportsLogprobs: true,  supportsCoT: true  },
  { provider: "anthropic", model: "claude-opus-4-6",   label: "Claude Opus 4.6",   supportsLogprobs: false, supportsCoT: true  },
  { provider: "anthropic", model: "claude-sonnet-4-6", label: "Claude Sonnet 4.6", supportsLogprobs: false, supportsCoT: true  },
  { provider: "gemini",    model: "gemini-2.5-pro",    label: "Gemini 2.5 Pro",    supportsLogprobs: false, supportsCoT: false },
];

export const DECEPTION_TYPE_COLORS: Record<DeceptionType, string> = {
  factual_error:   "#f85149",
  omission:        "#d29922",
  overconfidence:  "#ff7b72",
  sycophancy:      "#bc8cff",
  evasion:         "#79c0ff",
  contradiction:   "#ffa657",
  false_expertise: "#f0883e",
  none:            "#3fb950",
};

export const DECEPTION_TYPE_LABELS: Record<DeceptionType, string> = {
  factual_error:   "Factual Error",
  omission:        "Omission",
  overconfidence:  "Overconfidence",
  sycophancy:      "Sycophancy",
  evasion:         "Evasion",
  contradiction:   "Contradiction",
  false_expertise: "False Expertise",
  none:            "Honest",
};
