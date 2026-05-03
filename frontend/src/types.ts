export type DeceptionType =
  | "factual_error" | "omission" | "overconfidence"
  | "sycophancy" | "evasion" | "contradiction"
  | "false_expertise" | "none";

export interface TokenRisk { index: number; token: string; risk_score: number; }

export interface DeceptionBreakdownItem {
  type: string;
  label: string;
  score: number;
  description: string;
  severity: "low" | "moderate" | "high";
}

export interface DeceptionResult {
  score: number;
  deception_type: DeceptionType;
  confidence: number;
  explanation: string;
  per_token_scores: number[];
  high_risk_tokens: TokenRisk[];
  type_scores: Record<string, number>;
  signal_contributions: Record<string, number>;
  raw_signals: Record<string, unknown>;
  behavioral_signals?: { entropy?: number; consistency?: number; cot_contradiction?: number; confidence_mismatch?: number; sycophancy_score?: number; omission_score?: number };
  deception_breakdown?: DeceptionBreakdownItem[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  deception?: DeceptionResult;
  tokens?: string[];
}

export type ModelProvider = "openai" | "anthropic" | "gemini" | "groq";

export interface ModelOption {
  provider: ModelProvider;
  model: string;
  label: string;
  supportsLogprobs: boolean;
  supportsCoT: boolean;
  speed: "fast" | "medium" | "slow";
  tokensPerSec: string;
  unavailable?: boolean;   // true = show in list but disabled with reason
  unavailableReason?: string;
}

export const MODEL_OPTIONS: ModelOption[] = [
  { provider: "groq",     model: "llama-3.3-70b-versatile",                label: "LLaMA 3.3 70B",    supportsLogprobs: false, supportsCoT: false, speed: "fast",   tokensPerSec: "280" },
  { provider: "groq",     model: "llama-3.1-8b-instant",                   label: "LLaMA 3.1 8B",     supportsLogprobs: false, supportsCoT: false, speed: "fast",   tokensPerSec: "560" },
  { provider: "groq",     model: "meta-llama/llama-4-scout-17b-16e-instruct", label: "LLaMA 4 Scout", supportsLogprobs: false, supportsCoT: false, speed: "fast",   tokensPerSec: "750" },
  { provider: "groq",     model: "qwen/qwen3-32b",                         label: "Qwen3 32B",        supportsLogprobs: false, supportsCoT: false, speed: "fast",   tokensPerSec: "400" },
  { provider: "groq",     model: "openai/gpt-oss-20b",                     label: "GPT OSS 20B",      supportsLogprobs: false, supportsCoT: false, speed: "fast",   tokensPerSec: "1000" },
  { provider: "gemini",   model: "gemini-2.5-flash",                       label: "Gemini 2.5 Flash", supportsLogprobs: false, supportsCoT: false, speed: "fast",   tokensPerSec: "—", unavailable: true, unavailableReason: "API key needs project access — enable Gemini API at console.cloud.google.com" },
  { provider: "gemini",   model: "gemini-2.5-pro",                         label: "Gemini 2.5 Pro",   supportsLogprobs: false, supportsCoT: false, speed: "medium", tokensPerSec: "—", unavailable: true, unavailableReason: "API key needs project access — enable Gemini API at console.cloud.google.com" },
  { provider: "openai",   model: "gpt-4o",                                 label: "GPT-4o",           supportsLogprobs: true,  supportsCoT: false, speed: "medium", tokensPerSec: "—" },
  { provider: "anthropic",model: "claude-3-sonnet-4.6",                    label: "Claude Sonnet 4.6",supportsLogprobs: false, supportsCoT: true,  speed: "medium", tokensPerSec: "—" },
];

export const PROVIDER_META: Record<ModelProvider, { label: string; color: string; dim: string }> = {
  groq:      { label: "Groq",      color: "#f97316", dim: "rgba(249,115,22,0.12)" },
  gemini:    { label: "Gemini",    color: "#3b82f6", dim: "rgba(59,130,246,0.12)" },
  openai:    { label: "OpenAI",    color: "#22c55e", dim: "rgba(34,197,94,0.12)"  },
  anthropic: { label: "Anthropic", color: "#a855f7", dim: "rgba(168,85,247,0.12)" },
};

export const TYPE_COLOR: Record<string, string> = {
  factual_error:   "#ef4444",
  omission:        "#eab308",
  overconfidence:  "#f97316",
  sycophancy:      "#a855f7",
  evasion:         "#3b82f6",
  contradiction:   "#f59e0b",
  false_expertise: "#ec4899",
  none:            "#22c55e",
};

export const TYPE_LABEL: Record<string, string> = {
  factual_error:   "Factual Error",
  omission:        "Omission",
  overconfidence:  "Overconfidence",
  sycophancy:      "Sycophancy",
  evasion:         "Evasion",
  contradiction:   "Contradiction",
  false_expertise: "False Expertise",
  none:            "Honest",
};

// Short single-char sigils — no emoji
export const TYPE_SIGIL: Record<string, string> = {
  factual_error:   "F",
  omission:        "O",
  overconfidence:  "C",
  sycophancy:      "S",
  evasion:         "E",
  contradiction:   "X",
  false_expertise: "K",
  none:            "H",
};

export function scoreToColor(s: number): string {
  if (s < 0.25) return "#22c55e";
  if (s < 0.5)  return "#eab308";
  if (s < 0.75) return "#f97316";
  return "#ef4444";
}

export function scoreToLabel(s: number): string {
  if (s < 0.25) return "Honest";
  if (s < 0.5)  return "Moderate";
  if (s < 0.75) return "High Risk";
  return "Deceptive";
}
