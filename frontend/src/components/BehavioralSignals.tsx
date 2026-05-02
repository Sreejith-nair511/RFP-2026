/**
 * BehavioralSignals — dedicated panel showing all real-time behavioral
 * signal values extracted from the latest response.
 *
 * Signals displayed:
 *   consistency        – cross-sample Jaccard similarity (3 samples @ T=0.8)
 *   entropy            – token distribution entropy (logprobs when available)
 *   confidence_mismatch – overconfident language vs. actual uncertainty
 *   sycophancy_score   – explicit agreement phrase density
 *   omission_score     – response brevity relative to prompt complexity
 */

import React, { memo } from "react";
import { motion } from "framer-motion";
import { DeceptionResult, scoreToColor } from "../types";

interface Props {
  result: DeceptionResult | null;
  isLoading?: boolean;
}

interface SignalDef {
  key: string;
  label: string;
  description: string;
  invert?: boolean; // high value = good (e.g. consistency)
}

const SIGNALS: SignalDef[] = [
  {
    key: "consistency",
    label: "Consistency",
    description: "Cross-sample agreement across 3 independent samples at temperature 0.8. Low = unstable answers.",
    invert: true,
  },
  {
    key: "entropy",
    label: "Token Entropy",
    description: "Uncertainty in token distribution. High entropy suggests evasion or confabulation.",
  },
  {
    key: "confidence_mismatch",
    label: "Overconfidence",
    description: "Density of overconfident language ('definitely', 'guaranteed') relative to hedging words.",
  },
  {
    key: "sycophancy_score",
    label: "Sycophancy",
    description: "Density of explicit agreement phrases ('you're right', 'absolutely correct').",
  },
  {
    key: "omission_score",
    label: "Omission Risk",
    description: "Response brevity relative to prompt complexity. Short answers to complex questions.",
  },
];

function signalToRisk(value: number, invert: boolean): number {
  // For inverted signals (consistency), high value = low risk
  return invert ? 1 - value : value;
}

function riskColor(risk: number): string {
  if (risk < 0.25) return "#22c55e";
  if (risk < 0.5)  return "#eab308";
  if (risk < 0.75) return "#f97316";
  return "#ef4444";
}

function riskLabel(risk: number): string {
  if (risk < 0.25) return "Low";
  if (risk < 0.5)  return "Moderate";
  if (risk < 0.75) return "High";
  return "Critical";
}

const SignalRow = memo(({
  def, value, index,
}: {
  def: SignalDef;
  value: number | null;
  index: number;
}) => {
  const risk = value != null ? signalToRisk(value, def.invert ?? false) : null;
  const color = risk != null ? riskColor(risk) : "#4a5568";
  const pct = value != null ? Math.round(value * 100) : null;

  return (
    <div className="group relative">
      <div className="flex items-center gap-2.5">
        {/* Label */}
        <span className="text-2xs text-ink3 w-28 flex-shrink-0 truncate">{def.label}</span>

        {/* Bar track */}
        <div className="flex-1 h-1.5 bg-border2 rounded-full overflow-hidden">
          {value != null ? (
            <motion.div
              className="h-full rounded-full"
              style={{ backgroundColor: color }}
              initial={{ width: 0 }}
              animate={{ width: `${value * 100}%` }}
              transition={{ duration: 0.6, delay: index * 0.05, ease: [0.16, 1, 0.3, 1] }}
            />
          ) : (
            <div className="h-full w-8 bg-border rounded-full animate-pulse" />
          )}
        </div>

        {/* Value */}
        <div className="flex items-center gap-1.5 w-16 flex-shrink-0 justify-end">
          {pct != null && risk != null ? (
            <>
              <span
                className="text-2xs font-mono font-semibold"
                style={{ color }}
              >
                {pct}%
              </span>
              <span
                className="text-2xs px-1 py-px rounded-sm font-medium"
                style={{ color, background: color + "18" }}
              >
                {riskLabel(risk)}
              </span>
            </>
          ) : (
            <span className="text-2xs text-ink3">—</span>
          )}
        </div>
      </div>

      {/* Tooltip on hover */}
      <div className="absolute left-0 top-full mt-1 z-50 hidden group-hover:block
                      bg-bg border border-border rounded-md px-2.5 py-2 text-2xs text-ink3
                      shadow-panel max-w-[220px] leading-relaxed pointer-events-none">
        <span className="text-ink font-medium block mb-0.5">{def.label}</span>
        {def.description}
      </div>
    </div>
  );
});

export const BehavioralSignals: React.FC<Props> = memo(({ result, isLoading }) => {
  const bs = result?.behavioral_signals ?? {};

  // Merge type_scores into behavioral signals for display
  const signalValues: Record<string, number | null> = {
    consistency:          bs.consistency        ?? null,
    entropy:              bs.entropy            ?? null,
    confidence_mismatch:  (result?.type_scores?.overconfidence) ?? null,
    sycophancy_score:     (result?.type_scores?.sycophancy)     ?? null,
    omission_score:       (result?.type_scores?.omission)       ?? null,
  };

  const hasAnyData = Object.values(signalValues).some(v => v != null);

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">
          Behavioral Signals
        </span>
        {result && (
          <span
            className="text-2xs font-mono px-1.5 py-px rounded border"
            style={{
              color: scoreToColor(result.score),
              borderColor: scoreToColor(result.score) + "40",
              background: scoreToColor(result.score) + "12",
            }}
          >
            {(result.score * 100).toFixed(0)}% risk
          </span>
        )}
      </div>

      <div className="p-3 space-y-2.5">
        {isLoading ? (
          // Loading skeleton
          <div className="space-y-2.5">
            {SIGNALS.map((_, i) => (
              <div key={i} className="flex items-center gap-2.5">
                <div className="w-28 h-2 bg-border rounded animate-pulse" />
                <div className="flex-1 h-1.5 bg-border rounded-full animate-pulse" />
                <div className="w-16 h-2 bg-border rounded animate-pulse" />
              </div>
            ))}
          </div>
        ) : !hasAnyData ? (
          <p className="text-2xs text-ink3 text-center py-4">
            Send a message to see behavioral signals.
          </p>
        ) : (
          SIGNALS.map((def, i) => (
            <SignalRow
              key={def.key}
              def={def}
              value={signalValues[def.key]}
              index={i}
            />
          ))
        )}

        {/* Legend */}
        {hasAnyData && (
          <div className="flex items-center gap-3 pt-2 border-t border-border">
            {[
              { label: "Low",      color: "#22c55e" },
              { label: "Moderate", color: "#eab308" },
              { label: "High",     color: "#f97316" },
              { label: "Critical", color: "#ef4444" },
            ].map(l => (
              <div key={l.label} className="flex items-center gap-1">
                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: l.color }} />
                <span className="text-2xs text-ink3">{l.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});
