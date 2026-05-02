/**
 * DeceptionHeatmap — enhanced per-token risk overlay.
 *
 * Improvements:
 *  - Richer tooltip: shows risk %, signal contribution label, and risk tier
 *  - Smoother gradient interpolation (HSL-based)
 *  - No layout shift: tooltip is fixed-position relative to viewport
 *  - High-risk words get a subtle underline pulse instead of ring
 *  - Works correctly with streaming text (no key collisions)
 */

import React, { useState, useRef, useCallback, memo } from "react";
import { motion } from "framer-motion";
import { ChatMessage, TYPE_COLOR, TYPE_LABEL } from "../types";

interface Props { message: ChatMessage; }

// ── Color math ────────────────────────────────────────────────────────────────
// Green (#22c55e) → Yellow (#eab308) → Orange (#f97316) → Red (#ef4444)
function riskToColor(r: number): string {
  if (r < 0.33) {
    // green → yellow
    const t = r / 0.33;
    const rv = Math.round(34  + t * (234 - 34));
    const g  = Math.round(197 + t * (179 - 197));
    const b  = Math.round(94  + t * (8   - 94));
    return `rgb(${rv},${g},${b})`;
  }
  if (r < 0.66) {
    // yellow → orange
    const t = (r - 0.33) / 0.33;
    const rv = Math.round(234 + t * (249 - 234));
    const g  = Math.round(179 + t * (115 - 179));
    const b  = Math.round(8   + t * (22  - 8));
    return `rgb(${rv},${g},${b})`;
  }
  // orange → red
  const t = (r - 0.66) / 0.34;
  const rv = Math.round(249 + t * (239 - 249));
  const g  = Math.round(115 + t * (68  - 115));
  const b  = Math.round(22  + t * (68  - 22));
  return `rgb(${rv},${g},${b})`;
}

function riskTier(r: number): string {
  if (r < 0.25) return "Low";
  if (r < 0.5)  return "Moderate";
  if (r < 0.75) return "High";
  return "Critical";
}

// Map risk score to a human-readable signal label
function signalLabel(r: number, token: string): string {
  const t = token.toLowerCase().replace(/[.,!?;:'"]/g, "");
  const overconfident = ["definitely", "certainly", "absolutely", "guaranteed",
    "always", "never", "impossible", "100%", "undoubtedly"];
  const hedging = ["might", "could", "perhaps", "possibly", "maybe", "likely", "probably"];
  if (overconfident.includes(t)) return "Overconfident language";
  if (hedging.includes(t))       return "Hedging / uncertainty";
  if (r > 0.7)  return "High deception risk";
  if (r > 0.5)  return "Moderate risk signal";
  if (r < 0.2)  return "Low risk";
  return "Neutral";
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  token: string;
  risk: number;
}

export const DeceptionHeatmap: React.FC<Props> = memo(({ message }) => {
  const [tip, setTip] = useState<TooltipState>({
    visible: false, x: 0, y: 0, token: "", risk: 0,
  });

  const { content, deception } = message;
  const scores = deception?.per_token_scores ?? [];

  // Split into word + whitespace tokens
  const words = content.split(/(\s+)/);
  let wi = 0;
  const wordScores = words.map(w => {
    if (/^\s+$/.test(w)) return 0;
    return scores[wi++] ?? (deception?.score ?? 0);
  });

  const onEnter = useCallback((
    e: React.MouseEvent<HTMLSpanElement>,
    token: string,
    risk: number,
  ) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setTip({ visible: true, x: rect.left, y: rect.top - 44, token, risk });
  }, []);

  const onLeave = useCallback(() => {
    setTip(t => ({ ...t, visible: false }));
  }, []);

  // Dominant deception type for context
  const domType = deception?.deception_type;

  return (
    <div className="relative">
      <p className="text-sm leading-7 text-ink font-mono whitespace-pre-wrap break-words select-text">
        {words.map((word, i) => {
          if (/^\s+$/.test(word)) return <span key={`ws-${i}`}>{word}</span>;

          const risk = wordScores[i] ?? 0;
          const color = riskToColor(risk);
          const bgAlpha = Math.round((0.05 + risk * 0.38) * 255)
            .toString(16).padStart(2, "0");
          const isHigh = risk > 0.65;

          return (
            <motion.span
              key={`w-${i}`}
              initial={{ backgroundColor: "transparent" }}
              animate={{ backgroundColor: `${color}${bgAlpha}` }}
              transition={{ duration: 0.35, delay: i * 0.002 }}
              style={{
                color,
                borderRadius: "3px",
                padding: "1px 2px",
                cursor: "help",
                textDecoration: isHigh ? `underline ${color}60` : "none",
                textDecorationStyle: isHigh ? "dotted" : undefined,
                textUnderlineOffset: "3px",
              }}
              onMouseEnter={e => onEnter(e, word, risk)}
              onMouseLeave={onLeave}
            >
              {word}
            </motion.span>
          );
        })}
      </p>

      {/* Fixed-position tooltip — no layout shift */}
      {tip.visible && (
        <div
          className="fixed z-[9999] bg-bg border border-border rounded-md px-2.5 py-2
                     text-2xs shadow-panel pointer-events-none"
          style={{ left: tip.x, top: tip.y, transform: "translateY(-4px)" }}
        >
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono font-semibold" style={{ color: riskToColor(tip.risk) }}>
              {(tip.risk * 100).toFixed(1)}%
            </span>
            <span
              className="px-1 py-px rounded-sm text-2xs font-medium"
              style={{
                color: riskToColor(tip.risk),
                background: riskToColor(tip.risk) + "20",
              }}
            >
              {riskTier(tip.risk)}
            </span>
          </div>
          <p className="text-ink3">{signalLabel(tip.risk, tip.token)}</p>
          {domType && domType !== "none" && (
            <p className="text-ink3 mt-0.5">
              Context:{" "}
              <span style={{ color: TYPE_COLOR[domType] }}>
                {TYPE_LABEL[domType]}
              </span>
            </p>
          )}
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-2 mt-3 pt-2 border-t border-border">
        <span className="text-2xs text-ink3 flex-shrink-0">Risk</span>
        <div className="flex gap-px flex-1">
          {Array.from({ length: 28 }, (_, i) => i / 27).map((v, i) => (
            <div
              key={i}
              className="flex-1 h-1.5 rounded-sm"
              style={{ backgroundColor: riskToColor(v) }}
            />
          ))}
        </div>
        <div className="flex gap-3 flex-shrink-0">
          {["Low", "High"].map((l, i) => (
            <span key={l} className="text-2xs text-ink3">{l}</span>
          ))}
        </div>
        {deception && deception.high_risk_tokens.length > 0 && (
          <span className="text-2xs text-ink3 flex-shrink-0 ml-auto">
            {deception.high_risk_tokens.length} flagged
          </span>
        )}
      </div>
    </div>
  );
});
