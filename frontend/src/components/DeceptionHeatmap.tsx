/**
 * DeceptionHeatmap — renders the assistant response with per-token colour
 * overlays (green → red) based on per_token_scores from the backend.
 *
 * Tokens with risk > 0.7 get a pulsing highlight and tooltip.
 * Falls back to word-level colouring when token boundaries aren't available.
 */

import React, { useState } from "react";
import { motion } from "framer-motion";
import { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
}

/** Interpolate between green (#3fb950) and red (#f85149) via yellow (#d29922). */
function riskToColor(risk: number): string {
  if (risk < 0.5) {
    // green → yellow
    const t = risk / 0.5;
    const r = Math.round(0x3f + t * (0xd2 - 0x3f));
    const g = Math.round(0xb9 + t * (0x99 - 0xb9));
    const b = Math.round(0x50 + t * (0x22 - 0x50));
    return `rgb(${r},${g},${b})`;
  } else {
    // yellow → red
    const t = (risk - 0.5) / 0.5;
    const r = Math.round(0xd2 + t * (0xf8 - 0xd2));
    const g = Math.round(0x99 + t * (0x51 - 0x99));
    const b = Math.round(0x22 + t * (0x49 - 0x22));
    return `rgb(${r},${g},${b})`;
  }
}

function riskToOpacity(risk: number): number {
  // Low-risk tokens are nearly transparent; high-risk are vivid
  return 0.08 + risk * 0.55;
}

interface TooltipState {
  visible: boolean;
  x: number;
  y: number;
  token: string;
  risk: number;
}

export const DeceptionHeatmap: React.FC<Props> = ({ message }) => {
  const [tooltip, setTooltip] = useState<TooltipState>({
    visible: false, x: 0, y: 0, token: "", risk: 0,
  });

  const { content, deception } = message;
  const perTokenScores = deception?.per_token_scores ?? [];

  // Split content into tokens (words + punctuation)
  const words = content.split(/(\s+)/);

  // Map word index → score (skip whitespace tokens)
  let wordIdx = 0;
  const wordScores: number[] = [];
  for (const w of words) {
    if (/^\s+$/.test(w)) {
      wordScores.push(0);
    } else {
      wordScores.push(perTokenScores[wordIdx] ?? (deception?.score ?? 0));
      wordIdx++;
    }
  }

  const handleMouseEnter = (
    e: React.MouseEvent<HTMLSpanElement>,
    token: string,
    risk: number
  ) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    setTooltip({ visible: true, x: rect.left, y: rect.top - 32, token, risk });
  };

  const handleMouseLeave = () =>
    setTooltip((t) => ({ ...t, visible: false }));

  return (
    <div className="relative">
      <p className="text-sm leading-relaxed text-ds-text font-mono whitespace-pre-wrap break-words">
        {words.map((word, i) => {
          if (/^\s+$/.test(word)) return <span key={i}>{word}</span>;

          const risk = wordScores[i] ?? 0;
          const color = riskToColor(risk);
          const opacity = riskToOpacity(risk);
          const isHighRisk = risk > 0.7;

          return (
            <motion.span
              key={i}
              initial={{ backgroundColor: "transparent" }}
              animate={{
                backgroundColor: `rgba(${color.slice(4, -1)}, ${opacity})`,
              }}
              transition={{ duration: 0.4, delay: i * 0.005 }}
              style={{ color, borderRadius: "2px", padding: "0 1px" }}
              className={isHighRisk ? "ring-1 ring-ds-red/40 cursor-help" : ""}
              onMouseEnter={(e) => handleMouseEnter(e, word, risk)}
              onMouseLeave={handleMouseLeave}
            >
              {word}
            </motion.span>
          );
        })}
      </p>

      {/* Floating tooltip */}
      {tooltip.visible && (
        <div
          className="fixed z-50 bg-ds-bg border border-ds-border rounded px-2 py-1 text-xs text-ds-text shadow-lg pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <span className="text-ds-muted">risk: </span>
          <span
            style={{ color: riskToColor(tooltip.risk) }}
            className="font-mono font-bold"
          >
            {(tooltip.risk * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-2 mt-2 pt-2 border-t border-ds-border">
        <span className="text-xs text-ds-muted">Token risk:</span>
        <div className="flex gap-0.5">
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((v) => (
            <div
              key={v}
              className="w-4 h-2 rounded-sm"
              style={{ backgroundColor: riskToColor(v) }}
            />
          ))}
        </div>
        <span className="text-xs text-ds-muted">low → high</span>
      </div>
    </div>
  );
};
