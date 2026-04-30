import React, { useState, useRef } from "react";
import { motion } from "framer-motion";
import { ChatMessage } from "../types";

interface Props { message: ChatMessage; }

function riskToRgb(r: number): [number, number, number] {
  if (r < 0.5) {
    const t = r / 0.5;
    return [
      Math.round(34  + t * (234 - 34)),
      Math.round(197 + t * (179 - 197)),
      Math.round(94  + t * (8   - 94)),
    ];
  }
  const t = (r - 0.5) / 0.5;
  return [
    Math.round(234 + t * (239 - 234)),
    Math.round(179 + t * (68  - 179)),
    Math.round(8   + t * (68  - 8)),
  ];
}

function riskColor(r: number): string {
  const [rv, g, b] = riskToRgb(r);
  return `rgb(${rv},${g},${b})`;
}

interface Tip { visible: boolean; x: number; y: number; token: string; risk: number; }

export const DeceptionHeatmap: React.FC<Props> = ({ message }) => {
  const [tip, setTip] = useState<Tip>({ visible: false, x: 0, y: 0, token: "", risk: 0 });
  const ref = useRef<HTMLDivElement>(null);
  const { content, deception } = message;
  const scores = deception?.per_token_scores ?? [];

  const words = content.split(/(\s+)/);
  let wi = 0;
  const wordScores = words.map(w => {
    if (/^\s+$/.test(w)) return 0;
    return scores[wi++] ?? (deception?.score ?? 0);
  });

  const onEnter = (e: React.MouseEvent<HTMLSpanElement>, token: string, risk: number) => {
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const cRect = ref.current?.getBoundingClientRect();
    setTip({ visible: true, x: rect.left - (cRect?.left ?? 0), y: rect.top - (cRect?.top ?? 0) - 30, token, risk });
  };

  return (
    <div ref={ref} className="relative">
      <p className="text-sm leading-7 text-ink font-mono whitespace-pre-wrap break-words">
        {words.map((word, i) => {
          if (/^\s+$/.test(word)) return <span key={i}>{word}</span>;
          const risk = wordScores[i] ?? 0;
          const [rv, g, b] = riskToRgb(risk);
          const bgAlpha = Math.round((0.06 + risk * 0.4) * 255).toString(16).padStart(2, "0");
          return (
            <motion.span
              key={i}
              initial={{ backgroundColor: "transparent" }}
              animate={{ backgroundColor: `rgb(${rv},${g},${b},${bgAlpha})` }}
              transition={{ duration: 0.4, delay: i * 0.003 }}
              style={{
                color: riskColor(risk),
                borderRadius: "3px",
                padding: "1px 2px",
                cursor: risk > 0.6 ? "help" : "default",
              }}
              onMouseEnter={e => onEnter(e, word, risk)}
              onMouseLeave={() => setTip(t => ({ ...t, visible: false }))}
            >
              {word}
            </motion.span>
          );
        })}
      </p>

      {/* Tooltip */}
      {tip.visible && (
        <div
          className="absolute z-50 bg-bg border border-border rounded-md px-2 py-1 text-2xs shadow-panel pointer-events-none whitespace-nowrap"
          style={{ left: tip.x, top: tip.y }}
        >
          <span className="text-ink3">risk </span>
          <span className="font-mono font-semibold" style={{ color: riskColor(tip.risk) }}>
            {(tip.risk * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-2 mt-3 pt-2 border-t border-border">
        <span className="text-2xs text-ink3">Risk</span>
        <div className="flex gap-px">
          {Array.from({ length: 24 }, (_, i) => i / 23).map((v, i) => (
            <div key={i} className="w-2 h-1.5 rounded-sm" style={{ backgroundColor: riskColor(v) }} />
          ))}
        </div>
        <span className="text-2xs text-ink3">low — high</span>
        {deception && deception.high_risk_tokens.length > 0 && (
          <span className="ml-auto text-2xs text-ink3">
            {deception.high_risk_tokens.length} flagged
          </span>
        )}
      </div>
    </div>
  );
};
