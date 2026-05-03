/**
 * DeceptionBreakdown — shows every deception signal that fired,
 * with label, score bar, severity badge, and plain-English description.
 *
 * This is the "what was detected" panel — the most important panel
 * for understanding WHY a response scored the way it did.
 */

import React, { memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { DeceptionResult, DeceptionBreakdownItem, TYPE_COLOR, TYPE_LABEL, scoreToColor } from "../types";

interface Props {
  result: DeceptionResult | null;
  isLoading?: boolean;
}

const SEVERITY_STYLE: Record<string, { text: string; bg: string }> = {
  low:      { text: "text-emerald-400", bg: "bg-emerald-950/60 border-emerald-800" },
  moderate: { text: "text-amber-400",   bg: "bg-amber-950/60   border-amber-800"   },
  high:     { text: "text-red-400",     bg: "bg-red-950/60     border-red-800"     },
};

const BreakdownItem = memo(({
  item, index,
}: {
  item: DeceptionBreakdownItem;
  index: number;
}) => {
  const color = TYPE_COLOR[item.type] ?? scoreToColor(item.score);
  const sev   = SEVERITY_STYLE[item.severity] ?? SEVERITY_STYLE.moderate;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, delay: index * 0.05 }}
      className="bg-bg border border-border rounded-md p-3 space-y-2"
    >
      {/* Header row */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
          <span className="text-xs font-semibold text-ink truncate">{item.label}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`text-2xs px-1.5 py-px rounded border font-medium ${sev.bg} ${sev.text}`}>
            {item.severity}
          </span>
          <span className="text-2xs font-mono font-bold" style={{ color }}>
            {(item.score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Score bar */}
      <div className="h-1 bg-border2 rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(item.score * 100, 100)}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>

      {/* Description */}
      <p className="text-2xs text-ink3 leading-relaxed">{item.description}</p>
    </motion.div>
  );
});

export const DeceptionBreakdown: React.FC<Props> = memo(({ result, isLoading }) => {
  const items = result?.deception_breakdown ?? [];
  const score = result?.score ?? 0;

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">
          What Was Detected
        </span>
        {result && (
          <span
            className="text-2xs font-mono font-bold px-2 py-0.5 rounded-sm"
            style={{
              color: scoreToColor(score),
              background: scoreToColor(score) + "18",
            }}
          >
            {items.length} signal{items.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      <div className="p-3 space-y-2">
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2].map(i => (
              <div key={i} className="bg-bg border border-border rounded-md p-3 space-y-2 animate-pulse">
                <div className="flex justify-between">
                  <div className="h-3 w-32 bg-border rounded" />
                  <div className="h-3 w-12 bg-border rounded" />
                </div>
                <div className="h-1 bg-border rounded-full" />
                <div className="h-2 w-full bg-border rounded" />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="py-6 text-center space-y-1">
            {result ? (
              <>
                <div
                  className="w-8 h-8 rounded-full mx-auto flex items-center justify-center"
                  style={{ background: "#22c55e18", border: "1px solid #22c55e40" }}
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M2.5 7L5.5 10L11.5 4" stroke="#22c55e" strokeWidth="1.5"
                      strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <p className="text-xs font-medium text-emerald-400">No deception signals detected</p>
                <p className="text-2xs text-ink3">Response appears honest and well-calibrated.</p>
              </>
            ) : (
              <p className="text-2xs text-ink3">Send a message to see deception analysis.</p>
            )}
          </div>
        ) : (
          <AnimatePresence>
            {items.map((item, i) => (
              <BreakdownItem key={item.type + i} item={item} index={i} />
            ))}
          </AnimatePresence>
        )}

        {/* Overall score summary */}
        {result && items.length > 0 && (
          <div
            className="mt-1 pt-2 border-t border-border flex items-center justify-between"
          >
            <span className="text-2xs text-ink3">Overall deception risk</span>
            <div className="flex items-center gap-2">
              <div className="w-20 h-1 bg-border2 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: scoreToColor(score) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${score * 100}%` }}
                  transition={{ duration: 0.6 }}
                />
              </div>
              <span
                className="text-2xs font-mono font-bold"
                style={{ color: scoreToColor(score) }}
              >
                {(score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
