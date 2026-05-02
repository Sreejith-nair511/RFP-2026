/**
 * DeceptionScoreGauge — enhanced animated radial gauge.
 *
 * Improvements over v1:
 *  - Smooth spring animation on score change (not just initial mount)
 *  - Color zones painted on the track (green/yellow/orange/red)
 *  - Tooltip explaining what the score means
 *  - Risk level badge with background fill
 *  - Confidence outer ring with label
 *  - Type breakdown bars with animated fills
 *  - Accessible title/desc for screen readers
 */

import React, { memo, useState } from "react";
import { motion, AnimatePresence, useSpring, useTransform, useMotionValue } from "framer-motion";
import { DeceptionResult, TYPE_COLOR, TYPE_LABEL, TYPE_SIGIL, scoreToColor, scoreToLabel } from "../types";

interface Props { result: DeceptionResult | null; isLoading?: boolean; }

// ── Arc math ─────────────────────────────────────────────────────────────────
function arcPath(pct: number, r: number, cx: number, cy: number): string {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const clamped = Math.max(0.001, Math.min(0.999, pct));
  const s = -180, e = s + clamped * 180;
  const x1 = cx + r * Math.cos(toRad(s)), y1 = cy + r * Math.sin(toRad(s));
  const x2 = cx + r * Math.cos(toRad(e)), y2 = cy + r * Math.sin(toRad(e));
  return `M ${x1} ${y1} A ${r} ${r} 0 ${clamped > 0.5 ? 1 : 0} 1 ${x2} ${y2}`;
}

// Color zone arc — from startPct to endPct
function zoneArc(startPct: number, endPct: number, r: number, cx: number, cy: number): string {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const s = -180 + startPct * 180, e = -180 + endPct * 180;
  const x1 = cx + r * Math.cos(toRad(s)), y1 = cy + r * Math.sin(toRad(s));
  const x2 = cx + r * Math.cos(toRad(e)), y2 = cy + r * Math.sin(toRad(e));
  const large = (endPct - startPct) > 0.5 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
}

const ZONES = [
  { start: 0,    end: 0.25, color: "#22c55e", label: "Honest"    },
  { start: 0.25, end: 0.5,  color: "#eab308", label: "Moderate"  },
  { start: 0.5,  end: 0.75, color: "#f97316", label: "High Risk" },
  { start: 0.75, end: 1.0,  color: "#ef4444", label: "Deceptive" },
];

// ── Component ─────────────────────────────────────────────────────────────────
export const DeceptionScoreGauge: React.FC<Props> = memo(({ result, isLoading }) => {
  const [showTooltip, setShowTooltip] = useState(false);
  const score    = result?.score ?? 0;
  const color    = scoreToColor(score);
  const label    = scoreToLabel(score);
  const dominant = result?.deception_type ?? "none";
  const cx = 100, cy = 90, r = 70;

  const topTypes = Object.entries(result?.type_scores ?? {})
    .filter(([t]) => t !== "none")
    .sort(([, a], [, b]) => b - a)
    .slice(0, 4);

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">
            Deception Score
          </span>
          {/* Info tooltip trigger */}
          <button
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            className="relative w-3.5 h-3.5 rounded-full border border-ink3 flex items-center
                       justify-center text-ink3 hover:border-accent hover:text-accent transition-colors"
            aria-label="Score explanation"
          >
            <span className="text-[8px] font-bold leading-none">?</span>
            <AnimatePresence>
              {showTooltip && (
                <motion.div
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="absolute left-5 top-0 z-50 bg-bg border border-border rounded-md
                             px-2.5 py-2 text-2xs text-ink3 shadow-panel w-52 leading-relaxed
                             pointer-events-none"
                >
                  Score represents the probability of deceptive behavior based on multi-signal
                  analysis: consistency sampling, linguistic markers, and fusion scoring.
                </motion.div>
              )}
            </AnimatePresence>
          </button>
        </div>

        <AnimatePresence mode="wait">
          {result && (
            <motion.div
              key={dominant}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 px-2 py-0.5 rounded-sm"
              style={{ background: TYPE_COLOR[dominant] + "18" }}
            >
              <span
                className="text-2xs font-mono font-bold"
                style={{ color: TYPE_COLOR[dominant] }}
              >
                {TYPE_SIGIL[dominant]}
              </span>
              <span className="text-2xs font-medium" style={{ color: TYPE_COLOR[dominant] }}>
                {TYPE_LABEL[dominant]}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="p-3 space-y-3">
        {/* SVG Gauge */}
        <div className="flex justify-center relative">
          <svg viewBox="0 0 200 102" className="w-full max-w-[210px]" role="img"
            aria-label={`Deception score: ${Math.round(score * 100)}%`}>
            <defs>
              <filter id="gauge-glow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Color zone tracks (faint) */}
            {ZONES.map(z => (
              <path
                key={z.label}
                d={zoneArc(z.start, z.end, r, cx, cy)}
                fill="none"
                stroke={z.color}
                strokeWidth="12"
                strokeLinecap="butt"
                opacity={0.12}
              />
            ))}

            {/* Dark base track */}
            <path
              d={arcPath(1, r, cx, cy)}
              fill="none"
              stroke="#1e2730"
              strokeWidth="10"
              strokeLinecap="round"
            />

            {/* Score fill — animated */}
            <motion.path
              d={arcPath(Math.max(score, 0.005), r, cx, cy)}
              fill="none"
              stroke={color}
              strokeWidth="10"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: score }}
              transition={{ duration: 1.0, ease: [0.16, 1, 0.3, 1] }}
            />

            {/* Glow on fill */}
            <motion.path
              d={arcPath(Math.max(score, 0.005), r, cx, cy)}
              fill="none"
              stroke={color}
              strokeWidth="4"
              strokeLinecap="round"
              opacity={0.4}
              filter="url(#gauge-glow)"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: score }}
              transition={{ duration: 1.0, ease: [0.16, 1, 0.3, 1] }}
            />

            {/* Confidence outer ring */}
            {result && result.confidence > 0 && (
              <motion.path
                d={arcPath(result.confidence, r + 11, cx, cy)}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="2"
                strokeLinecap="round"
                opacity={0.35}
                initial={{ pathLength: 0 }}
                animate={{ pathLength: result.confidence }}
                transition={{ duration: 0.8, delay: 0.3 }}
              />
            )}

            {/* Tick marks at zone boundaries */}
            {[0, 0.25, 0.5, 0.75, 1].map(v => {
              const angle = -180 + v * 180;
              const rad = (angle * Math.PI) / 180;
              const x1 = cx + (r - 7) * Math.cos(rad), y1 = cy + (r - 7) * Math.sin(rad);
              const x2 = cx + (r - 2) * Math.cos(rad), y2 = cy + (r - 2) * Math.sin(rad);
              return (
                <line key={v} x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke="#263040" strokeWidth="1.5" />
              );
            })}

            {/* Zone labels at bottom */}
            {[
              { v: 0,    txt: "0" },
              { v: 0.5,  txt: "50" },
              { v: 1,    txt: "100" },
            ].map(({ v, txt }) => {
              const angle = -180 + v * 180;
              const rad = (angle * Math.PI) / 180;
              const lx = cx + (r + 18) * Math.cos(rad);
              const ly = cy + (r + 18) * Math.sin(rad);
              return (
                <text key={v} x={lx} y={ly} textAnchor="middle"
                  fill="#4a5568" fontSize="7" fontFamily="Inter, sans-serif">
                  {txt}
                </text>
              );
            })}

            {/* Score number */}
            <motion.text
              x={cx} y={cy + 4}
              textAnchor="middle"
              fill={isLoading ? "#4a5568" : color}
              fontSize="30"
              fontWeight="700"
              fontFamily="'JetBrains Mono', monospace"
              letterSpacing="-1"
              animate={{ fill: isLoading ? "#4a5568" : color }}
              transition={{ duration: 0.4 }}
            >
              {isLoading ? "—" : Math.round(score * 100)}
            </motion.text>

            {/* Risk label */}
            <text x={cx} y={cy + 18} textAnchor="middle"
              fill="#4a5568" fontSize="7.5" fontWeight="600"
              fontFamily="Inter, sans-serif" letterSpacing="1.5">
              {isLoading ? "ANALYSING" : label.toUpperCase()}
            </text>

            {/* Confidence label */}
            {result && result.confidence > 0 && (
              <text x={cx} y={cy + 28} textAnchor="middle"
                fill="#3b82f6" fontSize="6.5" fontFamily="Inter, sans-serif" opacity={0.6}>
                {(result.confidence * 100).toFixed(0)}% conf
              </text>
            )}
          </svg>
        </div>

        {/* Risk level badge */}
        {result && (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-center"
          >
            <div
              className="px-3 py-1 rounded-md text-xs font-semibold"
              style={{ color, background: color + "18", border: `1px solid ${color}30` }}
            >
              {label} Risk
            </div>
          </motion.div>
        )}

        {/* Type breakdown bars */}
        {topTypes.length > 0 && (
          <div className="space-y-1.5 pt-2 border-t border-border">
            <span className="text-2xs text-ink3 uppercase tracking-wider">Type Breakdown</span>
            {topTypes.map(([type, val], i) => (
              <div key={type} className="flex items-center gap-2">
                <span
                  className="text-2xs font-mono font-bold w-4 text-center flex-shrink-0"
                  style={{ color: TYPE_COLOR[type] }}
                >
                  {TYPE_SIGIL[type]}
                </span>
                <span className="text-2xs text-ink3 w-[72px] truncate flex-shrink-0">
                  {TYPE_LABEL[type]}
                </span>
                <div className="flex-1 h-1.5 bg-border2 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: TYPE_COLOR[type] }}
                    initial={{ width: 0 }}
                    animate={{ width: `${val * 100}%` }}
                    transition={{ duration: 0.5, delay: i * 0.06, ease: "easeOut" }}
                  />
                </div>
                <span className="text-2xs font-mono text-ink3 w-6 text-right flex-shrink-0">
                  {(val * 100).toFixed(0)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Explanation */}
        {result?.explanation && (
          <p className="text-2xs text-ink3 leading-relaxed pt-2 border-t border-border">
            {result.explanation}
          </p>
        )}
      </div>
    </div>
  );
});
