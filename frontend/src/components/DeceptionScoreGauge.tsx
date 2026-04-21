/**
 * DeceptionScoreGauge — animated radial gauge showing the 0–1 deception score.
 *
 * Colour transitions: green (honest) → yellow (moderate) → red (deceptive).
 * Also shows the dominant deception type and confidence band.
 */

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  DeceptionResult,
  DECEPTION_TYPE_COLORS,
  DECEPTION_TYPE_LABELS,
} from "../types";

interface Props {
  result: DeceptionResult | null;
  isLoading?: boolean;
}

function scoreToColor(score: number): string {
  if (score < 0.3) return "#3fb950";   // green
  if (score < 0.55) return "#d29922";  // yellow
  if (score < 0.75) return "#f0883e";  // orange
  return "#f85149";                    // red
}

function scoreToLabel(score: number): string {
  if (score < 0.25) return "HONEST";
  if (score < 0.55) return "MODERATE";
  if (score < 0.75) return "HIGH RISK";
  return "DECEPTIVE";
}

/** Converts a 0–1 score to SVG arc path for a semicircle gauge. */
function arcPath(score: number, r: number, cx: number, cy: number): string {
  const startAngle = -180;
  const endAngle   = startAngle + score * 180;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const x1 = cx + r * Math.cos(toRad(startAngle));
  const y1 = cy + r * Math.sin(toRad(startAngle));
  const x2 = cx + r * Math.cos(toRad(endAngle));
  const y2 = cy + r * Math.sin(toRad(endAngle));
  const largeArc = score > 0.5 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}

export const DeceptionScoreGauge: React.FC<Props> = ({ result, isLoading }) => {
  const score = result?.score ?? 0;
  const color = scoreToColor(score);
  const label = scoreToLabel(score);
  const dominantType = result?.deception_type ?? "none";

  const cx = 100, cy = 100, r = 70;
  const trackPath = arcPath(1, r, cx, cy);
  const fillPath  = arcPath(score, r, cx, cy);

  return (
    <div className="bg-ds-surface border border-ds-border rounded-xl p-4 flex flex-col items-center gap-3">
      <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider self-start">
        Deception Score
      </h2>

      {/* SVG Gauge */}
      <div className="relative w-48 h-28">
        <svg viewBox="0 0 200 110" className="w-full h-full">
          {/* Track */}
          <path
            d={trackPath}
            fill="none"
            stroke="#30363d"
            strokeWidth="14"
            strokeLinecap="round"
          />
          {/* Fill — animated */}
          <motion.path
            d={fillPath}
            fill="none"
            stroke={color}
            strokeWidth="14"
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: score }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
          {/* Score text */}
          <text
            x={cx}
            y={cy + 10}
            textAnchor="middle"
            fill={color}
            fontSize="28"
            fontWeight="bold"
            fontFamily="monospace"
          >
            {isLoading ? "…" : (score * 100).toFixed(0)}
          </text>
          <text
            x={cx}
            y={cy + 28}
            textAnchor="middle"
            fill="#8b949e"
            fontSize="10"
          >
            {isLoading ? "analysing" : label}
          </text>
        </svg>
      </div>

      {/* Dominant type badge */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            key={dominantType}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="flex items-center gap-2"
          >
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: DECEPTION_TYPE_COLORS[dominantType] }}
            />
            <span className="text-sm font-medium text-ds-text">
              {DECEPTION_TYPE_LABELS[dominantType]}
            </span>
            <span className="text-xs text-ds-muted">
              ({(result.confidence * 100).toFixed(0)}% conf.)
            </span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Type breakdown mini-bars */}
      {result && (
        <div className="w-full space-y-1 pt-1">
          {Object.entries(result.type_scores)
            .filter(([t]) => t !== "none")
            .sort(([, a], [, b]) => b - a)
            .slice(0, 4)
            .map(([type, val]) => (
              <div key={type} className="flex items-center gap-2">
                <span className="text-xs text-ds-muted w-24 truncate">
                  {DECEPTION_TYPE_LABELS[type as keyof typeof DECEPTION_TYPE_LABELS] ?? type}
                </span>
                <div className="flex-1 h-1.5 bg-ds-bg rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: DECEPTION_TYPE_COLORS[type as keyof typeof DECEPTION_TYPE_COLORS] ?? "#8b949e" }}
                    initial={{ width: 0 }}
                    animate={{ width: `${val * 100}%` }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                  />
                </div>
                <span className="text-xs text-ds-muted w-8 text-right">
                  {(val * 100).toFixed(0)}%
                </span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};
