import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { DeceptionResult, TYPE_COLOR, TYPE_LABEL, TYPE_SIGIL, scoreToColor, scoreToLabel } from "../types";

interface Props { result: DeceptionResult | null; isLoading?: boolean; }

function arcPath(pct: number, r: number, cx: number, cy: number): string {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const s = -180, e = s + pct * 180;
  const x1 = cx + r * Math.cos(toRad(s)), y1 = cy + r * Math.sin(toRad(s));
  const x2 = cx + r * Math.cos(toRad(e)), y2 = cy + r * Math.sin(toRad(e));
  return `M ${x1} ${y1} A ${r} ${r} 0 ${pct > 0.5 ? 1 : 0} 1 ${x2} ${y2}`;
}

export const DeceptionScoreGauge: React.FC<Props> = ({ result, isLoading }) => {
  const score = result?.score ?? 0;
  const color = scoreToColor(score);
  const label = scoreToLabel(score);
  const dominant = result?.deception_type ?? "none";
  const cx = 100, cy = 88, r = 68;

  const topTypes = Object.entries(result?.type_scores ?? {})
    .filter(([t]) => t !== "none")
    .sort(([, a], [, b]) => b - a)
    .slice(0, 4);

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Deception Score</span>
        <AnimatePresence mode="wait">
          {result && (
            <motion.div
              key={dominant}
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex items-center gap-1.5"
            >
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ backgroundColor: TYPE_COLOR[dominant] }}
              />
              <span className="text-2xs font-medium" style={{ color: TYPE_COLOR[dominant] }}>
                {TYPE_LABEL[dominant]}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="p-3 space-y-3">
        {/* SVG Gauge */}
        <div className="flex justify-center">
          <svg viewBox="0 0 200 100" className="w-full max-w-[200px]">
            <defs>
              <filter id="ds-glow">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>

            {/* Background track */}
            <path d={arcPath(1, r, cx, cy)} fill="none" stroke="#1e2730" strokeWidth="12" strokeLinecap="round" />

            {/* Score fill */}
            <motion.path
              d={arcPath(Math.max(score, 0.01), r, cx, cy)}
              fill="none"
              stroke={color}
              strokeWidth="12"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: score }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            />

            {/* Glow layer */}
            <motion.path
              d={arcPath(Math.max(score, 0.01), r, cx, cy)}
              fill="none"
              stroke={color}
              strokeWidth="5"
              strokeLinecap="round"
              opacity={0.35}
              filter="url(#ds-glow)"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: score }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
            />

            {/* Confidence outer ring */}
            {result && (
              <path
                d={arcPath(result.confidence, r + 10, cx, cy)}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="1.5"
                strokeLinecap="round"
                opacity={0.3}
              />
            )}

            {/* Tick marks */}
            {[0, 0.25, 0.5, 0.75, 1].map(v => {
              const angle = -180 + v * 180;
              const rad = (angle * Math.PI) / 180;
              const x1 = cx + (r - 8) * Math.cos(rad), y1 = cy + (r - 8) * Math.sin(rad);
              const x2 = cx + (r - 3) * Math.cos(rad), y2 = cy + (r - 3) * Math.sin(rad);
              return <line key={v} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#263040" strokeWidth="1.5" />;
            })}

            {/* Score number */}
            <text
              x={cx} y={cy + 6}
              textAnchor="middle"
              fill={isLoading ? "#4a5568" : color}
              fontSize="28"
              fontWeight="700"
              fontFamily="'JetBrains Mono', monospace"
              letterSpacing="-1"
            >
              {isLoading ? "—" : Math.round(score * 100)}
            </text>

            {/* Label */}
            <text
              x={cx} y={cy + 20}
              textAnchor="middle"
              fill="#4a5568"
              fontSize="8"
              fontWeight="600"
              fontFamily="Inter, sans-serif"
              letterSpacing="1.5"
            >
              {isLoading ? "ANALYSING" : label.toUpperCase()}
            </text>
          </svg>
        </div>

        {/* Confidence bar */}
        {result && (
          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <span className="text-2xs text-ink3">Confidence</span>
              <span className="text-2xs font-mono text-ink2">{(result.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="h-0.5 bg-border2 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-accent"
                initial={{ width: 0 }}
                animate={{ width: `${result.confidence * 100}%` }}
                transition={{ duration: 0.7, ease: "easeOut" }}
              />
            </div>
          </div>
        )}

        {/* Type breakdown */}
        {topTypes.length > 0 && (
          <div className="space-y-1.5 pt-2 border-t border-border">
            {topTypes.map(([type, val]) => (
              <div key={type} className="flex items-center gap-2">
                <span
                  className="text-2xs font-mono font-bold w-4 text-center flex-shrink-0"
                  style={{ color: TYPE_COLOR[type] }}
                >
                  {TYPE_SIGIL[type]}
                </span>
                <span className="text-2xs text-ink3 w-20 truncate">{TYPE_LABEL[type]}</span>
                <div className="flex-1 h-0.5 bg-border2 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: TYPE_COLOR[type] }}
                    initial={{ width: 0 }}
                    animate={{ width: `${val * 100}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </div>
                <span className="text-2xs font-mono text-ink3 w-6 text-right">
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
};
