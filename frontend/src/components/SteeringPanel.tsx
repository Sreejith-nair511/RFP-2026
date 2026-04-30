import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { DeceptionResult, TYPE_LABEL } from "../types";

interface Props {
  enabled: boolean; onToggle: () => void;
  strength: number; onStrengthChange: (v: number) => void;
  lastResult: DeceptionResult | null;
  steeredResult: DeceptionResult | null;
}

const PROMPTS: Record<string, string> = {
  factual_error:   "Be precise and accurate. Acknowledge uncertainty explicitly. Do not state unverified claims as facts.",
  omission:        "Provide complete information. Do not omit relevant caveats, risks, or counterarguments.",
  overconfidence:  "Calibrate confidence carefully. Distinguish established facts from uncertain claims.",
  sycophancy:      "Prioritise truthfulness over agreement. Correct factual errors even if the user disagrees.",
  evasion:         "Answer the question directly. Do not deflect or change the subject.",
  contradiction:   "Ensure logical consistency throughout. Check for internal contradictions before responding.",
  false_expertise: "Be transparent about the limits of your knowledge. Do not fabricate expertise.",
  none:            "Continue responding honestly and accurately.",
};

export const SteeringPanel: React.FC<Props> = ({
  enabled, onToggle, strength, onStrengthChange, lastResult, steeredResult,
}) => {
  const [expanded, setExpanded] = useState(false);
  const dominant = lastResult?.deception_type ?? "none";
  const prompt = PROMPTS[dominant] ?? PROMPTS.none;
  const improvement = lastResult && steeredResult ? lastResult.score - steeredResult.score : null;

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Steering</span>
        <button
          onClick={onToggle}
          className={`relative w-8 h-4 rounded-full transition-colors ${enabled ? "bg-accent" : "bg-border2"}`}
          aria-label="Toggle steering"
        >
          <motion.div
            className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm"
            animate={{ x: enabled ? 16 : 2 }}
            transition={{ type: "spring", stiffness: 600, damping: 35 }}
          />
        </button>
      </div>

      <div className="p-3 space-y-3">
        {/* Strength slider */}
        <div className="space-y-1.5">
          <div className="flex justify-between items-center">
            <span className="text-2xs text-ink3">Strength</span>
            <span className="text-2xs font-mono text-ink2">{(strength * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.05} value={strength}
            onChange={e => onStrengthChange(parseFloat(e.target.value))}
            disabled={!enabled}
            className="w-full"
          />
          <div className="flex justify-between text-2xs text-ink3">
            <span>Subtle</span><span>Aggressive</span>
          </div>
        </div>

        {/* Active prompt */}
        <div>
          <button
            onClick={() => setExpanded(v => !v)}
            className="flex items-center gap-1.5 text-2xs text-ink3 hover:text-accent transition-colors"
          >
            <svg
              width="8" height="8" viewBox="0 0 8 8" fill="none"
              className={`transition-transform ${expanded ? "rotate-90" : ""}`}
            >
              <path d="M2 1.5L5.5 4L2 6.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Active prompt
          </button>
          <AnimatePresence>
            {expanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.15 }}
                className="overflow-hidden"
              >
                <div className="mt-2 p-2.5 bg-bg border border-border rounded-md">
                  <p className="text-2xs text-ink3 mb-1 uppercase tracking-wider">
                    {TYPE_LABEL[dominant] ?? dominant}
                  </p>
                  <p className="text-2xs text-ink2 font-mono leading-relaxed">{prompt}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Before / after */}
        {lastResult && (
          <div className="space-y-2 pt-2 border-t border-border">
            <span className="text-2xs text-ink3 uppercase tracking-wider">Before / After</span>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { label: "Without", score: lastResult.score,    color: "text-danger"  },
                { label: "With",    score: steeredResult?.score, color: "text-success" },
              ].map(({ label, score, color }) => (
                <div key={label} className="bg-bg border border-border rounded-md p-2 text-center">
                  <p className="text-2xs text-ink3 mb-0.5">{label}</p>
                  <p className={`text-lg font-mono font-bold ${color}`}>
                    {score != null ? `${(score * 100).toFixed(0)}` : "—"}
                    {score != null && <span className="text-xs font-normal">%</span>}
                  </p>
                </div>
              ))}
            </div>
            {improvement != null && improvement > 0.01 && (
              <p className="text-2xs text-success text-center">
                {(improvement * 100).toFixed(1)}% reduction in deception score
              </p>
            )}
          </div>
        )}

        <p className="text-2xs text-ink3 leading-relaxed">
          Translates shadow model deception directions into system prompt injections.
        </p>
      </div>
    </div>
  );
};
