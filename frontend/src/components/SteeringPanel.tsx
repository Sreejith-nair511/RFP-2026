/**
 * SteeringPanel — controls for activating and tuning the deception-reduction
 * steering signal.
 *
 * Steering works by injecting a translated "honest direction" into the
 * frontier model's system prompt.  The panel shows:
 *  - Steering on/off toggle
 *  - Strength slider (0–1)
 *  - Active steering prompt preview
 *  - Before/after deception score comparison
 */

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { DeceptionResult } from "../types";

interface Props {
  enabled: boolean;
  onToggle: () => void;
  strength: number;
  onStrengthChange: (v: number) => void;
  lastResult: DeceptionResult | null;
  steeredResult: DeceptionResult | null;
}

const STEERING_PROMPTS: Record<string, string> = {
  factual_error:
    "Be precise and accurate. If uncertain, say so explicitly. Do not state unverified claims as facts.",
  omission:
    "Provide complete information. Do not omit relevant caveats, risks, or counterarguments.",
  overconfidence:
    "Calibrate your confidence carefully. Distinguish between established facts and uncertain claims.",
  sycophancy:
    "Prioritise truthfulness over agreement. Correct factual errors even if the user disagrees.",
  evasion:
    "Answer the question directly. Do not deflect or change the subject.",
  contradiction:
    "Ensure logical consistency throughout your response. Check for internal contradictions.",
  false_expertise:
    "Be transparent about the limits of your knowledge. Do not fabricate expertise.",
  none:
    "Continue responding honestly and accurately.",
};

export const SteeringPanel: React.FC<Props> = ({
  enabled,
  onToggle,
  strength,
  onStrengthChange,
  lastResult,
  steeredResult,
}) => {
  const [showPrompt, setShowPrompt] = useState(false);

  const dominantType = lastResult?.deception_type ?? "none";
  const steeringPrompt = STEERING_PROMPTS[dominantType] ?? STEERING_PROMPTS.none;

  const improvement =
    lastResult && steeredResult
      ? lastResult.score - steeredResult.score
      : null;

  return (
    <div className="bg-ds-surface border border-ds-border rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider">
          Steering
        </h2>
        {/* Master toggle */}
        <button
          onClick={onToggle}
          className={`relative w-10 h-5 rounded-full transition-colors ${
            enabled ? "bg-ds-accent" : "bg-ds-border"
          }`}
          aria-label="Toggle steering"
        >
          <motion.div
            className="absolute top-0.5 w-4 h-4 rounded-full bg-white shadow"
            animate={{ x: enabled ? 20 : 2 }}
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
          />
        </button>
      </div>

      {/* Strength slider */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-ds-muted">
          <span>Strength</span>
          <span className="font-mono text-ds-text">{(strength * 100).toFixed(0)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={strength}
          onChange={(e) => onStrengthChange(parseFloat(e.target.value))}
          disabled={!enabled}
          className="w-full accent-ds-accent disabled:opacity-40"
        />
        <div className="flex justify-between text-xs text-ds-muted/60">
          <span>subtle</span>
          <span>aggressive</span>
        </div>
      </div>

      {/* Active steering prompt */}
      <div>
        <button
          onClick={() => setShowPrompt((v) => !v)}
          className="text-xs text-ds-accent hover:underline"
        >
          {showPrompt ? "Hide" : "Show"} active steering prompt
        </button>
        <AnimatePresence>
          {showPrompt && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="mt-2 p-3 bg-ds-bg border border-ds-border rounded-lg">
                <p className="text-xs text-ds-muted mb-1 uppercase tracking-wider">
                  Injected system prompt ({dominantType.replace("_", " ")}):
                </p>
                <p className="text-xs text-ds-text font-mono leading-relaxed">
                  {steeringPrompt}
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Before / after comparison */}
      {lastResult && (
        <div className="space-y-2">
          <p className="text-xs text-ds-muted uppercase tracking-wider">
            Before / After
          </p>
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-ds-bg border border-ds-border rounded-lg p-2 text-center">
              <p className="text-xs text-ds-muted">Without steering</p>
              <p className="text-lg font-mono font-bold text-ds-red">
                {(lastResult.score * 100).toFixed(0)}%
              </p>
            </div>
            <div className="bg-ds-bg border border-ds-border rounded-lg p-2 text-center">
              <p className="text-xs text-ds-muted">With steering</p>
              <p className="text-lg font-mono font-bold text-ds-green">
                {steeredResult
                  ? `${(steeredResult.score * 100).toFixed(0)}%`
                  : "—"}
              </p>
            </div>
          </div>
          {improvement !== null && improvement > 0 && (
            <p className="text-xs text-ds-green text-center">
              ↓ {(improvement * 100).toFixed(1)}% reduction in deception score
            </p>
          )}
        </div>
      )}

      {/* Explanation */}
      <p className="text-xs text-ds-muted/70 leading-relaxed">
        Steering translates the shadow model's deception direction into a
        natural-language system prompt injected before each request.
        Higher strength = more aggressive intervention.
      </p>
    </div>
  );
};
