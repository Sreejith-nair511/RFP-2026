/**
 * LayerProbeViz — bar chart showing per-layer deception probe scores for
 * open-weight models (LLaMA, Mistral, Qwen).
 *
 * Interpretability insight: the layer where the probe score peaks is where
 * the model "knows" the truth but chooses to suppress it.  This is the
 * key finding we highlight in the grant application.
 */

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface LayerScore {
  layer: number;
  score: number;
  label?: string;
}

interface Props {
  layerScores: LayerScore[];
  modelName?: string;
}

function scoreToColor(score: number): string {
  if (score < 0.3) return "#3fb950";
  if (score < 0.55) return "#d29922";
  if (score < 0.75) return "#f0883e";
  return "#f85149";
}

const CustomTooltip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d: LayerScore = payload[0].payload;
  return (
    <div className="bg-ds-bg border border-ds-border rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-ds-muted">Layer {d.layer}</p>
      <p className="font-mono font-bold" style={{ color: scoreToColor(d.score) }}>
        {(d.score * 100).toFixed(1)}% deceptive
      </p>
      {d.label && <p className="text-ds-muted/70 mt-0.5">{d.label}</p>}
    </div>
  );
};

// Mock data for when no real data is available (demo mode)
const MOCK_LAYER_SCORES: LayerScore[] = Array.from({ length: 32 }, (_, i) => {
  // Simulate a realistic pattern: deception signal peaks in middle layers
  const x = i / 31;
  const base = 0.1 + 0.6 * Math.exp(-((x - 0.55) ** 2) / 0.04);
  return {
    layer: i,
    score: Math.min(1, base + (Math.random() - 0.5) * 0.08),
    label: i === 17 ? "Peak: model 'knows' truth here" : undefined,
  };
});

export const LayerProbeViz: React.FC<Props> = ({
  layerScores,
  modelName = "Open-weight model",
}) => {
  const data = layerScores.length > 0 ? layerScores : MOCK_LAYER_SCORES;
  const peakLayer = data.reduce((best, d) => (d.score > best.score ? d : best), data[0]);

  return (
    <div className="bg-ds-surface border border-ds-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider">
          Layer Probe Scores
        </h2>
        <span className="text-xs text-ds-muted">{modelName}</span>
      </div>

      {layerScores.length === 0 && (
        <p className="text-xs text-ds-yellow mb-2">
          ⚠ Demo data — connect an open-weight model for real activations.
        </p>
      )}

      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" vertical={false} />
          <XAxis
            dataKey="layer"
            tick={{ fill: "#8b949e", fontSize: 10 }}
            interval={3}
            label={{ value: "Layer", position: "insideBottomRight", fill: "#8b949e", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#8b949e", fontSize: 10 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            x={peakLayer.layer}
            stroke="#bc8cff"
            strokeDasharray="4 2"
            label={{ value: `peak L${peakLayer.layer}`, fill: "#bc8cff", fontSize: 10 }}
          />
          <Bar dataKey="score" radius={[2, 2, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={scoreToColor(entry.score)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Interpretability annotation */}
      <div className="mt-3 pt-3 border-t border-ds-border text-xs text-ds-muted space-y-1">
        <p>
          <span className="text-ds-purple font-semibold">Peak layer {peakLayer.layer}</span>
          {" "}— this is where the model's internal representation most strongly encodes
          deceptive intent. RepE steering vectors are extracted here.
        </p>
        <p className="text-ds-muted/60">
          Score = linear probe accuracy on honest vs. deceptive hidden states (TruthfulQA).
        </p>
      </div>
    </div>
  );
};
