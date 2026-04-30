import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ResponsiveContainer, ReferenceLine,
} from "recharts";
import { scoreToColor } from "../types";

interface LayerScore { layer: number; score: number; label?: string; }
interface Props { layerScores: LayerScore[]; modelName?: string; }

const MOCK: LayerScore[] = Array.from({ length: 32 }, (_, i) => {
  const x = i / 31;
  const base = 0.08 + 0.62 * Math.exp(-((x - 0.55) ** 2) / 0.04);
  return {
    layer: i,
    score: Math.min(1, base + Math.sin(i * 1.3) * 0.05),
    label: i === 17 ? "Peak" : undefined,
  };
});

const Tip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d: LayerScore = payload[0].payload;
  return (
    <div className="bg-bg border border-border rounded-md px-2.5 py-1.5 text-2xs shadow-panel">
      <p className="text-ink3">Layer {d.layer}</p>
      <p className="font-mono font-semibold" style={{ color: scoreToColor(d.score) }}>
        {(d.score * 100).toFixed(1)}%
      </p>
      {d.label && <p className="text-violet">{d.label}</p>}
    </div>
  );
};

export const LayerProbeViz: React.FC<Props> = ({ layerScores, modelName }) => {
  const data = layerScores.length > 0 ? layerScores : MOCK;
  const peak = data.reduce((b, d) => d.score > b.score ? d : b, data[0]);
  const isDemo = layerScores.length === 0;

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Layer Probes</span>
        <div className="flex items-center gap-2">
          {isDemo && (
            <span className="text-2xs text-warn border border-warn/30 rounded px-1.5 py-px">
              demo
            </span>
          )}
          {modelName && (
            <span className="text-2xs text-ink3 truncate max-w-[90px]">{modelName}</span>
          )}
        </div>
      </div>

      <div className="p-3">
        <ResponsiveContainer width="100%" height={130}>
          <BarChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: -28 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="#1e2730" vertical={false} />
            <XAxis dataKey="layer" tick={{ fill: "#4a5568", fontSize: 8 }} interval={7} />
            <YAxis
              domain={[0, 1]}
              tickFormatter={v => `${(v * 100).toFixed(0)}`}
              tick={{ fill: "#4a5568", fontSize: 8 }}
            />
            <Tooltip content={<Tip />} />
            <ReferenceLine
              x={peak.layer}
              stroke="#a855f7"
              strokeDasharray="3 3"
              label={{ value: `L${peak.layer}`, fill: "#a855f7", fontSize: 8 }}
            />
            <Bar dataKey="score" radius={[1, 1, 0, 0]}>
              {data.map((d, i) => <Cell key={i} fill={scoreToColor(d.score)} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <p className="text-2xs text-ink3 mt-2 pt-2 border-t border-border leading-relaxed">
          <span className="text-violet font-medium">Peak layer {peak.layer}</span>
          {" "}— where the model's residual stream most strongly encodes deceptive intent.
          RepE steering vectors are extracted at this layer.
        </p>
      </div>
    </div>
  );
};
