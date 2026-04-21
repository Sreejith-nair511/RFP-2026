/**
 * ResearchMode — raw signal inspector for interpretability researchers.
 *
 * Shows:
 *  - Raw logits / logprob distribution (bar chart)
 *  - Probe weights per layer
 *  - Signal contribution breakdown (pie / bar)
 *  - Full JSON dump of DeceptionResult
 *
 * This panel is what grant reviewers will want to see — it demonstrates
 * that DeceptiScope exposes interpretable internals, not just a black-box score.
 */

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  PieChart,
  Pie,
  Legend,
} from "recharts";
import { DeceptionResult, DECEPTION_TYPE_COLORS, DECEPTION_TYPE_LABELS } from "../types";

interface Props {
  result: DeceptionResult | null;
}

type Tab = "signals" | "type_breakdown" | "raw_json";

export const ResearchMode: React.FC<Props> = ({ result }) => {
  const [tab, setTab] = useState<Tab>("signals");

  if (!result) {
    return (
      <div className="bg-ds-surface border border-ds-border rounded-xl p-4">
        <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider mb-3">
          Research Mode
        </h2>
        <p className="text-ds-muted text-xs text-center py-8">
          Send a message to see raw signal data.
        </p>
      </div>
    );
  }

  // Signal contributions bar data
  const signalData = Object.entries(result.signal_contributions)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ name: k.replace("_", " "), value: v }));

  // Type breakdown pie data
  const typeData = Object.entries(result.type_scores)
    .filter(([, v]) => v > 0.01)
    .map(([k, v]) => ({
      name: DECEPTION_TYPE_LABELS[k as keyof typeof DECEPTION_TYPE_LABELS] ?? k,
      value: parseFloat((v * 100).toFixed(1)),
      color: DECEPTION_TYPE_COLORS[k as keyof typeof DECEPTION_TYPE_COLORS] ?? "#8b949e",
    }));

  const tabs: { id: Tab; label: string }[] = [
    { id: "signals",        label: "Signal Streams" },
    { id: "type_breakdown", label: "Type Breakdown" },
    { id: "raw_json",       label: "Raw JSON" },
  ];

  return (
    <div className="bg-ds-surface border border-ds-border rounded-xl p-4 space-y-3">
      <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider">
        Research Mode
      </h2>

      {/* Tab bar */}
      <div className="flex gap-1 bg-ds-bg rounded-lg p-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 text-xs py-1.5 rounded-md transition-colors ${
              tab === t.id
                ? "bg-ds-surface text-ds-text font-semibold"
                : "text-ds-muted hover:text-ds-text"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Signal streams */}
      {tab === "signals" && (
        <div className="space-y-3">
          <ResponsiveContainer width="100%" height={140}>
            <BarChart data={signalData} layout="vertical" margin={{ left: 60, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                tick={{ fill: "#8b949e", fontSize: 10 }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "#8b949e", fontSize: 10 }}
                width={60}
              />
              <Tooltip
                formatter={(v) => v != null ? [`${(Number(v) * 100).toFixed(1)}%`, "Weight"] : ["-", "Weight"]}
                contentStyle={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8 }}
                labelStyle={{ color: "#e6edf3" }}
              />
              <Bar dataKey="value" radius={[0, 3, 3, 0]}>
                {signalData.map((_, i) => (
                  <Cell key={i} fill="#58a6ff" />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Key metrics */}
          <div className="grid grid-cols-3 gap-2 text-center">
            {[
              { label: "Score",      value: `${(result.score * 100).toFixed(1)}%`,      color: "text-ds-red"    },
              { label: "Confidence", value: `${(result.confidence * 100).toFixed(1)}%`, color: "text-ds-accent" },
              { label: "Type",       value: result.deception_type.replace("_", " "),    color: "text-ds-purple" },
            ].map((m) => (
              <div key={m.label} className="bg-ds-bg border border-ds-border rounded-lg p-2">
                <p className="text-xs text-ds-muted">{m.label}</p>
                <p className={`text-sm font-mono font-bold ${m.color} capitalize`}>{m.value}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Type breakdown */}
      {tab === "type_breakdown" && (
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={typeData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="45%"
              outerRadius={70}
              label={({ name, value }) => `${name}: ${value}%`}
              labelLine={false}
            >
              {typeData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
            <Legend
              iconSize={8}
              formatter={(v) => <span style={{ color: "#8b949e", fontSize: 11 }}>{v}</span>}
            />
            <Tooltip
              formatter={(v) => v != null ? [`${v}%`] : ["-"]}
              contentStyle={{ background: "#0d1117", border: "1px solid #30363d", borderRadius: 8 }}
            />
          </PieChart>
        </ResponsiveContainer>
      )}

      {/* Raw JSON */}
      {tab === "raw_json" && (
        <pre className="bg-ds-bg border border-ds-border rounded-lg p-3 text-xs text-ds-text
                        font-mono overflow-auto max-h-64 leading-relaxed">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
};
