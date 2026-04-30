import React, { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
  PieChart, Pie, Legend,
} from "recharts";
import { DeceptionResult, TYPE_COLOR, TYPE_LABEL } from "../types";

interface Props { result: DeceptionResult | null; }
type Tab = "signals" | "types" | "json";

const TABS: { id: Tab; label: string }[] = [
  { id: "signals", label: "Signals" },
  { id: "types",   label: "Types"   },
  { id: "json",    label: "Raw"     },
];

export const ResearchMode: React.FC<Props> = ({ result }) => {
  const [tab, setTab] = useState<Tab>("signals");

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header + tabs */}
      <div className="flex items-center border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest px-3 py-2.5 border-r border-border">
          Research
        </span>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-3 py-2.5 text-2xs font-medium transition-colors relative
              ${tab === t.id ? "text-ink" : "text-ink3 hover:text-ink2"}`}
          >
            {t.label}
            {tab === t.id && (
              <div className="absolute bottom-0 left-0 right-0 h-px bg-accent" />
            )}
          </button>
        ))}
      </div>

      <div className="p-3">
        {!result ? (
          <p className="text-2xs text-ink3 text-center py-8">
            Send a message to see signal data.
          </p>
        ) : (
          <>
            {/* ── Signals ── */}
            {tab === "signals" && (
              <div className="space-y-3">
                {/* Metric cards */}
                <div className="grid grid-cols-3 gap-1.5">
                  {[
                    { label: "Score",  value: `${(result.score * 100).toFixed(1)}%`,      color: "text-danger"  },
                    { label: "Conf",   value: `${(result.confidence * 100).toFixed(0)}%`, color: "text-accent"  },
                    { label: "Streams",value: Object.values(result.signal_contributions).filter(v => v > 0.01).length.toString(), color: "text-violet" },
                  ].map(m => (
                    <div key={m.label} className="bg-bg border border-border rounded-md p-2 text-center">
                      <p className="text-2xs text-ink3 mb-0.5">{m.label}</p>
                      <p className={`text-sm font-mono font-bold ${m.color}`}>{m.value}</p>
                    </div>
                  ))}
                </div>

                {/* Signal contributions */}
                {Object.keys(result.signal_contributions).length > 0 && (
                  <div>
                    <p className="text-2xs text-ink3 mb-1.5 uppercase tracking-wider">Signal Streams</p>
                    <ResponsiveContainer width="100%" height={90}>
                      <BarChart
                        data={Object.entries(result.signal_contributions)
                          .filter(([, v]) => v > 0)
                          .map(([k, v]) => ({ name: k.replace(/_/g, " "), value: v }))}
                        layout="vertical"
                        margin={{ left: 64, right: 4 }}
                      >
                        <CartesianGrid strokeDasharray="2 4" stroke="#1e2730" horizontal={false} />
                        <XAxis
                          type="number" domain={[0, 1]}
                          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                          tick={{ fill: "#4a5568", fontSize: 8 }}
                        />
                        <YAxis
                          type="category" dataKey="name"
                          tick={{ fill: "#4a5568", fontSize: 8 }}
                          width={64}
                        />
                        <Tooltip
                          formatter={(v) => v != null ? [`${(Number(v) * 100).toFixed(1)}%`, "Weight"] : ["-", "Weight"]}
                          contentStyle={{ background: "#080c10", border: "1px solid #1e2730", borderRadius: 6, fontSize: 10 }}
                        />
                        <Bar dataKey="value" radius={[0, 2, 2, 0]}>
                          {Object.entries(result.signal_contributions)
                            .filter(([, v]) => v > 0)
                            .map((_, i) => <Cell key={i} fill="#3b82f6" />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Behavioral signals */}
                {result.behavioral_signals && (
                  <div>
                    <p className="text-2xs text-ink3 mb-1.5 uppercase tracking-wider">Behavioral Signals</p>
                    <div className="space-y-1.5">
                      {Object.entries(result.behavioral_signals)
                        .filter(([, v]) => v != null)
                        .map(([k, v]) => (
                          <div key={k} className="flex items-center gap-2">
                            <span className="text-2xs text-ink3 w-24 truncate">{k.replace(/_/g, " ")}</span>
                            <div className="flex-1 h-0.5 bg-border2 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-accent rounded-full"
                                style={{ width: `${(v as number) * 100}%` }}
                              />
                            </div>
                            <span className="text-2xs font-mono text-ink3 w-7 text-right">
                              {((v as number) * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── Types ── */}
            {tab === "types" && (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={Object.entries(result.type_scores)
                      .filter(([, v]) => v > 0.01)
                      .map(([k, v]) => ({
                        name:  TYPE_LABEL[k] ?? k,
                        value: parseFloat((v * 100).toFixed(1)),
                        color: TYPE_COLOR[k] ?? "#7a8899",
                      }))}
                    dataKey="value" nameKey="name"
                    cx="50%" cy="42%" outerRadius={60}
                    label={({ value }) => `${value}%`}
                    labelLine={false}
                  >
                    {Object.entries(result.type_scores)
                      .filter(([, v]) => v > 0.01)
                      .map(([k], i) => (
                        <Cell key={i} fill={TYPE_COLOR[k] ?? "#7a8899"} />
                      ))}
                  </Pie>
                  <Legend
                    iconSize={6}
                    formatter={v => <span style={{ color: "#7a8899", fontSize: 9 }}>{v}</span>}
                  />
                  <Tooltip
                    formatter={(v) => v != null ? [`${v}%`] : ["-"]}
                    contentStyle={{ background: "#080c10", border: "1px solid #1e2730", borderRadius: 6, fontSize: 10 }}
                  />
                </PieChart>
              </ResponsiveContainer>
            )}

            {/* ── Raw JSON ── */}
            {tab === "json" && (
              <pre className="bg-bg border border-border rounded-md p-2.5 text-2xs text-ink2
                              font-mono overflow-auto max-h-64 leading-relaxed">
                {JSON.stringify(result, null, 2)}
              </pre>
            )}
          </>
        )}
      </div>
    </div>
  );
};
