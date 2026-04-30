import React from "react";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from "recharts";
import { ChatMessage, TYPE_COLOR, scoreToColor } from "../types";

interface Props { messages: ChatMessage[]; }

const Tip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-bg border border-border rounded-md px-2.5 py-2 text-2xs shadow-panel max-w-[180px]">
      <p className="text-ink3 mb-0.5">Turn {d.turn}</p>
      <p className="font-mono font-semibold" style={{ color: d.color }}>
        {(d.score * 100).toFixed(1)}%
      </p>
      <p className="text-ink3 capitalize">{d.type.replace(/_/g, " ")}</p>
      <p className="text-ink3 mt-1 truncate opacity-60">{d.preview}</p>
    </div>
  );
};

export const ConsistencyTimeline: React.FC<Props> = ({ messages }) => {
  const rows = messages.filter(m => m.role === "assistant" && m.deception);
  const data = rows.map((m, i) => ({
    turn:    i + 1,
    score:   m.deception!.score,
    type:    m.deception!.deception_type,
    color:   TYPE_COLOR[m.deception!.deception_type] ?? "#7a8899",
    preview: m.content.slice(0, 48) + (m.content.length > 48 ? "…" : ""),
  }));

  const avg  = data.length ? data.reduce((s, d) => s + d.score, 0) / data.length : 0;
  const peak = data.length ? Math.max(...data.map(d => d.score)) : 0;
  const trend = data.length > 1 ? data[data.length - 1].score - data[0].score : 0;

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Timeline</span>
        {data.length > 1 && (
          <span className={`text-2xs font-mono ${trend > 0 ? "text-danger" : "text-success"}`}>
            {trend > 0 ? "+" : ""}{(trend * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="p-3">
        {data.length === 0 ? (
          <div className="h-28 flex items-center justify-center text-2xs text-ink3">
            No data yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={120}>
            <AreaChart data={data} margin={{ top: 4, right: 2, bottom: 0, left: -30 }}>
              <defs>
                <linearGradient id="tl-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#3b82f6" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" stroke="#1e2730" />
              <XAxis dataKey="turn" tick={{ fill: "#4a5568", fontSize: 9 }} />
              <YAxis
                domain={[0, 1]}
                tickFormatter={v => `${(v * 100).toFixed(0)}`}
                tick={{ fill: "#4a5568", fontSize: 9 }}
              />
              <Tooltip content={<Tip />} />
              <ReferenceLine y={0.5}  stroke="#eab308" strokeDasharray="3 3" strokeOpacity={0.4} />
              <ReferenceLine y={0.75} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.4} />
              <Area
                type="monotone"
                dataKey="score"
                stroke="#3b82f6"
                strokeWidth={1.5}
                fill="url(#tl-grad)"
                dot={(props: any) => (
                  <circle
                    key={props.index}
                    cx={props.cx} cy={props.cy} r={3}
                    fill={data[props.index]?.color ?? "#3b82f6"}
                    stroke="#080c10"
                    strokeWidth={1.5}
                  />
                )}
                activeDot={{ r: 5, stroke: "#3b82f6", strokeWidth: 1.5, fill: "#080c10" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}

        {data.length > 0 && (
          <div className="flex gap-4 mt-2 pt-2 border-t border-border text-2xs text-ink3">
            <span>Avg <span className="font-mono text-ink2">{(avg * 100).toFixed(1)}%</span></span>
            <span>Peak <span className="font-mono text-danger">{(peak * 100).toFixed(1)}%</span></span>
            <span>Turns <span className="font-mono text-ink2">{data.length}</span></span>
          </div>
        )}
      </div>
    </div>
  );
};
