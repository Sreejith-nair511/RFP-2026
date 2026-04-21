/**
 * ConsistencyTimeline — line chart of deception score across conversation turns.
 *
 * Helps researchers spot drift: does the model become more deceptive as the
 * conversation progresses?  Each data point is one assistant message.
 */

import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { ChatMessage, DECEPTION_TYPE_COLORS } from "../types";

interface Props {
  messages: ChatMessage[];
}

interface DataPoint {
  turn: number;
  score: number;
  type: string;
  preview: string;
}

const CustomTooltip: React.FC<any> = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d: DataPoint = payload[0].payload;
  return (
    <div className="bg-ds-bg border border-ds-border rounded-lg px-3 py-2 text-xs shadow-lg max-w-xs">
      <p className="text-ds-muted mb-1">Turn {d.turn}</p>
      <p className="text-ds-text font-mono font-bold">
        Score: {(d.score * 100).toFixed(1)}%
      </p>
      <p className="text-ds-muted capitalize">{d.type.replace("_", " ")}</p>
      <p className="text-ds-muted/70 mt-1 truncate">{d.preview}</p>
    </div>
  );
};

export const ConsistencyTimeline: React.FC<Props> = ({ messages }) => {
  const assistantMessages = messages.filter(
    (m) => m.role === "assistant" && m.deception
  );

  const data: DataPoint[] = assistantMessages.map((m, i) => ({
    turn:    i + 1,
    score:   m.deception!.score,
    type:    m.deception!.deception_type,
    preview: m.content.slice(0, 60) + (m.content.length > 60 ? "…" : ""),
  }));

  if (data.length === 0) {
    return (
      <div className="bg-ds-surface border border-ds-border rounded-xl p-4">
        <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider mb-3">
          Consistency Timeline
        </h2>
        <p className="text-ds-muted text-xs text-center py-8">
          No assistant messages yet.
        </p>
      </div>
    );
  }

  // Colour each dot by deception type
  const dotColors = data.map(
    (d) => DECEPTION_TYPE_COLORS[d.type as keyof typeof DECEPTION_TYPE_COLORS] ?? "#8b949e"
  );

  return (
    <div className="bg-ds-surface border border-ds-border rounded-xl p-4">
      <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider mb-3">
        Consistency Timeline
      </h2>

      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
          <XAxis
            dataKey="turn"
            tick={{ fill: "#8b949e", fontSize: 11 }}
            label={{ value: "Turn", position: "insideBottomRight", fill: "#8b949e", fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: "#8b949e", fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Risk threshold lines */}
          <ReferenceLine y={0.55} stroke="#d29922" strokeDasharray="4 2" label={{ value: "moderate", fill: "#d29922", fontSize: 10 }} />
          <ReferenceLine y={0.75} stroke="#f85149" strokeDasharray="4 2" label={{ value: "high", fill: "#f85149", fontSize: 10 }} />

          <Line
            type="monotone"
            dataKey="score"
            stroke="#58a6ff"
            strokeWidth={2}
            dot={(props: any) => {
              const { cx, cy, index } = props;
              return (
                <circle
                  key={index}
                  cx={cx}
                  cy={cy}
                  r={4}
                  fill={dotColors[index] ?? "#58a6ff"}
                  stroke="#0d1117"
                  strokeWidth={1.5}
                />
              );
            }}
            activeDot={{ r: 6, stroke: "#58a6ff", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Summary stats */}
      {data.length > 1 && (
        <div className="flex gap-4 mt-3 pt-3 border-t border-ds-border text-xs text-ds-muted">
          <span>
            Avg:{" "}
            <span className="text-ds-text font-mono">
              {((data.reduce((s, d) => s + d.score, 0) / data.length) * 100).toFixed(1)}%
            </span>
          </span>
          <span>
            Peak:{" "}
            <span className="text-ds-red font-mono">
              {(Math.max(...data.map((d) => d.score)) * 100).toFixed(1)}%
            </span>
          </span>
          <span>
            Trend:{" "}
            <span
              className={
                data[data.length - 1].score > data[0].score
                  ? "text-ds-red"
                  : "text-ds-green"
              }
            >
              {data[data.length - 1].score > data[0].score ? "↑ increasing" : "↓ decreasing"}
            </span>
          </span>
        </div>
      )}
    </div>
  );
};
