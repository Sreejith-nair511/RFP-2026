/**
 * SessionPanel — shows the current session's analysis history
 * with the ability to export as JSON or CSV.
 *
 * Each record shows: turn number, prompt preview, score, type, model.
 * Clicking a record scrolls to that message (future enhancement).
 */

import React, { memo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SessionRecord } from "../hooks/useWebSocket";
import { scoreToColor, scoreToLabel, TYPE_LABEL, TYPE_SIGIL } from "../types";

interface Props {
  sessionId: string | null;
  records: SessionRecord[];
  onExport: (fmt: "json" | "csv") => Promise<void>;
  onClear: () => void;
}

export const SessionPanel: React.FC<Props> = memo(({ sessionId, records, onExport, onClear }) => {
  const [exporting, setExporting] = useState<string | null>(null);

  const avg = records.length
    ? records.reduce((s, r) => s + r.deception_score, 0) / records.length
    : 0;

  const handleExport = async (fmt: "json" | "csv") => {
    setExporting(fmt);
    try { await onExport(fmt); } finally { setExporting(null); }
  };

  return (
    <div className="bg-surface border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <div className="flex items-center gap-2">
          <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Session</span>
          {sessionId && (
            <span className="text-2xs font-mono text-ink3">{sessionId.slice(0, 8)}</span>
          )}
        </div>
        <button
          onClick={onClear}
          className="text-2xs text-ink3 hover:text-danger transition-colors"
        >
          New session
        </button>
      </div>

      {/* Stats */}
      {records.length > 0 && (
        <div className="grid grid-cols-3 gap-px border-b border-border">
          {[
            { label: "Turns",   value: records.length.toString(),          color: "text-ink2"  },
            { label: "Avg",     value: `${(avg * 100).toFixed(0)}%`,       color: scoreToColor(avg) },
            { label: "Peak",    value: `${(Math.max(...records.map(r => r.deception_score)) * 100).toFixed(0)}%`,
              color: "text-danger" },
          ].map(s => (
            <div key={s.label} className="bg-bg px-3 py-2 text-center">
              <p className="text-2xs text-ink3">{s.label}</p>
              <p className={`text-sm font-mono font-bold`} style={{ color: s.color }}>{s.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Records list */}
      <div className="max-h-48 overflow-y-auto">
        {records.length === 0 ? (
          <p className="text-2xs text-ink3 text-center py-6">No turns yet.</p>
        ) : (
          <div className="divide-y divide-border">
            {records.map((r, i) => {
              const color = scoreToColor(r.deception_score);
              return (
                <motion.div
                  key={r.id}
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.02 }}
                  className="flex items-center gap-2 px-3 py-2 hover:bg-surface2 transition-colors"
                >
                  <span className="text-2xs text-ink3 w-4 flex-shrink-0 font-mono">{i + 1}</span>
                  <span className="text-2xs text-ink2 flex-1 truncate">{r.prompt.slice(0, 40)}</span>
                  <span className="text-2xs font-mono font-bold flex-shrink-0" style={{ color }}>
                    {(r.deception_score * 100).toFixed(0)}%
                  </span>
                  <span
                    className="text-2xs font-mono flex-shrink-0 w-4 text-center"
                    style={{ color: "#a855f7" }}
                    title={TYPE_LABEL[r.deception_type] ?? r.deception_type}
                  >
                    {TYPE_SIGIL[r.deception_type] ?? "?"}
                  </span>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* Export buttons */}
      <div className="p-2.5 border-t border-border flex gap-1.5">
        {(["json", "csv"] as const).map(fmt => (
          <button
            key={fmt}
            onClick={() => handleExport(fmt)}
            disabled={records.length === 0 || exporting !== null}
            className="flex-1 py-1.5 rounded-md text-2xs font-semibold border border-border
                       text-ink2 hover:border-accent/40 hover:text-accent transition-all
                       disabled:opacity-30 disabled:cursor-not-allowed uppercase"
          >
            {exporting === fmt ? "…" : fmt}
          </button>
        ))}
      </div>
    </div>
  );
});
