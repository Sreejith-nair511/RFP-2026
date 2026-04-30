import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MODEL_OPTIONS, ModelOption, ModelProvider, PROVIDER_META } from "../types";
import { ConnectionStatus } from "../hooks/useWebSocket";

interface Props {
  selectedModel: ModelOption | null;
  status: ConnectionStatus;
  onSelect: (o: ModelOption) => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

const PROVIDERS: ModelProvider[] = ["groq", "gemini", "openai", "anthropic"];

const STATUS_DOT: Record<ConnectionStatus, string> = {
  connected:    "bg-success",
  connecting:   "bg-warn animate-pulse",
  disconnected: "bg-ink3",
  error:        "bg-danger",
};

const STATUS_TEXT: Record<ConnectionStatus, string> = {
  connected:    "Connected",
  connecting:   "Connecting",
  disconnected: "Offline",
  error:        "Error",
};

export const ModelSelector: React.FC<Props> = ({
  selectedModel, status, onSelect, onConnect, onDisconnect,
}) => {
  const [activeProvider, setActiveProvider] = useState<ModelProvider>("groq");
  const filtered = MODEL_OPTIONS.filter(m => m.provider === activeProvider);
  const meta = PROVIDER_META[activeProvider];

  return (
    <div className="flex flex-col bg-surface border border-border rounded-lg overflow-hidden">

      {/* Header row */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
        <span className="text-2xs font-semibold text-ink2 uppercase tracking-widest">Model</span>
        <div className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOT[status]}`} />
          <span className="text-2xs text-ink3">{STATUS_TEXT[status]}</span>
        </div>
      </div>

      {/* Provider tabs */}
      <div className="flex border-b border-border">
        {PROVIDERS.map(p => {
          const m = PROVIDER_META[p];
          const active = activeProvider === p;
          const has = MODEL_OPTIONS.some(o => o.provider === p);
          return (
            <button
              key={p}
              onClick={() => setActiveProvider(p)}
              disabled={!has}
              className={`flex-1 py-2 text-2xs font-medium transition-colors relative
                ${active ? "text-ink" : "text-ink3 hover:text-ink2"}
                ${!has ? "opacity-30 cursor-not-allowed" : "cursor-pointer"}
              `}
            >
              {m.label}
              {active && (
                <motion.div
                  layoutId="providerUnderline"
                  className="absolute bottom-0 left-0 right-0 h-px"
                  style={{ background: m.color }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Model list */}
      <div className="overflow-y-auto" style={{ maxHeight: 220 }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={activeProvider}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className="p-1.5 space-y-0.5"
          >
            {filtered.map(opt => {
              const sel = selectedModel?.model === opt.model;
              return (
                <button
                  key={opt.model}
                  onClick={() => onSelect(opt)}
                  className={`w-full text-left px-2.5 py-2 rounded-md transition-all group
                    ${sel
                      ? "bg-surface2 border border-border2"
                      : "hover:bg-surface2 border border-transparent"
                    }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      {/* Selection indicator */}
                      <div
                        className={`w-1 h-1 rounded-full flex-shrink-0 transition-all ${sel ? "opacity-100" : "opacity-0"}`}
                        style={{ backgroundColor: meta.color }}
                      />
                      <span className={`text-sm font-medium truncate ${sel ? "text-ink" : "text-ink2 group-hover:text-ink"}`}>
                        {opt.label}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {opt.tokensPerSec !== "—" && (
                        <span className="text-2xs text-ink3 font-mono">{opt.tokensPerSec} t/s</span>
                      )}
                      <div className="flex gap-1">
                        {opt.supportsLogprobs && (
                          <span className="text-2xs px-1 py-px rounded bg-surface2 border border-border text-ink3 font-mono">lp</span>
                        )}
                        {opt.supportsCoT && (
                          <span className="text-2xs px-1 py-px rounded bg-surface2 border border-border text-ink3 font-mono">cot</span>
                        )}
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Action row */}
      <div className="p-2 border-t border-border flex gap-1.5">
        <button
          onClick={onConnect}
          disabled={!selectedModel || status === "connecting"}
          className="flex-1 py-2 rounded-md text-xs font-semibold transition-all
                     disabled:opacity-30 disabled:cursor-not-allowed"
          style={selectedModel ? {
            background: PROVIDER_META[selectedModel.provider].dim,
            border: `1px solid ${PROVIDER_META[selectedModel.provider].color}33`,
            color: PROVIDER_META[selectedModel.provider].color,
          } : {
            background: "transparent",
            border: "1px solid #1e2730",
            color: "#4a5568",
          }}
        >
          {status === "connecting" ? "Connecting…" : status === "connected" ? "Reconnect" : "Connect"}
        </button>
        {status === "connected" && (
          <button
            onClick={onDisconnect}
            className="px-3 py-2 rounded-md border border-border text-xs text-ink3
                       hover:border-danger/40 hover:text-danger transition-all"
          >
            Disconnect
          </button>
        )}
      </div>
    </div>
  );
};
