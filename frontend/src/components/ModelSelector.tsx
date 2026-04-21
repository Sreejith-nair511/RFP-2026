/**
 * ModelSelector — lets the user pick a frontier model and connect.
 *
 * Shows provider badges, capability tags (logprobs / CoT), and
 * connection status indicator.
 */

import React from "react";
import { motion } from "framer-motion";
import { MODEL_OPTIONS, ModelOption, ModelProvider } from "../types";
import { ConnectionStatus } from "../hooks/useWebSocket";

interface Props {
  selectedModel: ModelOption | null;
  status: ConnectionStatus;
  onSelect: (option: ModelOption) => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

const PROVIDER_COLORS: Record<ModelProvider, string> = {
  openai:    "bg-green-900 text-green-300 border-green-700",
  anthropic: "bg-orange-900 text-orange-300 border-orange-700",
  gemini:    "bg-blue-900  text-blue-300  border-blue-700",
};

const STATUS_DOT: Record<ConnectionStatus, string> = {
  connected:    "bg-ds-green",
  connecting:   "bg-ds-yellow animate-pulse",
  disconnected: "bg-ds-muted",
  error:        "bg-ds-red",
};

export const ModelSelector: React.FC<Props> = ({
  selectedModel,
  status,
  onSelect,
  onConnect,
  onDisconnect,
}) => {
  return (
    <div className="bg-ds-surface border border-ds-border rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ds-text uppercase tracking-wider">
          Model
        </h2>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${STATUS_DOT[status]}`} />
          <span className="text-xs text-ds-muted capitalize">{status}</span>
        </div>
      </div>

      {/* Model grid */}
      <div className="grid grid-cols-1 gap-2">
        {MODEL_OPTIONS.map((opt) => {
          const isSelected = selectedModel?.model === opt.model;
          return (
            <motion.button
              key={opt.model}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              onClick={() => onSelect(opt)}
              className={`
                flex items-center justify-between px-3 py-2 rounded-lg border text-left
                transition-colors text-sm
                ${isSelected
                  ? "border-ds-accent bg-blue-950 text-ds-text"
                  : "border-ds-border bg-ds-bg text-ds-muted hover:border-ds-accent hover:text-ds-text"
                }
              `}
            >
              <div className="flex items-center gap-2">
                <span
                  className={`text-xs px-1.5 py-0.5 rounded border font-mono ${PROVIDER_COLORS[opt.provider]}`}
                >
                  {opt.provider}
                </span>
                <span className="font-medium">{opt.label}</span>
              </div>
              <div className="flex gap-1">
                {opt.supportsLogprobs && (
                  <span className="text-xs bg-ds-surface border border-ds-border px-1.5 py-0.5 rounded text-ds-muted">
                    logprobs
                  </span>
                )}
                {opt.supportsCoT && (
                  <span className="text-xs bg-ds-surface border border-ds-border px-1.5 py-0.5 rounded text-ds-muted">
                    CoT
                  </span>
                )}
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* Connect / Disconnect */}
      <div className="flex gap-2 pt-1">
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={onConnect}
          disabled={!selectedModel || status === "connecting"}
          className="flex-1 py-2 rounded-lg bg-ds-accent text-ds-bg font-semibold text-sm
                     disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-400 transition-colors"
        >
          {status === "connecting" ? "Connecting…" : "Connect"}
        </motion.button>
        {status === "connected" && (
          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={onDisconnect}
            className="px-4 py-2 rounded-lg border border-ds-red text-ds-red text-sm
                       hover:bg-red-950 transition-colors"
          >
            Disconnect
          </motion.button>
        )}
      </div>
    </div>
  );
};
