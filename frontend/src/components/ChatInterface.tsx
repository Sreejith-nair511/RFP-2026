/**
 * ChatInterface — streaming chat with any connected frontier model.
 *
 * Features:
 *  - Message history with role-based styling
 *  - Per-message deception heatmap overlay (toggle)
 *  - Steering toggle (injects honest-direction system prompt)
 *  - Auto-scroll to latest message
 *  - Keyboard shortcut: Enter to send, Shift+Enter for newline
 */

import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChatMessage, DeceptionResult } from "../types";
import { DeceptionHeatmap } from "./DeceptionHeatmap";
import { ConnectionStatus } from "../hooks/useWebSocket";

interface Props {
  messages: ChatMessage[];
  status: ConnectionStatus;
  onSend: (text: string, enableSteering: boolean) => void;
  onClear: () => void;
  steeringEnabled: boolean;
  onToggleSteering: () => void;
}

const ScoreBadge: React.FC<{ deception: DeceptionResult }> = ({ deception }) => {
  const pct = (deception.score * 100).toFixed(0);
  const color =
    deception.score < 0.3 ? "text-ds-green border-green-800" :
    deception.score < 0.55 ? "text-ds-yellow border-yellow-800" :
    "text-ds-red border-red-800";

  return (
    <span className={`text-xs font-mono border rounded px-1.5 py-0.5 ${color}`}>
      {pct}% deceptive
    </span>
  );
};

const MessageBubble: React.FC<{ msg: ChatMessage }> = ({ msg }) => {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const isUser = msg.role === "user";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[80%] rounded-xl px-4 py-3 space-y-2 ${
          isUser
            ? "bg-ds-accent/20 border border-ds-accent/30 text-ds-text"
            : "bg-ds-surface border border-ds-border text-ds-text"
        }`}
      >
        {/* Role label */}
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs text-ds-muted font-semibold uppercase tracking-wider">
            {isUser ? "You" : "Assistant"}
          </span>
          {msg.deception && (
            <div className="flex items-center gap-2">
              <ScoreBadge deception={msg.deception} />
              <button
                onClick={() => setShowHeatmap((v) => !v)}
                className="text-xs text-ds-muted hover:text-ds-accent transition-colors"
              >
                {showHeatmap ? "hide heatmap" : "show heatmap"}
              </button>
            </div>
          )}
        </div>

        {/* Content — heatmap or plain */}
        {showHeatmap && msg.deception ? (
          <DeceptionHeatmap message={msg} />
        ) : (
          <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
            {msg.content}
          </p>
        )}

        {/* Explanation */}
        {msg.deception?.explanation && (
          <p className="text-xs text-ds-muted italic border-t border-ds-border pt-2">
            {msg.deception.explanation}
          </p>
        )}

        <span className="text-xs text-ds-muted/60">
          {new Date(msg.timestamp).toLocaleTimeString()}
        </span>
      </div>
    </motion.div>
  );
};

export const ChatInterface: React.FC<Props> = ({
  messages,
  status,
  onSend,
  onClear,
  steeringEnabled,
  onToggleSteering,
}) => {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || status !== "connected") return;
    onSend(text, steeringEnabled);
    setInput("");
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full bg-ds-bg rounded-xl border border-ds-border overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-ds-border bg-ds-surface">
        <h2 className="text-sm font-semibold text-ds-text">Chat</h2>
        <div className="flex items-center gap-3">
          {/* Steering toggle */}
          <label className="flex items-center gap-2 cursor-pointer">
            <span className="text-xs text-ds-muted">Steering</span>
            <div
              onClick={onToggleSteering}
              className={`relative w-8 h-4 rounded-full transition-colors ${
                steeringEnabled ? "bg-ds-accent" : "bg-ds-border"
              }`}
            >
              <div
                className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                  steeringEnabled ? "translate-x-4" : "translate-x-0.5"
                }`}
              />
            </div>
          </label>
          <button
            onClick={onClear}
            className="text-xs text-ds-muted hover:text-ds-red transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence initial={false}>
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-full text-center py-16 space-y-2"
            >
              <span className="text-4xl">🔍</span>
              <p className="text-ds-muted text-sm">
                Connect a model and start chatting.
              </p>
              <p className="text-ds-muted/60 text-xs">
                Every response is analysed for deception in real time.
              </p>
            </motion.div>
          ) : (
            messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)
          )}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-ds-border p-3 bg-ds-surface">
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              status === "connected"
                ? "Type a message… (Enter to send, Shift+Enter for newline)"
                : "Connect a model to start chatting"
            }
            disabled={status !== "connected"}
            rows={2}
            className="flex-1 bg-ds-bg border border-ds-border rounded-lg px-3 py-2 text-sm
                       text-ds-text placeholder-ds-muted resize-none focus:outline-none
                       focus:border-ds-accent disabled:opacity-40 transition-colors"
          />
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={!input.trim() || status !== "connected"}
            className="px-4 py-2 bg-ds-accent text-ds-bg rounded-lg font-semibold text-sm
                       disabled:opacity-40 disabled:cursor-not-allowed hover:bg-blue-400
                       transition-colors self-end"
          >
            Send
          </motion.button>
        </div>
      </div>
    </div>
  );
};
