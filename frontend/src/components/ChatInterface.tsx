import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChatMessage, TYPE_COLOR, TYPE_LABEL, TYPE_SIGIL, scoreToColor, scoreToLabel } from "../types";
import { DeceptionHeatmap } from "./DeceptionHeatmap";
import { ConnectionStatus } from "../hooks/useWebSocket";

interface Props {
  messages: ChatMessage[];
  status: ConnectionStatus;
  onSend: (text: string, steering: boolean) => void;
  onClear: () => void;
  steeringEnabled: boolean;
  onToggleSteering: () => void;
  isTyping: boolean;
}

const SUGGESTIONS = [
  "What is the capital of Australia?",
  "Are vaccines safe?",
  "Will this investment definitely return 20%?",
  "Is this supplement completely risk-free?",
];

const TypingDots = () => (
  <motion.div
    initial={{ opacity: 0, y: 4 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0 }}
    className="flex justify-start"
  >
    <div className="bg-surface border border-border rounded-lg rounded-tl-sm px-3 py-2.5">
      <div className="flex gap-1 items-center">
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            className="w-1 h-1 rounded-full bg-ink3"
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
    </div>
  </motion.div>
);

const ScoreChip: React.FC<{ score: number; type: string }> = ({ score, type }) => {
  const color = scoreToColor(score);
  return (
    <div
      className="flex items-center gap-1.5 px-2 py-0.5 rounded-sm border"
      style={{ borderColor: color + "30", background: color + "0d" }}
    >
      <span className="text-2xs font-mono font-bold" style={{ color }}>
        {TYPE_SIGIL[type]}
      </span>
      <span className="text-2xs font-medium" style={{ color }}>
        {TYPE_LABEL[type]}
      </span>
      <span className="text-2xs font-mono ml-1" style={{ color }}>
        {(score * 100).toFixed(0)}%
      </span>
    </div>
  );
};

const MessageBubble: React.FC<{ msg: ChatMessage }> = ({ msg }) => {
  const [heatmap, setHeatmap] = useState(false);
  const isUser = msg.role === "user";
  const d = msg.deception;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div className={`max-w-[82%] flex flex-col gap-1 ${isUser ? "items-end" : "items-start"}`}>

        {/* Role */}
        <span className={`text-2xs font-semibold tracking-wider px-0.5
          ${isUser ? "text-accent" : "text-ink3"}`}
        >
          {isUser ? "You" : "Assistant"}
        </span>

        {/* Bubble */}
        <div className={`rounded-lg px-3.5 py-2.5 space-y-2
          ${isUser
            ? "bg-surface2 border border-border2 rounded-tr-sm"
            : "bg-surface  border border-border  rounded-tl-sm"
          }`}
        >
          {/* Deception badge */}
          {d && (
            <div className="flex items-center gap-2 flex-wrap">
              <ScoreChip score={d.score} type={d.deception_type} />
              <button
                onClick={() => setHeatmap(v => !v)}
                className="text-2xs text-ink3 hover:text-accent transition-colors ml-auto"
              >
                {heatmap ? "Plain text" : "Heatmap"}
              </button>
            </div>
          )}

          {/* Content */}
          {heatmap && d ? (
            <DeceptionHeatmap message={msg} />
          ) : (
            <p className="text-sm text-ink leading-relaxed whitespace-pre-wrap break-words">
              {msg.content}
            </p>
          )}

          {/* Signal row */}
          {d?.behavioral_signals && (
            <div className="flex gap-3 pt-1.5 border-t border-border">
              {d.behavioral_signals.entropy != null && (
                <span className="text-2xs text-ink3">
                  entropy{" "}
                  <span className="font-mono text-ink2">
                    {(d.behavioral_signals.entropy * 100).toFixed(0)}%
                  </span>
                </span>
              )}
              {d.behavioral_signals.consistency != null && (
                <span className="text-2xs text-ink3">
                  consistency{" "}
                  <span className="font-mono text-ink2">
                    {(d.behavioral_signals.consistency * 100).toFixed(0)}%
                  </span>
                </span>
              )}
            </div>
          )}
        </div>

        <span className="text-2xs text-ink3 px-0.5">
          {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
    </motion.div>
  );
};

export const ChatInterface: React.FC<Props> = ({
  messages, status, onSend, onClear, steeringEnabled, onToggleSteering, isTyping,
}) => {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const send = () => {
    const t = input.trim();
    if (!t || status === "connecting") return;
    onSend(t, steeringEnabled);
    setInput("");
    textareaRef.current?.focus();
  };

  const canSend = input.trim().length > 0 && status !== "connecting";

  return (
    <div className="flex flex-col h-full bg-bg border border-border rounded-lg overflow-hidden">

      {/* Toolbar */}
      <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-border bg-surface">
        <div className="flex items-center gap-2.5">
          <span className="text-sm font-semibold text-ink">Chat</span>
          <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-sm border text-2xs font-mono
            ${status === "connected"
              ? "border-success/30 bg-success/8 text-success"
              : status === "connecting"
              ? "border-warn/30 bg-warn/8 text-warn"
              : "border-border text-ink3"
            }`}
          >
            <span className={`w-1 h-1 rounded-full ${
              status === "connected" ? "bg-success" :
              status === "connecting" ? "bg-warn animate-pulse" : "bg-ink3"
            }`} />
            {status}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Steering toggle */}
          <div className="flex items-center gap-2">
            <span className="text-2xs text-ink3 select-none">Steering</span>
            <button
              onClick={onToggleSteering}
              className={`relative w-8 h-4 rounded-full transition-colors ${
                steeringEnabled ? "bg-accent" : "bg-border2"
              }`}
            >
              <motion.div
                className="absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm"
                animate={{ x: steeringEnabled ? 16 : 2 }}
                transition={{ type: "spring", stiffness: 600, damping: 35 }}
              />
            </button>
          </div>

          <button
            onClick={onClear}
            className="text-2xs text-ink3 hover:text-danger transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        <AnimatePresence initial={false}>
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-full py-16 space-y-4 text-center"
            >
              {/* Logo mark */}
              <div className="w-10 h-10 rounded-lg bg-surface border border-border flex items-center justify-center">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <circle cx="9" cy="9" r="7" stroke="#263040" strokeWidth="1.5" />
                  <circle cx="9" cy="9" r="3.5" stroke="#3b82f6" strokeWidth="1.5" />
                  <line x1="14" y1="14" x2="16.5" y2="16.5" stroke="#3b82f6" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
              </div>

              <div className="space-y-1">
                <p className="text-sm font-medium text-ink2">Select a model and connect</p>
                <p className="text-xs text-ink3 max-w-xs leading-relaxed">
                  Every response is analysed for deception using behavioral probing and fusion scoring.
                </p>
              </div>

              {/* Suggestion chips */}
              <div className="flex flex-wrap gap-1.5 justify-center max-w-sm">
                {SUGGESTIONS.map(s => (
                  <button
                    key={s}
                    onClick={() => setInput(s)}
                    className="text-2xs text-ink3 border border-border rounded-md px-2 py-1
                               hover:border-accent/40 hover:text-accent transition-all"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </motion.div>
          ) : (
            messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)
          )}
          {isTyping && <TypingDots />}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border bg-surface px-3 py-2.5">
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={
              status === "connecting" ? "Connecting…" :
              status === "connected"  ? "Send a message  (Enter)" :
              "Connect a model to begin"
            }
            rows={2}
            className="flex-1 bg-bg border border-border rounded-md px-3 py-2 text-sm text-ink
                       placeholder-ink3 resize-none focus:outline-none focus:border-accent/50
                       transition-colors leading-relaxed"
          />
          <button
            onClick={send}
            disabled={!canSend}
            className="px-3.5 py-2 bg-accent text-white rounded-md text-sm font-semibold
                       disabled:opacity-25 disabled:cursor-not-allowed hover:bg-blue-500
                       transition-colors self-end"
          >
            Send
          </button>
        </div>
        <p className="text-2xs text-ink3 mt-1.5 px-0.5">
          Shift+Enter for newline
          {steeringEnabled && <span className="ml-2 text-accent/60">Steering active</span>}
        </p>
      </div>
    </div>
  );
};
