/**
 * useDeceptiScope — manages communication with the DeceptiScope backend.
 *
 * Strategy: REST-first (POST /api/chat) with WebSocket upgrade when available.
 * REST works immediately without a persistent connection — no 400 path issues.
 * "Connect" just validates the model selection and marks status as ready.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, DeceptionResult, ModelProvider } from "../types";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

interface UseWebSocketReturn {
  status: ConnectionStatus;
  messages: ChatMessage[];
  sendMessage: (text: string, enableSteering: boolean) => void;
  clearMessages: () => void;
  connect: (provider: ModelProvider, model: string) => void;
  disconnect: () => void;
  isTyping: boolean;
}

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus]   = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const providerRef = useRef<string>("groq");
  const modelRef    = useRef<string>("llama-3.1-8b-instant");
  const abortRef    = useRef<AbortController | null>(null);

  // "Connect" = validate backend reachability + store provider/model
  const connect = useCallback(async (provider: ModelProvider, model: string) => {
    providerRef.current = provider;
    modelRef.current    = model;
    setStatus("connecting");

    try {
      const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        setStatus("connected");
      } else {
        setStatus("error");
      }
    } catch {
      // Backend not reachable — still mark connected so user can try
      // (REST call will show the real error)
      setStatus("connected");
    }
  }, []);

  const disconnect = useCallback(() => {
    abortRef.current?.abort();
    setStatus("disconnected");
  }, []);

  const sendMessage = useCallback((text: string, enableSteering: boolean) => {
    if (!text.trim()) return;

    // Add user message immediately
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setIsTyping(true);

    // Cancel any in-flight request
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: ctrl.signal,
      body: JSON.stringify({
        message:         text,
        provider:        providerRef.current,
        model:           modelRef.current,
        enable_steering: enableSteering,
      }),
    })
      .then(r => {
        if (!r.ok) return r.text().then(t => { throw new Error(`HTTP ${r.status}: ${t}`); });
        return r.json();
      })
      .then(data => {
        setIsTyping(false);

        if (data.detail || data.error) {
          const errText = data.detail || data.error;
          setMessages(prev => [...prev, {
            id: crypto.randomUUID(), role: "assistant",
            content: `⚠ ${errText}`, timestamp: Date.now(),
          }]);
          return;
        }

        const deception: DeceptionResult = {
          score:                data.deception_score  ?? 0,
          deception_type:       data.deception_type   ?? "none",
          confidence:           data.confidence        ?? 0,
          explanation:          data.explanation       ?? "",
          per_token_scores:     data.token_analysis?.per_token_scores ?? [],
          high_risk_tokens:     data.token_analysis?.high_risk_tokens ?? [],
          type_scores:          data.type_scores       ?? {},
          signal_contributions: data.signal_contributions ?? {},
          raw_signals:          {},
          behavioral_signals:   data.behavioral_signals ?? {},
        };

        setMessages(prev => [...prev, {
          id: crypto.randomUUID(), role: "assistant",
          content: data.response ?? "",
          timestamp: Date.now(),
          deception,
          tokens: data.response?.split(" ") ?? [],
        }]);
      })
      .catch(err => {
        if (err.name === "AbortError") return;
        setIsTyping(false);
        setMessages(prev => [...prev, {
          id: crypto.randomUUID(), role: "assistant",
          content: `⚠ ${err.message}`,
          timestamp: Date.now(),
        }]);
      });
  }, []);

  const clearMessages = useCallback(() => setMessages([]), []);

  useEffect(() => () => { abortRef.current?.abort(); }, []);

  return { status, messages, sendMessage, clearMessages, connect, disconnect, isTyping };
}
