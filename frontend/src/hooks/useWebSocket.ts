import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, DeceptionResult, ModelProvider } from "../types";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

export interface SessionRecord {
  id: string;
  timestamp: number;
  prompt: string;
  response: string;
  deception_score: number;
  deception_type: string;
  confidence: number;
  explanation: string;
  model: string;
  provider: string;
}

interface UseWebSocketReturn {
  status: ConnectionStatus;
  messages: ChatMessage[];
  sendMessage: (text: string, enableSteering: boolean) => void;
  clearMessages: () => void;
  connect: (provider: ModelProvider, model: string) => void;
  disconnect: () => void;
  isTyping: boolean;
  sessionId: string | null;
  sessionRecords: SessionRecord[];
  exportSession: (fmt: "json" | "csv") => Promise<void>;
  clearSession: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus]     = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId]         = useState<string | null>(null);
  const [sessionRecords, setSessionRecords] = useState<SessionRecord[]>([]);

  const providerRef = useRef<string>("groq");
  const modelRef    = useRef<string>("llama-3.3-70b-versatile");
  const abortRef    = useRef<AbortController | null>(null);

  // Create a session on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/sessions`, { method: "POST" })
      .then(r => r.json())
      .then(d => setSessionId(d.session_id))
      .catch(() => {}); // silent — session is optional
  }, []);

  const connect = useCallback(async (provider: ModelProvider, model: string) => {
    providerRef.current = provider;
    modelRef.current    = model;
    setStatus("connecting");
    try {
      const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(5000) });
      setStatus(res.ok ? "connected" : "error");
    } catch {
      setStatus("connected"); // optimistic — REST call will surface real errors
    }
  }, []);

  const disconnect = useCallback(() => {
    abortRef.current?.abort();
    setStatus("disconnected");
  }, []);

  const sendMessage = useCallback((text: string, enableSteering: boolean) => {
    if (!text.trim()) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(), role: "user",
      content: text, timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setIsTyping(true);

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
        session_id:      sessionId,
      }),
    })
      .then(r => {
        if (!r.ok) return r.json().then(d => { throw new Error(d.detail || `HTTP ${r.status}`); });
        return r.json();
      })
      .then(data => {
        setIsTyping(false);

        if (data.detail || data.error) {
          setMessages(prev => [...prev, {
            id: crypto.randomUUID(), role: "assistant",
            content: `Error: ${data.detail || data.error}`, timestamp: Date.now(),
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
          deception_breakdown:  data.deception_breakdown ?? [],
        };

        setMessages(prev => [...prev, {
          id: crypto.randomUUID(), role: "assistant",
          content: data.response ?? "",
          timestamp: Date.now(),
          deception,
          tokens: data.response?.split(" ") ?? [],
        }]);

        // Refresh session records
        if (sessionId) {
          fetch(`${API_BASE}/api/sessions/${sessionId}`)
            .then(r => r.json())
            .then(d => setSessionRecords(d.records ?? []))
            .catch(() => {});
        }
      })
      .catch(err => {
        if (err.name === "AbortError") return;
        setIsTyping(false);
        setMessages(prev => [...prev, {
          id: crypto.randomUUID(), role: "assistant",
          content: `Error: ${err.message}`,
          timestamp: Date.now(),
        }]);
      });
  }, [sessionId]);

  const clearMessages = useCallback(() => setMessages([]), []);

  const exportSession = useCallback(async (fmt: "json" | "csv") => {
    if (!sessionId) return;
    const url = `${API_BASE}/api/sessions/${sessionId}/export?fmt=${fmt}`;
    const r = await fetch(url);
    const blob = await r.blob();
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `deceptiscope-session-${sessionId.slice(0, 8)}.${fmt}`,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  }, [sessionId]);

  const clearSession = useCallback(() => {
    if (sessionId) {
      fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
    }
    // Create a new session
    fetch(`${API_BASE}/api/sessions`, { method: "POST" })
      .then(r => r.json())
      .then(d => { setSessionId(d.session_id); setSessionRecords([]); })
      .catch(() => {});
    setMessages([]);
  }, [sessionId]);

  useEffect(() => () => { abortRef.current?.abort(); }, []);

  return {
    status, messages, sendMessage, clearMessages,
    connect, disconnect, isTyping,
    sessionId, sessionRecords, exportSession, clearSession,
  };
}
