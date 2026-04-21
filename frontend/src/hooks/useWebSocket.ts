/**
 * useWebSocket — manages the WebSocket connection to the DeceptiScope backend.
 *
 * Handles:
 *  - Connection lifecycle (open / close / reconnect)
 *  - Sending chat messages
 *  - Parsing incoming DeceptionResult payloads
 *  - Exposing connection status for the UI
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ChatMessage, DeceptionResult, ModelProvider } from "../types";

const WS_BASE = process.env.REACT_APP_WS_URL || "ws://localhost:8000";

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";

interface UseWebSocketReturn {
  status: ConnectionStatus;
  messages: ChatMessage[];
  sendMessage: (text: string, enableSteering: boolean) => void;
  clearMessages: () => void;
  connect: (provider: ModelProvider, model: string) => void;
  disconnect: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const pendingUserMsg = useRef<ChatMessage | null>(null);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("disconnected");
  }, []);

  const connect = useCallback(
    (provider: ModelProvider, model: string) => {
      // Close any existing connection first
      wsRef.current?.close();

      const url = `${WS_BASE}/ws/chat/${provider}_${model}`;
      setStatus("connecting");

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
      };

      ws.onclose = () => {
        setStatus("disconnected");
      };

      ws.onerror = () => {
        setStatus("error");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.error) {
            console.error("Backend error:", data.error);
            return;
          }

          // Build the assistant message with deception analysis
          const deception: DeceptionResult = {
            score:               data.deception_score ?? 0,
            deception_type:      data.deception_type  ?? "none",
            confidence:          data.confidence       ?? 0,
            explanation:         data.explanation      ?? "",
            per_token_scores:    data.token_analysis?.per_token_scores ?? [],
            high_risk_tokens:    data.token_analysis?.high_risk_tokens ?? [],
            type_scores:         data.type_scores      ?? {},
            signal_contributions: data.signal_contributions ?? {},
            raw_signals:         data.raw_signals      ?? {},
          };

          const assistantMsg: ChatMessage = {
            id:        crypto.randomUUID(),
            role:      "assistant",
            content:   data.response ?? "",
            timestamp: Date.now(),
            deception,
            tokens:    data.response?.split(" ") ?? [],
          };

          setMessages((prev) => [...prev, assistantMsg]);
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err);
        }
      };
    },
    []
  );

  const sendMessage = useCallback(
    (text: string, enableSteering: boolean) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.warn("WebSocket not connected");
        return;
      }

      // Optimistically add user message to the list
      const userMsg: ChatMessage = {
        id:        crypto.randomUUID(),
        role:      "user",
        content:   text,
        timestamp: Date.now(),
      };
      pendingUserMsg.current = userMsg;
      setMessages((prev) => [...prev, userMsg]);

      wsRef.current.send(
        JSON.stringify({ message: text, enable_steering: enableSteering })
      );
    },
    []
  );

  const clearMessages = useCallback(() => setMessages([]), []);

  // Cleanup on unmount
  useEffect(() => () => wsRef.current?.close(), []);

  return { status, messages, sendMessage, clearMessages, connect, disconnect };
}
