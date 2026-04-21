/**
 * DeceptiScope v2 — Main Dashboard
 *
 * Layout:
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  Header                                                 │
 *  ├──────────────┬──────────────────────────┬──────────────┤
 *  │ Left sidebar │   Chat (centre)          │ Right panel  │
 *  │  ModelSelect │                          │  Score gauge │
 *  │  Steering    │                          │  Timeline    │
 *  │  Export      │                          │  LayerProbe  │
 *  │              │                          │  Research    │
 *  └──────────────┴──────────────────────────┴──────────────┘
 */

import React, { useState, useCallback } from "react";
import { ModelSelector } from "./components/ModelSelector";
import { ChatInterface } from "./components/ChatInterface";
import { DeceptionScoreGauge } from "./components/DeceptionScoreGauge";
import { ConsistencyTimeline } from "./components/ConsistencyTimeline";
import { LayerProbeViz } from "./components/LayerProbeViz";
import { SteeringPanel } from "./components/SteeringPanel";
import { ResearchMode } from "./components/ResearchMode";
import { ExportReport } from "./components/ExportReport";
import { useWebSocket } from "./hooks/useWebSocket";
import { MODEL_OPTIONS, ModelOption, DeceptionResult } from "./types";

export default function App() {
  const { status, messages, sendMessage, clearMessages, connect, disconnect } =
    useWebSocket();

  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(
    MODEL_OPTIONS[0]
  );
  const [steeringEnabled, setSteeringEnabled] = useState(true);
  const [steeringStrength, setSteeringStrength] = useState(0.6);

  // Latest assistant deception result for the right panel
  const latestResult: DeceptionResult | null =
    [...messages]
      .reverse()
      .find((m) => m.role === "assistant" && m.deception)
      ?.deception ?? null;

  const handleConnect = useCallback(() => {
    if (!selectedModel) return;
    connect(selectedModel.provider, selectedModel.model);
  }, [selectedModel, connect]);

  const handleSend = useCallback(
    (text: string, steering: boolean) => {
      sendMessage(text, steering);
    },
    [sendMessage]
  );

  return (
    <div className="min-h-screen bg-ds-bg text-ds-text flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="border-b border-ds-border bg-ds-surface px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xl">🔍</span>
          <div>
            <h1 className="text-base font-bold text-ds-text tracking-tight">
              DeceptiScope <span className="text-ds-accent">v2</span>
            </h1>
            <p className="text-xs text-ds-muted">
              AI Interpretability · Deception Detection · Frontier + Open Models
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs text-ds-muted">
          <span className="hidden md:block">
            Schmidt Sciences 2026 Interpretability RFP
          </span>
          {selectedModel && (
            <span className="bg-ds-bg border border-ds-border rounded px-2 py-1 font-mono">
              {selectedModel.label}
            </span>
          )}
        </div>
      </header>

      {/* ── Main layout ────────────────────────────────────────────── */}
      <main className="flex-1 flex gap-4 p-4 overflow-hidden">
        {/* Left sidebar */}
        <aside className="w-64 flex-shrink-0 flex flex-col gap-4 overflow-y-auto">
          <ModelSelector
            selectedModel={selectedModel}
            status={status}
            onSelect={setSelectedModel}
            onConnect={handleConnect}
            onDisconnect={disconnect}
          />
          <SteeringPanel
            enabled={steeringEnabled}
            onToggle={() => setSteeringEnabled((v) => !v)}
            strength={steeringStrength}
            onStrengthChange={setSteeringStrength}
            lastResult={latestResult}
            steeredResult={null}
          />
          <ExportReport
            messages={messages}
            modelLabel={selectedModel?.label ?? "Unknown"}
          />
        </aside>

        {/* Centre — chat */}
        <section className="flex-1 min-w-0">
          <ChatInterface
            messages={messages}
            status={status}
            onSend={handleSend}
            onClear={clearMessages}
            steeringEnabled={steeringEnabled}
            onToggleSteering={() => setSteeringEnabled((v) => !v)}
          />
        </section>

        {/* Right panel — analysis */}
        <aside className="w-72 flex-shrink-0 flex flex-col gap-4 overflow-y-auto">
          <DeceptionScoreGauge result={latestResult} />
          <ConsistencyTimeline messages={messages} />
          <LayerProbeViz layerScores={[]} modelName={selectedModel?.label} />
          <ResearchMode result={latestResult} />
        </aside>
      </main>
    </div>
  );
}
