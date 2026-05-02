import React, { useState, useCallback, lazy, Suspense, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ModelSelector }       from "./components/ModelSelector";
import { ChatInterface }       from "./components/ChatInterface";
import { DeceptionScoreGauge } from "./components/DeceptionScoreGauge";
import { ConsistencyTimeline } from "./components/ConsistencyTimeline";
import { LayerProbeViz }       from "./components/LayerProbeViz";
import { SteeringPanel }       from "./components/SteeringPanel";
import { ResearchMode }        from "./components/ResearchMode";
import { ExportReport }        from "./components/ExportReport";
import { BehavioralSignals }   from "./components/BehavioralSignals";
import { useWebSocket }        from "./hooks/useWebSocket";
import {
  MODEL_OPTIONS, ModelOption, DeceptionResult,
  PROVIDER_META, scoreToColor, scoreToLabel,
} from "./types";

// Lazy-load the architecture page — keeps main bundle small
const ArchitecturePage = lazy(() =>
  import("./pages/ArchitecturePage").then(m => ({ default: m.ArchitecturePage }))
);

type RightTab = "analysis" | "signals" | "research" | "layers";
type AppPage  = "dashboard" | "architecture";

// ── Shared icons ──────────────────────────────────────────────────────────────
const Logo = memo(() => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
    <circle cx="10" cy="10" r="8.5" stroke="#263040" strokeWidth="1.5" />
    <circle cx="10" cy="10" r="4"   stroke="#3b82f6" strokeWidth="1.5" />
    <line x1="15" y1="15" x2="18" y2="18" stroke="#3b82f6" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="10" cy="10" r="1.5" fill="#3b82f6" />
  </svg>
));

const Chevron = memo(({ dir }: { dir: "left" | "right" }) => (
  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
    <path
      d={dir === "left" ? "M6.5 2L3.5 5L6.5 8" : "M3.5 2L6.5 5L3.5 8"}
      stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"
    />
  </svg>
));

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const {
    status, messages, sendMessage, clearMessages,
    connect, disconnect, isTyping,
  } = useWebSocket();

  const [selectedModel, setSelectedModel]       = useState<ModelOption | null>(MODEL_OPTIONS[0]);
  const [steeringEnabled, setSteeringEnabled]   = useState(true);
  const [steeringStrength, setSteeringStrength] = useState(0.6);
  const [rightTab, setRightTab]   = useState<RightTab>("analysis");
  const [leftOpen, setLeftOpen]   = useState(true);
  const [page, setPage]           = useState<AppPage>("dashboard");

  const latestResult: DeceptionResult | null =
    [...messages].reverse().find(m => m.role === "assistant" && m.deception)?.deception ?? null;

  const handleConnect = useCallback(() => {
    if (selectedModel) connect(selectedModel.provider, selectedModel.model);
  }, [selectedModel, connect]);

  const providerColor = selectedModel ? PROVIDER_META[selectedModel.provider].color : "#3b82f6";

  const RIGHT_TABS: { id: RightTab; label: string }[] = [
    { id: "analysis", label: "Score"    },
    { id: "signals",  label: "Signals"  },
    { id: "research", label: "Research" },
    { id: "layers",   label: "Layers"   },
  ];

  return (
    <div className="h-screen bg-bg text-ink flex flex-col overflow-hidden font-sans">

      {/* ── Topbar ──────────────────────────────────────────────────────────── */}
      <header className="flex-shrink-0 h-11 border-b border-border bg-surface flex items-center px-4 gap-4">

        {/* Brand */}
        <div className="flex items-center gap-2.5 flex-shrink-0">
          <Logo />
          <span className="text-sm font-semibold text-ink tracking-tight">DeceptiScope</span>
          <span
            className="text-2xs font-mono px-1.5 py-px rounded border"
            style={{
              color: providerColor,
              borderColor: providerColor + "40",
              background: providerColor + "12",
            }}
          >
            v2
          </span>
        </div>

        <div className="w-px h-4 bg-border flex-shrink-0" />

        {/* Page nav */}
        <nav className="flex items-center gap-1">
          {(["dashboard", "architecture"] as AppPage[]).map(p => (
            <button
              key={p}
              onClick={() => setPage(p)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors
                ${page === p
                  ? "bg-surface2 text-ink"
                  : "text-ink3 hover:text-ink2 hover:bg-surface2/50"
                }`}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </nav>

        {/* Active model pill */}
        <AnimatePresence mode="wait">
          {selectedModel && page === "dashboard" && (
            <motion.div
              key={selectedModel.model}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 text-xs text-ink2"
            >
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: providerColor }}
              />
              <span className="font-medium">{selectedModel.label}</span>
              <span className="text-ink3">·</span>
              <span className="text-ink3">{PROVIDER_META[selectedModel.provider].label}</span>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="flex-1" />

        {/* Live score badge */}
        <AnimatePresence>
          {latestResult && page === "dashboard" && (
            <motion.div
              key={Math.round(latestResult.score * 100)}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-2 px-2.5 py-1 rounded-md border text-xs font-mono"
              style={{
                borderColor: scoreToColor(latestResult.score) + "35",
                background:  scoreToColor(latestResult.score) + "10",
                color:       scoreToColor(latestResult.score),
              }}
            >
              <span className="font-bold">{(latestResult.score * 100).toFixed(0)}%</span>
              <span className="text-2xs opacity-70">{scoreToLabel(latestResult.score)}</span>
            </motion.div>
          )}
        </AnimatePresence>

        <span className="hidden lg:block text-2xs text-ink3 border border-border rounded px-2 py-1">
          Schmidt Sciences 2026
        </span>
      </header>

      {/* ── Page content ────────────────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {page === "architecture" ? (
          <motion.div
            key="architecture"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="flex-1 overflow-hidden"
          >
            <Suspense
              fallback={
                <div className="flex-1 flex items-center justify-center text-ink3 text-xs">
                  Loading…
                </div>
              }
            >
              <div className="h-full overflow-y-auto">
                <ArchitecturePage />
              </div>
            </Suspense>
          </motion.div>
        ) : (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="flex-1 flex overflow-hidden"
          >
            {/* Left sidebar */}
            <AnimatePresence initial={false}>
              {leftOpen && (
                <motion.aside
                  key="left"
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 264, opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
                  className="flex-shrink-0 border-r border-border overflow-hidden"
                >
                  <div className="w-[264px] h-full overflow-y-auto p-2.5 space-y-2">
                    <ModelSelector
                      selectedModel={selectedModel}
                      status={status}
                      onSelect={setSelectedModel}
                      onConnect={handleConnect}
                      onDisconnect={disconnect}
                    />
                    <SteeringPanel
                      enabled={steeringEnabled}
                      onToggle={() => setSteeringEnabled(v => !v)}
                      strength={steeringStrength}
                      onStrengthChange={setSteeringStrength}
                      lastResult={latestResult}
                      steeredResult={null}
                    />
                    <ExportReport
                      messages={messages}
                      modelLabel={selectedModel?.label ?? "Unknown"}
                    />
                  </div>
                </motion.aside>
              )}
            </AnimatePresence>

            {/* Sidebar toggle */}
            <button
              onClick={() => setLeftOpen(v => !v)}
              title={leftOpen ? "Collapse sidebar" : "Expand sidebar"}
              className="flex-shrink-0 w-5 border-r border-border bg-surface hover:bg-surface2
                         flex items-center justify-center text-ink3 hover:text-ink2 transition-colors"
            >
              <Chevron dir={leftOpen ? "left" : "right"} />
            </button>

            {/* Chat */}
            <main className="flex-1 min-w-0 p-2.5">
              <ChatInterface
                messages={messages}
                status={status}
                onSend={sendMessage}
                onClear={clearMessages}
                steeringEnabled={steeringEnabled}
                onToggleSteering={() => setSteeringEnabled(v => !v)}
                isTyping={isTyping}
              />
            </main>

            {/* Right panel */}
            <aside className="flex-shrink-0 w-[272px] border-l border-border flex flex-col overflow-hidden">

              {/* Tab bar */}
              <div className="flex-shrink-0 flex border-b border-border bg-surface">
                {RIGHT_TABS.map(t => (
                  <button
                    key={t.id}
                    onClick={() => setRightTab(t.id)}
                    className={`flex-1 py-2.5 text-2xs font-semibold uppercase tracking-wider
                                transition-colors relative
                      ${rightTab === t.id ? "text-ink" : "text-ink3 hover:text-ink2"}`}
                  >
                    {t.label}
                    {rightTab === t.id && (
                      <motion.div
                        layoutId="rightTabLine"
                        className="absolute bottom-0 left-0 right-0 h-px bg-accent"
                      />
                    )}
                  </button>
                ))}
              </div>

              {/* Tab content */}
              <div className="flex-1 overflow-y-auto p-2.5 space-y-2">
                <AnimatePresence mode="wait">
                  {rightTab === "analysis" && (
                    <motion.div
                      key="analysis"
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.12 }}
                      className="space-y-2"
                    >
                      <DeceptionScoreGauge result={latestResult} isLoading={isTyping} />
                      <ConsistencyTimeline messages={messages} />
                    </motion.div>
                  )}

                  {rightTab === "signals" && (
                    <motion.div
                      key="signals"
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.12 }}
                    >
                      <BehavioralSignals result={latestResult} isLoading={isTyping} />
                    </motion.div>
                  )}

                  {rightTab === "research" && (
                    <motion.div
                      key="research"
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.12 }}
                    >
                      <ResearchMode result={latestResult} />
                    </motion.div>
                  )}

                  {rightTab === "layers" && (
                    <motion.div
                      key="layers"
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.12 }}
                    >
                      <LayerProbeViz layerScores={[]} modelName={selectedModel?.label} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </aside>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
