import React, { useState, useCallback, useEffect, lazy, Suspense, memo } from "react";
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
import { DeceptionBreakdown }  from "./components/DeceptionBreakdown";
import { SessionPanel }        from "./components/SessionPanel";
import { ThemePicker }         from "./components/ThemePicker";
import { useWebSocket }        from "./hooks/useWebSocket";
import { THEMES, Theme, applyTheme, loadSavedTheme } from "./themes";
import {
  MODEL_OPTIONS, ModelOption, DeceptionResult,
  PROVIDER_META, scoreToColor, scoreToLabel,
} from "./types";

const ArchitecturePage = lazy(() =>
  import("./pages/ArchitecturePage").then(m => ({ default: m.ArchitecturePage }))
);

type RightTab  = "analysis" | "breakdown" | "signals" | "research" | "layers";
type AppPage   = "dashboard" | "architecture";
type MobileTab = "chat" | "score" | "detected" | "settings";

// ── Icons ─────────────────────────────────────────────────────────────────────
const Logo = memo(() => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden>
    <circle cx="10" cy="10" r="8.5" stroke="var(--border2)" strokeWidth="1.5" />
    <circle cx="10" cy="10" r="4"   stroke="var(--accent)"  strokeWidth="1.5" />
    <line x1="15" y1="15" x2="18" y2="18" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="10" cy="10" r="1.5" fill="var(--accent)" />
  </svg>
));

const ChevronIcon = memo(({ dir }: { dir: "left" | "right" }) => (
  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
    <path d={dir === "left" ? "M6.5 2L3.5 5L6.5 8" : "M3.5 2L6.5 5L3.5 8"}
      stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
));

// Mobile nav icons
const ChatIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <path d="M3 4h14a1 1 0 011 1v8a1 1 0 01-1 1H6l-3 3V5a1 1 0 011-1z"
      stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);
const ScoreIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="1.5" />
    <path d="M10 6v4l3 2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);
const DetectedIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <path d="M10 3L12.5 8H17L13.5 11.5L15 16.5L10 13.5L5 16.5L6.5 11.5L3 8H7.5L10 3Z"
      stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);
const SettingsIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <circle cx="10" cy="10" r="2.5" stroke="currentColor" strokeWidth="1.5" />
    <path d="M10 3v2M10 15v2M3 10h2M15 10h2M5.05 5.05l1.41 1.41M13.54 13.54l1.41 1.41M5.05 14.95l1.41-1.41M13.54 6.46l1.41-1.41"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const {
    status, messages, sendMessage, clearMessages,
    connect, disconnect, isTyping,
    sessionId, sessionRecords, exportSession, clearSession,
  } = useWebSocket();

  const [selectedModel, setSelectedModel]       = useState<ModelOption | null>(MODEL_OPTIONS[0]);
  const [steeringEnabled, setSteeringEnabled]   = useState(true);
  const [steeringStrength, setSteeringStrength] = useState(0.6);
  const [rightTab, setRightTab]   = useState<RightTab>("analysis");
  const [leftOpen, setLeftOpen]   = useState(true);
  const [page, setPage]           = useState<AppPage>("dashboard");
  const [mobileTab, setMobileTab] = useState<MobileTab>("chat");
  const [mobileSheet, setMobileSheet] = useState<"settings" | "score" | "detected" | null>(null);
  const [theme, setTheme]         = useState<Theme>(() => loadSavedTheme());

  // Apply saved theme on mount
  useEffect(() => { applyTheme(theme); }, []);

  const handleThemeChange = useCallback((t: Theme) => {
    applyTheme(t);
    setTheme(t);
  }, []);

  const latestResult: DeceptionResult | null =
    [...messages].reverse().find(m => m.role === "assistant" && m.deception)?.deception ?? null;

  const handleConnect = useCallback(() => {
    if (selectedModel) connect(selectedModel.provider, selectedModel.model);
  }, [selectedModel, connect]);

  const providerColor = selectedModel ? PROVIDER_META[selectedModel.provider].color : "var(--accent)";

  const RIGHT_TABS: { id: RightTab; label: string }[] = [
    { id: "analysis",  label: "Score"    },
    { id: "breakdown", label: "Detected" },
    { id: "signals",   label: "Signals"  },
    { id: "research",  label: "Research" },
    { id: "layers",    label: "Layers"   },
  ];

  const MOBILE_TABS: { id: MobileTab; label: string; icon: React.ReactNode }[] = [
    { id: "chat",     label: "Chat",     icon: <ChatIcon />     },
    { id: "score",    label: "Score",    icon: <ScoreIcon />    },
    { id: "detected", label: "Detected", icon: <DetectedIcon /> },
    { id: "settings", label: "Settings", icon: <SettingsIcon /> },
  ];

  // Right panel content — shared between desktop and mobile sheet
  const RightPanelContent = () => (
    <div className="space-y-2 p-2.5">
      <AnimatePresence mode="wait">
        {rightTab === "analysis" && (
          <motion.div key="analysis" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.12 }} className="space-y-2">
            <DeceptionScoreGauge result={latestResult} isLoading={isTyping} />
            <ConsistencyTimeline messages={messages} />
          </motion.div>
        )}
        {rightTab === "breakdown" && (
          <motion.div key="breakdown" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            <DeceptionBreakdown result={latestResult} isLoading={isTyping} />
          </motion.div>
        )}
        {rightTab === "signals" && (
          <motion.div key="signals" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            <BehavioralSignals result={latestResult} isLoading={isTyping} />
          </motion.div>
        )}
        {rightTab === "research" && (
          <motion.div key="research" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            <ResearchMode result={latestResult} />
          </motion.div>
        )}
        {rightTab === "layers" && (
          <motion.div key="layers" initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
            <LayerProbeViz layerScores={[]} modelName={selectedModel?.label} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  // Settings panel — shared between desktop sidebar and mobile sheet
  const SettingsContent = () => (
    <div className="space-y-2 p-2.5">
      <ModelSelector selectedModel={selectedModel} status={status}
        onSelect={setSelectedModel} onConnect={handleConnect} onDisconnect={disconnect} />
      <SteeringPanel enabled={steeringEnabled} onToggle={() => setSteeringEnabled(v => !v)}
        strength={steeringStrength} onStrengthChange={setSteeringStrength}
        lastResult={latestResult} steeredResult={null} />
      <SessionPanel sessionId={sessionId} records={sessionRecords}
        onExport={exportSession} onClear={clearSession} />
      <ExportReport messages={messages} modelLabel={selectedModel?.label ?? "Unknown"} />
    </div>
  );

  return (
    <div className="h-screen bg-bg text-ink flex flex-col overflow-hidden font-sans">

      {/* ── Topbar ──────────────────────────────────────────────────────────── */}
      <header className="flex-shrink-0 h-11 border-b border-border bg-surface flex items-center px-3 gap-3">
        {/* Brand */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <Logo />
          <span className="text-sm font-semibold text-ink tracking-tight">DeceptiScope</span>
          <span className="text-2xs font-mono px-1.5 py-px rounded border hidden xs:inline"
            style={{ color: providerColor, borderColor: providerColor + "40", background: providerColor + "12" }}>
            v2
          </span>
        </div>

        <div className="w-px h-4 bg-border flex-shrink-0 hidden sm:block" />

        {/* Page nav — desktop only */}
        <nav className="hidden sm:flex items-center gap-1">
          {(["dashboard", "architecture"] as AppPage[]).map(p => (
            <button key={p} onClick={() => setPage(p)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors
                ${page === p ? "bg-surface2 text-ink" : "text-ink3 hover:text-ink2 hover:bg-surface2"}`}>
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </nav>

        {/* Active model pill */}
        <AnimatePresence mode="wait">
          {selectedModel && page === "dashboard" && (
            <motion.div key={selectedModel.model} initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
              className="hidden md:flex items-center gap-1.5 text-xs text-ink2">
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: providerColor }} />
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
            <motion.div key={Math.round(latestResult.score * 100)}
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
              className="flex items-center gap-1.5 px-2 py-1 rounded-md border text-xs font-mono"
              style={{
                borderColor: scoreToColor(latestResult.score) + "35",
                background:  scoreToColor(latestResult.score) + "10",
                color:       scoreToColor(latestResult.score),
              }}>
              <span className="font-bold">{(latestResult.score * 100).toFixed(0)}%</span>
              <span className="text-2xs opacity-70 hidden xs:inline">{scoreToLabel(latestResult.score)}</span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Theme picker */}
        <ThemePicker currentTheme={theme} onThemeChange={handleThemeChange} />

        <span className="hidden xl:block text-2xs text-ink3 border border-border rounded px-2 py-1">
          Schmidt Sciences 2026
        </span>
      </header>

      {/* ── Page content ────────────────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {page === "architecture" ? (
          <motion.div key="architecture" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.15 }} className="flex-1 overflow-hidden">
            <Suspense fallback={
              <div className="flex-1 flex items-center justify-center text-ink3 text-xs">Loading…</div>
            }>
              <div className="h-full overflow-y-auto"><ArchitecturePage /></div>
            </Suspense>
          </motion.div>
        ) : (
          <motion.div key="dashboard" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.15 }}
            className="flex-1 flex overflow-hidden">

            {/* ── Desktop left sidebar ── */}
            <AnimatePresence initial={false}>
              {leftOpen && (
                <motion.aside key="left"
                  initial={{ width: 0, opacity: 0 }} animate={{ width: 264, opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={{ duration: 0.18, ease: [0.4, 0, 0.2, 1] }}
                  className="desktop-sidebar flex-shrink-0 border-r border-border overflow-hidden">
                  <div className="w-[264px] h-full overflow-y-auto">
                    <SettingsContent />
                  </div>
                </motion.aside>
              )}
            </AnimatePresence>

            {/* Sidebar toggle — desktop */}
            <button onClick={() => setLeftOpen(v => !v)}
              title={leftOpen ? "Collapse" : "Expand"}
              className="desktop-sidebar flex-shrink-0 w-5 border-r border-border bg-surface
                         hover:bg-surface2 flex items-center justify-center text-ink3
                         hover:text-ink2 transition-colors">
              <ChevronIcon dir={leftOpen ? "left" : "right"} />
            </button>

            {/* ── Chat (centre) ── */}
            <main className={`flex-1 min-w-0 p-2.5 mobile-chat-wrapper
              ${mobileTab !== "chat" ? "hidden md:block" : ""}`}>
              <ChatInterface messages={messages} status={status}
                onSend={sendMessage} onClear={clearSession}
                steeringEnabled={steeringEnabled}
                onToggleSteering={() => setSteeringEnabled(v => !v)}
                isTyping={isTyping} />
            </main>

            {/* ── Desktop right panel ── */}
            <aside className="desktop-right-panel flex-shrink-0 w-[272px] border-l border-border
                              flex flex-col overflow-hidden">
              {/* Tab bar */}
              <div className="flex-shrink-0 flex border-b border-border bg-surface overflow-x-auto">
                {RIGHT_TABS.map(t => (
                  <button key={t.id} onClick={() => setRightTab(t.id)}
                    className={`flex-shrink-0 px-2 py-2.5 text-2xs font-semibold uppercase
                                tracking-wider transition-colors relative whitespace-nowrap
                      ${rightTab === t.id ? "text-ink" : "text-ink3 hover:text-ink2"}`}>
                    {t.label}
                    {rightTab === t.id && (
                      <motion.div layoutId="rightTabLine"
                        className="absolute bottom-0 left-0 right-0 h-px bg-accent" />
                    )}
                  </button>
                ))}
              </div>
              <div className="flex-1 overflow-y-auto">
                <RightPanelContent />
              </div>
            </aside>

            {/* ── Mobile: Score tab ── */}
            {mobileTab === "score" && (
              <div className="md:hidden flex-1 overflow-y-auto p-3 space-y-3">
                <DeceptionScoreGauge result={latestResult} isLoading={isTyping} />
                <ConsistencyTimeline messages={messages} />
                <BehavioralSignals result={latestResult} isLoading={isTyping} />
              </div>
            )}

            {/* ── Mobile: Detected tab ── */}
            {mobileTab === "detected" && (
              <div className="md:hidden flex-1 overflow-y-auto p-3 space-y-3">
                <DeceptionBreakdown result={latestResult} isLoading={isTyping} />
                <ResearchMode result={latestResult} />
              </div>
            )}

            {/* ── Mobile: Settings tab ── */}
            {mobileTab === "settings" && (
              <div className="md:hidden flex-1 overflow-y-auto">
                <SettingsContent />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Mobile bottom navigation ─────────────────────────────────────────── */}
      <nav className="mobile-nav">
        {MOBILE_TABS.map(t => {
          const isActive = mobileTab === t.id;
          return (
            <button key={t.id} onClick={() => setMobileTab(t.id)}
              className="flex-1 flex flex-col items-center gap-0.5 py-1 transition-colors"
              style={{ color: isActive ? "var(--accent)" : "var(--ink3)" }}>
              <span className={`transition-transform ${isActive ? "scale-110" : ""}`}>
                {t.icon}
              </span>
              <span className="text-[9px] font-medium">{t.label}</span>
              {isActive && (
                <motion.div layoutId="mobileTabDot"
                  className="w-1 h-1 rounded-full bg-accent absolute bottom-1" />
              )}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
