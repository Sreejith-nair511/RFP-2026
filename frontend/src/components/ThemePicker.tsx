/**
 * ThemePicker — floating theme selector.
 * Shows swatches for all 8 themes, applies instantly on click.
 */

import React, { useState, useRef, useEffect, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { THEMES, Theme, applyTheme } from "../themes";

interface Props {
  currentTheme: Theme;
  onThemeChange: (t: Theme) => void;
}

export const ThemePicker: React.FC<Props> = memo(({ currentTheme, onThemeChange }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const select = (t: Theme) => {
    applyTheme(t);
    onThemeChange(t);
    setOpen(false);
  };

  return (
    <div ref={ref} className="relative">
      {/* Trigger button */}
      <button
        onClick={() => setOpen(v => !v)}
        title="Change theme"
        className="flex items-center gap-1.5 px-2 py-1 rounded-md border border-[var(--border)]
                   bg-[var(--surface)] hover:bg-[var(--surface2)] transition-colors"
        aria-label="Theme picker"
      >
        {/* Current theme swatch */}
        <div className="flex gap-0.5">
          {currentTheme.preview.map((c, i) => (
            <div key={i} className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: c }} />
          ))}
        </div>
        <span className="text-2xs text-[var(--ink2)] hidden sm:block">{currentTheme.label}</span>
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none" className="text-[var(--ink3)]">
          <path d="M1 2.5L4 5.5L7 2.5" stroke="currentColor" strokeWidth="1.2"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {/* Dropdown */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.97 }}
            transition={{ duration: 0.12 }}
            className="absolute right-0 top-full mt-1.5 z-[9999] w-56 rounded-xl border
                       shadow-[0_8px_32px_rgba(0,0,0,0.5)] overflow-hidden"
            style={{
              background: "var(--surface)",
              borderColor: "var(--border)",
            }}
          >
            <div className="px-3 py-2 border-b" style={{ borderColor: "var(--border)" }}>
              <span className="text-2xs font-semibold uppercase tracking-widest"
                style={{ color: "var(--ink2)" }}>
                Theme
              </span>
            </div>
            <div className="p-1.5 grid grid-cols-2 gap-1">
              {THEMES.map(t => {
                const isActive = t.id === currentTheme.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => select(t)}
                    className="flex items-center gap-2 px-2 py-2 rounded-lg text-left
                               transition-all hover:opacity-90"
                    style={{
                      background: isActive ? "var(--accent)" + "22" : "transparent",
                      border: `1px solid ${isActive ? "var(--accent)" + "60" : "transparent"}`,
                    }}
                  >
                    {/* Swatch */}
                    <div className="flex gap-0.5 flex-shrink-0">
                      {t.preview.map((c, i) => (
                        <div key={i} className="w-3 h-3 rounded-sm" style={{ backgroundColor: c }} />
                      ))}
                    </div>
                    <div className="min-w-0">
                      <p className="text-2xs font-semibold truncate"
                        style={{ color: isActive ? "var(--accent)" : "var(--ink)" }}>
                        {t.label}
                      </p>
                      <p className="text-[9px] truncate" style={{ color: "var(--ink3)" }}>
                        {t.description}
                      </p>
                    </div>
                    {isActive && (
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none"
                        className="ml-auto flex-shrink-0" style={{ color: "var(--accent)" }}>
                        <path d="M2 5L4.5 7.5L8.5 3" stroke="currentColor" strokeWidth="1.5"
                          strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    )}
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});
