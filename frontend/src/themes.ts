/**
 * Theme definitions for DeceptiScope v2.
 * Each theme is a set of CSS custom property values applied to :root.
 * All components use var(--*) tokens so switching themes is instant.
 */

export interface Theme {
  id: string;
  label: string;
  description: string;
  preview: string[]; // 3 hex colors for the swatch
  vars: Record<string, string>;
}

export const THEMES: Theme[] = [
  // ── 1. Midnight (default dark) ──────────────────────────────────────────
  {
    id: "midnight",
    label: "Midnight",
    description: "Deep dark — easy on the eyes",
    preview: ["#080c10", "#0e1318", "#3b82f6"],
    vars: {
      "--bg":       "#080c10",
      "--surface":  "#0e1318",
      "--surface2": "#141a21",
      "--border":   "#1e2730",
      "--border2":  "#263040",
      "--ink":      "#d4dce8",
      "--ink2":     "#7a8899",
      "--ink3":     "#4a5568",
      "--accent":   "#3b82f6",
      "--success":  "#22c55e",
      "--warn":     "#eab308",
      "--danger":   "#ef4444",
      "--violet":   "#a855f7",
      "--fire":     "#f97316",
    },
  },

  // ── 2. Arctic (light) ────────────────────────────────────────────────────
  {
    id: "arctic",
    label: "Arctic",
    description: "Clean light — high contrast",
    preview: ["#f8fafc", "#ffffff", "#2563eb"],
    vars: {
      "--bg":       "#f0f4f8",
      "--surface":  "#ffffff",
      "--surface2": "#f1f5f9",
      "--border":   "#cbd5e1",
      "--border2":  "#94a3b8",
      "--ink":      "#0f172a",
      "--ink2":     "#334155",
      "--ink3":     "#64748b",
      "--accent":   "#2563eb",
      "--success":  "#16a34a",
      "--warn":     "#ca8a04",
      "--danger":   "#dc2626",
      "--violet":   "#7c3aed",
      "--fire":     "#ea580c",
    },
  },

  // ── 3. Forest (green dark) ───────────────────────────────────────────────
  {
    id: "forest",
    label: "Forest",
    description: "Deep green — calm and focused",
    preview: ["#0a1a0f", "#0f2318", "#22c55e"],
    vars: {
      "--bg":       "#0a1a0f",
      "--surface":  "#0f2318",
      "--surface2": "#162d1f",
      "--border":   "#1e3d28",
      "--border2":  "#2a5238",
      "--ink":      "#d1fae5",
      "--ink2":     "#6ee7b7",
      "--ink3":     "#34d399",
      "--accent":   "#22c55e",
      "--success":  "#4ade80",
      "--warn":     "#fbbf24",
      "--danger":   "#f87171",
      "--violet":   "#c084fc",
      "--fire":     "#fb923c",
    },
  },

  // ── 4. Crimson (red dark) ────────────────────────────────────────────────
  {
    id: "crimson",
    label: "Crimson",
    description: "Bold red — high alert mode",
    preview: ["#120a0a", "#1e0f0f", "#ef4444"],
    vars: {
      "--bg":       "#120a0a",
      "--surface":  "#1e0f0f",
      "--surface2": "#2a1515",
      "--border":   "#3d1e1e",
      "--border2":  "#522828",
      "--ink":      "#fde8e8",
      "--ink2":     "#fca5a5",
      "--ink3":     "#f87171",
      "--accent":   "#ef4444",
      "--success":  "#22c55e",
      "--warn":     "#fbbf24",
      "--danger":   "#ff6b6b",
      "--violet":   "#c084fc",
      "--fire":     "#fb923c",
    },
  },

  // ── 5. Violet (purple dark) ──────────────────────────────────────────────
  {
    id: "violet",
    label: "Violet",
    description: "Deep purple — research mode",
    preview: ["#0d0a1a", "#150f2a", "#a855f7"],
    vars: {
      "--bg":       "#0d0a1a",
      "--surface":  "#150f2a",
      "--surface2": "#1e1538",
      "--border":   "#2d1f52",
      "--border2":  "#3d2a6e",
      "--ink":      "#ede9fe",
      "--ink2":     "#c4b5fd",
      "--ink3":     "#8b5cf6",
      "--accent":   "#a855f7",
      "--success":  "#34d399",
      "--warn":     "#fbbf24",
      "--danger":   "#f87171",
      "--violet":   "#e879f9",
      "--fire":     "#fb923c",
    },
  },

  // ── 6. Slate (neutral light) ─────────────────────────────────────────────
  {
    id: "slate",
    label: "Slate",
    description: "Warm neutral — paper-like",
    preview: ["#f5f0eb", "#fffdf9", "#0f766e"],
    vars: {
      "--bg":       "#f5f0eb",
      "--surface":  "#fffdf9",
      "--surface2": "#f0ebe4",
      "--border":   "#d6cfc6",
      "--border2":  "#b8b0a6",
      "--ink":      "#1c1917",
      "--ink2":     "#44403c",
      "--ink3":     "#78716c",
      "--accent":   "#0f766e",
      "--success":  "#15803d",
      "--warn":     "#b45309",
      "--danger":   "#b91c1c",
      "--violet":   "#6d28d9",
      "--fire":     "#c2410c",
    },
  },

  // ── 7. Neon (cyberpunk) ──────────────────────────────────────────────────
  {
    id: "neon",
    label: "Neon",
    description: "Cyberpunk — high contrast neon",
    preview: ["#000000", "#0a0a0a", "#00ff88"],
    vars: {
      "--bg":       "#000000",
      "--surface":  "#0a0a0a",
      "--surface2": "#111111",
      "--border":   "#1a1a1a",
      "--border2":  "#222222",
      "--ink":      "#e0ffe0",
      "--ink2":     "#00ff88",
      "--ink3":     "#00cc66",
      "--accent":   "#00ff88",
      "--success":  "#00ff88",
      "--warn":     "#ffff00",
      "--danger":   "#ff0055",
      "--violet":   "#cc00ff",
      "--fire":     "#ff6600",
    },
  },

  // ── 8. Ocean (blue-teal dark) ────────────────────────────────────────────
  {
    id: "ocean",
    label: "Ocean",
    description: "Deep blue — calm and deep",
    preview: ["#050d1a", "#0a1628", "#06b6d4"],
    vars: {
      "--bg":       "#050d1a",
      "--surface":  "#0a1628",
      "--surface2": "#0f1f38",
      "--border":   "#162d4e",
      "--border2":  "#1e3d68",
      "--ink":      "#e0f2fe",
      "--ink2":     "#7dd3fc",
      "--ink3":     "#38bdf8",
      "--accent":   "#06b6d4",
      "--success":  "#34d399",
      "--warn":     "#fbbf24",
      "--danger":   "#f87171",
      "--violet":   "#818cf8",
      "--fire":     "#fb923c",
    },
  },
];

export const DEFAULT_THEME_ID = "midnight";

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  Object.entries(theme.vars).forEach(([key, value]) => {
    root.style.setProperty(key, value);
  });
  root.setAttribute("data-theme", theme.id);
  // Persist
  try { localStorage.setItem("ds-theme", theme.id); } catch {}
}

export function loadSavedTheme(): Theme {
  try {
    const saved = localStorage.getItem("ds-theme");
    if (saved) {
      const found = THEMES.find(t => t.id === saved);
      if (found) return found;
    }
  } catch {}
  return THEMES.find(t => t.id === DEFAULT_THEME_ID)!;
}
