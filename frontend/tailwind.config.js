/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // DeceptiScope brand palette
        "ds-bg":      "#0d1117",
        "ds-surface": "#161b22",
        "ds-border":  "#30363d",
        "ds-text":    "#e6edf3",
        "ds-muted":   "#8b949e",
        "ds-accent":  "#58a6ff",
        "ds-green":   "#3fb950",
        "ds-yellow":  "#d29922",
        "ds-red":     "#f85149",
        "ds-purple":  "#bc8cff",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
    },
  },
  plugins: [],
};
