/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg:       "#080c10",
        surface:  "#0e1318",
        surface2: "#141a21",
        border:   "#1e2730",
        border2:  "#263040",
        ink:      "#d4dce8",
        ink2:     "#7a8899",
        ink3:     "#4a5568",
        accent:   "#3b82f6",
        success:  "#22c55e",
        warn:     "#eab308",
        danger:   "#ef4444",
        violet:   "#a855f7",
        fire:     "#f97316",
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      fontSize: {
        "2xs": ["10px", { lineHeight: "14px" }],
        "xs":  ["11px", { lineHeight: "16px" }],
        "sm":  ["12px", { lineHeight: "18px" }],
        "base":["13px", { lineHeight: "20px" }],
        "md":  ["14px", { lineHeight: "20px" }],
        "lg":  ["15px", { lineHeight: "22px" }],
      },
      borderRadius: {
        DEFAULT: "8px",
        sm: "5px",
        md: "8px",
        lg: "12px",
        xl: "16px",
      },
      boxShadow: {
        card:  "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
        panel: "0 4px 24px rgba(0,0,0,0.5)",
        glow:  "0 0 20px rgba(59,130,246,0.15)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "fade-in":    "fadeIn 0.15s ease-out",
        "slide-up":   "slideUp 0.2s ease-out",
      },
      keyframes: {
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: "translateY(6px)" }, to: { opacity: 1, transform: "translateY(0)" } },
      },
    },
  },
  plugins: [],
};
