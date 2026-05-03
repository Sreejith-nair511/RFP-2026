/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      // All colors reference CSS variables — theme switching is instant
      colors: {
        bg:       "var(--bg)",
        surface:  "var(--surface)",
        surface2: "var(--surface2)",
        border:   "var(--border)",
        border2:  "var(--border2)",
        ink:      "var(--ink)",
        ink2:     "var(--ink2)",
        ink3:     "var(--ink3)",
        accent:   "var(--accent)",
        success:  "var(--success)",
        warn:     "var(--warn)",
        danger:   "var(--danger)",
        violet:   "var(--violet)",
        fire:     "var(--fire)",
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
        card:  "0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)",
        panel: "0 4px 24px rgba(0,0,0,0.4)",
        glow:  "0 0 20px rgba(59,130,246,0.15)",
      },
      screens: {
        xs: "480px",
        sm: "640px",
        md: "768px",
        lg: "1024px",
        xl: "1280px",
      },
    },
  },
  plugins: [],
};
