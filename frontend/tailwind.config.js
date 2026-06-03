/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        studio: {
          data: "#3b82f6",      // blue
          ml: "#10b981",        // emerald
          dl: "#a855f7",        // purple
          rag: "#f59e0b",       // amber
          agent: "#ef4444",     // red
          deploy: "#06b6d4",    // cyan
          hub: "#64748b",       // slate
          settings: "#94a3b8",  // slate-light
        },
        bg: {
          DEFAULT: "#0b1020",
          panel: "#121833",
          card: "#1a2142",
          border: "#262d4e",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
