/** @type {import('tailwindcss').Config} */

// "Viridis lab" theme.
//
// The studios follow the viridis colour ramp (matplotlib's default colormap) in
// workflow order: Data -> ML -> DL -> RAG -> Agents -> Deploy. So an accent
// colour tells you where you are in the pipeline. Each stop is lifted in
// lightness so it stays readable as text on the dark graphite surfaces.
const viridis = {
  data: "#A18BF2", // purple
  ml: "#6FA0E8", // blue
  dl: "#3CC3B4", // teal
  rag: "#5CCB6F", // green
  agent: "#B6DE3A", // lime
  deploy: "#F5E14A", // yellow
};

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        studio: {
          ...viridis,
          hub: "#9AA1AB",
          settings: "#8B929C",
        },
        // Neutral graphite surfaces (no blue cast).
        bg: {
          DEFAULT: "#121417",
          panel: "#171A1E",
          card: "#1C2025",
          border: "#2A2F36",
        },
        // Text that sits on a solid accent fill.
        "on-accent": "#121417",
        // The app's text colours are written as `slate-*` throughout; remap the
        // scale to cool neutral ink so every page picks up the new palette.
        slate: {
          50: "#F4F5F7",
          100: "#E8EAED",
          200: "#D3D7DC",
          300: "#B5BBC3",
          400: "#959CA6",
          500: "#737A84",
          600: "#545A62",
          700: "#3B4047",
          800: "#272B31",
          900: "#1A1D21",
          950: "#0F1114",
        },
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      borderRadius: {
        panel: "10px",
      },
    },
  },
  plugins: [],
};
