import { useMemo, type ComponentType } from "react";
import PlotImport from "react-plotly.js";

// react-plotly.js is a CommonJS package. Depending on the bundler (Vite 8 /
// Rolldown in dev vs. build), the default import is either the component or a
// `{ default: Component }` wrapper, so unwrap it once here.
const Plot = ((PlotImport as unknown as { default?: unknown }).default ?? PlotImport) as ComponentType<
  Record<string, unknown>
>;

interface Props {
  figure: unknown;
  height?: number;
}

// Matches tailwind.config.js: the lifted viridis stops, in workflow order.
const COLORWAY = ["#A18BF2", "#6FA0E8", "#3CC3B4", "#5CCB6F", "#B6DE3A", "#F5E14A"];
const GRID = "#2A2F36";
const INK = "#B5BBC3";

type Obj = Record<string, unknown>;

const BASE_LAYOUT: Obj = {
  margin: { l: 40, r: 20, t: 40, b: 40 },
};

/** Applied on top of the backend's figure so every chart matches the app theme. */
function themeLayout(layout: Obj): Obj {
  const axis = (a: unknown): Obj => ({ ...((a as Obj) ?? {}), gridcolor: GRID, zerolinecolor: GRID, linecolor: GRID });
  const out: Obj = {
    ...BASE_LAYOUT,
    ...layout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { ...((layout.font as Obj) ?? {}), color: INK, family: '"IBM Plex Sans", system-ui, sans-serif', size: 12 },
    colorway: COLORWAY,
    xaxis: axis(layout.xaxis),
    yaxis: axis(layout.yaxis),
  };
  // Heatmaps drawn with a shared colour axis (e.g. the confusion matrix).
  if (layout.coloraxis) out.coloraxis = { ...(layout.coloraxis as Obj), colorscale: "Viridis" };
  return out;
}

// Plotly's default qualitative palette. The backend's figures (plotly express)
// bake these into each trace, so a layout colorway alone can't override them.
const PLOTLY_DEFAULTS = [
  "#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a",
  "#19d3f3", "#ff6692", "#b6e880", "#ff97ff", "#fecb52",
];

function remapColour(c: unknown): unknown {
  if (typeof c !== "string") return c;
  const i = PLOTLY_DEFAULTS.indexOf(c.toLowerCase());
  return i === -1 ? c : COLORWAY[i % COLORWAY.length];
}

function themeTrace(trace: unknown): unknown {
  if (!trace || typeof trace !== "object") return trace;
  const t: Obj = { ...(trace as Obj) };
  if ("colorscale" in t) t.colorscale = "Viridis";
  if (t.marker && typeof t.marker === "object") {
    const m = { ...(t.marker as Obj) };
    m.color = Array.isArray(m.color) ? m.color.map(remapColour) : remapColour(m.color);
    if ("colorscale" in m) m.colorscale = "Viridis";
    t.marker = m;
  }
  if (t.line && typeof t.line === "object") {
    t.line = { ...(t.line as Obj), color: remapColour((t.line as Obj).color) };
  }
  return t;
}

export default function PlotlyChart({ figure, height = 320 }: Props) {
  const fig = useMemo(() => {
    const f = (figure as { data?: unknown[]; layout?: Obj }) ?? {};
    return {
      data: (f.data ?? []).map(themeTrace),
      layout: { ...themeLayout(f.layout ?? {}), height },
    };
  }, [figure, height]);

  return (
    <Plot
      data={fig.data}
      layout={fig.layout}
      useResizeHandler
      style={{ width: "100%", height }}
      config={{ displaylogo: false, responsive: true }}
    />
  );
}
