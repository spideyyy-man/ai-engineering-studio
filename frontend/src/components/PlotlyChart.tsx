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

const DARK_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#cbd5e1", family: "Inter, system-ui, sans-serif", size: 12 },
  xaxis: { gridcolor: "#262d4e", zerolinecolor: "#262d4e" },
  yaxis: { gridcolor: "#262d4e", zerolinecolor: "#262d4e" },
  margin: { l: 40, r: 20, t: 40, b: 40 },
};

export default function PlotlyChart({ figure, height = 320 }: Props) {
  const fig = useMemo(() => {
    const f = (figure as { data?: unknown[]; layout?: Record<string, unknown> }) ?? {};
    return {
      data: f.data ?? [],
      layout: { ...DARK_LAYOUT, ...(f.layout ?? {}), height },
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
