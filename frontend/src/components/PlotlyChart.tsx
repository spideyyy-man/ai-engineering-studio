import { useMemo } from "react";
import Plot from "react-plotly.js";

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
      // @ts-expect-error — react-plotly types are loose
      data={fig.data}
      layout={fig.layout}
      useResizeHandler
      style={{ width: "100%", height }}
      config={{ displaylogo: false, responsive: true }}
    />
  );
}
