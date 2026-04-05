import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";

const Plot = createPlotlyComponent(Plotly);

export function PlotlyChart({ figure, title }) {
  return (
    <Plot
      className="plotly-chart"
      data={figure.data}
      layout={{
        autosize: true,
        margin: { l: 32, r: 20, t: 56, b: 32 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(255,255,255,0.86)",
        ...figure.layout,
        title: figure.layout?.title || { text: title },
      }}
      config={{ displaylogo: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}
