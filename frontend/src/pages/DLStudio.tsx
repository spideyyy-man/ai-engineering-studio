import { useEffect, useMemo, useState } from "react";
import { Network, Play, Download, Code, Boxes, BarChart3, CheckCircle2, XCircle, Info } from "lucide-react";

import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import StatTile from "@/components/StatTile";
import EmptyState from "@/components/EmptyState";
import CodeBlock from "@/components/CodeBlock";
import PlotlyChart from "@/components/PlotlyChart";
import SectionTabs from "@/components/SectionTabs";
import LayerEditor from "@/components/LayerEditor";

import { useProjectStore } from "@/store/projectStore";
import { useDataStore } from "@/store/dataStore";
import { useDLStore } from "@/store/dlStore";
import { api } from "@/api/client";

type DataType = "tabular" | "image" | "text" | "timeseries";
type TopSection = "build" | "results";

const DATA_TYPES: { key: DataType; label: string; hint: string }[] = [
  { key: "tabular", label: "Tabular", hint: "rows × columns — trains in-studio" },
  { key: "image", label: "Image", hint: "CNN / transfer — design + export" },
  { key: "text", label: "Text", hint: "embedding + RNN / transformer" },
  { key: "timeseries", label: "Time series", hint: "sequence forecasting" },
];

export default function DLStudio() {
  const projectId = useProjectStore((s) => s.currentId);

  const datasets = useDataStore((s) => s.datasets);
  const currentDatasetId = useDataStore((s) => s.currentDatasetId);
  const summary = useDataStore((s) => s.summary);
  const setCurrentDataset = useDataStore((s) => s.setCurrent);
  const fetchDatasets = useDataStore((s) => s.fetchDatasets);
  const loadDataset = useDataStore((s) => s.loadAll);
  const targetCol = useDataStore((s) => s.targetCol);
  const setTarget = useDataStore((s) => s.setTarget);

  const {
    catalog, architectures, dataType, architecture, isRegression,
    spec, validation, code, training, error, result,
    fetchCatalog, fetchArchitectures, setDataType, setRegression,
    scaffold, setApi, setCompile, setTraining, train, reset,
  } = useDLStore();

  const [section, setSection] = useState<TopSection>("build");
  const [nClasses, setNClasses] = useState(2);
  const [shapeText, setShapeText] = useState("28,28,3");

  useEffect(() => {
    void fetchCatalog();
    void fetchArchitectures("tabular");
  }, [fetchCatalog, fetchArchitectures]);

  useEffect(() => {
    if (projectId) void fetchDatasets(projectId);
  }, [projectId, fetchDatasets]);

  useEffect(() => {
    if (currentDatasetId) void loadDataset(currentDatasetId, targetCol ?? undefined);
  }, [currentDatasetId, targetCol, loadDataset]);

  useEffect(() => {
    if (result?.trained) setSection("results");
  }, [result]);

  const featureCount = useMemo(() => {
    if (!summary) return 0;
    return summary.columns.filter((c) => c.name !== targetCol).length;
  }, [summary, targetCol]);

  function inputShapeForScaffold(): number[] {
    if (dataType === "tabular") return [Math.max(1, featureCount)];
    const parsed = shapeText.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n));
    return parsed.length ? parsed : [1];
  }

  function onScaffold(arch: string) {
    const nc = dataType === "tabular" && isRegression ? null : nClasses;
    void scaffold(arch, inputShapeForScaffold(), nc);
  }

  const lossOptions = useMemo(() => {
    if (!catalog || !spec) return [];
    const t = spec.task;
    const family =
      t === "tabular_regression" || t === "timeseries_forecast"
        ? "regression"
        : (spec.n_classes ?? 2) <= 2
        ? "binary_classification"
        : "multiclass_classification";
    return catalog.losses_by_task[family] ?? [];
  }, [catalog, spec]);

  if (!projectId) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-12">
        <EmptyState title="Select or create a project" description="Use the project switcher in the sidebar." />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-8 py-8">
      <PageHeader
        title="Deep Learning Studio"
        subtitle="Design a Keras network layer-by-layer (Sequential or Functional), validate its shapes live, generate code, and train tabular models in-studio."
        accent="text-studio-dl"
      />

      {catalog && !catalog.tf_available && (
        <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-700/40 bg-amber-900/15 px-4 py-3 text-sm text-amber-100">
          <Info size={16} className="mt-0.5 shrink-0" />
          <span>
            TensorFlow isn't installed in the backend, so designing, shape-validation and code
            export are fully live — but in-studio training is disabled. Install the optional deep-learning
            dependencies with <code className="rounded bg-black/30 px-1">pip install -r requirements-dl.txt</code> (Python 3.11–3.12),
            or set <code className="rounded bg-black/30 px-1">INSTALL_DL: "true"</code> in docker-compose.yml.
          </span>
        </div>
      )}

      <SectionTabs
        accent="border-studio-dl text-studio-dl"
        tabs={[
          { key: "build", label: "Build & Train", icon: <Boxes size={14} /> },
          { key: "results", label: "Results & Export", icon: <BarChart3 size={14} />, disabled: !result?.trained },
        ]}
        active={section}
        onChange={(k) => setSection(k as TopSection)}
      />

      <div className="mt-6">
        {section === "build" ? (
          <div className="grid gap-6 lg:grid-cols-[1fr_1.6fr]">
            {/* ---------------- left: data + preset */}
            <div className="space-y-6">
              <Card title="Data type" subtitle="Drives which architectures and layers are offered.">
                <div className="grid grid-cols-2 gap-2">
                  {DATA_TYPES.map((dt) => (
                    <button
                      key={dt.key}
                      onClick={() => setDataType(dt.key)}
                      className={`rounded-lg border p-2.5 text-left transition ${
                        dataType === dt.key
                          ? "border-studio-dl bg-studio-dl/5"
                          : "border-bg-border bg-bg-panel hover:border-slate-500"
                      }`}
                    >
                      <div className="text-sm font-medium text-slate-100">{dt.label}</div>
                      <div className="mt-0.5 text-[11px] text-slate-500">{dt.hint}</div>
                    </button>
                  ))}
                </div>
              </Card>

              {dataType === "tabular" ? (
                <>
                  <Card title="Dataset" subtitle="Pick from this project's uploaded datasets.">
                    {datasets.length === 0 ? (
                      <p className="text-sm text-slate-500">No datasets — upload one in Data &amp; EDA first.</p>
                    ) : (
                      <select
                        className="input"
                        value={currentDatasetId ?? ""}
                        onChange={(e) => setCurrentDataset(e.target.value || null)}
                      >
                        <option value="">— Pick a dataset —</option>
                        {datasets.map((d) => (
                          <option key={d.id} value={d.id}>
                            {d.name} ({d.n_rows} × {d.n_cols})
                          </option>
                        ))}
                      </select>
                    )}
                  </Card>

                  <Card title="Target column" subtitle="The column the network predicts.">
                    {!summary ? (
                      <p className="text-sm text-slate-500">Pick a dataset to choose a target.</p>
                    ) : (
                      <>
                        <select
                          className="input"
                          value={targetCol ?? ""}
                          onChange={(e) => setTarget(e.target.value || null)}
                        >
                          <option value="">— Pick a target —</option>
                          {summary.columns.map((c) => (
                            <option key={c.name} value={c.name}>
                              {c.name} ({c.dtype})
                            </option>
                          ))}
                        </select>
                        <div className="mt-3 space-y-2">
                          <label className="flex items-center gap-2 text-sm text-slate-200">
                            <input
                              type="checkbox"
                              checked={isRegression}
                              onChange={(e) => setRegression(e.target.checked)}
                            />
                            Regression (predict a continuous number)
                          </label>
                          {!isRegression && (
                            <div>
                              <label className="label">Number of classes</label>
                              <input
                                type="number"
                                className="input"
                                min={2}
                                value={nClasses}
                                onChange={(e) => setNClasses(Math.max(2, parseInt(e.target.value, 10) || 2))}
                              />
                            </div>
                          )}
                          <p className="text-[11px] text-slate-500">
                            Input is auto-bound to the post-preprocessing feature count at train time
                            ({featureCount} raw feature column{featureCount === 1 ? "" : "s"}).
                          </p>
                        </div>
                      </>
                    )}
                  </Card>
                </>
              ) : (
                <Card title="Input shape" subtitle="Shape of one sample, excluding the batch axis.">
                  <input
                    className="input font-mono"
                    value={shapeText}
                    onChange={(e) => setShapeText(e.target.value)}
                    placeholder={dataType === "image" ? "28,28,3" : "100"}
                  />
                  <p className="mt-2 text-[11px] text-slate-500">
                    {dataType === "image"
                      ? "e.g. 28,28,3 for 28×28 RGB images."
                      : dataType === "text"
                      ? "Sequence length, e.g. 100 (tokens). Embedding turns it into (100, dim)."
                      : "Window length × features, e.g. 30,1 for a univariate window of 30."}
                  </p>
                  {!isRegression && (
                    <div className="mt-3">
                      <label className="label">Number of classes</label>
                      <input
                        type="number" className="input" min={2} value={nClasses}
                        onChange={(e) => setNClasses(Math.max(2, parseInt(e.target.value, 10) || 2))}
                      />
                    </div>
                  )}
                </Card>
              )}

              <Card title="Architecture preset" subtitle="A smart starting point — fully editable after.">
                <ul className="space-y-2">
                  {architectures.map((a) => (
                    <li key={a.key}>
                      <button
                        onClick={() => onScaffold(a.key)}
                        className={`w-full rounded-lg border p-3 text-left transition ${
                          architecture === a.key
                            ? "border-studio-dl bg-studio-dl/5"
                            : "border-bg-border bg-bg-panel hover:border-slate-500"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-slate-100">{a.label}</span>
                          <span className="pill">{a.api}</span>
                        </div>
                        <p className="mt-1 text-xs text-slate-400">{a.description}</p>
                        {!a.trainable_here && (
                          <p className="mt-1 text-[10px] text-amber-300/80">design + code export</p>
                        )}
                      </button>
                    </li>
                  ))}
                </ul>
              </Card>
            </div>

            {/* ---------------- right: builder */}
            <div className="space-y-6">
              {!spec ? (
                <Card title="Network builder">
                  <EmptyState
                    icon={<Network size={28} />}
                    title="Pick an architecture preset"
                    description="Choose a data type and a preset on the left to scaffold an editable network."
                  />
                </Card>
              ) : (
                <>
                  {/* summary bar */}
                  <Card
                    title="Network"
                    subtitle={`${spec.layers.length} layers · ${spec.task}`}
                    action={
                      <div className="flex items-center gap-1 rounded-lg border border-bg-border p-0.5">
                        {(["sequential", "functional"] as const).map((m) => (
                          <button
                            key={m}
                            onClick={() => setApi(m)}
                            className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                              spec.api === m ? "bg-studio-dl/20 text-studio-dl" : "text-slate-400 hover:text-slate-200"
                            }`}
                          >
                            {m}
                          </button>
                        ))}
                      </div>
                    }
                  >
                    <div className="grid grid-cols-3 gap-3">
                      <StatTile
                        label="Output shape"
                        value={validation?.output_shape ? `(${validation.output_shape.join(", ")})` : "—"}
                        accent="text-studio-dl"
                      />
                      <StatTile label="Trainable params" value={validation?.total_params_human ?? "—"} accent="text-studio-dl" />
                      <StatTile
                        label="Status"
                        value={validation ? (validation.ok ? "valid" : "invalid") : "…"}
                        accent={validation?.ok ? "text-emerald-400" : "text-red-400"}
                      />
                    </div>

                    {validation && validation.errors.length > 0 && (
                      <div className="mt-3 space-y-1 rounded-lg border border-red-700/40 bg-red-900/15 p-3 text-xs text-red-200">
                        {validation.errors.map((e, i) => (
                          <div key={i} className="flex items-start gap-1.5">
                            <XCircle size={13} className="mt-0.5 shrink-0" /> {e}
                          </div>
                        ))}
                      </div>
                    )}
                    {validation && validation.ok && validation.warnings.length > 0 && (
                      <div className="mt-3 space-y-1 rounded-lg border border-amber-700/40 bg-amber-900/15 p-3 text-xs text-amber-100">
                        {validation.warnings.map((w, i) => (
                          <div key={i}>⚠ {w}</div>
                        ))}
                      </div>
                    )}
                  </Card>

                  <Card title="Layers" subtitle="Add, remove, reorder and tune each layer. Shapes update live.">
                    <LayerEditor />
                  </Card>

                  <Card title="Compile" subtitle="Optimizer, learning rate and loss.">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <label className="label">Optimizer</label>
                        <select
                          className="input"
                          value={spec.compile.optimizer}
                          onChange={(e) => setCompile("optimizer", e.target.value)}
                        >
                          {(catalog?.optimizers ?? []).map((o) => (
                            <option key={o.key} value={o.key}>{o.label}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="label">Learning rate</label>
                        <input
                          type="number" step={0.0001} min={0} className="input"
                          value={spec.compile.learning_rate}
                          onChange={(e) => setCompile("learning_rate", parseFloat(e.target.value) || 0.001)}
                        />
                      </div>
                      <div className="sm:col-span-2">
                        <label className="label">Loss</label>
                        <select
                          className="input"
                          value={spec.compile.loss}
                          onChange={(e) => setCompile("loss", e.target.value)}
                        >
                          {lossOptions.map((l) => (
                            <option key={l.key} value={l.key}>{l.label}</option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <p className="mt-2 text-[11px] text-slate-500">
                      Metrics: <span className="font-mono">{spec.compile.metrics.join(", ") || "—"}</span> (auto-set by task)
                    </p>
                  </Card>

                  <Card title="Training" subtitle="Epochs, batch, and callbacks.">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div>
                        <label className="label">Epochs</label>
                        <input
                          type="number" min={1} className="input" value={spec.training.epochs}
                          onChange={(e) => setTraining("epochs", parseInt(e.target.value, 10) || 1)}
                        />
                      </div>
                      <div>
                        <label className="label">Batch size</label>
                        <input
                          type="number" min={1} className="input" value={spec.training.batch_size}
                          onChange={(e) => setTraining("batch_size", parseInt(e.target.value, 10) || 1)}
                        />
                      </div>
                      <div className="sm:col-span-2">
                        <label className="label flex items-center justify-between">
                          <span>Validation split</span>
                          <span className="font-mono text-xs text-slate-500">
                            {(spec.training.validation_split * 100).toFixed(0)}%
                          </span>
                        </label>
                        <input
                          type="range" min={0.05} max={0.5} step={0.05}
                          value={spec.training.validation_split}
                          onChange={(e) => setTraining("validation_split", parseFloat(e.target.value))}
                          className="w-full"
                        />
                      </div>
                      <label className="flex items-center gap-2 text-sm text-slate-200">
                        <input
                          type="checkbox" checked={spec.training.early_stopping}
                          onChange={(e) => setTraining("early_stopping", e.target.checked)}
                        />
                        Early stopping
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-200">
                        <input
                          type="checkbox" checked={spec.training.reduce_lr_on_plateau}
                          onChange={(e) => setTraining("reduce_lr_on_plateau", e.target.checked)}
                        />
                        Reduce LR on plateau
                      </label>
                    </div>
                  </Card>

                  {code && (
                    <Card title="Generated Keras code" subtitle="Live — reflects every edit above. Runnable as-is.">
                      <CodeBlock code={code} filename="model.py" />
                    </Card>
                  )}

                  {error && (
                    <div className="rounded-lg border border-red-700/40 bg-red-900/20 px-4 py-3 text-sm text-red-200">
                      {error}
                    </div>
                  )}
                  {result && !result.trained && result.message && (
                    <div className="rounded-lg border border-amber-700/40 bg-amber-900/15 px-4 py-3 text-sm text-amber-100">
                      {result.message}
                    </div>
                  )}

                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      className="btn-primary bg-studio-dl hover:opacity-90"
                      disabled={
                        training ||
                        !validation?.ok ||
                        (dataType === "tabular" && (!currentDatasetId || !targetCol))
                      }
                      onClick={() => {
                        if (!projectId) return;
                        void train(projectId, currentDatasetId ?? "", targetCol);
                      }}
                    >
                      <Play size={16} /> {training ? "Training…" : "Train network"}
                    </button>
                    <button className="btn-ghost" onClick={reset}>Reset</button>
                    {dataType !== "tabular" && (
                      <span className="text-xs text-slate-500">
                        Non-tabular: validates &amp; exports code (train via the generated script).
                      </span>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        ) : (
          <ResultsSection />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- Results

function ResultsSection() {
  const result = useDLStore((s) => s.result);
  if (!result || !result.trained) {
    return <EmptyState title="No trained model yet" description="Build a valid network and train it first." />;
  }

  const apiBase = (api.defaults.baseURL ?? "").replace(/\/$/, "");
  const modelUrl = result.model_download_url
    ? apiBase + result.model_download_url.replace(/^\/api\/v1/, "")
    : null;
  const codeUrl = result.code_download_url
    ? apiBase + result.code_download_url.replace(/^\/api\/v1/, "")
    : null;

  const m = result.metrics as Record<string, unknown>;
  const fmt = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x.toFixed(4) : "—");
  const isReg = result.task === "tabular_regression" || result.task === "timeseries_forecast";
  const tiles = isReg
    ? [
        { label: "R²", value: fmt(m.r2) },
        { label: "MAE", value: fmt(m.mae) },
        { label: "RMSE", value: fmt(m.rmse) },
        { label: "MSE", value: fmt(m.mse) },
      ]
    : [
        { label: "Accuracy", value: fmt(m.accuracy) },
        { label: "Precision", value: fmt(m.precision) },
        { label: "Recall", value: fmt(m.recall) },
        { label: "F1", value: fmt(m.f1) },
      ];

  return (
    <div className="space-y-6">
      <Card
        title={`${result.architecture ?? "custom"} — ${result.task}`}
        subtitle={`${result.n_train} train / ${result.n_test} test · ${result.epochs_ran} epochs ran · ${result.api} API`}
        action={
          <div className="flex gap-2">
            {modelUrl && (
              <a className="btn-primary bg-studio-dl" href={modelUrl} target="_blank" rel="noreferrer">
                <Download size={16} /> model.keras
              </a>
            )}
            {codeUrl && (
              <a className="btn-ghost" href={codeUrl} target="_blank" rel="noreferrer">
                <Code size={16} /> model.py
              </a>
            )}
          </div>
        }
      >
        <div className="mb-3 flex items-center gap-2 text-sm text-emerald-300">
          <CheckCircle2 size={16} /> {result.message}
        </div>
        <div className="grid gap-3 sm:grid-cols-4">
          {tiles.map((t) => (
            <StatTile key={t.label} label={t.label} value={t.value} accent="text-studio-dl" />
          ))}
        </div>
      </Card>

      {Object.keys(result.figures).length > 0 && (
        <Card title="Diagnostics" subtitle="Learning curves and held-out evaluation.">
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(result.figures).map(([name, fig]) => (
              <div key={name} className="rounded-lg border border-bg-border bg-bg-panel/40 p-2">
                <PlotlyChart figure={fig} height={320} />
              </div>
            ))}
          </div>
        </Card>
      )}

      <Card title="Training script" subtitle="The exact Keras architecture you trained, as runnable code.">
        <CodeBlock code={result.code} filename={result.code_filename} />
      </Card>
    </div>
  );
}
