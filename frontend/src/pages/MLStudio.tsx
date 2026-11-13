import { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit,
  Play,
  Download,
  Sliders,
  BarChart3,
  Code,
} from "lucide-react";

import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import StatTile from "@/components/StatTile";
import EmptyState from "@/components/EmptyState";
import CodeBlock from "@/components/CodeBlock";
import PlotlyChart from "@/components/PlotlyChart";
import SectionTabs from "@/components/SectionTabs";

import { useProjectStore } from "@/store/projectStore";
import { useDataStore } from "@/store/dataStore";
import { useMLStore } from "@/store/mlStore";
import HyperparamForm from "@/components/HyperparamForm";
import { api } from "@/api/client";
import type { AlgorithmInfo } from "@/api/ml";

type TopSection = "configure" | "results";

export default function MLStudio() {
  const projectId = useProjectStore((s) => s.currentId);

  // Datasets (reuse data store)
  const datasets = useDataStore((s) => s.datasets);
  const currentDatasetId = useDataStore((s) => s.currentDatasetId);
  const summary = useDataStore((s) => s.summary);
  const setCurrentDataset = useDataStore((s) => s.setCurrent);
  const fetchDatasets = useDataStore((s) => s.fetchDatasets);
  const loadDataset = useDataStore((s) => s.loadAll);
  const targetCol = useDataStore((s) => s.targetCol);
  const setTarget = useDataStore((s) => s.setTarget);

  // ML store
  const {
    algorithms,
    task,
    taskRationale,
    algoKey,
    hparams,
    testSize,
    randomState,
    cvStrategy,
    cvFolds,
    hptMode,
    hptNIter,
    training,
    error,
    result,
    fetchAlgorithms,
    detectTask,
    selectAlgo,
    setHparam,
    setTestSize,
    setRandomState,
    setCVStrategy,
    setCVFolds,
    setHPTMode,
    setHPTNIter,
    train,
    reset,
  } = useMLStore();

  const [section, setSection] = useState<TopSection>("configure");

  // Bootstrap
  useEffect(() => {
    void fetchAlgorithms();
  }, [fetchAlgorithms]);

  useEffect(() => {
    if (projectId) void fetchDatasets(projectId);
  }, [projectId, fetchDatasets]);

  useEffect(() => {
    if (currentDatasetId) void loadDataset(currentDatasetId, targetCol ?? undefined);
  }, [currentDatasetId, targetCol, loadDataset]);

  // Re-detect task whenever dataset or target changes
  useEffect(() => {
    if (currentDatasetId) void detectTask(currentDatasetId, targetCol);
  }, [currentDatasetId, targetCol, detectTask]);

  // Show results section automatically after training
  useEffect(() => {
    if (result) setSection("results");
  }, [result]);

  const algoPool: AlgorithmInfo[] = useMemo(() => {
    if (task === "unknown") return algorithms;
    return algorithms.filter((a) => a.task === task);
  }, [algorithms, task]);

  const selectedAlgo = algorithms.find((a) => a.key === algoKey);

  if (!projectId) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-12">
        <EmptyState
          title="Select or create a project"
          description="Use the project switcher in the sidebar."
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-8 py-8">
      <PageHeader
        title="Machine Learning Studio"
        subtitle="Pick a dataset, choose an algorithm, tune hyperparameters, train, and download a ready-to-use pickle + sample code."
        accent="text-studio-ml"
      />

      <SectionTabs
        accent="border-studio-ml text-studio-ml"
        tabs={[
          { key: "configure", label: "Configure & Train", icon: <Sliders size={14} /> },
          {
            key: "results",
            label: "Results & Export",
            icon: <BarChart3 size={14} />,
            disabled: !result,
          },
        ]}
        active={section}
        onChange={(k) => setSection(k as TopSection)}
      />

      <div className="mt-6">
        {section === "configure" ? (
          <ConfigureSection
            datasets={datasets}
            currentDatasetId={currentDatasetId}
            onPickDataset={setCurrentDataset}
            summaryColumns={summary?.columns ?? []}
            targetCol={targetCol}
            setTarget={setTarget}
            task={task}
            taskRationale={taskRationale}
            algoPool={algoPool}
            selectedAlgo={selectedAlgo}
            algoKey={algoKey}
            onPickAlgo={selectAlgo}
            hparams={hparams}
            onHparam={setHparam}
            testSize={testSize}
            setTestSize={setTestSize}
            randomState={randomState}
            setRandomState={setRandomState}
            cvStrategy={cvStrategy}
            setCVStrategy={setCVStrategy}
            cvFolds={cvFolds}
            setCVFolds={setCVFolds}
            hptMode={hptMode}
            setHPTMode={setHPTMode}
            hptNIter={hptNIter}
            setHPTNIter={setHPTNIter}
            training={training}
            error={error}
            onTrain={() => {
              if (!currentDatasetId || !projectId) return;
              void train(currentDatasetId, projectId, targetCol);
            }}
            onReset={reset}
          />
        ) : (
          <ResultsSection />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- Configure

interface ConfigureProps {
  datasets: { id: string; name: string; n_rows: number; n_cols: number }[];
  currentDatasetId: string | null;
  onPickDataset: (id: string | null) => void;
  summaryColumns: { name: string; dtype: string }[];
  targetCol: string | null;
  setTarget: (col: string | null) => void;
  task: string;
  taskRationale: string;
  algoPool: AlgorithmInfo[];
  selectedAlgo: AlgorithmInfo | undefined;
  algoKey: string | null;
  onPickAlgo: (k: string | null) => void;
  hparams: Record<string, unknown>;
  onHparam: (k: string, v: unknown) => void;
  testSize: number;
  setTestSize: (v: number) => void;
  randomState: number;
  setRandomState: (v: number) => void;
  // Slice A
  cvStrategy: string;
  setCVStrategy: (v: import("@/api/ml").CVStrategy) => void;
  cvFolds: number;
  setCVFolds: (v: number) => void;
  hptMode: string;
  setHPTMode: (v: import("@/api/ml").HPTMode) => void;
  hptNIter: number;
  setHPTNIter: (v: number) => void;
  training: boolean;
  error: string | null;
  onTrain: () => void;
  onReset: () => void;
}

function ConfigureSection(p: ConfigureProps) {
  const supervised = p.task === "classification" || p.task === "regression";
  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
      {/* Left: dataset + target */}
      <div className="space-y-6">
        <Card title="Dataset" subtitle="Pick from this project's uploaded datasets.">
          {p.datasets.length === 0 ? (
            <p className="text-sm text-slate-500">
              No datasets in this project. Upload one in Data &amp; EDA first.
            </p>
          ) : (
            <select
              className="input"
              value={p.currentDatasetId ?? ""}
              onChange={(e) => p.onPickDataset(e.target.value || null)}
            >
              <option value="">— Pick a dataset —</option>
              {p.datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.n_rows} × {d.n_cols})
                </option>
              ))}
            </select>
          )}
        </Card>

        <Card title="Target column" subtitle="Leave blank for unsupervised clustering.">
          {p.summaryColumns.length === 0 ? (
            <p className="text-sm text-slate-500">Pick a dataset to choose a target.</p>
          ) : (
            <>
              <select
                className="input"
                value={p.targetCol ?? ""}
                onChange={(e) => p.setTarget(e.target.value || null)}
              >
                <option value="">— Unsupervised / not set —</option>
                {p.summaryColumns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}  ({c.dtype})
                  </option>
                ))}
              </select>

              {p.task !== "unknown" && (
                <div className="mt-3 rounded-lg border border-bg-border bg-bg-panel/60 p-3 text-xs text-slate-300">
                  <span className="font-mono uppercase text-studio-ml">{p.task}</span>
                  <p className="mt-1 text-slate-400">{p.taskRationale}</p>
                </div>
              )}
            </>
          )}
        </Card>

        <Card title="Training" subtitle="Split, CV, hyperparameter tuning, seed.">
          {supervised ? (
            <div className="space-y-3">
              <div>
                <label className="label">Test set size</label>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0.05}
                    max={0.5}
                    step={0.05}
                    value={p.testSize}
                    onChange={(e) => p.setTestSize(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <span className="w-12 text-right font-mono text-sm">
                    {(p.testSize * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Slice A: CV strategy */}
              <div>
                <label className="label">Cross-validation</label>
                <select
                  className="input"
                  value={p.cvStrategy}
                  onChange={(e) => p.setCVStrategy(e.target.value as import("@/api/ml").CVStrategy)}
                >
                  <option value="none">None (simple train-test split)</option>
                  <option value="kfold">K-fold</option>
                  <option value="stratified_kfold">Stratified K-fold (classification)</option>
                  <option value="time_series">Time Series Split</option>
                  <option value="group_kfold">Group K-fold</option>
                  <option value="leave_one_out">Leave-One-Out (small N only)</option>
                </select>
              </div>
              {p.cvStrategy !== "none" && p.cvStrategy !== "leave_one_out" && (
                <div>
                  <label className="label flex items-center justify-between">
                    <span>CV folds</span>
                    <span className="font-mono text-xs text-slate-500">{p.cvFolds}</span>
                  </label>
                  <input
                    type="range" min={2} max={10} step={1}
                    value={p.cvFolds}
                    onChange={(e) => p.setCVFolds(parseInt(e.target.value, 10))}
                    className="w-full"
                  />
                </div>
              )}

              {/* Slice A: HPT mode */}
              <div>
                <label className="label">Hyperparameter tuning</label>
                <select
                  className="input"
                  value={p.hptMode}
                  onChange={(e) => p.setHPTMode(e.target.value as import("@/api/ml").HPTMode)}
                >
                  <option value="manual">Manual (use form values)</option>
                  <option value="grid">Grid Search (auto-grid from spec)</option>
                  <option value="random">Randomised Search (auto-grid + n_iter)</option>
                </select>
              </div>
              {p.hptMode === "random" && (
                <div>
                  <label className="label flex items-center justify-between">
                    <span>HPT iterations (n_iter)</span>
                    <span className="font-mono text-xs text-slate-500">{p.hptNIter}</span>
                  </label>
                  <input
                    type="range" min={5} max={100} step={5}
                    value={p.hptNIter}
                    onChange={(e) => p.setHPTNIter(parseInt(e.target.value, 10))}
                    className="w-full"
                  />
                </div>
              )}
              {p.hptMode !== "manual" && (
                <p className="rounded-md border border-bg-border bg-bg-panel/60 p-2 text-xs text-slate-400">
                  Search space is auto-built from the algorithm's hyperparameter spec.
                  Form values below are ignored; best params will be surfaced in Results.
                </p>
              )}

              <div>
                <label className="label">Random seed</label>
                <input
                  type="number"
                  className="input"
                  value={p.randomState}
                  onChange={(e) => p.setRandomState(parseInt(e.target.value, 10) || 0)}
                />
              </div>
            </div>
          ) : (
            <div>
              <label className="label">Random seed</label>
              <input
                type="number"
                className="input"
                value={p.randomState}
                onChange={(e) => p.setRandomState(parseInt(e.target.value, 10) || 0)}
              />
            </div>
          )}
        </Card>
      </div>

      {/* Right: algorithm + hparams + train */}
      <div className="space-y-6">
        <Card
          title="Algorithm"
          subtitle={
            p.task === "unknown"
              ? "Pick a dataset and target first."
              : `Showing ${p.algoPool.length} algorithm${p.algoPool.length === 1 ? "" : "s"} for ${p.task}.`
          }
        >
          {p.algoPool.length === 0 ? (
            <EmptyState
              icon={<BrainCircuit size={28} />}
              title="No algorithms available"
              description={p.task === "unknown" ? "Pick a dataset above." : "No algorithms registered for this task yet."}
            />
          ) : (
            <ul className="grid gap-2 sm:grid-cols-2">
              {p.algoPool.map((a) => {
                const active = a.key === p.algoKey;
                return (
                  <li key={a.key}>
                    <button
                      type="button"
                      className={`w-full rounded-lg border p-3 text-left transition ${
                        active
                          ? "border-studio-ml bg-studio-ml/5"
                          : "border-bg-border bg-bg-panel hover:border-slate-500"
                      }`}
                      onClick={() => p.onPickAlgo(a.key)}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-slate-100">{a.label}</span>
                        {active && <span className="pill">selected</span>}
                      </div>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        <span
                          className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${
                            a.needs_scaling
                              ? "border-sky-700/40 bg-sky-900/30 text-sky-200"
                              : "border-slate-700/40 bg-slate-800/40 text-slate-400"
                          }`}
                          title={
                            a.needs_scaling
                              ? "StandardScaler will be applied automatically when you train."
                              : "Tree-based / NB algorithm — scaling is skipped (no effect)."
                          }
                        >
                          {a.needs_scaling ? "auto-scale ✓" : "no-scale needed"}
                        </span>
                        <span className="inline-flex items-center gap-1 rounded-full border border-emerald-700/40 bg-emerald-900/30 px-1.5 py-0.5 text-[10px] font-medium text-emerald-200"
                              title="Numeric imputation + categorical imputation + one-hot encoding are always auto-applied.">
                          auto-impute + encode ✓
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-slate-400">{a.description}</p>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </Card>

        {p.selectedAlgo && (
          <Card
            title="Hyperparameters"
            subtitle={`Defaults for ${p.selectedAlgo.label}. Override what you want.`}
          >
            <HyperparamForm
              spec={p.selectedAlgo.params}
              values={p.hparams}
              onChange={p.onHparam}
            />
            {Object.keys(p.selectedAlgo.fixed_params).length > 0 && (
              <p className="mt-3 text-xs text-slate-500">
                Fixed (always-applied):{" "}
                <span className="font-mono">
                  {Object.entries(p.selectedAlgo.fixed_params)
                    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                    .join(", ")}
                </span>
              </p>
            )}
          </Card>
        )}

        {p.error && (
          <div className="rounded-lg border border-red-700/40 bg-red-900/20 px-4 py-3 text-sm text-red-200">
            {p.error}
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <button
            className="btn-primary bg-studio-ml hover:bg-emerald-500"
            disabled={!p.currentDatasetId || !p.algoKey || p.training}
            onClick={p.onTrain}
          >
            <Play size={16} /> {p.training ? "Training…" : "Train model"}
          </button>
          <button className="btn-ghost" onClick={p.onReset}>
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------- Results

function ResultsSection() {
  const result = useMLStore((s) => s.result);
  if (!result) {
    return (
      <EmptyState
        title="No trained model yet"
        description="Configure and train a model first."
      />
    );
  }

  const apiBase = (api.defaults.baseURL ?? "").replace(/\/$/, "");
  const pickleUrl = apiBase + result.pickle_download_url.replace(/^\/api\/v1/, "");
  const sampleUrl = apiBase + result.sample_download_url.replace(/^\/api\/v1/, "");

  const metricTiles = buildMetricTiles(result);

  return (
    <div className="space-y-6">
      <Card
        title={`${result.algorithm_label} — ${result.task}`}
        subtitle={`${result.n_train} train / ${result.n_test} test rows · target: ${
          result.target_column ?? "(unsupervised)"
        }`}
        action={
          <div className="flex gap-2">
            <a
              className="btn-primary"
              href={pickleUrl}
              target="_blank"
              rel="noreferrer"
            >
              <Download size={16} /> model.pkl
            </a>
            <a className="btn-ghost" href={sampleUrl} target="_blank" rel="noreferrer">
              <Code size={16} /> sample_usage.py
            </a>
          </div>
        }
      >
        <div className="grid gap-3 sm:grid-cols-4">
          {metricTiles.map((t) => (
            <StatTile
              key={t.label}
              label={t.label}
              value={t.value}
              hint={t.hint}
              accent="text-studio-ml"
            />
          ))}
        </div>
      </Card>

      {(result.cv_used || result.hpt_mode === "grid" || result.hpt_mode === "random") && (
        <Card
          title="Cross-validation & tuning"
          subtitle={cvSubtitle(result)}
        >
          <div className="grid gap-4 lg:grid-cols-2">
            {/* Left: CV summary */}
            <div className="space-y-2 text-sm">
              <div className="flex items-baseline justify-between rounded-lg border border-bg-border bg-bg-panel/60 p-3">
                <span className="text-slate-400">CV mean score</span>
                <span className="font-mono text-xl font-semibold text-studio-ml">
                  {fmtNum(result.cv_mean_score)}
                  {result.cv_std_score !== null && result.cv_std_score !== undefined && (
                    <span className="ml-1 text-base text-slate-400">
                      ± {fmtNum(result.cv_std_score)}
                    </span>
                  )}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded border border-bg-border bg-bg-panel/60 px-2 py-1">
                  <div className="text-slate-500">Strategy</div>
                  <div className="font-mono text-slate-100">{result.cv_strategy ?? "none"}</div>
                </div>
                <div className="rounded border border-bg-border bg-bg-panel/60 px-2 py-1">
                  <div className="text-slate-500">Folds</div>
                  <div className="font-mono text-slate-100">{result.cv_folds ?? "—"}</div>
                </div>
                <div className="rounded border border-bg-border bg-bg-panel/60 px-2 py-1">
                  <div className="text-slate-500">Scoring</div>
                  <div className="font-mono text-slate-100">{result.cv_scoring ?? "—"}</div>
                </div>
                <div className="rounded border border-bg-border bg-bg-panel/60 px-2 py-1">
                  <div className="text-slate-500">HPT mode</div>
                  <div className="font-mono text-slate-100">{result.hpt_mode ?? "manual"}</div>
                </div>
              </div>
              {result.hpt_best_params && Object.keys(result.hpt_best_params).length > 0 && (
                <div className="rounded-lg border border-emerald-700/40 bg-emerald-900/15 p-3">
                  <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-emerald-200">
                    ★ Best params
                  </div>
                  <div className="space-y-0.5 font-mono text-xs text-slate-100">
                    {Object.entries(result.hpt_best_params).map(([k, v]) => (
                      <div key={k}>
                        <span className="text-emerald-300">{k}</span>
                        <span className="text-slate-500"> = </span>
                        <span>{JSON.stringify(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right: top HPT trials table */}
            {result.hpt_cv_results && result.hpt_cv_results.length > 0 && (
              <div className="overflow-x-auto rounded-lg border border-bg-border bg-bg-panel/40">
                <table className="w-full text-xs">
                  <thead className="bg-bg-card text-[10px] uppercase tracking-wide text-slate-400">
                    <tr>
                      <th className="px-2 py-1.5 text-left">Rank</th>
                      <th className="px-2 py-1.5 text-left">Mean ± std</th>
                      <th className="px-2 py-1.5 text-left">Params</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.hpt_cv_results.map((r, i) => (
                      <tr
                        key={i}
                        className={`border-t border-bg-border ${
                          r.rank === 1 ? "bg-emerald-900/15 text-emerald-100" : "text-slate-200"
                        }`}
                      >
                        <td className="px-2 py-1 font-mono">{r.rank}</td>
                        <td className="px-2 py-1 font-mono whitespace-nowrap">
                          {fmtNum(r.mean_score)} ± {fmtNum(r.std_score)}
                        </td>
                        <td className="px-2 py-1 font-mono">
                          {Object.entries(r.params)
                            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                            .join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </Card>
      )}

      {Object.keys(result.figures).length > 0 && (
        <Card title="Diagnostics" subtitle="Auto-generated plots for this model.">
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(result.figures).map(([name, fig]) => (
              <div
                key={name}
                className="rounded-lg border border-bg-border bg-bg-panel/40 p-2"
              >
                <PlotlyChart figure={fig} height={340} />
              </div>
            ))}
          </div>
        </Card>
      )}

      {result.preprocessing_applied && (
        <Card
          title="Auto-applied preprocessing"
          subtitle={`Baked into the pickle — feed the model raw rows, it handles the rest. ${result.preprocessing_applied.numeric_columns} numeric · ${result.preprocessing_applied.categorical_columns} categorical column${result.preprocessing_applied.categorical_columns === 1 ? "" : "s"}.`}
        >
          <ul className="space-y-2">
            {result.preprocessing_applied.steps.map((s) => (
              <li
                key={s.kind}
                className={`flex items-start gap-3 rounded-lg border p-3 text-sm ${
                  s.applied
                    ? "border-emerald-700/40 bg-emerald-900/15"
                    : "border-bg-border bg-bg-panel/40"
                }`}
              >
                <span
                  className={`mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full text-[10px] font-bold ${
                    s.applied ? "bg-emerald-600 text-white" : "bg-bg-card text-slate-500"
                  }`}
                  title={s.applied ? "applied" : "skipped"}
                >
                  {s.applied ? "✓" : "—"}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-2">
                    <span className="font-mono text-xs uppercase tracking-wide text-slate-400">
                      {s.kind}
                    </span>
                    <code className="font-mono text-xs text-slate-100">{s.method}</code>
                  </div>
                  <p className="mt-1 text-xs text-slate-400">{s.rationale}</p>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card
        title="How to use this model"
        subtitle={`Save model.pkl alongside this script. The pickle bundles preprocessing + the ${result.algorithm_label} estimator.`}
      >
        <CodeBlock code={result.sample_code} filename={result.sample_filename} />
        <div className="mt-3 rounded-lg border border-bg-border bg-bg-panel/60 p-3 text-xs text-slate-400">
          <span className="font-mono">Hyperparameters (effective):</span>{" "}
          {Object.entries(result.hparams_effective)
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
            .join(", ") || "(sklearn defaults)"}
        </div>
      </Card>
    </div>
  );
}

function fmtNum(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(4) : "—";
  return String(v);
}

function cvSubtitle(r: NonNullable<ReturnType<typeof useMLStore.getState>["result"]>): string {
  const parts: string[] = [];
  if (r.cv_used && r.cv_strategy) {
    parts.push(`${r.cv_strategy} (${r.cv_folds ?? "?"} folds)`);
  }
  if (r.hpt_mode && r.hpt_mode !== "manual") {
    parts.push(`${r.hpt_mode === "grid" ? "Grid Search" : "Randomised Search"}`);
    if (r.hpt_cv_results) parts.push(`${r.hpt_cv_results.length} trials`);
  }
  return parts.join(" · ") || "Cross-validated";
}

interface Tile { label: string; value: string; hint?: string }

function buildMetricTiles(r: ReturnType<typeof useMLStore.getState>["result"]): Tile[] {
  if (!r) return [];
  const m = r.metrics as Record<string, unknown>;
  const fmt = (x: unknown) =>
    typeof x === "number" ? (Number.isFinite(x) ? x.toFixed(4) : "—") : "—";

  if (r.task === "classification") {
    return [
      { label: "Accuracy", value: fmt(m.accuracy) },
      {
        label: `Precision (${m.average ?? "weighted"})`,
        value: fmt(m.precision),
      },
      {
        label: `Recall (${m.average ?? "weighted"})`,
        value: fmt(m.recall),
      },
      { label: "F1", value: fmt(m.f1), hint: `Macro F1: ${fmt(m.f1_macro)}` },
      { label: "ROC AUC", value: fmt(m.roc_auc) },
      { label: "Classes", value: String(m.n_classes ?? "—") },
    ];
  }
  if (r.task === "regression") {
    return [
      { label: "R²", value: fmt(m.r2) },
      { label: "MAE", value: fmt(m.mae) },
      { label: "MSE", value: fmt(m.mse) },
      { label: "RMSE", value: fmt(m.rmse) },
      { label: "Max error", value: fmt(m.max_error) },
    ];
  }
  // clustering
  return [
    { label: "Clusters", value: String(m.n_clusters ?? "—") },
    { label: "Noise points", value: String(m.n_noise ?? "—") },
    { label: "Silhouette", value: fmt(m.silhouette) },
    { label: "Davies-Bouldin", value: fmt(m.davies_bouldin) },
    { label: "Calinski-Harabasz", value: fmt(m.calinski_harabasz) },
    { label: "Inertia (KMeans)", value: fmt(m.inertia) },
  ];
}
