import { useEffect, useMemo, useRef, useState } from "react";
import {
  Trash2,
  LayoutDashboard,
  BarChart3,
  ShieldAlert,
  Sparkles,
  ChevronRight,
  Check,
  Zap,
  Code2,
  History as HistoryIcon,
  ScatterChart,
} from "lucide-react";

import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import EmptyState from "@/components/EmptyState";
import FileUpload from "@/components/FileUpload";
import DataTable from "@/components/DataTable";
import PlotlyChart from "@/components/PlotlyChart";
import RecommendationsList from "@/components/RecommendationsList";
import QualityIssueList, { issueKey } from "@/components/QualityIssueList";
import FixDialog from "@/components/FixDialog";
import BaselineCard from "@/components/BaselineCard";
import StatsDiffPanel from "@/components/StatsDiffPanel";
import HistoryPanel from "@/components/HistoryPanel";
import PipelineExportDialog from "@/components/PipelineExportDialog";
import CleanDatasetDialog from "@/components/CleanDatasetDialog";
import CockpitBar from "@/components/CockpitBar";
import BackToTop from "@/components/BackToTop";
import PrettySelect from "@/components/PrettySelect";

import { dataApi } from "@/api";
import { useProjectStore } from "@/store/projectStore";
import { useDataStore } from "@/store/dataStore";
import {
  formatBytes,
  formatNumber,
  shortId,
} from "@/lib/format";
import type {
  AutoFixResponse,
  BaselineResponse,
  EmbeddingVizResponse,
  HistoryEntry,
  QualityIssue,
  RecommendedFix,
  StatsDiff,
  TaskDetectionResponse,
  TransformResponse,
} from "@/api/types";

type SectionKey = "overview" | "visuals" | "quality" | "preprocessing" | "history";

const SECTION_ACCENT: Record<SectionKey, string> = {
  overview: "border-l-studio-data",
  visuals: "border-l-studio-dl",
  quality: "border-l-studio-agent",
  preprocessing: "border-l-studio-rag",
  history: "border-l-amber-400",
};

export default function DataStudio() {
  const projectId = useProjectStore((s) => s.currentId);

  const datasets = useDataStore((s) => s.datasets);
  const currentDatasetId = useDataStore((s) => s.currentDatasetId);
  const summary = useDataStore((s) => s.summary);
  const profile = useDataStore((s) => s.profile);
  const quality = useDataStore((s) => s.quality);
  const recs = useDataStore((s) => s.recommendations);
  const targetCol = useDataStore((s) => s.targetCol);
  const loading = useDataStore((s) => s.loading);
  const error = useDataStore((s) => s.error);
  const setTarget = useDataStore((s) => s.setTarget);
  const setCurrent = useDataStore((s) => s.setCurrent);
  const fetchDatasets = useDataStore((s) => s.fetchDatasets);
  const upload = useDataStore((s) => s.upload);
  const loadAll = useDataStore((s) => s.loadAll);
  const remove = useDataStore((s) => s.remove);
  const reset = useDataStore((s) => s.reset);

  const [section, setSection] = useState<SectionKey>("overview");
  const [selectedRecs, setSelectedRecs] = useState<Set<number>>(new Set());

  const [fixIssue, setFixIssue] = useState<QualityIssue | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const [activeFig, setActiveFig] = useState<string | null>(null);
  const [tsne, setTsne] = useState<EmbeddingVizResponse | null>(null);
  const [tsneLoading, setTsneLoading] = useState(false);
  const [tsneError, setTsneError] = useState<string | null>(null);

  const [recsByIssue, setRecsByIssue] = useState<Record<string, RecommendedFix | null>>({});
  const [inplacePref, setInplacePref] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem("aies.fix.inplace") === "1";
  });
  useEffect(() => {
    localStorage.setItem("aies.fix.inplace", inplacePref ? "1" : "0");
  }, [inplacePref]);

  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [autoFixing, setAutoFixing] = useState(false);
  const [autoFixResult, setAutoFixResult] = useState<AutoFixResponse | null>(null);

  const [taskDetect, setTaskDetect] = useState<TaskDetectionResponse | null>(null);
  const [taskLoading, setTaskLoading] = useState(false);
  const [baseline, setBaseline] = useState<BaselineResponse | null>(null);
  const [baselineLoading, setBaselineLoading] = useState(false);

  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [undoing, setUndoing] = useState(false);

  // Stat diff from the most recent transform — shown INSIDE Problem Columns.
  const [lastDiff, setLastDiff] = useState<{ summary: string; diff: StatsDiff } | null>(null);

  const [exportOpen, setExportOpen] = useState(false);
  const [cleanOpen, setCleanOpen] = useState(false);

  // Smooth-scroll target for section transitions
  const sectionAnchorRef = useRef<HTMLDivElement | null>(null);

  function setSectionAndScroll(k: string) {
    setSection(k as SectionKey);
    requestAnimationFrame(() => {
      sectionAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  // ---- Boot
  useEffect(() => { if (projectId) void fetchDatasets(projectId); }, [projectId, fetchDatasets]);
  useEffect(() => {
    if (currentDatasetId) void loadAll(currentDatasetId, targetCol ?? undefined);
    else reset();
  }, [currentDatasetId, targetCol, loadAll, reset]);
  useEffect(() => { setSelectedRecs(new Set(recs.map((_, idx) => idx))); }, [recs]);

  useEffect(() => {
    if (!currentDatasetId) { setHistory([]); return; }
    void dataApi.history(currentDatasetId)
      .then((r) => setHistory(r.entries))
      .catch(() => setHistory([]));
  }, [currentDatasetId]);

  useEffect(() => {
    if (!currentDatasetId) { setTaskDetect(null); return; }
    setTaskLoading(true);
    dataApi.detectTask(currentDatasetId, targetCol)
      .then(setTaskDetect)
      .catch(() => setTaskDetect(null))
      .finally(() => setTaskLoading(false));
  }, [currentDatasetId, targetCol]);

  useEffect(() => {
    if (!currentDatasetId || !targetCol) { setBaseline(null); return; }
    setBaselineLoading(true);
    dataApi.baseline(currentDatasetId, targetCol)
      .then(setBaseline)
      .catch(() => setBaseline(null))
      .finally(() => setBaselineLoading(false));
  }, [currentDatasetId, targetCol]);

  useEffect(() => {
    setRecsByIssue({});
    if (!quality || !currentDatasetId) return;
    let cancelled = false;
    Promise.all(
      quality.issues.map(async (i) => {
        try {
          const r = await dataApi.recommendedFix(currentDatasetId, i.code, i.column);
          return [issueKey(i), r] as const;
        } catch { return [issueKey(i), null] as const; }
      })
    ).then((pairs) => {
      if (cancelled) return;
      setRecsByIssue(Object.fromEntries(pairs));
    });
    return () => { cancelled = true; };
  }, [quality, currentDatasetId]);

  const figureEntries = useMemo(
    () => (profile?.figures ? Object.entries(profile.figures) : []),
    [profile?.figures]
  );

  useEffect(() => {
    if (figureEntries.length > 0 && (!activeFig || !figureEntries.find(([k]) => k === activeFig))) {
      setActiveFig(figureEntries[0][0]);
    }
  }, [figureEntries, activeFig]);

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(id);
  }, [toast]);

  useEffect(() => {
    setTsne(null);
    setTsneError(null);
  }, [currentDatasetId, targetCol]);

  async function refreshHistory() {
    if (!currentDatasetId) return;
    const r = await dataApi.history(currentDatasetId);
    setHistory(r.entries);
  }

  function handleApplied(result: TransformResponse) {
    setLastDiff({ summary: result.summary, diff: result.stats_diff || {} });
    if (result.inplace) {
      if (currentDatasetId) void loadAll(currentDatasetId, targetCol ?? undefined);
      void refreshHistory();
      setToast(`✓ ${result.summary}  ·  ${result.n_rows_before}→${result.n_rows_after} rows, ${result.n_cols_before}→${result.n_cols_after} cols`);
    } else if (result.new_dataset) {
      setCurrent(result.new_dataset.id);
      if (projectId) void fetchDatasets(projectId);
      setToast(`✓ ${result.summary}  ·  new dataset "${result.new_dataset.name}" created`);
    }
  }

  async function handleQuickFix(issue: QualityIssue, rec: RecommendedFix) {
    if (!currentDatasetId) return;
    const key = issueKey(issue);
    setBusyKey(key);
    try {
      const r = await dataApi.transform(currentDatasetId, {
        operation: rec.operation, params: rec.params, inplace: inplacePref,
      });
      handleApplied(r);
    } catch (e) {
      setToast(`✗ Quick Fix failed: ${(e as Error).message}`);
    } finally {
      setBusyKey(null);
    }
  }

  async function handleAutoFixAll() {
    if (!currentDatasetId || !quality) return;
    setAutoFixing(true);
    setAutoFixResult(null);
    try {
      const r = await dataApi.autoFix(currentDatasetId, inplacePref, targetCol);
      setAutoFixResult(r);
      if (r.inplace) {
        if (currentDatasetId) void loadAll(currentDatasetId, targetCol ?? undefined);
        void refreshHistory();
      } else if (r.new_dataset) {
        setCurrent(r.new_dataset.id);
        if (projectId) void fetchDatasets(projectId);
      }
      setToast(`✓ Auto-fix: ${r.n_applied} applied, ${r.n_skipped} skipped, ${r.n_failed} failed`);
    } catch (e) {
      setToast(`✗ Auto-fix failed: ${(e as Error).message}`);
    } finally {
      setAutoFixing(false);
    }
  }

  async function handleUndo() {
    if (!currentDatasetId) return;
    setUndoing(true);
    try {
      const u = await dataApi.undo(currentDatasetId);
      setToast(`↶ Undid step ${u.undone_step} (${u.operation}) — ${u.n_rows_after}×${u.n_cols_after}, ${u.history_remaining} remaining`);
      void loadAll(currentDatasetId, targetCol ?? undefined);
      void refreshHistory();
      setLastDiff(null);
    } catch (e) {
      setToast(`✗ Undo failed: ${(e as Error).message}`);
    } finally {
      setUndoing(false);
    }
  }

  async function loadTSNE() {
    if (!currentDatasetId) return;
    setTsneLoading(true);
    setTsneError(null);
    try {
      const r = await dataApi.embeddingViz(currentDatasetId, targetCol);
      setTsne(r);
    } catch (e) {
      setTsneError((e as Error).message);
    } finally {
      setTsneLoading(false);
    }
  }

  const recommendableCount = useMemo(
    () => quality?.issues.filter((i) => recsByIssue[issueKey(i)]).length ?? 0,
    [quality, recsByIssue]
  );

  const issuesTotal = quality?.issues.length ?? 0;

  const tabs = [
    { key: "overview",      label: "Overview",        icon: <LayoutDashboard size={14} /> },
    { key: "visuals",       label: "Visualizations",  icon: <BarChart3 size={14} />,    badge: figureEntries.length || undefined },
    { key: "quality",       label: "Problem Columns", icon: <ShieldAlert size={14} />,  badge: issuesTotal || undefined },
    { key: "preprocessing", label: "Preprocessing",   icon: <Sparkles size={14} />,     badge: recs.length || undefined },
    { key: "history",       label: "History",         icon: <HistoryIcon size={14} />,  badge: history.length || undefined },
  ];

  if (!projectId) {
    return (
      <div className="mx-auto max-w-3xl px-8 py-12">
        <EmptyState
          title="Select or create a project"
          description="Use the project switcher in the sidebar — every dataset belongs to a project."
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-8 py-8">
      <PageHeader
        title="Data Engineering & EDA"
        subtitle="Upload, profile, diagnose, fix — every transform is tracked and replayable."
        accent="text-studio-data"
        action={
          summary && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                className="flex items-center gap-2 rounded-lg bg-emerald-400 px-3.5 py-2 text-sm font-semibold text-on-accent hover:bg-emerald-300"
                onClick={() => setCleanOpen(true)}
                title="One-click: iteratively apply the recommended fix for every problem until none remain. Produces a brand-new ML-ready dataset."
              >
                <Sparkles size={15} /> Make Training-Ready
              </button>
              <button
                className="btn-ghost"
                onClick={() => setExportOpen(true)}
                title="Export the applied transforms as preprocess.py"
              >
                <Code2 size={14} /> Export pipeline
                {history.length > 0 && (
                  <span className="ml-1 rounded-full bg-emerald-700/30 px-1.5 text-xs">
                    {history.length}
                  </span>
                )}
              </button>
            </div>
          )
        }
      />

      <div className="grid gap-6 lg:grid-cols-[340px_1fr]">
        {/* ─── Left rail ─── */}
        <div className="space-y-5">
          <Card title="Upload">
            <FileUpload
              onFile={async (f) => { await upload(projectId, f); }}
              hint="CSV, TSV, XLSX, Parquet, JSON · ≤ 200 MB"
            />
          </Card>

          <Card title="Datasets" subtitle={`${datasets.length} in this project`}>
            {datasets.length === 0 ? (
              <p className="text-sm text-slate-500">No datasets yet.</p>
            ) : (
              <ul className="space-y-1.5">
                {datasets.map((d) => {
                  const active = d.id === currentDatasetId;
                  return (
                    <li
                      key={d.id}
                      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-sm transition ${
                        active
                          ? "border-studio-data bg-studio-data/5"
                          : "border-bg-border bg-bg-panel hover:border-slate-500"
                      }`}
                    >
                      <button
                        className="min-w-0 flex-1 text-left"
                        onClick={() => setCurrent(d.id)}
                      >
                        <div className="truncate font-medium text-slate-100">{d.name}</div>
                        <div className="mt-0.5 text-xs text-slate-500">
                          {formatNumber(d.n_rows)} × {d.n_cols} · {formatBytes(d.size_bytes)} · {shortId(d.id)}
                        </div>
                      </button>
                      <button
                        className="ml-2 rounded p-1 text-slate-500 hover:bg-bg-card hover:text-red-400"
                        onClick={() => {
                          if (confirm(`Delete "${d.name}"?`)) void remove(d.id);
                        }}
                      >
                        <Trash2 size={14} />
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </Card>

          {summary && (
            <Card
              title="Target column"
              subtitle="Drives task detection + baseline + class-imbalance/leakage checks."
            >
              <select
                className="input"
                value={targetCol ?? ""}
                onChange={(e) => setTarget(e.target.value || null)}
              >
                <option value="">— Unsupervised / not set —</option>
                {summary.columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}  ({c.dtype})
                  </option>
                ))}
              </select>
            </Card>
          )}
        </div>

        {/* ─── Right pane ─── */}
        <div className="space-y-4">
          {error && (
            <div className="rounded-lg border border-red-700/40 bg-red-900/20 px-4 py-3 text-sm text-red-200">
              {error}
            </div>
          )}

          {!summary ? (
            <EmptyState
              title={loading ? "Loading…" : "Select or upload a dataset"}
              description="The cockpit, sections, and tools appear here once a dataset is selected."
            />
          ) : (
            <>
              <CockpitBar
                task={taskDetect}
                taskLoading={taskLoading}
                summary={summary}
                profile={profile}
                quality={quality}
                tabs={tabs}
                activeSection={section}
                onSectionChange={setSectionAndScroll}
              />

              <div ref={sectionAnchorRef} className="scroll-mt-32" />

              <SectionPanel sectionKey={section}>
                {section === "overview" && (
                  <div className="grid gap-3 lg:grid-cols-[1fr_320px]">
                    <Card title="Schema & sample" subtitle="First 10 rows with detected dtypes.">
                      <DataTable
                        columns={summary.columns.map((c) => ({
                          key: c.name,
                          label: `${c.name}  ·  ${c.dtype}`,
                        }))}
                        rows={summary.sample}
                      />
                    </Card>
                    <div className="space-y-3">
                      <BaselineCard
                        baseline={baseline}
                        loading={baselineLoading}
                        onRefresh={() => {
                          if (!currentDatasetId || !targetCol) return;
                          setBaselineLoading(true);
                          dataApi.baseline(currentDatasetId, targetCol)
                            .then(setBaseline)
                            .catch(() => setBaseline(null))
                            .finally(() => setBaselineLoading(false));
                        }}
                      />
                      {!targetCol && (
                        <p className="rounded-lg border border-dashed border-bg-border bg-bg-panel/40 p-3 text-xs text-slate-400">
                          Set a target column on the left to compute the baseline score.
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {section === "visuals" && (
                  <div className="space-y-3">
                    <VisualsView
                      entries={figureEntries}
                      activeKey={activeFig}
                      onPick={setActiveFig}
                    />

                    <Card
                      title="t-SNE projection"
                      subtitle="2-D embedding of the whole feature space. Colour by target if set. Sampled to 2000 rows."
                      action={
                        !tsne && (
                          <button
                            className="btn-primary bg-studio-dl"
                            onClick={loadTSNE}
                            disabled={tsneLoading}
                          >
                            <ScatterChart size={14} />
                            {tsneLoading ? "Computing…" : "Compute t-SNE"}
                          </button>
                        )
                      }
                    >
                      {tsneError && (
                        <div className="mb-3 rounded-lg border border-red-700/40 bg-red-900/20 px-3 py-2 text-sm text-red-200">
                          {tsneError}
                        </div>
                      )}
                      {tsne ? (
                        <>
                          <div className="rounded-lg border border-bg-border bg-bg-panel/40 p-2">
                            <PlotlyChart figure={tsne.figure} height={460} />
                          </div>
                          <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
                            <span>{tsne.n_points} points · perplexity {tsne.perplexity}</span>
                            <button className="btn-ghost py-1 text-xs" onClick={loadTSNE}>
                              Recompute
                            </button>
                          </div>
                        </>
                      ) : tsneLoading ? (
                        <p className="text-sm text-slate-400">
                          Crunching the embedding — t-SNE is slow on first run, give it ~10–60 s.
                        </p>
                      ) : (
                        <p className="text-sm text-slate-500">
                          Click <span className="font-mono">Compute t-SNE</span> to see whether your classes
                          (or natural clusters) actually separate in feature space.
                        </p>
                      )}
                    </Card>
                  </div>
                )}

                {section === "quality" && (
                  <Card
                    title="Problem columns"
                    subtitle="Each issue has a recommended method. Quick Fix applies it directly; Fix opens manual configuration."
                  >
                    {quality ? (
                      <>
                        {quality.issues.length > 0 && (
                          <div className="mb-4 space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-bg-border bg-bg-panel/60 p-3">
                              <div className="flex flex-wrap items-center gap-2">
                                <button
                                  className="flex items-center gap-2 rounded-lg bg-emerald-700/30 px-3 py-1.5 text-sm font-semibold text-emerald-100 hover:bg-emerald-700/50 disabled:cursor-not-allowed disabled:opacity-50"
                                  onClick={handleAutoFixAll}
                                  disabled={autoFixing || recommendableCount === 0}
                                  title="One round, applies recommended fix per issue."
                                >
                                  <Zap size={14} />
                                  {autoFixing ? "Running…" : `Auto-fix all (${recommendableCount})`}
                                </button>
                                <button
                                  className="flex items-center gap-2 rounded-lg bg-emerald-400 px-3 py-1.5 text-sm font-semibold text-on-accent hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                                  onClick={() => setCleanOpen(true)}
                                  disabled={autoFixing || recommendableCount === 0}
                                  title="Iterative loop — keeps applying fixes until no fixable issues remain. Always creates a new dataset."
                                >
                                  <Sparkles size={14} />
                                  Create clean dataset
                                </button>
                              </div>
                              <label className="flex items-center gap-2 text-xs text-slate-300">
                                <input
                                  type="checkbox"
                                  checked={inplacePref}
                                  onChange={(e) => setInplacePref(e.target.checked)}
                                />
                                <span className="font-mono">inplace = {inplacePref ? "true" : "false"}</span>
                                <span className="text-slate-500">
                                  ({inplacePref ? "overwrite + history" : "new dataset"})
                                </span>
                              </label>
                            </div>
                            <p className="px-1 text-xs text-slate-500">
                              <span className="font-semibold text-slate-300">Auto-fix all</span> runs one round on the current dataset (respects the toggle).{" "}
                              <span className="font-semibold text-slate-300">Create clean dataset</span> loops until convergence — produces a brand-new dataset with zero fixable issues.
                            </p>
                          </div>
                        )}

                        {/* Latest change — kept INSIDE this section so it's contextual */}
                        {lastDiff && Object.keys(lastDiff.diff).length > 0 && (
                          <div className="mb-4 rounded-lg border border-emerald-700/30 bg-emerald-900/10 p-3">
                            <div className="mb-2 flex items-center justify-between text-xs text-emerald-200">
                              <span>★ Latest change: <span className="font-mono">{lastDiff.summary}</span></span>
                              <button
                                className="text-emerald-300 hover:text-emerald-100"
                                onClick={() => setLastDiff(null)}
                              >
                                dismiss
                              </button>
                            </div>
                            <StatsDiffPanel diff={lastDiff.diff} title="" />
                          </div>
                        )}

                        <QualityIssueList
                          issues={quality.issues}
                          recommendations={recsByIssue}
                          onFix={(issue) => setFixIssue(issue)}
                          onQuickFix={(issue, rec) => void handleQuickFix(issue, rec)}
                          busyKey={busyKey}
                        />

                        {autoFixResult && (
                          <div className="mt-4 rounded-lg border border-bg-border bg-bg-panel/60 p-3 text-xs">
                            <div className="mb-2 flex flex-wrap items-center gap-2 text-slate-300">
                              <Sparkles size={12} className="text-emerald-300" />
                              <span className="font-semibold">
                                {autoFixResult.rounds_run && autoFixResult.rounds_run > 1
                                  ? "Clean-dataset report:"
                                  : "Auto-fix report:"}
                              </span>
                              <span>
                                {autoFixResult.n_applied} applied · {autoFixResult.n_skipped} skipped · {autoFixResult.n_failed} failed
                              </span>
                              {autoFixResult.rounds_run !== undefined && autoFixResult.rounds_run > 1 && (
                                <span className="rounded-full border border-emerald-700/40 bg-emerald-900/30 px-1.5 text-emerald-200">
                                  {autoFixResult.rounds_run} rounds
                                </span>
                              )}
                              {autoFixResult.final_issues_count !== undefined &&
                               autoFixResult.final_issues_count !== null && (
                                <span
                                  className={`rounded-full border px-1.5 ${
                                    autoFixResult.final_issues_count === 0
                                      ? "border-emerald-700/40 bg-emerald-900/30 text-emerald-200"
                                      : "border-amber-700/40 bg-amber-900/30 text-amber-200"
                                  }`}
                                >
                                  {autoFixResult.final_issues_count === 0
                                    ? "✓ 0 issues remain"
                                    : `${autoFixResult.final_issues_count} unfixable`}
                                </span>
                              )}
                              <span className="text-slate-500">
                                · rows {autoFixResult.n_rows_before}→{autoFixResult.n_rows_after}, cols {autoFixResult.n_cols_before}→{autoFixResult.n_cols_after}
                              </span>
                              <button
                                className="ml-auto text-slate-500 hover:text-slate-100"
                                onClick={() => setAutoFixResult(null)}
                              >
                                dismiss
                              </button>
                            </div>
                            {autoFixResult.unfixable && autoFixResult.unfixable.length > 0 && (
                              <div className="mb-2 flex flex-wrap items-center gap-1.5 rounded-md border border-amber-700/30 bg-amber-900/15 px-2 py-1 text-amber-100">
                                <span className="font-semibold">Left unfixed:</span>
                                {autoFixResult.unfixable.map((c) => (
                                  <span key={c} className="rounded bg-amber-800/40 px-1.5 font-mono">{c}</span>
                                ))}
                                <span className="text-amber-300/80">— handled at training time or needs manual config.</span>
                              </div>
                            )}
                            <ul className="space-y-1">
                              {autoFixResult.steps.map((s, i) => (
                                <li
                                  key={i}
                                  className={`flex items-start gap-2 rounded border border-bg-border px-2 py-1 ${
                                    s.status === "applied"
                                      ? "bg-emerald-900/15"
                                      : s.status === "failed"
                                      ? "bg-red-900/15"
                                      : "bg-bg-panel/50"
                                  }`}
                                >
                                  <span
                                    className={`mt-0.5 inline-block h-1.5 w-1.5 shrink-0 rounded-full ${
                                      s.status === "applied"
                                        ? "bg-emerald-400"
                                        : s.status === "failed"
                                        ? "bg-red-400"
                                        : "bg-slate-500"
                                    }`}
                                  />
                                  <div className="min-w-0 flex-1">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="font-mono text-slate-400">{s.issue_code}</span>
                                      {s.column && <span className="pill">{s.column}</span>}
                                      {s.operation && (
                                        <span className="font-mono text-emerald-300">{s.operation}</span>
                                      )}
                                    </div>
                                    {s.summary && (
                                      <p className="mt-0.5 text-slate-200">{s.summary}</p>
                                    )}
                                    {s.error && (
                                      <p className="mt-0.5 text-red-300">{s.error}</p>
                                    )}
                                  </div>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </>
                    ) : (
                      <p className="text-sm text-slate-500">Loading…</p>
                    )}
                  </Card>
                )}

                {section === "preprocessing" && (
                  <Card
                    title="Preprocessing recommendations"
                    subtitle="Reference list — the same operations get baked into ML/DL trainers. To actually alter your data, use Problem Columns → Fix."
                  >
                    <RecommendationsList
                      recommendations={recs}
                      selected={selectedRecs}
                      onToggle={(idx) => {
                        setSelectedRecs((prev) => {
                          const next = new Set(prev);
                          if (next.has(idx)) next.delete(idx);
                          else next.add(idx);
                          return next;
                        });
                      }}
                    />
                  </Card>
                )}

                {section === "history" && (
                  <Card
                    title="Transform history"
                    subtitle="Every in-place transform is recorded as a step with a snapshot. Undo restores the previous snapshot. Export the full pipeline as code via the button at the top."
                  >
                    <HistoryPanel
                      entries={history}
                      onUndo={handleUndo}
                      busy={undoing}
                    />
                  </Card>
                )}
              </SectionPanel>
            </>
          )}
        </div>
      </div>

      {toast && (
        <div className="pointer-events-none fixed bottom-6 left-1/2 z-40 -translate-x-1/2 rounded-xl border border-emerald-700/40 bg-emerald-900/80 px-4 py-2 text-sm text-emerald-100 shadow-xl backdrop-blur">
          {toast}
        </div>
      )}

      <FixDialog
        open={!!fixIssue}
        onClose={() => setFixIssue(null)}
        issue={fixIssue}
        datasetId={currentDatasetId ?? ""}
        summary={summary}
        onApplied={handleApplied}
      />

      <PipelineExportDialog
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        datasetId={currentDatasetId ?? ""}
      />

      <CleanDatasetDialog
        open={cleanOpen}
        onClose={() => setCleanOpen(false)}
        datasetId={currentDatasetId ?? ""}
        datasetName={
          datasets.find((d) => d.id === currentDatasetId)?.name ?? "dataset"
        }
        targetCol={targetCol}
        onApplied={(r) => {
          setAutoFixResult(r);
          if (r.new_dataset) {
            setCurrent(r.new_dataset.id);
            if (projectId) void fetchDatasets(projectId);
          }
          setToast(
            `✓ Clean dataset: ${r.n_applied} ops in ${r.rounds_run ?? 1} round${r.rounds_run === 1 ? "" : "s"} · final issues ${r.final_issues_count ?? "?"}`
          );
        }}
      />

      <BackToTop />
    </div>
  );
}

// ---------------------------------------------------------------------------- Panel + Visuals

function SectionPanel({
  sectionKey,
  children,
}: {
  sectionKey: SectionKey;
  children: React.ReactNode;
}) {
  return (
    <div
      className={`rounded-xl border border-bg-border bg-bg-panel/40 border-l-4 ${SECTION_ACCENT[sectionKey]}`}
    >
      <div className="p-2">{children}</div>
    </div>
  );
}

function VisualsView({
  entries,
  activeKey,
  onPick,
}: {
  entries: [string, unknown][];
  activeKey: string | null;
  onPick: (k: string) => void;
}) {
  if (entries.length === 0) {
    return (
      <Card title="Auto EDA">
        <p className="text-sm text-slate-500">No plots generated yet.</p>
      </Card>
    );
  }

  const active = entries.find(([k]) => k === activeKey) ?? entries[0];

  return (
    <Card
      title="Visualizations"
      subtitle="Pick a chart from the dropdown or the strip — it expands to full width below."
    >
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {/* Primary: prominent dropdown */}
        <PrettySelect
          label="Chart"
          value={active[0]}
          size="md"
          accent="focus-within:border-studio-dl hover:border-studio-dl/60"
          options={entries.map(([key]) => ({ value: key, label: prettyName(key) }))}
          onChange={onPick}
          title="Pick a chart"
        />

        <span className="hidden text-xs text-slate-500 md:inline">or</span>

        {/* Secondary: strip (visual context, scrollable) */}
        <div className="-mx-1 flex max-w-full gap-2 overflow-x-auto px-1 pb-1">
          {entries.map(([key]) => {
            const on = key === activeKey;
            return (
              <button
                key={key}
                onClick={() => onPick(key)}
                className={`group flex shrink-0 items-center gap-2 rounded-md border px-2 py-1 text-[11px] transition ${
                  on
                    ? "border-studio-dl bg-studio-dl/10 text-slate-100"
                    : "border-bg-border bg-bg-panel text-slate-400 hover:border-slate-500 hover:text-slate-100"
                }`}
              >
                {on && (
                  <span className="grid h-3 w-3 place-items-center rounded-full bg-studio-dl text-on-accent">
                    <Check size={9} />
                  </span>
                )}
                <span className="font-medium">{prettyName(key)}</span>
                {!on && <ChevronRight size={11} className="opacity-50 group-hover:opacity-100" />}
              </button>
            );
          })}
        </div>
      </div>

      <div className="rounded-lg border border-bg-border bg-bg-panel/40 p-2">
        <PlotlyChart figure={active[1]} height={460} />
      </div>
    </Card>
  );
}

function prettyName(key: string): string {
  if (key.startsWith("hist::")) return `Distribution · ${key.slice(6)}`;
  if (key.startsWith("cats::")) return `Top categories · ${key.slice(6)}`;
  const map: Record<string, string> = {
    boxplots: "Numeric boxplots",
    missing_bar: "Missing per column",
    corr_heatmap: "Correlation heatmap",
  };
  return map[key] ?? key.replace(/_/g, " ");
}
