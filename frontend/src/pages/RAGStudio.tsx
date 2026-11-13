import { useEffect, useMemo, useState } from "react";
import {
  Database, Search, Loader2, Trash2, Play, Sparkles, Code2, ChevronDown, ChevronRight,
} from "lucide-react";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import PrettySelect from "@/components/PrettySelect";
import { useProjectStore } from "@/store/projectStore";
import {
  ragApi, type RagCatalog, type RagIndex, type RagIngestResponse, type RagQueryResponse, type ParamSpec,
} from "@/api/rag";

const SAMPLE = `Hybrid search fuses BM25 lexical retrieval with dense vector retrieval using reciprocal rank fusion (RRF).
Dense retrieval uses an embedding model and cosine similarity over an ANN index such as HNSW.
BM25 is a sparse keyword scoring function good for exact terms and rare tokens.
A cross-encoder reranker reorders the top candidates for precision.
MMR selects diverse, non-redundant results. Recursive chunking splits on a hierarchy of separators.
Contextual retrieval prepends an LLM-written blurb to each chunk before embedding.`;

// strategy preset -> query-stage wiring (the taxonomy as presets over one DAG)
const PRESETS: Record<string, Partial<{ transform: string; retriever: string; rerank: string; compress: string }>> = {
  naive: { transform: "none", retriever: "dense", rerank: "none", compress: "none" },
  advanced: { transform: "rewrite", retriever: "hybrid", rerank: "cross_encoder", compress: "compression" },
  hybrid: { transform: "none", retriever: "hybrid", rerank: "none", compress: "none" },
  rag_fusion: { transform: "rag_fusion", retriever: "hybrid", rerank: "none", compress: "none" },
  hyde: { transform: "hyde", retriever: "dense", rerank: "none", compress: "none" },
  contextual: { transform: "none", retriever: "hybrid", rerank: "cross_encoder", compress: "none" },
  parent_document: { transform: "none", retriever: "dense", rerank: "none", compress: "none" },
  conversational: { transform: "rewrite", retriever: "hybrid", rerank: "none", compress: "none" },
};

const num = (v: unknown, d: number) => (v === "" || v == null || isNaN(Number(v)) ? d : Number(v));

export default function RAGStudio() {
  const projectId = useProjectStore((s) => s.currentId);
  const [cat, setCat] = useState<RagCatalog | null>(null);
  const [indexes, setIndexes] = useState<RagIndex[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [showIR, setShowIR] = useState(false);

  // ── pipeline config (the IR) ──
  const [strategy, setStrategy] = useState("advanced");
  // index-time
  const [name, setName] = useState("My knowledge base");
  const [srcKind, setSrcKind] = useState("text");
  const [srcText, setSrcText] = useState(SAMPLE);
  const [chunkImpl, setChunkImpl] = useState("recursive");
  const [chunkP, setChunkP] = useState<Record<string, unknown>>({ size: 512, overlap: 64 });
  const [embedKey, setEmbedKey] = useState("tfidf");
  const [storeImpl, setStoreImpl] = useState("faiss");
  const [annImpl, setAnnImpl] = useState("hnsw");
  // query-time
  const [transformImpl, setTransformImpl] = useState("rewrite");
  const [transformN, setTransformN] = useState(4);
  const [retriever, setRetriever] = useState("hybrid");
  const [k, setK] = useState(8);
  const [fusion, setFusion] = useState("rrf");
  const [alpha, setAlpha] = useState(0.5);
  const [rerankImpl, setRerankImpl] = useState("cross_encoder");
  const [rerankTopN, setRerankTopN] = useState(8);
  const [compressImpl, setCompressImpl] = useState("compression");
  const [targetTokens, setTargetTokens] = useState(2000);
  const [genProvider, setGenProvider] = useState("ollama");
  const [genModel, setGenModel] = useState("llama3");
  const [citations, setCitations] = useState(true);
  // runtime
  const [question, setQuestion] = useState("How does hybrid search combine BM25 and dense vectors?");
  const [activeIdx, setActiveIdx] = useState("");
  const [building, setBuilding] = useState(false);
  const [buildRes, setBuildRes] = useState<RagIngestResponse | null>(null);
  const [querying, setQuerying] = useState(false);
  const [result, setResult] = useState<RagQueryResponse | null>(null);

  useEffect(() => { ragApi.catalog().then(setCat).catch(() => setCat(null)); }, []);
  const refreshIndexes = () => {
    if (!projectId) return;
    ragApi.indexes(projectId).then((rows) => {
      setIndexes(rows);
      setActiveIdx((cur) => (rows.find((r) => r.id === cur) ? cur : rows[0]?.id ?? ""));
    });
  };
  useEffect(refreshIndexes, [projectId]);

  function applyStrategy(key: string) {
    setStrategy(key);
    const p = PRESETS[key];
    if (p) {
      if (p.transform) setTransformImpl(p.transform);
      if (p.retriever) setRetriever(p.retriever);
      if (p.rerank) setRerankImpl(p.rerank);
      if (p.compress) setCompressImpl(p.compress);
    }
  }

  // embedding options: TF-IDF (real, offline) + every catalog model
  const embedOpts = useMemo(() => {
    const base = [{ value: "tfidf", label: "TF-IDF — offline, built-in ★" }];
    const more = (cat?.embeddings ?? []).map((e) => ({
      value: e.key, label: `${e.label} · ${e.provider}${e.dims ? ` · ${e.dims}d` : ""}`,
    }));
    return [...base, ...more];
  }, [cat]);
  const selEmbed = useMemo(
    () => (cat?.embeddings ?? []).find((e) => e.key === embedKey),
    [cat, embedKey]
  );

  const opts = (arr?: { key: string; label: string; recommended?: boolean }[]) =>
    (arr ?? []).map((x) => ({ value: x.key, label: x.label + (x.recommended ? " ★" : "") }));

  const chunkSpec = cat?.chunking.find((c) => c.key === chunkImpl)?.params;

  // the IR / config object (dev face)
  const ir = useMemo(() => ({
    strategy,
    index: {
      source: srcKind,
      chunk: { impl: chunkImpl, ...chunkP },
      embed: embedKey === "tfidf" ? { provider: "tfidf", model: "tfidf" } : { provider: selEmbed?.provider, model: embedKey },
      store: { impl: storeImpl, ann: annImpl },
    },
    query: {
      transform: { impl: transformImpl, n: transformN },
      retrieve: { impl: retriever, k, fusion, alpha },
      rerank: { impl: rerankImpl, top_n: rerankTopN },
      compress: { impl: compressImpl, target_tokens: targetTokens },
      generate: { provider: genProvider, model: genModel, citations },
    },
  }), [strategy, srcKind, chunkImpl, chunkP, embedKey, selEmbed, storeImpl, annImpl,
       transformImpl, transformN, retriever, k, fusion, alpha, rerankImpl, rerankTopN,
       compressImpl, targetTokens, genProvider, genModel, citations]);

  async function build() {
    if (!projectId) return setErr("Pick or create a project first.");
    if (!srcText.trim()) return setErr("Add source text or a URL.");
    setBuilding(true); setErr(null); setBuildRes(null);
    try {
      const res = await ragApi.ingest({
        project_id: projectId, name, sources: [{ kind: srcKind, location: srcText.trim() }],
        chunking_strategy: chunkImpl, chunk_size: num(chunkP.size, 512), chunk_overlap: num(chunkP.overlap, 64),
        embedding_provider: embedKey === "tfidf" ? "tfidf" : (selEmbed?.provider ?? "tfidf"),
        embedding_model: embedKey, vector_store: storeImpl, ann_index: annImpl,
      });
      setBuildRes(res); setActiveIdx(res.index.id); refreshIndexes();
    } catch (e) { setErr((e as Error).message); } finally { setBuilding(false); }
  }

  async function runQuery() {
    if (!activeIdx) return setErr("Build or select an index first.");
    setQuerying(true); setErr(null); setResult(null);
    try {
      const res = await ragApi.query({
        index_id: activeIdx, question, strategy,
        query_transform: transformImpl, transform_n: transformN,
        retriever_type: retriever, k, fusion, hybrid_alpha: alpha,
        reranker: rerankImpl, rerank_top_n: rerankTopN,
        context_processor: compressImpl === "none" ? undefined : "compression",
        compress_target_tokens: targetTokens,
        generate_answer: true, citations, llm_provider: genProvider, llm_model: genModel,
      });
      setResult(res);
    } catch (e) { setErr((e as Error).message); } finally { setQuerying(false); }
  }

  return (
    <div className="mx-auto max-w-7xl px-8 py-8">
      <PageHeader title="RAG Engineering Studio" accent="text-studio-rag"
        subtitle="Build a retrieval pipeline step by step: split, embed and index your sources, then choose how questions are rewritten, retrieved, reranked and answered. Start from a strategy preset or set each step yourself." />

      {!projectId && <Card className="mb-4"><p className="text-sm text-amber-400">Select a project (top-left) to build an index.</p></Card>}
      {err && <Card className="mb-4"><p className="text-sm text-red-400">{err}</p></Card>}

      {/* strategy presets */}
      {cat && (
        <Card className="mb-4" title="Strategy" subtitle="A preset is just a saved wiring over the same pipeline. Pick one, then tweak any stage.">
          <div className="flex flex-wrap gap-2">
            {cat.strategies.map((s) => (
              <button key={s.key} onClick={() => applyStrategy(s.key)} title={`${s.wiring} — ${s.when}`}
                className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                  strategy === s.key ? "bg-studio-rag text-on-accent" : "border border-bg-border text-slate-300 hover:bg-bg-panel"
                }`}>
                {s.label}{s.recommended ? " ★" : ""}
              </button>
            ))}
          </div>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        {/* ───────── INDEX PIPELINE ───────── */}
        <Card title="Index pipeline" subtitle="Build the searchable index from your sources.">
          <div className="space-y-3">
            <div>
              <label className="label">Index name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
            </div>

            <Stage n={1} title="Source / Load">
              <PrettySelect value={srcKind} onChange={setSrcKind} block accent="focus-within:border-studio-rag"
                options={[{ value: "text", label: "Paste text" }, { value: "url", label: "URL" },
                          { value: "csv", label: "CSV path" }, { value: "txt", label: "Text file path" }]} />
              {srcKind === "text"
                ? <textarea className="input mt-2 min-h-[96px] font-mono text-xs" value={srcText} onChange={(e) => setSrcText(e.target.value)} />
                : <input className="input mt-2" placeholder={srcKind === "url" ? "https://…" : "C:\\path"} value={srcText} onChange={(e) => setSrcText(e.target.value)} />}
            </Stage>

            <Stage n={2} title="Chunk / Split">
              <PrettySelect value={chunkImpl} onChange={setChunkImpl} block accent="focus-within:border-studio-rag" options={opts(cat?.chunking)} />
              <ParamForm spec={chunkSpec} values={chunkP} onChange={setChunkP} />
            </Stage>

            <Stage n={3} title="Embed (model choice)">
              <PrettySelect value={embedKey} onChange={setEmbedKey} block accent="focus-within:border-studio-rag" options={embedOpts} />
              <p className="mt-1 text-[11px] text-slate-500">
                TF-IDF runs offline now. Neural models run if you start Ollama or set an OpenAI key — otherwise the build falls back with a note.
              </p>
            </Stage>

            <Stage n={4} title="Store / Index">
              <div className="grid grid-cols-2 gap-2">
                <PrettySelect label="DB" value={storeImpl} onChange={setStoreImpl} block accent="focus-within:border-studio-rag" options={opts(cat?.vector_stores)} />
                <PrettySelect label="ANN" value={annImpl} onChange={setAnnImpl} block accent="focus-within:border-studio-rag" options={opts(cat?.ann_indexes)} />
              </div>
            </Stage>

            <button className="btn-primary w-full bg-studio-rag" disabled={building || !projectId} onClick={build}>
              {building ? <Loader2 size={16} className="animate-spin" /> : <Database size={16} />}
              {building ? "Building…" : "Build index"}
            </button>
            {buildRes && (
              <div className="rounded-lg border border-bg-border bg-bg-panel p-3 text-xs text-slate-300">
                ✓ {buildRes.index.name} · {buildRes.n_chunks} chunks · {buildRes.embed_kind} ({buildRes.dim}d) · {buildRes.build_ms} ms
                {buildRes.notes.map((nt, i) => <p key={i} className="mt-1 text-amber-400">⚠ {nt}</p>)}
              </div>
            )}
            {indexes.length > 0 && (
              <div>
                <label className="label">Indexes</label>
                <ul className="space-y-1">
                  {indexes.map((ix) => (
                    <li key={ix.id} onClick={() => setActiveIdx(ix.id)}
                      className={`flex cursor-pointer items-center justify-between rounded-lg border p-2 text-xs ${
                        activeIdx === ix.id ? "border-studio-rag bg-bg-panel" : "border-bg-border hover:bg-bg-panel"}`}>
                      <span className="truncate text-slate-200">{ix.name} · {ix.n_chunks} chunks · {ix.embedding_model}</span>
                      <button className="text-slate-500 hover:text-red-400" onClick={(e) => { e.stopPropagation(); ragApi.deleteIndex(ix.id).then(refreshIndexes); }}><Trash2 size={13} /></button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Card>

        {/* ───────── QUERY PIPELINE ───────── */}
        <Card title="Query pipeline" subtitle="How a question is answered against the index.">
          <div className="space-y-3">
            <Stage n={5} title="Query transform">
              <div className="grid grid-cols-2 gap-2">
                <PrettySelect value={transformImpl} onChange={setTransformImpl} block accent="focus-within:border-studio-rag" options={opts(cat?.query_transforms)} />
                <NumField label="Variants (n)" value={transformN} onChange={setTransformN} />
              </div>
            </Stage>

            <Stage n={6} title="Retrieve">
              <PrettySelect value={retriever} onChange={setRetriever} block accent="focus-within:border-studio-rag" options={opts(cat?.retrievers)} />
              <div className="mt-2 grid grid-cols-3 gap-2">
                <NumField label="Top-k" value={k} onChange={setK} />
                <PrettySelect label="Fusion" value={fusion} onChange={setFusion} block accent="focus-within:border-studio-rag"
                  options={[{ value: "rrf", label: "RRF" }, { value: "weighted", label: "Weighted" }]} />
                <NumField label="α dense" value={alpha} onChange={setAlpha} step={0.05} />
              </div>
            </Stage>

            <Stage n={7} title="Rerank">
              <div className="grid grid-cols-2 gap-2">
                <PrettySelect value={rerankImpl} onChange={setRerankImpl} block accent="focus-within:border-studio-rag" options={opts(cat?.rerankers)} />
                <NumField label="Keep top-n" value={rerankTopN} onChange={setRerankTopN} />
              </div>
            </Stage>

            <Stage n={8} title="Compress / Filter">
              <div className="grid grid-cols-2 gap-2">
                <PrettySelect value={compressImpl} onChange={setCompressImpl} block accent="focus-within:border-studio-rag" options={opts(cat?.context_processors)} />
                <NumField label="Target tokens" value={targetTokens} onChange={setTargetTokens} />
              </div>
            </Stage>

            <Stage n={9} title="Generate (LLM choice)">
              <div className="grid grid-cols-2 gap-2">
                <PrettySelect label="Provider" value={genProvider} onChange={setGenProvider} block accent="focus-within:border-studio-rag"
                  options={[{ value: "ollama", label: "Ollama" }, { value: "openai", label: "OpenAI" }]} />
                <div><label className="label">Model</label><input className="input" value={genModel} onChange={(e) => setGenModel(e.target.value)} /></div>
              </div>
              <label className="mt-2 flex items-center gap-2 text-sm text-slate-300">
                <input type="checkbox" checked={citations} onChange={(e) => setCitations(e.target.checked)} /> Require inline citations
              </label>
            </Stage>

            <div>
              <label className="label">Question</label>
              <input className="input" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runQuery()} />
            </div>
            <button className="btn-primary w-full bg-studio-rag" disabled={querying || !activeIdx} onClick={runQuery}>
              {querying ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />} Run query
            </button>
          </div>
        </Card>
      </div>

      {/* ───────── RESULT + TRACE ───────── */}
      {result && (
        <Card className="mt-5" title="Result" subtitle={`${result.retriever} · ${result.latency_ms} ms`}>
          {result.notes.length > 0 && (
            <div className="mb-3 space-y-1">
              {result.notes.map((nt, i) => <p key={i} className="text-xs text-amber-400">⚠ {nt}</p>)}
            </div>
          )}
          {result.answer && (
            <div className="mb-4 rounded-lg border border-bg-border bg-bg-panel p-4">
              <div className="mb-2 flex items-center gap-2">
                <Sparkles size={15} className="text-studio-rag" />
                <span className="text-sm font-semibold text-slate-100">Answer</span>
                <span className="pill">{result.answer_kind}</span>
              </div>
              <p className="whitespace-pre-wrap text-sm text-slate-200">{result.answer}</p>
            </div>
          )}
          {Object.keys(result.trace).length > 0 && (
            <pre className="mb-4 overflow-auto rounded-lg border border-bg-border bg-bg-card p-3 text-[11px] text-slate-400">
{JSON.stringify(result.trace, null, 2)}
            </pre>
          )}
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-200"><Search size={15} /> Retrieved ({result.docs.length})</div>
          <ul className="space-y-2">
            {result.docs.map((d, i) => (
              <li key={i} className="rounded-lg border border-bg-border bg-bg-card p-3">
                <div className="mb-1 text-[11px] text-slate-500">#{i + 1} · <span className="pill">score {d.score}</span> {String(d.metadata?.source ?? "")}</div>
                <p className="text-xs text-slate-300">{d.content}</p>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* ───────── IR (dev face) ───────── */}
      <Card className="mt-5">
        <button className="flex w-full items-center gap-2 text-sm font-semibold text-slate-200" onClick={() => setShowIR((v) => !v)}>
          {showIR ? <ChevronDown size={16} /> : <ChevronRight size={16} />} <Code2 size={15} /> Pipeline IR (config-as-code)
        </button>
        {showIR && (
          <pre className="mt-3 overflow-auto rounded-lg border border-bg-border bg-bg-card p-3 text-[11px] text-slate-300">
{JSON.stringify(ir, null, 2)}
          </pre>
        )}
      </Card>
    </div>
  );
}

function Stage({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-bg-border bg-bg-panel/40 p-3">
      <div className="mb-2 flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-studio-rag/20 text-[10px] font-bold text-studio-rag">{n}</span>
        <span className="text-sm font-semibold text-slate-100">{title}</span>
      </div>
      {children}
    </div>
  );
}

function NumField({ label, value, onChange, step }: { label: string; value: number; onChange: (n: number) => void; step?: number }) {
  return (
    <div>
      <label className="label">{label}</label>
      <input type="number" step={step} className="input" value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

function ParamForm({ spec, values, onChange }: { spec?: Record<string, ParamSpec>; values: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void }) {
  if (!spec) return null;
  const set = (k: string, v: unknown) => onChange({ ...values, [k]: v });
  return (
    <div className="mt-2 grid grid-cols-2 gap-2">
      {Object.entries(spec).map(([key, p]) => {
        const v = values[key] ?? p.default;
        if (p.type === "bool")
          return (
            <label key={key} className="flex items-center gap-2 text-xs text-slate-300">
              <input type="checkbox" checked={Boolean(v)} onChange={(e) => set(key, e.target.checked)} /> {p.label ?? key}
            </label>
          );
        if (p.type === "select")
          return (
            <div key={key}>
              <label className="label">{p.label ?? key}</label>
              <select className="input" value={String(v)} onChange={(e) => set(key, e.target.value)}>
                {(p.options ?? []).map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </div>
          );
        return (
          <div key={key}>
            <label className="label">{p.label ?? key}</label>
            <input className="input" type={p.type === "text" ? "text" : "number"} step={p.step} value={String(v ?? "")}
              onChange={(e) => set(key, p.type === "text" ? e.target.value : Number(e.target.value))} />
          </div>
        );
      })}
    </div>
  );
}
