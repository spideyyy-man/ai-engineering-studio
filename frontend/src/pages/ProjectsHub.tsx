import { useState } from "react";
import { FolderPlus, Trash2, Folder } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useProjectStore } from "@/store/projectStore";
import PageHeader from "@/components/PageHeader";
import Card from "@/components/Card";
import EmptyState from "@/components/EmptyState";
import { shortId } from "@/lib/format";

export default function ProjectsHub() {
  const projects = useProjectStore((s) => s.projects);
  const loading = useProjectStore((s) => s.loading);
  const create = useProjectStore((s) => s.create);
  const remove = useProjectStore((s) => s.remove);
  const setCurrent = useProjectStore((s) => s.setCurrent);
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");

  async function onCreate() {
    if (!name.trim()) return;
    await create(name.trim(), desc.trim());
    setName("");
    setDesc("");
  }

  return (
    <div className="mx-auto max-w-5xl px-8 py-8">
      <PageHeader
        title="Projects"
        subtitle="Each project bundles its datasets, models, RAG indexes, agents, and deployments."
        accent="text-studio-hub"
      />

      <Card title="Create a project" className="mb-6">
        <div className="grid gap-3 sm:grid-cols-[1fr_2fr_auto]">
          <input
            className="input"
            placeholder="Name (e.g., Churn Prediction)"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <input
            className="input"
            placeholder="Description (optional)"
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
          />
          <button
            className="btn-primary bg-slate-100 hover:bg-white"
            onClick={onCreate}
            disabled={!name.trim() || loading}
          >
            <FolderPlus size={16} /> Create
          </button>
        </div>
      </Card>

      {projects.length === 0 ? (
        <EmptyState
          icon={<Folder size={32} />}
          title="No projects yet"
          description="Create one above to start uploading data and building models."
        />
      ) : (
        <ul className="grid gap-3 sm:grid-cols-2">
          {projects.map((p) => (
            <li key={p.id} className="card flex flex-col">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <button
                    className="truncate text-left text-base font-semibold text-slate-100 hover:underline"
                    onClick={() => {
                      setCurrent(p.id);
                      navigate("/data");
                    }}
                  >
                    {p.name}
                  </button>
                  <p className="mt-0.5 text-xs text-slate-500">{shortId(p.id)}</p>
                </div>
                <button
                  className="rounded p-1 text-slate-500 hover:bg-bg-panel hover:text-red-400"
                  onClick={() => {
                    if (confirm(`Delete "${p.name}" and all its artifacts?`)) {
                      void remove(p.id);
                    }
                  }}
                  title="Delete"
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <p className="mt-2 line-clamp-2 text-sm text-slate-400">
                {p.description || "No description."}
              </p>
              <div className="mt-3 text-xs text-slate-500">
                Created {new Date(p.created_at).toLocaleString()}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
