import { useState } from "react";
import { Plus } from "lucide-react";
import { useProjectStore } from "@/store/projectStore";

export default function ProjectSwitcher() {
  const projects = useProjectStore((s) => s.projects);
  const currentId = useProjectStore((s) => s.currentId);
  const setCurrent = useProjectStore((s) => s.setCurrent);
  const create = useProjectStore((s) => s.create);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");

  return (
    <div>
      <label className="label">Current project</label>
      <select
        className="input"
        value={currentId ?? ""}
        onChange={(e) => setCurrent(e.target.value || null)}
      >
        {projects.length === 0 && <option value="">No projects</option>}
        {projects.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>

      {creating ? (
        <div className="mt-2 flex gap-2">
          <input
            autoFocus
            className="input"
            placeholder="Project name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && name.trim()) {
                void create(name.trim()).finally(() => {
                  setCreating(false);
                  setName("");
                });
              }
              if (e.key === "Escape") {
                setCreating(false);
                setName("");
              }
            }}
          />
        </div>
      ) : (
        <button
          className="mt-2 flex w-full items-center justify-center gap-1 rounded-lg border border-dashed border-bg-border py-1.5 text-xs text-slate-400 hover:border-slate-500 hover:text-slate-200"
          onClick={() => setCreating(true)}
        >
          <Plus size={14} /> New project
        </button>
      )}
    </div>
  );
}
