import { create } from "zustand";
import type { Project } from "@/api/types";
import { projectsApi } from "@/api";

interface ProjectState {
  projects: Project[];
  currentId: string | null;
  loading: boolean;
  error: string | null;

  fetchAll: () => Promise<void>;
  create: (name: string, description?: string) => Promise<Project>;
  setCurrent: (id: string | null) => void;
  remove: (id: string) => Promise<void>;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  projects: [],
  currentId: null,
  loading: false,
  error: null,

  fetchAll: async () => {
    set({ loading: true, error: null });
    try {
      const projects = await projectsApi.list();
      set({
        projects,
        loading: false,
        currentId: get().currentId ?? projects[0]?.id ?? null,
      });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  create: async (name, description = "") => {
    const p = await projectsApi.create(name, description);
    set((s) => ({ projects: [p, ...s.projects], currentId: p.id }));
    return p;
  },

  setCurrent: (id) => set({ currentId: id }),

  remove: async (id) => {
    await projectsApi.remove(id);
    set((s) => ({
      projects: s.projects.filter((p) => p.id !== id),
      currentId: s.currentId === id ? null : s.currentId,
    }));
  },
}));
