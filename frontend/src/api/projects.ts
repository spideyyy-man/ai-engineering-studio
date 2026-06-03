import { api } from "./client";
import type { Project } from "./types";

export const projectsApi = {
  list: () => api.get<Project[]>("/projects").then((r) => r.data),
  create: (name: string, description = "") =>
    api.post<Project>("/projects", { name, description }).then((r) => r.data),
  get: (id: string) => api.get<Project>(`/projects/${id}`).then((r) => r.data),
  remove: (id: string) => api.delete(`/projects/${id}`).then((r) => r.data),
};
