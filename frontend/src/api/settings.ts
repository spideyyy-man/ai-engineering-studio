import { api } from "./client";
import type { AppSettings, ProviderInfo } from "./types";

export const settingsApi = {
  get: () => api.get<AppSettings>("/settings").then((r) => r.data),
  providers: () => api.get<ProviderInfo[]>("/settings/providers").then((r) => r.data),
  testLLM: (provider?: string) =>
    api
      .post<{ provider: string; ok: boolean }>("/settings/test-llm", null, {
        params: provider ? { provider } : {},
      })
      .then((r) => r.data),
};
