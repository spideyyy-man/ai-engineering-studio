import { useEffect, useState } from "react";
import StubPage from "./_StubPage";
import Card from "@/components/Card";
import { api } from "@/api/client";

interface Catalog {
  targets: { key: string; label: string; description: string }[];
}

export default function DeploymentStudio() {
  const [cat, setCat] = useState<Catalog | null>(null);

  useEffect(() => {
    api.get<Catalog>("/deploy/catalog").then((r) => setCat(r.data)).catch(() => setCat(null));
  }, []);

  return (
    <StubPage
      title="Deployment & MLOps Studio"
      subtitle="Generate FastAPI / Docker / Streamlit / Dash artifacts. Experiment registry, drift detection, monitoring."
      accent="text-studio-deploy"
      phase="Phase 5"
      capabilities={[
        "GET /api/v1/deploy/catalog — supported deployment targets",
      ]}
      comingSoon={[
        "FastAPI /predict generation per trained model",
        "Dockerfile + requirements + docker-compose",
        "Lightweight model registry & versioning",
        "Drift detection (KS test, PSI) on production inputs",
      ]}
      extra={
        cat && (
          <Card title="Deployment targets" className="mt-6">
            <ul className="grid gap-2 sm:grid-cols-2">
              {cat.targets.map((t) => (
                <li key={t.key} className="rounded-lg border border-bg-border bg-bg-panel p-3">
                  <div className="font-medium text-slate-100">{t.label}</div>
                  <div className="mt-0.5 text-xs text-slate-500">{t.description}</div>
                </li>
              ))}
            </ul>
          </Card>
        )
      }
    />
  );
}
