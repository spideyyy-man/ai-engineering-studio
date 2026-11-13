import { Routes, Route, NavLink, Navigate, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import clsx from "clsx";
import {
  Database,
  BrainCircuit,
  Network,
  BookOpen,
  Bot,
  Rocket,
  LayoutGrid,
  Settings as SettingsIcon,
  Menu,
  PinOff,
  Pin,
} from "lucide-react";

import ProjectsHub from "@/pages/ProjectsHub";
import DataStudio from "@/pages/DataStudio";
import MLStudio from "@/pages/MLStudio";
import DLStudio from "@/pages/DLStudio";
import RAGStudio from "@/pages/RAGStudio";
import AgentStudio from "@/pages/AgentStudio";
import DeploymentStudio from "@/pages/DeploymentStudio";
import SettingsPage from "@/pages/Settings";

import { useProjectStore } from "@/store/projectStore";
import ProjectSwitcher from "@/components/ProjectSwitcher";

interface NavItem {
  to: string;
  label: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement> & { size?: number | string }>;
  /** Text colour class for the active icon. */
  accent: string;
  /** Background colour class for the workflow tick. */
  tick?: string;
  planned?: boolean;
}

// The workflow studios, in pipeline order. Their colours walk the viridis
// ramp, so the sidebar reads like a colourbar from data to deployment.
const WORKFLOW: NavItem[] = [
  { to: "/data", label: "Data & EDA", icon: Database, accent: "text-studio-data", tick: "bg-studio-data" },
  { to: "/ml", label: "ML Studio", icon: BrainCircuit, accent: "text-studio-ml", tick: "bg-studio-ml" },
  { to: "/dl", label: "DL Studio", icon: Network, accent: "text-studio-dl", tick: "bg-studio-dl" },
  { to: "/rag", label: "RAG Studio", icon: BookOpen, accent: "text-studio-rag", tick: "bg-studio-rag" },
  { to: "/agents", label: "AI Agents", icon: Bot, accent: "text-studio-agent", tick: "bg-studio-agent", planned: true },
  { to: "/deploy", label: "Deployment", icon: Rocket, accent: "text-studio-deploy", tick: "bg-studio-deploy", planned: true },
];

const PROJECTS: NavItem = { to: "/projects", label: "Projects", icon: LayoutGrid, accent: "text-slate-100" };
const SETTINGS: NavItem = { to: "/settings", label: "Settings", icon: SettingsIcon, accent: "text-slate-100" };

/** The brand mark: a tiny viridis colourbar, one segment per workflow studio. */
function ColourbarMark() {
  return (
    <span aria-hidden className="flex h-8 w-2 shrink-0 flex-col overflow-hidden rounded-sm">
      {WORKFLOW.map((s) => (
        <span key={s.to} className={clsx("flex-1", s.tick)} />
      ))}
    </span>
  );
}

function NavRow({ item, expanded }: { item: NavItem; expanded: boolean }) {
  return (
    <NavLink
      to={item.to}
      title={item.planned ? `${item.label} (planned)` : item.label}
      className={({ isActive }) =>
        clsx(
          "group relative flex items-center gap-3 rounded-lg py-2 pl-3 pr-2.5 text-sm transition-colors",
          isActive
            ? "bg-bg-card font-medium text-slate-50"
            : "text-slate-400 hover:bg-bg-card/60 hover:text-slate-100"
        )
      }
    >
      {({ isActive }) => (
        <>
          {item.tick && (
            <span
              aria-hidden
              className={clsx(
                "absolute left-0 top-1/2 w-[3px] -translate-y-1/2 rounded-full transition-all",
                item.tick,
                isActive ? "h-5 opacity-100" : "h-2.5 opacity-50 group-hover:opacity-80"
              )}
            />
          )}
          <item.icon size={17} className={clsx("shrink-0", isActive ? item.accent : "text-slate-500 group-hover:text-slate-300")} />
          <span
            className={clsx(
              "flex flex-1 items-center justify-between gap-2 whitespace-nowrap transition-opacity",
              expanded ? "opacity-100" : "w-0 opacity-0"
            )}
          >
            {item.label}
            {item.planned && <span className="text-[11px] font-normal text-slate-500">Planned</span>}
          </span>
        </>
      )}
    </NavLink>
  );
}

export default function App() {
  const fetchAll = useProjectStore((s) => s.fetchAll);
  const navigate = useNavigate();

  // Sidebar mode: "pinned" (always open) or "collapsed" (icons-only; hover to peek).
  const [pinned, setPinned] = useState<boolean>(() => {
    if (typeof window === "undefined") return true;
    return localStorage.getItem("aies.sidebar.pinned") !== "0";
  });
  const [hovering, setHovering] = useState(false);
  const expanded = pinned || hovering;

  useEffect(() => {
    void fetchAll();
  }, [fetchAll]);

  useEffect(() => {
    localStorage.setItem("aies.sidebar.pinned", pinned ? "1" : "0");
  }, [pinned]);

  return (
    <div className="flex h-screen overflow-hidden bg-bg text-slate-100">
      {/* Sidebar */}
      <aside
        onMouseEnter={() => !pinned && setHovering(true)}
        onMouseLeave={() => !pinned && setHovering(false)}
        className={clsx(
          "relative z-20 flex flex-col border-r border-bg-border bg-bg-panel transition-[width] duration-200 ease-out",
          expanded ? "w-60" : "w-[64px]"
        )}
      >
        {/* Brand + pin */}
        <div className="flex items-center justify-between gap-2 px-4 pb-3 pt-4">
          <button
            className="flex items-center gap-3 overflow-hidden rounded-md text-left"
            onClick={() => navigate("/projects")}
            title="AI Engineering Studio"
          >
            <ColourbarMark />
            <span
              className={clsx(
                "leading-tight transition-opacity",
                expanded ? "opacity-100" : "w-0 opacity-0"
              )}
            >
              <span className="block whitespace-nowrap text-sm font-semibold text-slate-50">AI Engineering</span>
              <span className="block whitespace-nowrap text-sm text-slate-400">Studio</span>
            </span>
          </button>
          <button
            className={clsx(
              "rounded-md p-1.5 text-slate-500 transition-colors hover:bg-bg-card hover:text-slate-100",
              !expanded && "absolute right-2 top-4"
            )}
            title={pinned ? "Collapse sidebar" : "Keep sidebar open"}
            aria-label={pinned ? "Collapse sidebar" : "Keep sidebar open"}
            onClick={() => setPinned((p) => !p)}
          >
            {pinned ? <PinOff size={15} /> : <Pin size={15} />}
          </button>
        </div>

        {/* Project switcher (expanded only) */}
        <div
          className={clsx(
            "overflow-hidden transition-all",
            expanded ? "max-h-40 px-4 pb-3 opacity-100" : "max-h-0 opacity-0"
          )}
        >
          <ProjectSwitcher />
        </div>

        {/* Nav */}
        <nav className="flex flex-1 flex-col overflow-x-hidden overflow-y-auto border-t border-bg-border px-2 py-3">
          <NavRow item={PROJECTS} expanded={expanded} />
          <div className="mx-3 my-2 border-t border-bg-border" />
          <div className="space-y-0.5">
            {WORKFLOW.map((item) => (
              <NavRow key={item.to} item={item} expanded={expanded} />
            ))}
          </div>
          <div className="mt-auto pt-3">
            <NavRow item={SETTINGS} expanded={expanded} />
          </div>
        </nav>

        {/* Collapsed hint */}
        {!pinned && !hovering && (
          <Menu
            size={14}
            className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 text-slate-600"
          />
        )}
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<ProjectsHub />} />
          <Route path="/data" element={<DataStudio />} />
          <Route path="/ml" element={<MLStudio />} />
          <Route path="/dl" element={<DLStudio />} />
          <Route path="/rag" element={<RAGStudio />} />
          <Route path="/agents" element={<AgentStudio />} />
          <Route path="/deploy" element={<DeploymentStudio />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/projects" replace />} />
        </Routes>
      </main>
    </div>
  );
}
