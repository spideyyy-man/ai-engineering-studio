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
  accent: string;
}

const NAV: NavItem[] = [
  { to: "/projects", label: "Projects", icon: LayoutGrid, accent: "text-studio-hub" },
  { to: "/data", label: "Data & EDA", icon: Database, accent: "text-studio-data" },
  { to: "/ml", label: "ML Studio", icon: BrainCircuit, accent: "text-studio-ml" },
  { to: "/dl", label: "DL Studio", icon: Network, accent: "text-studio-dl" },
  { to: "/rag", label: "RAG Studio", icon: BookOpen, accent: "text-studio-rag" },
  { to: "/agents", label: "AI Agents", icon: Bot, accent: "text-studio-agent" },
  { to: "/deploy", label: "Deployment", icon: Rocket, accent: "text-studio-deploy" },
  { to: "/settings", label: "Settings", icon: SettingsIcon, accent: "text-studio-settings" },
];

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
          "relative z-20 flex flex-col border-r border-bg-border bg-bg-panel shadow-xl transition-[width] duration-200 ease-out",
          expanded ? "w-60" : "w-[64px]"
        )}
      >
        {/* Logo + collapse pin */}
        <div className="flex items-center justify-between gap-2 border-b border-bg-border px-3 py-3">
          <button
            className="flex items-center gap-2 overflow-hidden text-left"
            onClick={() => navigate("/projects")}
          >
            <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-gradient-to-br from-studio-data to-studio-dl text-sm font-bold">
              AI
            </span>
            <div
              className={clsx(
                "leading-tight transition-all",
                expanded ? "opacity-100" : "w-0 opacity-0"
              )}
            >
              <div className="whitespace-nowrap text-sm font-semibold">AI Engineering</div>
              <div className="whitespace-nowrap text-xs text-slate-400">Studio · v0.1</div>
            </div>
          </button>
          <button
            className={clsx(
              "rounded-md p-1.5 text-slate-400 hover:bg-bg-card hover:text-slate-100",
              !expanded && "absolute right-2 top-3"
            )}
            title={pinned ? "Collapse sidebar" : "Pin sidebar open"}
            onClick={() => setPinned((p) => !p)}
          >
            {pinned ? <PinOff size={15} /> : <Pin size={15} />}
          </button>
        </div>

        {/* Project switcher (expanded only) */}
        <div
          className={clsx(
            "overflow-hidden border-b border-bg-border transition-all",
            expanded ? "max-h-40 px-4 py-3 opacity-100" : "max-h-0 opacity-0"
          )}
        >
          <ProjectSwitcher />
        </div>

        {/* Nav */}
        <nav className="flex-1 space-y-0.5 overflow-x-hidden overflow-y-auto px-2 py-2">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              title={item.label}
              className={({ isActive }) =>
                clsx(
                  "group flex items-center gap-3 rounded-lg px-2.5 py-2 text-sm font-medium transition",
                  isActive
                    ? "bg-bg-card text-white"
                    : "text-slate-400 hover:bg-bg-card hover:text-slate-100"
                )
              }
            >
              <item.icon size={18} className={clsx("shrink-0", item.accent)} />
              <span
                className={clsx(
                  "whitespace-nowrap transition-all",
                  expanded ? "opacity-100" : "w-0 opacity-0"
                )}
              >
                {item.label}
              </span>
            </NavLink>
          ))}
        </nav>

        <div
          className={clsx(
            "overflow-hidden border-t border-bg-border text-xs text-slate-500 transition-all",
            expanded ? "px-5 py-3 opacity-100" : "max-h-0 opacity-0"
          )}
        >
          {pinned ? "Pinned · local mode" : "Hover to expand · click pin to lock"}
        </div>

        {/* Floating menu icon when collapsed — also serves as drag-target hint */}
        {!pinned && !hovering && (
          <Menu
            size={14}
            className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 text-slate-500"
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
