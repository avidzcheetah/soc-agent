"use client";

import React, { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import {
  RefreshCw,
  Server,
  Activity,
  Layers,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { api } from "@/lib/api";

const ROUTE_INFO: Record<string, { title: string; subtitle: string }> = {
  "/": {
    title: "Simulation Control Workspace",
    subtitle: "Interactive parameters, architecture flow, and decision monitoring",
  },
  "/overview": {
    title: "Research & Project Overview",
    subtitle: "High-level summary of SecBERT representation learning and PPO policy",
  },
  "/architecture": {
    title: "System Architecture Flow",
    subtitle: "Complete 8-stage interactive execution pipeline (React Flow)",
  },
  "/results": {
    title: "Results & Empirical Evaluation",
    subtitle: "Phase 1 ablation benchmarks, Top-2 accuracy, and PPO milestone audit",
  },
  "/system-health": {
    title: "System & Service Health",
    subtitle: "Technical monitoring of backend daemons, datasets, and model checkpoints",
  },
};

export function TopBar() {
  const pathname = usePathname();
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastChecked, setLastChecked] = useState<string>("Checking...");

  const currentRouteKey =
    Object.keys(ROUTE_INFO).find(
      (key) => pathname === key || (key !== "/" && pathname.startsWith(key))
    ) || "/";

  const { title, subtitle } = ROUTE_INFO[currentRouteKey] || {
    title: "SOC Agent",
    subtitle: "Transformer-Based Incident Response Research",
  };

  const checkHealth = async () => {
    setIsRefreshing(true);
    try {
      const res = await api.getHealth();
      setBackendOnline(res.status === "online");
      setLastChecked(new Date().toLocaleTimeString());
    } catch {
      setBackendOnline(false);
      setLastChecked(new Date().toLocaleTimeString());
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-800 bg-[#060a14]/90 px-6 backdrop-blur">
      {/* Title & Subtitle */}
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
          <span className="hidden md:inline-block px-1.5 py-0.5 rounded text-[10px] font-mono text-slate-400 bg-slate-800/80 border border-slate-700/60">
            SOC Agent
          </span>
        </div>
        <p className="text-[11px] text-slate-400 hidden sm:block">{subtitle}</p>
      </div>

      {/* Status Indicators */}
      <div className="flex items-center gap-3">
        {/* Simulation Status Pill */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px]">
          <Activity className="h-3.5 w-3.5 text-slate-400" />
          <span className="text-slate-400">Simulation:</span>
          <span className="font-mono font-medium text-amber-300">Idle / Standby</span>
        </div>

        {/* Backend Connection Indicator */}
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[11px] font-mono transition-colors ${
            backendOnline === true
              ? "bg-teal-500/10 text-teal-300 border-teal-500/30"
              : "bg-rose-500/10 text-rose-300 border-rose-500/30"
          }`}
          title={
            backendOnline === true
              ? "Python FastAPI backend verified online on port 8000"
              : "Backend not connected. Integration pending in Step 2."
          }
        >
          <span
            className={`h-2 w-2 rounded-full ${
              backendOnline === true ? "bg-teal-400 animate-pulse" : "bg-rose-400"
            }`}
          />
          <Server className="h-3 w-3" />
          <span className="hidden sm:inline">Backend:</span>
          <span className="font-bold">
            {backendOnline === true
              ? "Connected"
              : backendOnline === false
              ? "Not Connected"
              : "Checking..."}
          </span>
        </div>

        {/* Refresh Button */}
        <button
          onClick={checkHealth}
          disabled={isRefreshing}
          className="p-1.5 rounded-md bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
          title={`Refresh status (Last checked: ${lastChecked})`}
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
        </button>
      </div>
    </header>
  );
}
