"use client";

import React, { useState, useEffect } from "react";
import { Shield, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";

export function TopBar() {
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkHealth = async () => {
    setIsRefreshing(true);
    try {
      const res = await api.getHealth();
      setBackendOnline(res.status === "online");
    } catch {
      setBackendOnline(false);
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
    <header className="sticky top-0 z-30 flex h-16 w-full items-center justify-between border-b border-slate-200 bg-white px-6 shadow-sm">
      {/* Brand & Project Identity */}
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-teal-50 border border-teal-200 text-teal-700">
          <Shield className="h-5 w-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-base font-bold text-slate-900 tracking-tight font-mono">
              CORTEX-R
            </h1>
            <span className="hidden sm:inline-block text-[11px] text-slate-400 font-mono">
              |
            </span>
            <span className="text-xs text-slate-600 hidden md:inline font-medium">
              Cybersecurity Orchestration and Response through Transformer-guided EXecution and Reinforcement
            </span>
          </div>
        </div>
      </div>

      {/* System Status Pill */}
      <div className="flex items-center gap-3">
        <div
          className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-mono font-semibold border transition-colors ${
            backendOnline === true
              ? "bg-emerald-50 text-emerald-800 border-emerald-200"
              : backendOnline === false
              ? "bg-rose-50 text-rose-800 border-rose-200"
              : "bg-slate-100 text-slate-600 border-slate-200"
          }`}
        >
          <span
            className={`h-2 w-2 rounded-full ${
              backendOnline === true
                ? "bg-emerald-500 animate-pulse"
                : backendOnline === false
                ? "bg-rose-500"
                : "bg-slate-400"
            }`}
          />
          <span>
            System: {backendOnline === true ? "READY" : backendOnline === false ? "OFFLINE" : "CHECKING..."}
          </span>
        </div>

        <button
          onClick={checkHealth}
          disabled={isRefreshing}
          className="p-1.5 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-500 hover:text-slate-800 border border-slate-200 transition-colors"
          title="Refresh System Status"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
        </button>
      </div>
    </header>
  );
}
