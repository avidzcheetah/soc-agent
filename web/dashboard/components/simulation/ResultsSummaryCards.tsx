"use client";

import React from "react";
import { Download } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { SimulationResults } from "@/types";

interface Props {
  results?: SimulationResults | null;
  isRunning?: boolean;
}

export function ResultsSummaryCards({ results = null, isRunning = false }: Props) {
  const isCompleted = results !== null;

  const handleExportJSON = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `simulation-result-${results.run_id || "run"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const outcome = isRunning
    ? "Running"
    : isCompleted
    ? results.status === "completed"
      ? "Completed"
      : "Stopped"
    : "Standby";

  const steps = isCompleted ? `${results.total_steps}` : isRunning ? "In progress" : "—";

  const reward = isCompleted
    ? results.cumulative_reward > 0
      ? `+${results.cumulative_reward.toFixed(1)}`
      : `${results.cumulative_reward.toFixed(1)}`
    : "—";

  const duration = isCompleted
    ? `${results.execution_duration_sec || 0}s`
    : isRunning
    ? "Active"
    : "—";

  return (
    <Card className="p-4 border-slate-200 bg-white shadow-sm space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
          Current Simulation Results
        </h2>
        {isCompleted && (
          <span className="text-[11px] font-mono text-slate-500">
            Run: {results.run_id}
          </span>
        )}
      </div>

      {/* 5 Compact Metrics: Outcome | Completed Steps | Cumulative Reward | Duration | Export */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 items-center">
        {/* Outcome */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Outcome
          </span>
          <p
            className={`text-sm font-bold font-mono ${
              isCompleted && results.status === "completed"
                ? "text-emerald-700"
                : "text-slate-900"
            }`}
          >
            {outcome}
          </p>
        </div>

        {/* Completed Steps */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Completed Steps
          </span>
          <p className="text-sm font-bold font-mono text-slate-900">{steps}</p>
        </div>

        {/* Cumulative Reward */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Cumulative Reward
          </span>
          <p
            className={`text-sm font-bold font-mono ${
              isCompleted && results.cumulative_reward > 0
                ? "text-emerald-700"
                : "text-slate-900"
            }`}
          >
            {reward}
          </p>
        </div>

        {/* Duration */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Duration
          </span>
          <p className="text-sm font-bold font-mono text-slate-900">{duration}</p>
        </div>

        {/* Export Control */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200 flex flex-col justify-between">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Export Results
          </span>
          <button
            type="button"
            onClick={handleExportJSON}
            disabled={!isCompleted}
            className={`w-full py-1 px-2 rounded text-xs font-medium flex items-center justify-center gap-1.5 transition-colors ${
              isCompleted
                ? "bg-teal-600 hover:bg-teal-700 text-white shadow-sm cursor-pointer"
                : "bg-slate-200 text-slate-400 cursor-not-allowed"
            }`}
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>
    </Card>
  );
}
