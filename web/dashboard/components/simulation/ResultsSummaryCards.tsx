"use client";

import React from "react";
import {
  Award,
  Hash,
  Activity,
  ShieldCheck,
  Timer,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { SimulationResults } from "@/types";

interface Props {
  results?: SimulationResults | null;
  isRunning?: boolean;
}

export function ResultsSummaryCards({ results = null, isRunning = false }: Props) {
  const isCompleted = results !== null;

  const resultCards = [
    {
      label: "Simulation Outcome",
      value: isRunning
        ? "Replaying..."
        : isCompleted
        ? results.status === "completed"
          ? "Historical Replay Complete"
          : "Replay Stopped"
        : "Not Evaluated",
      desc: isCompleted
        ? `Action Match: ${(results.accuracy * 100).toFixed(1)}%`
        : "Status after episode termination",
      icon: Award,
      highlight: isCompleted && results.status === "completed",
    },
    {
      label: "Total Steps",
      value: isCompleted ? `${results.total_steps}` : "—",
      desc: isCompleted
        ? `${results.correct_steps} matching ground-truth`
        : "Environment interaction steps",
      icon: Hash,
      highlight: false,
    },
    {
      label: "Cumulative Return",
      value: isCompleted
        ? results.cumulative_reward > 0
          ? `+${results.cumulative_reward}`
          : `${results.cumulative_reward}`
        : "—",
      desc: isCompleted
        ? `Mean reward: ${results.mean_reward}`
        : "Sum of episode feedback (Σ r_t)",
      icon: Activity,
      highlight: isCompleted && results.cumulative_reward > 0,
    },
    {
      label: "Containment Outcome",
      value: isCompleted
        ? results.containment_outcome || "Not Available"
        : "—",
      desc: isCompleted
        ? "Bandit evaluates per-incident actions, not host containment"
        : "Threat mitigation effectiveness",
      icon: ShieldCheck,
      highlight: false,
    },
    {
      label: "Execution Duration",
      value: isCompleted
        ? `${results.execution_duration_sec}s`
        : isRunning
        ? "Active"
        : "—",
      desc: isCompleted
        ? `Paced SSE stream elapsed time`
        : "Inference & step elapsed time",
      icon: Timer,
      highlight: false,
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <Award className="h-3.5 w-3.5 text-teal-400" />
          <span>Simulation Results Summary</span>
        </h2>
        {isCompleted ? (
          <span className="text-[11px] text-emerald-400 font-mono flex items-center gap-1">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
            <span>Evaluation Metrics Verified (Run: {results.run_id})</span>
          </span>
        ) : (
          <span className="text-[11px] text-slate-400 font-mono flex items-center gap-1">
            <AlertCircle className="h-3 w-3 text-slate-400" />
            <span>Results will update upon simulation completion</span>
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
        {resultCards.map((c) => {
          const Icon = c.icon;
          return (
            <Card
              key={c.label}
              className={`p-3.5 border-slate-800/80 bg-slate-900/60 flex flex-col justify-between transition-colors ${
                c.highlight ? "border-teal-500/40 bg-teal-500/5" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                  {c.label}
                </span>
                <Icon
                  className={`h-3.5 w-3.5 ${
                    c.highlight ? "text-teal-400" : "text-slate-400"
                  }`}
                />
              </div>
              <div className="my-2">
                <p
                  className={`text-base font-bold font-mono ${
                    c.highlight ? "text-teal-300" : "text-slate-100"
                  }`}
                >
                  {c.value}
                </p>
              </div>
              <p className="text-[10px] text-slate-400 leading-tight">{c.desc}</p>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
