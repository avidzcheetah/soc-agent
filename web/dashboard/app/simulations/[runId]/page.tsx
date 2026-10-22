"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  PlaySquare,
  Clock,
  Layers,
  Award,
  Download,
  FileSpreadsheet,
  CheckCircle2,
  XCircle,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  ExternalLink,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { SimulationRewardChart } from "@/components/charts/SimulationRewardChart";
import { api } from "@/lib/api";
import { SimulationRun, SimulationStep } from "@/types";
import { formatDate } from "@/lib/utils";

export default function SimulationDetailPage() {
  const params = useParams();
  const runId = (params?.runId as string) || "";

  const [loading, setLoading] = useState(true);
  const [run, setRun] = useState<SimulationRun | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<Record<number, boolean>>({});

  useEffect(() => {
    async function loadDetail() {
      try {
        const data = await api.getSimulationDetail(runId);
        setRun(data);
      } catch (e) {
        console.error("Failed to load simulation detail", e);
      } finally {
        setLoading(false);
      }
    }
    if (runId) loadDetail();
  }, [runId]);

  const toggleStep = (stepNum: number) => {
    setExpandedSteps((prev) => ({
      ...prev,
      [stepNum]: !prev[stepNum],
    }));
  };


  if (loading) {
    return <LoadingState message={`Loading simulation trajectory ${runId}...`} />;
  }

  if (!run) {
    return (
      <div className="space-y-4">
        <Link
          href="/results"
          className="inline-flex items-center gap-1.5 text-xs text-teal-400 hover:text-teal-300 font-medium"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to Results & Simulations
        </Link>
        <Card className="text-center py-12">
          <p className="text-sm font-semibold text-slate-200">Simulation Run Not Found</p>
          <p className="text-xs text-slate-400 mt-1">ID: {runId}</p>
        </Card>
      </div>
    );
  }

  const exportJsonUrl = api.getSimulationExportUrl(runId, "json");
  const exportCsvUrl = api.getSimulationExportUrl(runId, "csv");

  return (
    <div className="space-y-6">
      {/* Back Link & Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            href="/results"
            className="p-2 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-800 transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold text-slate-100 font-mono">{run.id}</h1>
              <StatusBadge status={run.status} />
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700/60">
                {run.simulation_mode || "Historical Replay"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">{run.scenario}</p>
          </div>
        </div>

        {/* Export Controls */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <a
            href={exportJsonUrl}
            download={`${run.id}.json`}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs border border-slate-800 flex items-center gap-1.5 transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export JSON</span>
          </a>
          <a
            href={exportCsvUrl}
            download={`${run.id}_steps.csv`}
            className="px-3 py-1.5 rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-300 text-xs border border-teal-500/30 flex items-center gap-1.5 transition-colors font-medium"
          >
            <FileSpreadsheet className="h-3.5 w-3.5" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* Configuration & Parameter Metadata */}
      <Card className="p-4 bg-slate-900/60 border-slate-800">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
          <div>
            <span className="text-[10px] uppercase text-slate-400 block">Policy & Checkpoint</span>
            <span className="text-slate-200 font-semibold">{run.policy}</span>
            <span className="text-[11px] text-slate-400 block truncate">
              {run.checkpoint || (run.policy === "PPO" ? "v1.0-ppo-core (uncommitted)" : "EXP_PPO_ATGP_DEV (protocol)")}
            </span>
          </div>
          <div>
            <span className="text-[10px] uppercase text-slate-400 block">Encoder Representation</span>
            <span className="text-teal-400 font-semibold truncate block">
              {run.encoder || "SecBERT FP32 (EXP_20260727_001)"}
            </span>
            <span className="text-[11px] text-slate-400 block">768-dimensional frozen</span>
          </div>
          <div>
            <span className="text-[10px] uppercase text-slate-400 block">Random Seed / Target</span>
            <span className="text-slate-200">Seed: {run.seed !== undefined ? run.seed : 42}</span>
            <span className="text-[11px] text-slate-400 block truncate">Target: {run.target || "enterprise-lan"}</span>
          </div>
          <div>
            <span className="text-[10px] uppercase text-slate-400 block">Execution Started</span>
            <span className="text-slate-200 truncate block">{formatDate(run.start_time)}</span>
            <span className="text-[11px] text-slate-400 block">
              Steps: {run.steps_count} / {run.max_steps || run.steps_count}
            </span>
          </div>
        </div>
      </Card>

      {/* Operational Boundary Notice */}
      <div className="flex items-start gap-3 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-300/90 font-mono">
        <AlertCircle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
        <div>
          <span className="font-semibold text-amber-200">Operational Boundary Notice: </span>
          Host containment outcome, containment latency, and operational disruption are{" "}
          <strong className="underline">Not Available</strong> in this replay simulation. Sequential step rewards (+1.0 / -1.0) reflect policy alignment with ground-truth analyst actions.
        </div>
      </div>

      {/* Run Summary Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
        <Card className="text-center p-4">
          <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Total Steps</p>
          <p className="text-xl font-bold font-mono text-slate-100 mt-1">{run.steps_count}</p>
        </Card>
        <Card className="text-center p-4">
          <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Accuracy</p>
          <p className="text-xl font-bold font-mono text-teal-400 mt-1">
            {(run.accuracy * 100).toFixed(1)}%
          </p>
        </Card>
        <Card className="text-center p-4">
          <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Cum. Reward</p>
          <p className="text-xl font-bold font-mono text-sky-400 mt-1">
            {run.final_cumulative_reward > 0 ? `+${run.final_cumulative_reward}` : run.final_cumulative_reward}
          </p>
        </Card>
        <Card className="text-center p-4">
          <p className="text-[10px] text-slate-400 uppercase tracking-wider font-mono">Mean Step Reward</p>
          <p className="text-xl font-bold font-mono text-slate-200 mt-1">
            {run.mean_reward > 0 ? `+${run.mean_reward.toFixed(3)}` : run.mean_reward.toFixed(3)}
          </p>
        </Card>
      </div>

      {/* Cumulative Reward Chart */}
      {run.steps && run.steps.length > 0 && (
        <Card>
          <CardHeader
            title="Cumulative Reward Trajectory"
            subtitle="Progression of agent reward over sequential incident response steps"
          />
          <SimulationRewardChart steps={run.steps} />
        </Card>
      )}

      {/* Step-by-Step Chronological Timeline */}
      <Card>
        <CardHeader
          title="Sequential Step Timeline"
          subtitle={`Inspect each incident observation, policy decision, and software state change`}
        />

        <div className="space-y-3">
          {run.steps?.map((step) => {
            const isExpanded = !!expandedSteps[step.step];

            return (
              <div
                key={step.step}
                className="rounded-lg border border-slate-800 bg-slate-900/50 p-3.5 transition-colors hover:border-slate-700"
              >
                <div
                  onClick={() => toggleStep(step.step)}
                  className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 cursor-pointer select-none"
                >
                  <div className="flex items-center gap-3">
                    <span className="flex items-center justify-center h-6 w-6 rounded-full bg-slate-800 text-xs font-mono font-bold text-slate-300">
                      {step.step}
                    </span>
                    <div className="flex items-center gap-1.5">
                      {step.is_correct ? (
                        <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0" />
                      ) : (
                        <XCircle className="h-4 w-4 text-rose-400 shrink-0" />
                      )}
                      <span className="font-mono text-xs font-semibold text-slate-200">
                        {step.action_name} ({step.chosen_action})
                      </span>
                      {!step.is_correct && (
                        <span className="text-[11px] text-slate-400 font-mono">
                          (Target: {step.ground_truth_name})
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span
                      className={`font-bold px-2 py-0.5 rounded ${
                        step.reward > 0 ? "bg-teal-500/10 text-teal-300" : "bg-rose-500/10 text-rose-300"
                      }`}
                    >
                      {step.reward > 0 ? "+1.0" : "-1.0"}
                    </span>
                    <span className="text-slate-400">
                      Cum: <strong className="text-sky-400">{step.cumulative_reward}</strong>
                    </span>
                    {isExpanded ? (
                      <ChevronUp className="h-4 w-4 text-slate-400" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-slate-400" />
                    )}
                  </div>
                </div>

                {/* Expanded Step Details */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-2 text-xs">
                    <div>
                      <span className="text-[10px] uppercase font-mono text-slate-400 block">Incident Alert:</span>
                      <p className="text-slate-300 font-mono mt-0.5 bg-slate-950 p-2.5 rounded border border-slate-800">
                        {step.incident_summary}
                      </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                      <div>
                        <span className="text-[10px] uppercase font-mono text-slate-400 block">Simulated Environment State:</span>
                        <p className="text-slate-300 font-mono mt-0.5">{step.simulated_state}</p>
                      </div>
                      <div>
                        <span className="text-[10px] uppercase font-mono text-slate-400 block">Critic Estimated Value V(s):</span>
                        <p className="text-sky-400 font-mono mt-0.5">{step.critic_value.toFixed(3)}</p>
                      </div>
                    </div>

                    <div className="pt-2 flex justify-end">
                      <Link
                        href={`/incidents/${step.incident_id}`}
                        className="text-xs text-teal-400 hover:text-teal-300 inline-flex items-center gap-1 font-mono"
                      >
                        <span>View Raw Incident ({step.incident_id})</span>
                        <ExternalLink className="h-3 w-3" />
                      </Link>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
