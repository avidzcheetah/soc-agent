"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  PlaySquare,
  Clock,
  Layers,
  Award,
  ChevronRight,
  ExternalLink,
  ShieldAlert,
  Play,
  RotateCcw,
  CheckCircle2,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { EmptyState } from "@/components/ui/EmptyState";
import { api } from "@/lib/api";
import { SimulationRun } from "@/types";
import { formatDate } from "@/lib/utils";

export default function SimulationsListPage() {
  const [loading, setLoading] = useState(true);
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);
  const [execMessage, setExecMessage] = useState<string | null>(null);

  const loadSimulations = async () => {
    try {
      const res = await api.getSimulations();
      setRuns(res.simulations);
    } catch (e) {
      console.error("Failed to load simulation runs", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSimulations();
  }, []);

  const handleRunSimulation = async () => {
    setIsExecuting(true);
    setExecMessage("Executing safe contextual bandit rollout across benchmark incidents...");
    try {
      const newRun = await api.runSimulation("Interactive-Benchmark-Rollout", 32);
      setExecMessage(`Successfully executed simulation ${newRun.id} (32 steps)!`);
      await loadSimulations();
    } catch (e) {
      console.error("Simulation run error", e);
      setExecMessage("Simulation execution failed or API offline.");
    } finally {
      setIsExecuting(false);
      setTimeout(() => setExecMessage(null), 5000);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100">SOC Simulation Run Explorer</h1>
          <p className="text-xs text-slate-400 mt-1">
            Explore sequential rollout trajectories, state transitions, and software-simulated mitigation responses.
          </p>
        </div>

        {/* Safe Simulation Run Trigger */}
        <button
          onClick={handleRunSimulation}
          disabled={isExecuting}
          className="px-3.5 py-2 rounded-lg bg-teal-500/15 hover:bg-teal-500/25 text-teal-300 text-xs font-semibold border border-teal-500/40 transition-colors flex items-center gap-2 self-start sm:self-auto disabled:opacity-50"
        >
          <Play className={`h-3.5 w-3.5 ${isExecuting ? "animate-spin" : ""}`} />
          <span>{isExecuting ? "Running Simulation..." : "Execute Benchmark Simulation"}</span>
        </button>
      </div>

      {execMessage && (
        <div className="p-3 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-300 text-xs flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>{execMessage}</span>
        </div>
      )}

      {/* Safety Boundary Notice (Mandatory Rule) */}
      <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/60 text-xs text-slate-300 flex items-start gap-3">
        <ShieldAlert className="h-5 w-5 text-teal-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-semibold text-slate-100">Software Simulation Boundary & Safety Constraints</p>
          <p className="text-slate-400 leading-relaxed">
            The simulation runner operates within the Gymnasium contextual bandit environment. Actions (0-19) update internal software state, calculate proportional CISA rewards, and advance incident pointers. No live commands are sent to physical network adapters, firewalls, or real hosts. Destructive Caldera execution is disabled in this dashboard.
          </p>
        </div>
      </div>

      {/* Runs List Table */}
      <Card>
        <CardHeader
          title="Recorded Simulation Episodes"
          subtitle={`Showing ${runs.length} recorded rollout runs`}
        />

        {loading ? (
          <LoadingState message="Loading simulation runs..." />
        ) : runs.length === 0 ? (
          <EmptyState
            title="No simulation runs found"
            description="Click 'Execute Benchmark Simulation' to launch a software-simulated evaluation run."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="research-table">
              <thead>
                <tr>
                  <th className="w-44">Run ID</th>
                  <th>Scenario Description</th>
                  <th className="w-48">Policy Checkpoint</th>
                  <th className="w-24 text-center">Steps</th>
                  <th className="w-28 text-center">Accuracy</th>
                  <th className="w-28 text-center">Cum. Reward</th>
                  <th className="w-28 text-center">Status</th>
                  <th className="w-24 text-right">Details</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td className="font-mono text-xs text-teal-400 font-medium">
                      <Link href={`/simulations/${r.id}`} className="hover:underline">
                        {r.id}
                      </Link>
                    </td>
                    <td className="text-xs text-slate-200">
                      <p className="font-medium">{r.scenario}</p>
                      <p className="text-[10px] text-slate-400 mt-0.5 font-mono">
                        Started: {formatDate(r.start_time)}
                      </p>
                    </td>
                    <td className="text-xs font-mono text-slate-300">
                      {r.policy}
                    </td>
                    <td className="text-center font-mono text-xs text-slate-200">
                      {r.steps_count}
                    </td>
                    <td className="text-center font-mono text-xs font-semibold text-teal-400">
                      {(r.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="text-center font-mono text-xs font-bold text-sky-400">
                      {r.final_cumulative_reward > 0 ? `+${r.final_cumulative_reward}` : r.final_cumulative_reward}
                    </td>
                    <td className="text-center">
                      <StatusBadge status={r.status} />
                    </td>
                    <td className="text-right">
                      <Link
                        href={`/simulations/${r.id}`}
                        className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-teal-400 inline-flex transition-colors"
                        title="Inspect Simulation Timeline"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
