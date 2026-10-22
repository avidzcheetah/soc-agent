"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import {
  Award,
  ArrowLeft,
  FileSpreadsheet,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  XCircle,
  Clock,
  Download,
  Filter,
  BarChart3,
  Layers,
  HelpCircle,
  ExternalLink,
  ChevronRight,
  GitCompare,
  TrendingUp,
  Cpu,
  ShieldAlert,
  PlaySquare,
  Activity,
  FileText,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { PolicyComparisonChart } from "@/components/charts/PolicyComparisonChart";
import { ConfusionMatrixHeatmap } from "@/components/charts/ConfusionMatrixHeatmap";
import { PolicyTrajectoryComparisonChart } from "@/components/charts/PolicyTrajectoryComparisonChart";
import { api } from "@/lib/api";
import {
  SimulationRun,
  EvaluationResponse,
  ResearchExperimentsResponse,
  SimulationComparisonResponse,
} from "@/types";
import { formatPercent, formatDate } from "@/lib/utils";

export default function ResultsPage() {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<
    "simulations" | "comparison" | "classification" | "sequential"
  >("simulations");

  // Data states
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [evalData, setEvalData] = useState<EvaluationResponse | null>(null);
  const [experimentsData, setExperimentsData] = useState<ResearchExperimentsResponse | null>(null);

  // Filters for simulation runs table
  const [scenarioFilter, setScenarioFilter] = useState<string>("all");
  const [policyFilter, setPolicyFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  // Comparison selection
  const [runAId, setRunAId] = useState<string>("SIM-BENCH-PPO-01");
  const [runBId, setRunBId] = useState<string>("SIM-BENCH-ATGP-01");
  const [comparison, setComparison] = useState<SimulationComparisonResponse | null>(null);
  const [loadingComparison, setLoadingComparison] = useState(false);

  // Initial data loading
  useEffect(() => {
    async function loadAllData() {
      try {
        const [runsRes, evalRes, expRes] = await Promise.all([
          api.getSimulations(),
          api.getEvaluations(),
          api.getResearchExperiments(),
        ]);
        setRuns(runsRes.simulations || []);
        setEvalData(evalRes);
        setExperimentsData(expRes);

        // Pre-select comparison runs if available
        if (runsRes.simulations && runsRes.simulations.length >= 2) {
          const ppo = runsRes.simulations.find((r) => r.policy === "PPO");
          const atgp = runsRes.simulations.find((r) => r.policy === "PPO+ATGP");
          if (ppo) setRunAId(ppo.id);
          if (atgp) setRunBId(atgp.id);
        }
      } catch (e) {
        console.error("Failed to load evaluation data", e);
      } finally {
        setLoading(false);
      }
    }
    loadAllData();
  }, []);

  // Fetch comparison when selections change
  useEffect(() => {
    async function fetchComparison() {
      if (!runAId || !runBId) return;
      setLoadingComparison(true);
      try {
        const comp = await api.compareSimulations(runAId, runBId);
        setComparison(comp);
      } catch (e) {
        console.error("Failed to load policy comparison", e);
      } finally {
        setLoadingComparison(false);
      }
    }
    if (activeTab === "comparison") {
      fetchComparison();
    }
  }, [runAId, runBId, activeTab]);

  // Filtered runs list
  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      if (scenarioFilter !== "all" && !r.scenario.toLowerCase().includes(scenarioFilter.toLowerCase())) {
        return false;
      }
      if (policyFilter !== "all" && r.policy.toLowerCase() !== policyFilter.toLowerCase()) {
        return false;
      }
      if (statusFilter !== "all" && r.status.toLowerCase() !== statusFilter.toLowerCase()) {
        return false;
      }
      return true;
    });
  }, [runs, scenarioFilter, policyFilter, statusFilter]);

  if (loading) {
    return <LoadingState message="Loading empirical results and policy evaluation benchmarks..." />;
  }

  return (
    <div className="space-y-6 pb-16">
      {/* Header */}
      <div className="rounded-xl border border-slate-200 bg-white shadow-sm p-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Link
                href="/"
                className="inline-flex items-center gap-1 text-xs text-teal-700 hover:text-teal-300 font-medium"
              >
                <ArrowLeft className="h-3.5 w-3.5" /> Back to Simulation Control
              </Link>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              Research Results &amp; Policy Evaluation
            </h1>
            <p className="text-xs sm:text-sm text-slate-500 max-w-3xl leading-relaxed">
              Empirical evaluation interface for{" "}
              <strong className="text-slate-800">
                Optimizing Cybersecurity Incident Response Actions Using a Transformer-Based Agent
              </strong>
              . Rigorously separating held-out classification metrics from sequential simulation outcomes.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start lg:self-auto">
            <span className="px-2.5 py-1 rounded bg-teal-50 text-teal-800 border border-teal-200 text-xs font-mono">
              Mode: Historical Replay
            </span>
          </div>
        </div>
      </div>

      {/* Primary Tab Navigation */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 pb-3">
        <button
          onClick={() => setActiveTab("simulations")}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "simulations"
              ? "bg-teal-600 text-white font-bold shadow-sm"
              : "bg-slate-100 text-slate-600 hover:text-slate-900 hover:bg-slate-200"
          }`}
        >
          <PlaySquare className="h-3.5 w-3.5" />
          <span>Simulation Runs &amp; Episodes ({runs.length})</span>
        </button>

        <button
          onClick={() => setActiveTab("comparison")}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "comparison"
              ? "bg-teal-600 text-white font-bold shadow-sm"
              : "bg-slate-100 text-slate-600 hover:text-slate-900 hover:bg-slate-200"
          }`}
        >
          <GitCompare className="h-3.5 w-3.5" />
          <span>PPO vs. PPO+ATGP Policy Comparison</span>
        </button>

        <button
          onClick={() => setActiveTab("classification")}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "classification"
              ? "bg-teal-600 text-white font-bold shadow-sm"
              : "bg-slate-100 text-slate-600 hover:text-slate-900 hover:bg-slate-200"
          }`}
        >
          <Award className="h-3.5 w-3.5" />
          <span>Held-Out Classification Evaluation (SecBERT)</span>
        </button>

        <button
          onClick={() => setActiveTab("sequential")}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
            activeTab === "sequential"
              ? "bg-teal-600 text-white font-bold shadow-sm"
              : "bg-slate-100 text-slate-600 hover:text-slate-900 hover:bg-slate-200"
          }`}
        >
          <Activity className="h-3.5 w-3.5" />
          <span>Sequential Environment Benchmarks</span>
        </button>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: SIMULATION RUNS TABLE                                              */}
      {/* ========================================================================= */}
      {activeTab === "simulations" && (
        <div className="space-y-4">
          {/* Filters Bar */}
          <Card className="p-4 bg-white border-slate-200 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-1.5 text-slate-500 font-semibold uppercase text-[10px]">
                  <Filter className="h-3.5 w-3.5 text-teal-700" />
                  <span>Filters:</span>
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-slate-500">Scenario:</span>
                  <select
                    value={scenarioFilter}
                    onChange={(e) => setScenarioFilter(e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800 font-mono text-[11px]"
                  >
                    <option value="all">All Scenarios</option>
                    <option value="ransomware">Ransomware Containment</option>
                    <option value="c2">Command &amp; Control</option>
                    <option value="vulnerability">CVE Vulnerability</option>
                  </select>
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-slate-500">Policy:</span>
                  <select
                    value={policyFilter}
                    onChange={(e) => setPolicyFilter(e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800 font-mono text-[11px]"
                  >
                    <option value="all">All Policies</option>
                    <option value="PPO">PPO</option>
                    <option value="PPO+ATGP">PPO+ATGP</option>
                  </select>
                </div>

                <div className="flex items-center gap-1">
                  <span className="text-slate-500">Status:</span>
                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-800 font-mono text-[11px]"
                  >
                    <option value="all">All Statuses</option>
                    <option value="completed">Completed</option>
                    <option value="stopped">Stopped</option>
                    <option value="failed">Failed</option>
                    <option value="running">Running</option>
                  </select>
                </div>
              </div>

              <div className="text-[11px] font-mono text-slate-500 self-end sm:self-auto">
                Showing {filteredRuns.length} of {runs.length} Recorded Runs
              </div>
            </div>
          </Card>

          {/* Runs Table */}
          <Card className="overflow-hidden border-slate-200">
            <div className="overflow-x-auto">
              <table className="research-table w-full text-left text-xs">
                <thead className="bg-slate-100 text-slate-500 font-medium uppercase text-[10px] tracking-wider border-b border-slate-200">
                  <tr>
                    <th className="py-3 px-4">Run ID</th>
                    <th className="py-3 px-4">Scenario &amp; Mode</th>
                    <th className="py-3 px-4">Policy &amp; Checkpoint</th>
                    <th className="py-3 px-4 text-center">Steps</th>
                    <th className="py-3 px-4 text-center">Alignment Acc</th>
                    <th className="py-3 px-4 text-center">Cum. Reward</th>
                    <th className="py-3 px-4 text-center">Status</th>
                    <th className="py-3 px-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-sans">
                  {filteredRuns.map((r) => {
                    const statusLower = (r.status || "").toLowerCase();
                    const isCompleted = statusLower === "completed";
                    const isStopped = statusLower === "stopped";
                    const isFailed = statusLower === "failed";

                    const accuracyDisplay =
                      r.accuracy !== undefined && r.accuracy !== null
                        ? `${(r.accuracy * 100).toFixed(1)}%`
                        : "—";

                    const cumRewardDisplay =
                      r.final_cumulative_reward !== undefined && r.final_cumulative_reward !== null
                        ? r.final_cumulative_reward > 0
                          ? `+${r.final_cumulative_reward.toFixed(1)}`
                          : `${r.final_cumulative_reward.toFixed(1)}`
                        : "—";

                    return (
                      <tr key={r.id} className="hover:bg-slate-50 transition-colors">
                        <td className="py-3 px-4 font-mono font-bold text-teal-700">
                          <Link href={`/simulations/${r.id}`} className="hover:underline">
                            {r.id}
                          </Link>
                        </td>
                        <td className="py-3 px-4 text-slate-800">
                          <p className="font-medium text-xs">{r.scenario}</p>
                          <div className="flex items-center gap-1.5 mt-0.5 text-[10px] text-slate-500 font-mono">
                            <span className="px-1.5 py-0.2 rounded bg-slate-100 text-teal-700">
                              Historical Replay
                            </span>
                            <span>{r.start_time ? formatDate(r.start_time) : ""}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-slate-700 font-mono text-[11px]">
                          <span className="font-semibold text-slate-900">{r.policy}</span>
                          <span className="block text-[10px] text-slate-500 truncate max-w-[180px]">
                            {r.policy === "PPO" ? "v1.0-ppo-core" : "EXP_PPO_ATGP_DEV"}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center font-mono text-slate-800">
                          {r.steps_count}
                        </td>
                        <td className="py-3 px-4 text-center font-mono font-bold text-slate-900">
                          {accuracyDisplay}
                        </td>
                        <td
                          className={`py-3 px-4 text-center font-mono font-bold ${
                            r.final_cumulative_reward && r.final_cumulative_reward > 0
                              ? "text-emerald-700"
                              : "text-rose-700"
                          }`}
                        >
                          {cumRewardDisplay}
                        </td>
                        <td className="py-3 px-4 text-center">
                          <StatusBadge status={r.status} />
                        </td>
                        <td className="py-3 px-4 text-right">
                          <div className="flex items-center justify-end gap-1.5">
                            <Link
                              href={`/simulations/${r.id}`}
                              className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium transition-colors"
                            >
                              Report
                            </Link>
                            <a
                              href={api.getSimulationExportUrl(r.id, "json")}
                              download
                              className="p-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-500 hover:text-slate-800 transition-colors"
                              title="Export JSON"
                            >
                              <Download className="h-3 w-3" />
                            </a>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: PPO VS. PPO+ATGP POLICY COMPARISON                                 */}
      {/* ========================================================================= */}
      {activeTab === "comparison" && (
        <div className="space-y-6">
          {/* Comparison Controls */}
          <Card className="p-5 border-slate-200 bg-white space-y-4 shadow-sm">
            <div>
              <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <GitCompare className="h-4 w-4 text-teal-700" />
                <span>Fair Policy Comparison Configuration</span>
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Select two simulation episodes for direct head-to-head trajectory and reward comparison.
                Initial state, attack scenario, and random seed must match for valid comparison.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Policy A Selector */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
                <label className="text-xs font-semibold text-slate-700 block">
                  Policy A (Baseline: PPO)
                </label>
                <select
                  value={runAId}
                  onChange={(e) => setRunAId(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded p-2 text-xs text-slate-800 font-mono"
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.id} - {r.scenario} ({r.policy})
                    </option>
                  ))}
                </select>
                {comparison?.run_a && (
                  <div className="text-[11px] font-mono text-slate-500 space-y-0.5 pt-1">
                    <div>Policy: <strong className="text-slate-800">{comparison.run_a.policy}</strong></div>
                    <div>Checkpoint: <span className="text-teal-300">{comparison.run_a.checkpoint}</span></div>
                    <div>Seed: {comparison.run_a.seed} | Steps: {comparison.run_a.steps_count}</div>
                  </div>
                )}
              </div>

              {/* Policy B Selector */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
                <label className="text-xs font-semibold text-slate-700 block">
                  Policy B (Comparison: PPO+ATGP)
                </label>
                <select
                  value={runBId}
                  onChange={(e) => setRunBId(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-200 rounded p-2 text-xs text-slate-800 font-mono"
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.id} - {r.scenario} ({r.policy})
                    </option>
                  ))}
                </select>
                {comparison?.run_b && (
                  <div className="text-[11px] font-mono text-slate-500 space-y-0.5 pt-1">
                    <div>Policy: <strong className="text-slate-800">{comparison.run_b.policy}</strong></div>
                    <div>Checkpoint: <span className="text-teal-300">{comparison.run_b.checkpoint}</span></div>
                    <div>Seed: {comparison.run_b.seed} | Steps: {comparison.run_b.steps_count}</div>
                  </div>
                )}
              </div>
            </div>
          </Card>

          {/* Compatibility Verification Banner */}
          {comparison && (
            <div
              className={`p-4 rounded-xl border flex items-start gap-3 ${
                comparison.compatibility === "compatible"
                  ? "bg-emerald-950/30 border-emerald-800/40 text-emerald-300"
                  : comparison.compatibility === "partially_compatible"
                  ? "bg-amber-950/30 border-amber-800/40 text-amber-300"
                  : "bg-rose-950/30 border-rose-800/40 text-rose-300"
              }`}
            >
              {comparison.compatibility === "compatible" ? (
                <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-700 mt-0.5" />
              ) : (
                <AlertTriangle className="h-5 w-5 shrink-0 text-amber-700 mt-0.5" />
              )}
              <div className="space-y-1">
                <p className="font-bold text-xs uppercase tracking-wider">
                  Comparison Status: {comparison.compatibility.replace("_", " ")}
                </p>
                <ul className="text-xs space-y-0.5 list-disc list-inside">
                  {comparison.compatibility_reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {/* Side-by-Side Metric Difference Cards */}
          {comparison && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Card className="p-4 space-y-1 border-slate-200 bg-slate-50">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                  Alignment Accuracy Δ
                </span>
                <p
                  className={`text-lg font-bold font-mono ${
                    comparison.metrics_diff.accuracy_diff >= 0
                      ? "text-emerald-700"
                      : "text-rose-700"
                  }`}
                >
                  {comparison.metrics_diff.accuracy_diff >= 0 ? "+" : ""}
                  {(comparison.metrics_diff.accuracy_diff * 100).toFixed(1)}%
                </p>
                <p className="text-[10px] text-slate-500">
                  {comparison.run_a.policy} ({(comparison.run_a.accuracy * 100).toFixed(1)}%) vs{" "}
                  {comparison.run_b.policy} ({(comparison.run_b.accuracy * 100).toFixed(1)}%)
                </p>
              </Card>

              <Card className="p-4 space-y-1 border-slate-200 bg-slate-50">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                  Cumulative Return Δ
                </span>
                <p
                  className={`text-lg font-bold font-mono ${
                    comparison.metrics_diff.reward_diff >= 0
                      ? "text-emerald-700"
                      : "text-rose-700"
                  }`}
                >
                  {comparison.metrics_diff.reward_diff >= 0 ? "+" : ""}
                  {comparison.metrics_diff.reward_diff.toFixed(1)}
                </p>
                <p className="text-[10px] text-slate-500">
                  {comparison.run_a.cumulative_reward} vs {comparison.run_b.cumulative_reward}
                </p>
              </Card>

              <Card className="p-4 space-y-1 border-slate-200 bg-slate-50">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                  Mean Reward Per Step Δ
                </span>
                <p
                  className={`text-lg font-bold font-mono ${
                    comparison.metrics_diff.mean_reward_diff >= 0
                      ? "text-emerald-700"
                      : "text-rose-700"
                  }`}
                >
                  {comparison.metrics_diff.mean_reward_diff >= 0 ? "+" : ""}
                  {comparison.metrics_diff.mean_reward_diff.toFixed(3)}
                </p>
                <p className="text-[10px] text-slate-500">
                  {comparison.run_a.mean_reward} vs {comparison.run_b.mean_reward}
                </p>
              </Card>

              <Card className="p-4 space-y-1 border-slate-200 bg-slate-50">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                  Enterprise Containment
                </span>
                <p className="text-xs font-bold text-amber-700 font-mono mt-1">
                  Not Available
                </p>
                <p className="text-[10px] text-slate-500">
                  Contextual bandit formulation
                </p>
              </Card>
            </div>
          )}

          {/* Trajectory Comparison Chart */}
          {comparison && comparison.trajectory_comparison.length > 0 && (
            <Card className="p-5 border-slate-200 bg-white space-y-3">
              <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
                <div>
                  <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
                    Cumulative Reward Trajectory Comparison
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    Step-by-step reward accumulation across sequential test incidents.
                  </p>
                </div>
              </div>

              <PolicyTrajectoryComparisonChart
                trajectory={comparison.trajectory_comparison}
                policyAName={`${comparison.run_a.policy} (${comparison.run_a.id})`}
                policyBName={`${comparison.run_b.policy} (${comparison.run_b.id})`}
              />
            </Card>
          )}

          {/* Action Frequency Breakdown */}
          {comparison && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <Card className="p-4 border-slate-200 bg-slate-50 space-y-2">
                <h4 className="font-semibold text-slate-800 uppercase tracking-wider text-[11px]">
                  {comparison.run_a.policy} Action Selections
                </h4>
                <div className="space-y-1 font-mono text-[11px]">
                  {Object.entries(comparison.action_distribution.policy_a).map(([action, count]) => (
                    <div key={action} className="flex items-center justify-between p-1 rounded bg-slate-50">
                      <span className="text-slate-700">{action}</span>
                      <span className="text-teal-700 font-bold">{count} steps</span>
                    </div>
                  ))}
                </div>
              </Card>

              <Card className="p-4 border-slate-200 bg-slate-50 space-y-2">
                <h4 className="font-semibold text-slate-800 uppercase tracking-wider text-[11px]">
                  {comparison.run_b.policy} Action Selections
                </h4>
                <div className="space-y-1 font-mono text-[11px]">
                  {Object.entries(comparison.action_distribution.policy_b).map(([action, count]) => (
                    <div key={action} className="flex items-center justify-between p-1 rounded bg-slate-50">
                      <span className="text-slate-700">{action}</span>
                      <span className="text-teal-700 font-bold">{count} steps</span>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          )}

          {/* Scientific Caveat Notice */}
          <div className="p-4 rounded-xl border border-slate-200 bg-slate-50 text-[11px] text-slate-500 space-y-1">
            <span className="font-semibold text-slate-700 uppercase tracking-wider text-[10px] block">
              Statistical Significance &amp; Research Boundary Note:
            </span>
            <p>
              Single-trajectory comparisons provide qualitative execution insight. Statistical significance claims require multiple seeded evaluations (e.g., n ≥ 30) with confidence intervals. Checkpoint ATGP reflects developmental protocol targets.
            </p>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: HELD-OUT CLASSIFICATION EVALUATION                                 */}
      {/* ========================================================================= */}
      {activeTab === "classification" && evalData && (
        <div className="space-y-6">
          {/* Dataset & Experiment Metadata Card */}
          <Card className="p-5 border-slate-200 bg-white shadow-sm space-y-3">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <span className="text-[10px] font-mono text-teal-700 font-bold uppercase tracking-wider">
                  Strictly Held-Out Evaluation Split
                </span>
                <h2 className="text-sm font-bold text-slate-900 mt-0.5">
                  SecBERT Standalone Classifier: EXP_002G (FP32 Ablation)
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Evaluated on 1,539 strictly deduplicated test incidents (10% stratified split). Seed: 42. Checkpoint: EXP_20260727_001.
                </p>
              </div>
              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="px-2 py-1 rounded bg-slate-50 border border-slate-200 text-slate-700">
                  Samples: <strong>1,539</strong>
                </span>
                <span className="px-2 py-1 rounded bg-slate-50 border border-slate-200 text-slate-700">
                  Zero-Shot: <strong>Classes 2 &amp; 18</strong>
                </span>
              </div>
            </div>
          </Card>

          {/* Key Classification Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <Card className="p-3 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Accuracy (Top-1)
              </span>
              <p className="text-base font-bold text-teal-700 font-mono">89.93%</p>
              <span className="text-[10px] text-slate-500">1,384 / 1,539 correct</span>
            </Card>

            <Card className="p-3 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Macro F1
              </span>
              <p className="text-base font-bold text-slate-900 font-mono">0.7121</p>
              <span className="text-[10px] text-slate-500">Balanced unweighted</span>
            </Card>

            <Card className="p-3 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Weighted F1
              </span>
              <p className="text-base font-bold text-teal-700 font-mono">0.9454</p>
              <span className="text-[10px] text-slate-500">Class frequency weighted</span>
            </Card>

            <Card className="p-3 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                MCC Score
              </span>
              <p className="text-base font-bold text-teal-700 font-mono">0.9364</p>
              <span className="text-[10px] text-slate-500">Matthews correlation</span>
            </Card>

            <Card className="p-3 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Top-2 Accuracy
              </span>
              <p className="text-base font-bold text-emerald-700 font-mono">96.95%</p>
              <span className="text-[10px] text-slate-500">PPO policy action set</span>
            </Card>
          </div>

          {/* Model Architectures Comparison Bar Chart */}
          <Card className="p-5 border-slate-200 bg-slate-50 space-y-3">
            <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
              Architecture Benchmark Comparison (Supervised Classifier vs. PPO Policy)
            </h3>
            <PolicyComparisonChart architectures={evalData.architectures} />
          </Card>

          {/* Confusion Matrix Heatmap */}
          <Card className="p-5 border-slate-200 bg-slate-50 space-y-3">
            <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
              Confusion Matrix (8 Major Active Cybersecurity Classes)
            </h3>
            <ConfusionMatrixHeatmap data={evalData.confusion_matrix_sample} />
          </Card>

          {/* 20-Action Space Per-Class Metrics Table */}
          <Card className="overflow-hidden border-slate-200">
            <div className="p-4 border-b border-slate-200 bg-slate-50">
              <h3 className="text-xs font-semibold text-slate-900 uppercase tracking-wider">
                Full 20-Action Space Per-Class Breakdown (Test Split)
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="research-table w-full text-left text-xs">
                <thead className="bg-slate-100 text-slate-500 font-medium uppercase text-[10px] tracking-wider border-b border-slate-200">
                  <tr>
                    <th className="py-2.5 px-3">Index</th>
                    <th className="py-2.5 px-3">Canonical Action Name</th>
                    <th className="py-2.5 px-3 text-center">Test Support</th>
                    <th className="py-2.5 px-3 text-center">Precision</th>
                    <th className="py-2.5 px-3 text-center">Recall</th>
                    <th className="py-2.5 px-3 text-center">F1-Score</th>
                    <th className="py-2.5 px-3">Evaluation Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                  {evalData.per_class_metrics.map((m) => (
                    <tr key={m.index} className="hover:bg-slate-50">
                      <td className="py-2 px-3 text-slate-500">#{m.index}</td>
                      <td className="py-2 px-3 font-bold text-slate-800 font-sans">{m.action}</td>
                      <td className="py-2 px-3 text-center text-slate-700">{m.samples}</td>
                      <td className="py-2 px-3 text-center text-slate-700">{m.precision.toFixed(2)}</td>
                      <td className="py-2 px-3 text-center text-slate-700">{m.recall.toFixed(2)}</td>
                      <td className="py-2 px-3 text-center font-bold text-teal-700">{m.f1.toFixed(2)}</td>
                      <td className="py-2 px-3 text-[11px] font-sans text-slate-500">
                        {m.note || (m.samples > 100 ? "High Support Class" : "Standard Class")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Phase 1 Ablation Study Archive */}
          {experimentsData?.phase1_secbert && (
            <Card className="overflow-hidden border-slate-200">
              <div className="p-4 border-b border-slate-200 bg-slate-50">
                <h3 className="text-xs font-semibold text-slate-900 uppercase tracking-wider">
                  Phase 1 SecBERT Rigorous Ablation Study Archive
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="research-table w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-500 font-medium uppercase text-[10px] tracking-wider border-b border-slate-200">
                    <tr>
                      <th className="py-2.5 px-3">Experiment</th>
                      <th className="py-2.5 px-3">Run ID</th>
                      <th className="py-2.5 px-3">Intervention</th>
                      <th className="py-2.5 px-3 text-center">Macro F1</th>
                      <th className="py-2.5 px-3 text-center">Top-2 Acc</th>
                      <th className="py-2.5 px-3">Scientific Outcome</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-sans text-xs">
                    {experimentsData.phase1_secbert.map((exp) => (
                      <tr key={exp.logical_name} className="hover:bg-slate-50">
                        <td className="py-2.5 px-3 font-mono font-bold text-slate-800">
                          {exp.logical_name}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-500 text-[11px]">
                          {exp.system_id}
                        </td>
                        <td className="py-2.5 px-3 text-slate-700">{exp.description}</td>
                        <td className="py-2.5 px-3 text-center font-mono font-bold text-slate-900">
                          {exp.macro_f1.toFixed(4)}
                        </td>
                        <td className="py-2.5 px-3 text-center font-mono text-teal-700">
                          {exp.top2_accuracy ? `${(exp.top2_accuracy * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="py-2.5 px-3 text-[11px] text-slate-700">{exp.verdict}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: SEQUENTIAL ENVIRONMENT BENCHMARKS                                  */}
      {/* ========================================================================= */}
      {activeTab === "sequential" && evalData?.sequential_simulation_metrics && (
        <div className="space-y-6">
          <Card className="p-5 border-slate-200 bg-white shadow-sm space-y-3">
            <div>
              <span className="text-[10px] font-mono text-teal-700 font-bold uppercase tracking-wider">
                Gymnasium Contextual Bandit Sequential Rollout Environment
              </span>
              <h2 className="text-sm font-bold text-slate-900 mt-0.5">
                Aggregate Sequential Incident Evaluation Summary
              </h2>
              <p className="text-xs text-slate-500 mt-1">
                Reflects authentic sequential execution metrics across historical incident streams.
              </p>
            </div>
          </Card>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <Card className="p-4 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Recorded Episodes
              </span>
              <p className="text-xl font-bold text-slate-900 font-mono">
                {evalData.sequential_simulation_metrics.total_runs_recorded}
              </p>
              <span className="text-[10px] text-slate-500">Simulations saved</span>
            </Card>

            <Card className="p-4 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Mean Cumulative Return
              </span>
              <p className="text-xl font-bold text-emerald-700 font-mono">
                +{evalData.sequential_simulation_metrics.mean_cumulative_reward.toFixed(1)}
              </p>
              <span className="text-[10px] text-slate-500">
                Median: +{evalData.sequential_simulation_metrics.median_cumulative_reward.toFixed(1)}
              </span>
            </Card>

            <Card className="p-4 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Mean Steps / Episode
              </span>
              <p className="text-xl font-bold text-teal-700 font-mono">
                {evalData.sequential_simulation_metrics.mean_simulation_steps.toFixed(1)}
              </p>
              <span className="text-[10px] text-slate-500">Step horizon</span>
            </Card>

            <Card className="p-4 border-slate-200 bg-slate-50 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">
                Containment Success
              </span>
              <p className="text-sm font-bold text-amber-700 font-mono mt-1">
                Not Available
              </p>
              <span className="text-[10px] text-slate-500">Contextual bandit</span>
            </Card>
          </div>

          {/* Honest Boundary Table */}
          <Card className="p-5 border-slate-200 bg-slate-50 space-y-4">
            <h3 className="text-xs font-semibold text-slate-800 uppercase tracking-wider">
              Sequential Environment Supported &amp; Unavailable Metrics
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="space-y-2 p-3 rounded-lg bg-slate-50 border border-slate-200">
                <span className="font-semibold text-emerald-700 flex items-center gap-1.5">
                  <CheckCircle2 className="h-4 w-4" />
                  <span>Supported Operational Metrics</span>
                </span>
                <ul className="space-y-1 text-slate-700 font-mono text-[11px]">
                  <li>• Per-step reward feedback: +1.0 (Match) / -1.0 (Mismatch)</li>
                  <li>• Cumulative episodic return: Σ r_t</li>
                  <li>• Ground-truth CISA playbook alignment rate</li>
                  <li>• Step-by-step experience memory buffer</li>
                </ul>
              </div>

              <div className="space-y-2 p-3 rounded-lg bg-slate-50 border border-amber-900/40">
                <span className="font-semibold text-amber-700 flex items-center gap-1.5">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Explicitly Unavailable Metrics (Phase 2 Boundary)</span>
                </span>
                <ul className="space-y-1 text-slate-500 font-mono text-[11px]">
                  <li>• Host-level containment success rate: <strong>Not Available</strong></li>
                  <li>• Time to containment (dwell time): <strong>Not Available</strong></li>
                  <li>• Operational business disruption cost: <strong>Not Available</strong></li>
                  <li>• Live MITRE Caldera execution: <strong>Not Available</strong></li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
