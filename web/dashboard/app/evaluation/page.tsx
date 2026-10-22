"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  BarChart3,
  Layers,
  Award,
  AlertTriangle,
  Info,
  CheckCircle2,
  XCircle,
  HelpCircle,
  FileText,
  Sliders,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { PolicyComparisonChart } from "@/components/charts/PolicyComparisonChart";
import { ConfusionMatrixHeatmap } from "@/components/charts/ConfusionMatrixHeatmap";
import { api } from "@/lib/api";
import { EvaluationResponse } from "@/types";
import { formatPercent } from "@/lib/utils";

export default function EvaluationPage() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [activeTab, setActiveTab] = useState<"classification" | "operational" | "per_class" | "failure_cases">("classification");

  useEffect(() => {
    async function loadEvaluations() {
      try {
        const res = await api.getEvaluations();
        setData(res);
      } catch (e) {
        console.error("Failed to load evaluations", e);
      } finally {
        setLoading(false);
      }
    }
    loadEvaluations();
  }, []);

  if (loading) {
    return <LoadingState message="Compiling policy evaluation benchmarks..." />;
  }

  if (!data) {
    return (
      <Card className="text-center py-12">
        <p className="text-sm font-semibold text-slate-200">Evaluation Data Unavailable</p>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-900">Policy Evaluation & Benchmark Comparison</h1>
        <p className="text-xs text-slate-500 mt-1">
          Rigorously compare SecBERT standalone classification against PPO reinforcement learning policies on identical holdout test sets.
        </p>
      </div>

      {/* Controlled Evaluation Metadata Card */}
      <Card className="bg-white border-slate-200 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono uppercase tracking-wider text-teal-700 font-semibold">
                Evaluation Protocol: Controlled Holdout Test Evaluation
              </span>
              <StatusBadge status="Verified Ready" />
            </div>
            <p className="text-xs text-slate-600">
              Evaluated on 1,539 strictly deduplicated test incidents (10% stratified holdout). Seed: 42.
            </p>
          </div>
          <div className="flex items-center gap-2 font-mono text-xs text-slate-500">
            <span className="p-2 rounded bg-slate-50 border border-slate-200">
              Test Set: <strong className="text-slate-800">1,539 records</strong>
            </span>
            <span className="p-2 rounded bg-slate-50 border border-slate-200">
              Zero-Shot Excluded: <strong className="text-slate-800">Actions 2 & 18</strong>
            </span>
          </div>
        </div>
      </Card>

      {/* Architectures Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {data.architectures.map((arch) => (
          <Card
            key={arch.id}
            glow={arch.top_performing}
            className={`flex flex-col justify-between ${
              arch.top_performing ? "border-teal-400 bg-teal-50/20 shadow-sm" : ""
            }`}
          >
            <div>
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-sm font-bold text-slate-900">{arch.name}</h3>
                  <p className="text-[11px] text-slate-500 font-mono mt-0.5">{arch.checkpoint}</p>
                </div>
                {arch.top_performing ? (
                  <span className="px-2 py-0.5 rounded bg-teal-50 border border-teal-300 text-teal-800 text-[10px] font-bold uppercase tracking-wider">
                    Highest F1
                  </span>
                ) : (
                  <StatusBadge status={arch.accuracy !== null ? "Evaluated" : "Planned"} />
                )}
              </div>

              <p className="text-xs text-slate-500 mt-2">{arch.type}</p>

              {arch.accuracy !== null ? (
                <div className="grid grid-cols-2 gap-2 mt-4 text-xs font-mono">
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 uppercase">Weighted F1</span>
                    <p className="text-base font-bold text-teal-700 mt-0.5">
                      {formatPercent(arch.weighted_f1, 2)}
                    </p>
                  </div>
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 uppercase">Top-2 Acc</span>
                    <p className="text-base font-bold text-teal-700 mt-0.5">
                      {formatPercent(arch.top2_accuracy, 2)}
                    </p>
                  </div>
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 uppercase">Accuracy</span>
                    <p className="text-sm font-semibold text-slate-800 mt-0.5">
                      {formatPercent(arch.accuracy, 2)}
                    </p>
                  </div>
                  <div className="p-2 rounded bg-slate-50 border border-slate-200">
                    <span className="text-[10px] text-slate-500 uppercase">Macro F1</span>
                    <p className="text-sm font-semibold text-slate-800 mt-0.5">
                      {formatPercent(arch.macro_f1, 2)}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="mt-4 p-4 rounded-lg bg-slate-50 border border-dashed border-slate-200 text-center">
                  <p className="text-xs text-slate-500 font-mono">Not evaluated on test split</p>
                  <p className="text-[11px] text-slate-500 mt-1">{arch.notes}</p>
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-slate-100 text-[11px] text-slate-500">
              <span>{arch.status}</span>
            </div>
          </Card>
        ))}
      </div>

      {/* Tabs Control */}
      <div className="flex border-b border-slate-200 gap-1 text-xs font-medium">
        <button
          onClick={() => setActiveTab("classification")}
          className={`px-4 py-2.5 border-b-2 transition-colors ${
            activeTab === "classification"
              ? "border-teal-600 text-teal-800 font-bold"
              : "border-transparent text-slate-500 hover:text-slate-800"
          }`}
        >
          Classification Metrics & Side-by-Side Chart
        </button>
        <button
          onClick={() => setActiveTab("operational")}
          className={`px-4 py-2.5 border-b-2 transition-colors ${
            activeTab === "operational"
              ? "border-teal-600 text-teal-800 font-bold"
              : "border-transparent text-slate-500 hover:text-slate-800"
          }`}
        >
          Operational & Environment Metrics
        </button>
        <button
          onClick={() => setActiveTab("per_class")}
          className={`px-4 py-2.5 border-b-2 transition-colors ${
            activeTab === "per_class"
              ? "border-teal-600 text-teal-800 font-bold"
              : "border-transparent text-slate-500 hover:text-slate-800"
          }`}
        >
          Per-Class Precision, Recall, F1 (20 Actions)
        </button>
        <button
          onClick={() => setActiveTab("failure_cases")}
          className={`px-4 py-2.5 border-b-2 transition-colors ${
            activeTab === "failure_cases"
              ? "border-teal-600 text-teal-800 font-bold"
              : "border-transparent text-slate-500 hover:text-slate-800"
          }`}
        >
          Failure Case Analysis
        </button>
      </div>

      {/* TAB 1: Classification & Side-by-Side Comparison */}
      {activeTab === "classification" && (
        <div className="space-y-6">
          <Card>
            <CardHeader
              title="Side-by-Side Policy Performance Comparison"
              subtitle="Direct comparison across Top-1 Accuracy, Macro F1, Weighted F1, MCC, and Top-2 Accuracy"
            />
            <PolicyComparisonChart architectures={data.architectures} />
            <p className="text-[11px] text-slate-500 mt-3 border-t border-slate-100 pt-2 font-mono">
              Note: SecBERT classification represents direct argmax softmax output from the fine-tuned classification head. PPO policy optimizes exploration via entropy bonus (c2=0.01) yielding 78.4% top-1 accuracy while sustaining +0.568 mean reward.
            </p>
          </Card>

          {/* Confusion Matrix Heatmap */}
          <Card>
            <CardHeader
              title="Confusion Matrix (Major Incident Response Classes)"
              subtitle="True vs Predicted class distribution on 1,539 holdout test records"
            />
            <ConfusionMatrixHeatmap data={data.confusion_matrix_sample} />
          </Card>
        </div>
      )}

      {/* TAB 2: Operational Metrics */}
      {activeTab === "operational" && (
        <Card>
          <CardHeader
            title="Operational & Environment Metrics"
            subtitle="Distinguishing measured software bandit outcomes from planned live sequential containment"
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 block">Containment Success Rate</span>
              <p className="text-lg font-bold font-mono text-teal-700">{data.operational_metrics.containment_success_rate}</p>
              <p className="text-xs text-slate-500">Proportion of incidents receiving matching CISA containment action.</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 block">Response Success Rate</span>
              <p className="text-lg font-bold font-mono text-teal-700">{data.operational_metrics.response_success_rate}</p>
              <p className="text-xs text-slate-500">Action strictly matches expert labeled ground truth playbook.</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 block">Cumulative Reward (64-Step Rollout)</span>
              <p className="text-lg font-bold font-mono text-sky-700">{data.operational_metrics.cumulative_reward_64_steps}</p>
              <p className="text-xs text-slate-500">Summed episode reward under +1.0 (match) and -1.0 (mismatch) formulation.</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 block">Time to Containment</span>
              <p className="text-sm font-mono text-slate-600">{data.operational_metrics.time_to_containment}</p>
              <p className="text-xs text-slate-500">Contextual bandit advances immediately per step; physical host latency not measured.</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 block">Operational Disruption & Collateral Penalty</span>
              <p className="text-sm font-mono text-slate-600">{data.operational_metrics.operational_disruption}</p>
              <p className="text-xs text-slate-500">Requires multi-node enterprise topology (Phase 3 planned Caldera environment).</p>
            </div>

            <div className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
              <span className="text-[10px] uppercase font-mono text-slate-500 block">Incident Resolution Rate</span>
              <p className="text-lg font-bold font-mono text-teal-700">{data.operational_metrics.incident_resolution_rate}</p>
              <p className="text-xs text-slate-500">All presented alerts receive deterministic action execution and state advancement.</p>
            </div>
          </div>

          <div className="mt-4 p-3.5 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
            <p>
              <strong className="text-amber-950 font-semibold">Mandatory Data Rule:</strong> {data.operational_metrics.status_disclaimer}
            </p>
          </div>
        </Card>
      )}

      {/* TAB 3: Per-Class Breakdown Table */}
      {activeTab === "per_class" && (
        <Card>
          <CardHeader
            title="Per-Class Action Evaluation (All 20 Actions)"
            subtitle="Precision, Recall, and F1-Score breakdown on 1,539 test records"
          />
          <div className="overflow-x-auto">
            <table className="research-table">
              <thead>
                <tr>
                  <th className="w-16">Index</th>
                  <th>Action Name</th>
                  <th className="w-24 text-center">Test Samples</th>
                  <th className="w-24 text-center">Precision</th>
                  <th className="w-24 text-center">Recall</th>
                  <th className="w-24 text-center">F1-Score</th>
                  <th className="w-40">Status Note</th>
                </tr>
              </thead>
              <tbody>
                {data.per_class_metrics.map((pcm) => (
                  <tr key={pcm.index}>
                    <td className="font-mono text-xs text-slate-500">#{pcm.index}</td>
                    <td className="font-mono text-xs font-semibold text-slate-800">
                      {pcm.action}
                    </td>
                    <td className="text-center font-mono text-xs text-slate-600">
                      {pcm.samples}
                    </td>
                    <td className="text-center font-mono text-xs text-slate-600">
                      {pcm.samples > 0 ? pcm.precision.toFixed(2) : "—"}
                    </td>
                    <td className="text-center font-mono text-xs text-slate-600">
                      {pcm.samples > 0 ? pcm.recall.toFixed(2) : "—"}
                    </td>
                    <td className="text-center font-mono text-xs font-bold text-teal-700">
                      {pcm.samples > 0 ? pcm.f1.toFixed(2) : "—"}
                    </td>
                    <td className="text-xs text-slate-500">
                      {pcm.note ? (
                        <span className="px-2 py-0.5 rounded bg-amber-50 text-[10px] text-amber-800 border border-amber-200">
                          {pcm.note}
                        </span>
                      ) : pcm.samples < 10 ? (
                        <span className="text-[11px] text-slate-500">Minority Class</span>
                      ) : (
                        <span className="text-[11px] text-teal-700">Balanced</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* TAB 4: Failure Cases */}
      {activeTab === "failure_cases" && (
        <Card>
          <CardHeader
            title="Evaluation Failure Case Analysis"
            subtitle="Actual test misclassifications and semantic ambiguity analysis"
          />
          <div className="space-y-4">
            {data.failure_cases.map((fc) => (
              <div
                key={fc.id}
                className="p-4 rounded-lg bg-slate-50 border border-slate-200 space-y-3"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-rose-700">{fc.id}</span>
                    <span className="text-xs text-slate-500">Incident: {fc.incident_id}</span>
                  </div>
                  <span className="text-xs font-mono text-slate-500">
                    Critic V(s): {fc.critic_value}
                  </span>
                </div>

                <p className="text-xs font-mono text-slate-700 bg-white p-2.5 rounded border border-slate-200">
                  {fc.text}
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-2.5 rounded bg-teal-50 border border-teal-200">
                    <span className="text-[10px] uppercase font-mono text-teal-800 block">Ground Truth Target</span>
                    <span className="font-mono font-semibold text-teal-900 mt-0.5 block">{fc.ground_truth}</span>
                  </div>
                  <div className="p-2.5 rounded bg-rose-50 border border-rose-200">
                    <span className="text-[10px] uppercase font-mono text-rose-800 block">Policy Predicted Action</span>
                    <span className="font-mono font-semibold text-rose-900 mt-0.5 block">{fc.predicted}</span>
                  </div>
                </div>

                <div className="text-xs text-slate-600 bg-slate-100 p-2.5 rounded border border-slate-200">
                  <strong className="text-slate-800">Root Cause Analysis:</strong> {fc.root_cause}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
