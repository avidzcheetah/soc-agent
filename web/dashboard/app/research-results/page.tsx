"use client";

import React, { useState, useEffect } from "react";
import {
  FileSpreadsheet,
  Download,
  Printer,
  FileCode,
  Layers,
  Award,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
} from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LoadingState } from "@/components/ui/LoadingState";
import { api } from "@/lib/api";
import { ResearchExperimentsResponse, Phase1Experiment } from "@/types";

export default function ResearchResultsPage() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<ResearchExperimentsResponse | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await api.getResearchExperiments();
        setData(res);
      } catch (e) {
        console.error("Failed to load research experiments", e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const exportCsv = () => {
    if (!data) return;
    const headers = ["Logical Name", "System ID", "Description", "Macro F1", "Top-2 Acc", "Verdict"];
    const rows = data.phase1_secbert.map((e) => [
      e.logical_name,
      e.system_id,
      `"${e.description.replace(/"/g, '""')}"`,
      e.macro_f1,
      e.top2_accuracy || "N/A",
      `"${e.verdict.replace(/"/g, '""')}"`,
    ]);
    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "secbert_ablation_experiments.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportJson = () => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "research_experiments_archive.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  if (loading) {
    return <LoadingState message="Loading research experiment archives..." />;
  }

  if (!data) {
    return (
      <Card className="text-center py-12">
        <p className="text-sm font-semibold text-slate-200">Experiment Data Unavailable</p>
      </Card>
    );
  }

  return (
    <div className="space-y-8 print:p-0">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100">Research Experiment Archive</h1>
          <p className="text-xs text-slate-400 mt-1">
            Complete empirical ablation study records mapping Phase 1 SecBERT fine-tuning and Phase 2 PPO reinforcement learning.
          </p>
        </div>

        {/* Export Controls */}
        <div className="flex items-center gap-2 print:hidden self-start sm:self-auto">
          <button
            onClick={exportCsv}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs border border-slate-800 flex items-center gap-1.5 transition-colors font-medium"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={exportJson}
            className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs border border-slate-800 flex items-center gap-1.5 transition-colors font-medium"
          >
            <FileCode className="h-3.5 w-3.5" />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handlePrint}
            className="px-3 py-1.5 rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-300 text-xs border border-teal-500/30 flex items-center gap-1.5 transition-colors font-medium"
          >
            <Printer className="h-3.5 w-3.5" />
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* Locked Final Checkpoint Summary Card */}
      <Card glow className="bg-[#0b1222] border-teal-500/30">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-teal-400" />
              <span className="text-[10px] uppercase tracking-wider font-mono text-teal-400 font-bold">
                Officially Locked Phase 1 Model Checkpoint
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-100 mt-1">
              EXP_20260727_001 (EXP_002G / FP32 Ablation)
            </h3>
            <p className="text-xs text-slate-300 mt-0.5">
              Base: jackaduma/SecBERT fine-tuned with CISA phase prefixing, targeted augmentation, and FP32 precision.
            </p>
          </div>
          <div className="flex items-center gap-3 font-mono text-xs">
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-center">
              <span className="text-[9px] uppercase text-slate-400 block">Weighted F1</span>
              <span className="text-sm font-bold text-teal-400">0.9454</span>
            </div>
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-center">
              <span className="text-[9px] uppercase text-slate-400 block">MCC</span>
              <span className="text-sm font-bold text-teal-400">0.9364</span>
            </div>
            <div className="p-2.5 rounded bg-slate-900 border border-slate-800 text-center">
              <span className="text-[9px] uppercase text-slate-400 block">Top-2 Acc</span>
              <span className="text-sm font-bold text-teal-400">0.9695</span>
            </div>
          </div>
        </div>
      </Card>

      {/* Phase 1: SecBERT Strict Ablation Study (9 Experiments) */}
      <Card>
        <CardHeader
          title="Phase 1: SecBERT Fine-Tuning Ablation Study (EXP_001 – EXP_002G)"
          subtitle="Systematic isolation of each variable: sampling, label smoothing, phase tags, augmentation, and numerical precision"
        />

        <div className="overflow-x-auto">
          <table className="research-table">
            <thead>
              <tr>
                <th className="w-28">Logical Name</th>
                <th className="w-36">System ID</th>
                <th>Hypothesis & Intervention</th>
                <th className="w-24 text-center">Macro F1</th>
                <th className="w-24 text-center">Top-2 Acc</th>
                <th className="w-80">Empirical Verdict & Finding</th>
              </tr>
            </thead>
            <tbody>
              {data.phase1_secbert.map((exp) => {
                const isSelected = exp.logical_name === "EXP_002G";
                const isNegative = exp.verdict.toLowerCase().includes("negative");

                return (
                  <tr
                    key={exp.logical_name}
                    className={isSelected ? "bg-teal-500/10 hover:bg-teal-500/15" : ""}
                  >
                    <td className="font-mono text-xs font-bold text-slate-100">
                      {exp.logical_name}
                      {isSelected && (
                        <span className="ml-1 text-[9px] px-1.5 py-0.2 bg-teal-500/20 text-teal-300 rounded border border-teal-500/40">
                          LOCKED
                        </span>
                      )}
                    </td>
                    <td className="font-mono text-xs text-slate-400">{exp.system_id}</td>
                    <td className="text-xs text-slate-300">{exp.description}</td>
                    <td className="text-center font-mono text-xs font-bold">
                      <span
                        className={
                          exp.macro_f1 >= 0.71
                            ? "text-teal-400"
                            : isNegative
                            ? "text-rose-400"
                            : "text-slate-300"
                        }
                      >
                        {exp.macro_f1.toFixed(4)}
                      </span>
                    </td>
                    <td className="text-center font-mono text-xs text-slate-300">
                      {exp.top2_accuracy ? exp.top2_accuracy.toFixed(4) : "—"}
                    </td>
                    <td className="text-xs">
                      <span
                        className={
                          isSelected
                            ? "text-teal-300 font-semibold"
                            : isNegative
                            ? "text-rose-300"
                            : "text-slate-300"
                        }
                      >
                        {exp.verdict}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
          <Info className="h-4 w-4 text-sky-400 shrink-0 mt-0.5" />
          <p>
            <strong className="text-slate-300">Scientific Integrity (Rule 10):</strong> Negative results (such as EXP_002A where weighted cross-entropy produced double-weighting collapse, and EXP_002E where excessive LLM synthetic data degraded decision boundaries) are preserved faithfully as critical empirical contributions to cybersecurity NLP.
          </p>
        </div>
      </Card>

      {/* Phase 2: PPO Reinforcement Learning Milestone */}
      <Card>
        <CardHeader
          title="Phase 2: Deep Reinforcement Learning (PPO Agent v1.0 Core)"
          subtitle="Formal code audit score: 95/100 | Verification milestone completed on 2026-08-05"
        />

        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Milestone Tag</span>
              <span className="font-bold text-teal-400 block mt-0.5">v1.0-ppo-core</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Code Audit Score</span>
              <span className="font-bold text-teal-400 block mt-0.5">95 / 100 (Passed)</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">GAE Identities Verification</span>
              <span className="text-slate-200 block mt-0.5">&lt; 1e-7 Error Invariance</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Initial Policy Entropy</span>
              <span className="text-sky-400 block mt-0.5">2.99 (~ln 20 Uniform)</span>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-slate-200 mb-2 font-mono uppercase tracking-wider">
              Verified Pipeline Components:
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {data.phase2_ppo[0]?.components?.map((comp, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-2 p-2.5 rounded bg-slate-900/60 border border-slate-800/80 font-mono text-slate-300"
                >
                  <CheckCircle2 className="h-3.5 w-3.5 text-teal-400 shrink-0" />
                  <span>{comp}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Planned Future Extensions (ATGP & Sequential SOC) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader
            title="Adaptive Teacher-Guided Preservation (ATGP)"
            subtitle="Protocol Phase 2.5 Planned Extension"
          />
          <div className="space-y-3 text-xs text-slate-300">
            <p className="leading-relaxed">
              ATGP introduces an auxiliary knowledge distillation loss:
            </p>
            <div className="p-2.5 rounded bg-slate-950 font-mono text-teal-300 border border-slate-800 text-[11px]">
              L_total = L_PPO + alpha_t * D_KL(pi_SecBERT || pi_PPO)
            </div>
            <p className="text-slate-400 leading-relaxed">
              Where alpha_t decays dynamically as the agent gains environmental mastery. This prevents catastrophic forgetting of pre-trained cybersecurity semantics during initial exploration.
            </p>
            <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/60">
              <span>Status: Experimental Specification Documented</span>
              <StatusBadge status="Planned" />
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader
            title="Sequential SOC Multi-Host Environment"
            subtitle="Protocol Phase 3 Planned Deployment"
          />
          <div className="space-y-3 text-xs text-slate-300">
            <p className="leading-relaxed">
              In Phase 3, the agent transitions from per-alert contextual bandit evaluation to multi-step sequential incident mitigation across an enterprise subnet with live Wazuh SIEM alerts and Caldera attacks.
            </p>
            <div className="p-2.5 rounded bg-slate-950 font-mono text-sky-300 border border-slate-800 text-[11px]">
              State: s_t = [ SecBERT(alert_t) || Host_Context_t || Action_History_t ]
            </div>
            <p className="text-slate-400 leading-relaxed">
              Enables measuring operational disruption, containment time latency, and unnecessary isolation penalties.
            </p>
            <div className="pt-2 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/60">
              <span>Status: Infrastructure Architecture Planned</span>
              <StatusBadge status="Phase 3 Planned" />
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
