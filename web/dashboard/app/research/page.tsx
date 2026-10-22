"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  FileSpreadsheet,
  Award,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Layers,
  ArrowRight,
  Sparkles,
  BookOpen,
  Info,
  ChevronRight,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { LoadingState } from "@/components/ui/LoadingState";
import { api } from "@/lib/api";

export default function ResearchResultsPage() {
  const [loading, setLoading] = useState(true);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [checkpointMeta, setCheckpointMeta] = useState<any>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await api.getResearchExperiments();
        setExperiments(res.phase1_secbert || []);
        setCheckpointMeta(res.locked_checkpoint_metadata || null);
      } catch (e) {
        console.error("Failed to load research experiments", e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return <LoadingState message="Loading empirical research findings..." />;
  }

  return (
    <div className="space-y-8 pb-12">
      {/* Header Banner */}
      <div className="rounded-xl border border-slate-800 bg-gradient-to-r from-slate-900/90 via-[#0a1122]/80 to-slate-900/90 p-6">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-teal-500/10 text-teal-400 border border-teal-500/20">
              Department of Computer Engineering, University of Jaffna
            </span>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono text-slate-400 bg-slate-800/60 border border-slate-700/50">
              Thesis Ablation Archive
            </span>
          </div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
            Research Journey & Empirical Ablations
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 max-w-3xl leading-relaxed">
            Rigorous empirical ablation study tracking representation learning across SecBERT fine-tuning experiments,
            documenting negative findings, isolating semantic phase benefits, and verifying the contextual bandit RL formulation.
          </p>
        </div>
      </div>

      {/* Primary Research Metrics Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Card className="p-4 space-y-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Locked Final Model
          </span>
          <p className="text-lg font-bold text-slate-100 font-mono">EXP_002G (FP32)</p>
          <p className="text-[11px] text-teal-400 font-medium">Checkpoint: EXP_20260727_001</p>
        </Card>

        <Card className="p-4 space-y-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Weighted F1 Score
          </span>
          <p className="text-lg font-bold text-teal-400 font-mono">0.9454</p>
          <p className="text-[11px] text-slate-400 font-mono">MCC: 0.9364</p>
        </Card>

        <Card className="p-4 space-y-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Top-2 Accuracy
          </span>
          <p className="text-lg font-bold text-teal-400 font-mono">96.95%</p>
          <p className="text-[11px] text-slate-400">Foundation for downstream PPO</p>
        </Card>

        <Card className="p-4 space-y-1">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Benchmark Dataset
          </span>
          <p className="text-lg font-bold text-slate-100 font-mono">18,218 Records</p>
          <p className="text-[11px] text-slate-400">CISSM (16.7k) + rcATT (1.5k)</p>
        </Card>
      </div>

      {/* Complete Phase 1 Ablation Study Table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <FileSpreadsheet className="h-4 w-4 text-teal-400" />
            <span>Complete SecBERT Representation Ablation Study</span>
          </h2>
          <span className="text-[11px] text-slate-400 font-mono">Strict Single-Variable Isolation</span>
        </div>

        <Card className="overflow-hidden border-slate-800">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 font-medium uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Experiment</th>
                  <th className="py-3 px-4">Run ID</th>
                  <th className="py-3 px-4">Technique / Intervention</th>
                  <th className="py-3 px-4 text-center">Macro F1</th>
                  <th className="py-3 px-4 text-center">Top-2 Acc.</th>
                  <th className="py-3 px-4">Empirical Outcome & Scientific Verdict</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-sans">
                {experiments.map((exp) => {
                  const isLocked = exp.logical_name === "EXP_002G";
                  const isNegative = exp.verdict.toLowerCase().includes("negative");

                  return (
                    <tr
                      key={exp.logical_name}
                      className={`hover:bg-slate-900/50 transition-colors ${
                        isLocked
                          ? "bg-teal-500/5 font-semibold"
                          : isNegative
                          ? "bg-rose-500/5"
                          : ""
                      }`}
                    >
                      <td className="py-3 px-4 font-mono font-bold text-slate-200">
                        <div className="flex items-center gap-1.5">
                          {isLocked && <Award className="h-3.5 w-3.5 text-amber-400" />}
                          {isNegative && <AlertTriangle className="h-3.5 w-3.5 text-rose-400" />}
                          <span>{exp.logical_name}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 font-mono text-[11px] text-slate-400">
                        {exp.system_id}
                      </td>
                      <td className="py-3 px-4 text-slate-300 max-w-xs">
                        {exp.description}
                      </td>
                      <td className="py-3 px-4 text-center font-mono font-bold text-slate-100">
                        {exp.macro_f1?.toFixed(4) || "—"}
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-teal-400 font-semibold">
                        {exp.top2_accuracy ? `${(exp.top2_accuracy * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="py-3 px-4 text-slate-300">
                        <span
                          className={`inline-block text-[11px] ${
                            isLocked
                              ? "text-teal-300 font-semibold"
                              : isNegative
                              ? "text-rose-300"
                              : "text-slate-300"
                          }`}
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
        </Card>
      </div>

      {/* Research Deep Dives */}
      <div className="space-y-4 pt-4 border-t border-slate-800">
        <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <BookOpen className="h-4 w-4 text-teal-400" />
          <span>Core Academic Discoveries & Thesis Notes</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Deep Dive 1: Double-Weighting Problem */}
          <Card className="p-5 space-y-3 border-rose-500/20 bg-rose-500/5">
            <div className="flex items-center gap-2 text-rose-400">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <h3 className="text-xs font-bold uppercase tracking-wider">
                1. Preserving Negative Results: The Double-Weighting Problem (EXP_002A)
              </h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              In EXP_002A, adding inverse-frequency class weights to the <code className="text-slate-200">CrossEntropyLoss</code> caused a sharp performance drop:
              Accuracy plummeted from 0.899 to 0.801 (-9.8%), and Macro F1 fell from 0.625 to 0.574 (-0.051).
            </p>
            <div className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] text-slate-400 space-y-1 font-mono">
              <p className="text-slate-300 font-semibold font-sans">Root Cause Analysis:</p>
              <p>• Training already utilized a <code className="text-teal-400">WeightedRandomSampler</code> to equalize batch representation.</p>
              <p>• Compounding this with weighted loss doubly compensated minority classes: Class 17 oversampled ~456x with loss gradient amplified another 95x.</p>
              <p>• Caused catastrophic overfitting on rare samples and collapsed recall on majority classes (e.g. Class 8 recall dropped 0.81 → 0.54).</p>
            </div>
          </Card>

          {/* Deep Dive 2: Top-2 Accuracy */}
          <Card className="p-5 space-y-3 border-teal-500/20 bg-teal-500/5">
            <div className="flex items-center gap-2 text-teal-400">
              <Sparkles className="h-4 w-4 shrink-0" />
              <h3 className="text-xs font-bold uppercase tracking-wider">
                2. Significance of Top-2 Accuracy (96.95%) for Downstream PPO
              </h3>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              Evaluating the final SecBERT encoder (EXP_002G) revealed a Top-2 accuracy of <strong className="text-teal-300">96.95%</strong>.
              Even when SecBERT did not place the optimal action at the absolute highest logit, the correct mitigation action resided within its top two candidates 97% of the time.
            </p>
            <div className="p-3 rounded bg-slate-950 border border-slate-800 text-[11px] text-slate-400 space-y-1">
              <p className="text-slate-300 font-semibold">Thesis Architectural Impact:</p>
              <p>
                During reinforcement learning, the classifier head is discarded. This near-perfect Top-2 accuracy proves that SecBERT&apos;s 768-dimensional latent space encodes discriminative topology, enabling the PPO policy network to extract robust state representations during simulation.
              </p>
            </div>
          </Card>

          {/* Deep Dive 3: Contextual Bandit Formulation */}
          <Card className="p-5 space-y-3 border-slate-800">
            <div className="flex items-center gap-2 text-slate-200">
              <Layers className="h-4 w-4 text-teal-400 shrink-0" />
              <h3 className="text-xs font-bold uppercase tracking-wider">
                3. Formulation of the SOC Environment as a Contextual Bandit
              </h3>
            </div>
            <p className="text-xs text-slate-300 italic border-l-2 border-teal-500 pl-3 leading-relaxed">
              &ldquo;The environment is formulated as a contextual bandit, where each SOC alert is treated as an independent decision point.
              Consequently, the PPO agent optimizes per-incident action selection without modeling temporal dependencies between alerts.
              This formulation is appropriate because the objective of the proposed system is to recommend the most suitable response for each individual security incident rather than to learn long-horizon control policies.&rdquo;
            </p>
            <p className="text-[11px] text-slate-400">
              Selecting action a_t across the 20 canonical actions yields immediate feedback (+1.0 on match, -1.0 on mismatch) without dictating the arrival distribution or semantic content of subsequent uncorrelated enterprise alerts.
            </p>
          </Card>

          {/* Deep Dive 4: PPO v1.0 Core RL Architecture */}
          <Card className="p-5 space-y-3 border-slate-800">
            <div className="flex items-center gap-2 text-slate-200">
              <Cpu className="h-4 w-4 text-teal-400 shrink-0" />
              <h3 className="text-xs font-bold uppercase tracking-wider">
                4. PPO v1.0 Core Reinforcement Learning Specification
              </h3>
            </div>
            <div className="text-xs space-y-2 text-slate-300">
              <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Actor Network:</span>
                  <span className="text-teal-400">Linear(768→512→256→20)</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Critic Network:</span>
                  <span className="text-teal-400">Linear(768→512→256→1)</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">GAE Smoothing:</span>
                  <span className="text-slate-200">gamma=0.99, lambda=0.95</span>
                </div>
                <div className="p-2 rounded bg-slate-950 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Surrogate Loss:</span>
                  <span className="text-slate-200">epsilon=0.2, c2=0.01</span>
                </div>
              </div>
              <p className="text-[11px] text-slate-400 pt-1">
                Formal code audit completed on 2026-08-05 with a 95/100 audit score; mathematical GAE identity verified numerically to &lt; 1e-7 without gradient leakage.
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
