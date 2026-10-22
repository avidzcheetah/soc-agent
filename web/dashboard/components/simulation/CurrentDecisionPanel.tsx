"use client";

import React from "react";
import {
  BrainCircuit,
  Info,
  HelpCircle,
  ShieldCheck,
  Award,
  Layers,
  Activity,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Cpu,
  Zap,
} from "lucide-react";
import { Card } from "@/components/ui/Card";

interface Props {
  selectedPolicy?: "PPO" | "PPO+ATGP";
  currentDecision?: any | null;
  currentStep?: number | null;
  isRunning?: boolean;
}

export function CurrentDecisionPanel({
  selectedPolicy = "PPO",
  currentDecision = null,
  currentStep = null,
  isRunning = false,
}: Props) {
  const stepNumber = currentStep !== null ? currentStep : "—";

  // Honest reporting: Checkpoint weights are uncommitted on disk
  const confidence = "Not Available";
  const confidenceSubtitle = "Checkpoint uncommitted on disk";
  const criticValue = "Not Available";
  const containmentStatus = "Not Available (Contextual bandit benchmark replay)";

  const reward = currentDecision?.reward !== undefined ? currentDecision.reward : null;
  const rewardDisplay =
    reward !== null ? (reward > 0 ? `+${reward.toFixed(1)}` : `${reward.toFixed(1)}`) : "—";

  const isRewardPositive = reward !== null && reward > 0;

  return (
    <Card className="p-5 border-slate-800/80 bg-slate-900/60 space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h2 className="text-xs font-semibold text-slate-100 uppercase tracking-wider flex items-center gap-2">
            <BrainCircuit className="h-4 w-4 text-teal-400" />
            <span>Current Decision Inspection</span>
          </h2>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Step-by-step policy output, canonical CISA action, and authentic environment reward.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-teal-300 border border-slate-700">
            Historical Replay
          </span>
          {isRunning ? (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/30 flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-teal-400 animate-pulse" />
              <span>Step {stepNumber}</span>
            </span>
          ) : (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
              {currentDecision ? "Rollout Completed" : "Awaiting Execution"}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
        {/* Current Step */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            Simulation Step
          </span>
          <p className="text-base font-bold text-slate-100 font-mono">{stepNumber}</p>
          <span className="text-[10px] text-slate-400">Step counter</span>
        </div>

        {/* Selected Policy */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            Active Policy
          </span>
          <p className="text-base font-bold text-teal-300 font-mono">{selectedPolicy}</p>
          <span className="text-[10px] text-slate-400">Policy architecture</span>
        </div>

        {/* Action Probability / Confidence */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
              Policy Probabilities
            </span>
          </div>
          <p className="text-xs font-bold text-amber-400 font-mono">
            {currentDecision ? confidence : "—"}
          </p>
          <span className="text-[10px] text-slate-400 block truncate">
            {currentDecision ? confidenceSubtitle : "π(a|s) [20]"}
          </span>
        </div>

        {/* Reward */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            Environment Reward
          </span>
          <p
            className={`text-base font-bold font-mono ${
              reward === null
                ? "text-slate-400"
                : isRewardPositive
                ? "text-emerald-400"
                : "text-rose-400"
            }`}
          >
            {rewardDisplay}
          </p>
          <span className="text-[10px] text-slate-400">+1.0 / -1.0 contextual feedback</span>
        </div>
      </div>

      {/* Incident Details & Outcome Rows */}
      <div className="space-y-3 text-xs">
        {/* Incident Alert Context */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5 text-teal-400" />
              <span>Current Incident Alert Context</span>
            </span>
            {currentDecision?.cisa_phase && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-950/60 text-teal-300 border border-teal-800/60">
                Phase: {currentDecision.cisa_phase}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-300 leading-relaxed font-sans mt-1">
            {currentDecision?.incident_text ||
              currentDecision?.prepared_text ||
              "No active incident. Click 'Start Simulation' to trigger sequential alert processing from data/processed/test.csv."}
          </p>
        </div>

        {/* Model State & Selected Action */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {/* SecBERT & Critic Status */}
          <div className="p-3 rounded-lg bg-slate-950 border border-amber-900/40 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5" />
                <span>SecBERT State Model</span>
              </span>
              <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
                Uncommitted Checkpoint
              </span>
            </div>
            <p className="text-xs text-slate-300">
              State Embedding: <span className="font-mono text-amber-300">Unavailable</span> (768-dim architectural target)
            </p>
            <p className="text-[10px] text-slate-400">
              Critic Baseline V(s): <span className="font-mono text-amber-300">Not Available</span>
            </p>
          </div>

          {/* Selected Mitigation Action */}
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <Zap className="h-3.5 w-3.5 text-teal-400" />
                <span>Selected Mitigation Action</span>
              </span>
              {currentDecision?.chosen_action_phase && (
                <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-teal-300 border border-slate-700">
                  {currentDecision.chosen_action_phase}
                </span>
              )}
            </div>
            <p className="text-xs font-mono font-bold text-teal-300">
              {currentDecision?.chosen_action_name
                ? `${currentDecision.chosen_action_name} (#${currentDecision.chosen_action})`
                : "—"}
            </p>
            <p className="text-[10px] text-slate-400">
              {currentDecision?.ground_truth_name
                ? `Ground Truth Target: ${currentDecision.ground_truth_name} (#${currentDecision.ground_truth_action})`
                : "1 of 20 canonical CISA playbook actions"}
            </p>
          </div>
        </div>

        {/* Environment Transition & Containment */}
        <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Activity className="h-3.5 w-3.5 text-cyan-400" />
              <span>Environment State Transition & Containment</span>
            </span>
            <span className="text-[9px] font-mono text-slate-400">
              src/environment.py:step
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-tight">
            {currentDecision?.implementation ||
              (currentDecision
                ? "Benchmark incident evaluated. Immediate contextual bandit feedback returned without synthetic containment simulation."
                : "Awaiting next environment transition")}
          </p>
          <div className="pt-1 flex flex-wrap items-center justify-between text-[10px] text-slate-400 font-mono">
            <span>Host Containment: <strong className="text-amber-400/90">{containmentStatus}</strong></span>
            <span>Reward Logic: <strong className="text-slate-300">+1.0 Match / -1.0 Mismatch</strong></span>
          </div>
        </div>
      </div>
    </Card>
  );
}
