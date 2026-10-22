"use client";

import React from "react";
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

  const selectedAction = currentDecision?.chosen_action_name
    ? `${currentDecision.chosen_action_name} (#${currentDecision.chosen_action})`
    : currentDecision
    ? "Evaluating action space..."
    : "Standby — awaiting simulation trigger";

  const reward = currentDecision?.reward !== undefined ? currentDecision.reward : null;
  const rewardText =
    reward !== null ? (reward > 0 ? `+${reward.toFixed(1)}` : `${reward.toFixed(1)}`) : "—";

  // Environment state
  const envState = currentDecision?.incident_text
    ? currentDecision.incident_text
    : currentDecision?.target
    ? `Target: ${currentDecision.target} | Phase: ${currentDecision.cisa_phase || "Active"}`
    : "Environment normal. No active incident.";

  return (
    <Card className="p-4 border-slate-200 bg-white shadow-sm space-y-3 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-2">
        <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
          Current Decision
        </h2>
        <div className="flex items-center gap-2 font-mono text-xs text-slate-500">
          <span>Policy: <strong className="text-slate-800">{selectedPolicy}</strong></span>
          <span>•</span>
          <span>Step: <strong className="text-slate-800">{stepNumber}</strong></span>
        </div>
      </div>

      {/* 4 Fields: Step, Selected Action, Reward, Environment State */}
      <div className="space-y-2.5">
        {/* Selected Response Action */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Selected Response Action
          </span>
          <div className="text-sm font-mono font-bold text-slate-900">
            {selectedAction}
          </div>
          {currentDecision?.chosen_action_phase && (
            <div className="text-[11px] text-slate-500 font-mono mt-0.5">
              CISA Phase: {currentDecision.chosen_action_phase}
            </div>
          )}
        </div>

        {/* Reward */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Reward
          </span>
          <div className="flex items-baseline gap-2 font-mono">
            <span
              className={`text-base font-bold ${
                reward !== null
                  ? reward > 0
                    ? "text-emerald-700"
                    : "text-rose-700"
                  : "text-slate-400"
              }`}
            >
              {rewardText}
            </span>
            <span className="text-[11px] text-slate-500 font-sans">
              {reward !== null
                ? reward > 0
                  ? "Ground-truth alignment (+1.0)"
                  : "Mismatch penalty (-1.0)"
                : "Awaiting evaluation"}
            </span>
          </div>
        </div>

        {/* Environment State */}
        <div className="p-2.5 rounded-md bg-slate-50 border border-slate-200">
          <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider block mb-0.5">
            Environment State
          </span>
          <p className="text-xs text-slate-700 leading-relaxed font-sans line-clamp-3">
            {envState}
          </p>
        </div>
      </div>

      <div className="pt-2 text-[11px] text-slate-400 font-mono border-t border-slate-100 flex items-center justify-between">
        <span>Contextual bandit evaluation</span>
        <span>{isRunning ? "Running" : "Ready"}</span>
      </div>
    </Card>
  );
}
