"use client";

import React, { useState } from "react";
import {
  Play,
  Square,
  RotateCcw,
  Sliders,
  Flame,
  Settings,
  Cpu,
  Target,
  Hash,
  Loader2,
  CheckCircle2,
  ShieldCheck,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";

export interface SimulationConfig {
  scenario: string;
  policy: "PPO" | "PPO+ATGP";
  target: string;
  maxSteps: number;
  seed: number;
}

export interface ScenarioDefinition {
  id: string;
  name: string;
  calderaEvidence: string;
  agentResponses: string;
  targetDefault: string;
}

export const ATTACK_SCENARIOS: ScenarioDefinition[] = [
  {
    id: "scenario-c2",
    name: "C2 and data exfiltration",
    calderaEvidence:
      "Emulate system discovery, controlled outbound communication, and a simulated data-transfer attempt. Show process and network telemetry.",
    agentResponses:
      "Block destination IP, DNS sinkhole, or isolate host.",
    targetDefault: "Subnet 10.0.2.0/24 (DMZ Infrastructure)",
  },
  {
    id: "scenario-privilege-persistence",
    name: "Privilege escalation and persistence",
    calderaEvidence:
      "In an isolated Linux lab, emulate privilege escalation and creation of a test cron persistence entry. Show privilege and file-change events.",
    agentResponses:
      "Remove persistence, kill process, or isolate host.",
    targetDefault: "Linux Lab Host 1 (10.0.1.15)",
  },
  {
    id: "scenario-credential-lateral",
    name: "Credential abuse and lateral movement",
    calderaEvidence:
      "Use dedicated test credentials to emulate SSH access between lab hosts, followed by creation of a dummy database dump.",
    agentResponses:
      "Reset credentials, disable test account, or isolate host.",
    targetDefault: "Linux Lab Host 2 (10.0.1.16)",
  },
  {
    id: "scenario-web-rce",
    name: "Web RCE and reverse TCP tunnel",
    calderaEvidence:
      "Exploit a deliberately vulnerable test web service and emulate a controlled reverse connection. Show process and network events.",
    agentResponses:
      "Kill process, block destination, or isolate host.",
    targetDefault: "DMZ Ingress Web Cluster (10.0.2.10)",
  },
];

const TARGET_HOSTS = [
  "Subnet 10.0.2.0/24 (DMZ Infrastructure)",
  "Linux Lab Host 1 (10.0.1.15)",
  "Linux Lab Host 2 (10.0.1.16)",
  "DMZ Ingress Web Cluster (10.0.2.10)",
];

interface Props {
  config: SimulationConfig;
  onConfigChange: (config: SimulationConfig) => void;
  onStart: () => void;
  onStop: () => void;
  onReset?: () => void;
  isStarting?: boolean;
  isRunning?: boolean;
  activeRunId?: string | null;
}

export function SimulationConfigForm({
  config,
  onConfigChange,
  onStart,
  onStop,
  onReset,
  isStarting = false,
  isRunning = false,
  activeRunId = null,
}: Props) {
  const [isVerifying, setIsVerifying] = useState(false);
  const [envStatus, setEnvStatus] = useState("Tools Active");

  const selectedScenario =
    ATTACK_SCENARIOS.find((s) => s.id === config.scenario) || ATTACK_SCENARIOS[0];

  const handleScenarioChange = (scenarioId: string) => {
    const found = ATTACK_SCENARIOS.find((s) => s.id === scenarioId);
    onConfigChange({
      ...config,
      scenario: scenarioId,
      target: found ? found.targetDefault : config.target,
    });
  };

  const handleVerifyEnv = async () => {
    setIsVerifying(true);
    try {
      await api.getHealth();
      setEnvStatus("Environment Verified");
    } catch {
      setEnvStatus("Environment Degraded");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleResetEnv = async () => {
    try {
      await api.resetSimulations();
    } catch {
      // ignore
    }
    if (onReset) onReset();
  };

  return (
    <div className="space-y-3">
      {/* Section Header & Environment Control Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <h2 className="text-xs font-semibold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
          <Sliders className="h-4 w-4 text-teal-600" />
          <span>Simulation Control</span>
        </h2>

        {/* SOC Environment tools status & controls */}
        <div className="flex items-center gap-2 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 border border-slate-200 text-slate-600 text-[11px] font-mono">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
            <span>Wazuh • OSQuery • Caldera • SecBERT</span>
          </div>
          <button
            type="button"
            onClick={handleVerifyEnv}
            disabled={isVerifying}
            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium border border-slate-200 transition-colors"
          >
            {isVerifying ? "Checking..." : "Verify Environment"}
          </button>
          <button
            type="button"
            onClick={handleResetEnv}
            disabled={isRunning || isStarting}
            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-[11px] font-medium border border-slate-200 transition-colors disabled:opacity-50"
          >
            Reset Environment
          </button>
        </div>
      </div>

      {/* 2-Box Layout matching the ASCII Mockup */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Box: Attack Scenario Selection & Execution Controls */}
        <Card className="p-5 border-slate-200 bg-white shadow-sm flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <Flame className="h-4 w-4 text-rose-600" />
                <span>Attack Scenario</span>
              </label>
              <span className="text-[11px] text-slate-500 font-mono">
                {selectedScenario.id}
              </span>
            </div>

            <select
              value={config.scenario}
              disabled={isRunning || isStarting}
              onChange={(e) => handleScenarioChange(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-slate-50 border border-slate-300 text-xs font-medium text-slate-900 focus:outline-none focus:border-teal-500 disabled:opacity-60"
            >
              {ATTACK_SCENARIOS.map((sc) => (
                <option key={sc.id} value={sc.id}>
                  {sc.name}
                </option>
              ))}
            </select>

            {/* Caldera Evidence & Agent Response Preview */}
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2 text-xs">
              <div>
                <span className="font-semibold text-slate-700">Caldera Operation &amp; Observable Evidence:</span>
                <p className="text-slate-600 text-[11px] mt-0.5 leading-relaxed">
                  {selectedScenario.calderaEvidence}
                </p>
              </div>
              <div className="pt-2 border-t border-slate-200">
                <span className="font-semibold text-teal-800">Agent Response Examples:</span>
                <p className="text-teal-700 font-mono text-[11px] mt-0.5">
                  {selectedScenario.agentResponses}
                </p>
              </div>
            </div>
          </div>

          {/* Action Buttons: START SIMULATION, STOP, RESET */}
          <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onStart}
              disabled={isRunning || isStarting}
              className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
                !isRunning && !isStarting
                  ? "bg-teal-600 hover:bg-teal-700 text-white shadow-sm cursor-pointer"
                  : "bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed"
              }`}
            >
              {isStarting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Starting...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  <span>START SIMULATION</span>
                </>
              )}
            </button>

            <button
              type="button"
              onClick={onStop}
              disabled={!isRunning}
              className={`px-3.5 py-2 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-colors ${
                isRunning
                  ? "bg-rose-50 border-rose-300 text-rose-700 hover:bg-rose-100 cursor-pointer"
                  : "bg-slate-50 text-slate-400 border-slate-200 cursor-not-allowed"
              }`}
            >
              <Square className="h-3.5 w-3.5" />
              <span>STOP</span>
            </button>

            <button
              type="button"
              onClick={onReset}
              disabled={isRunning || isStarting}
              className="px-3.5 py-2 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-700 text-xs font-medium border border-slate-200 flex items-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span>RESET</span>
            </button>

            {activeRunId && isRunning && (
              <span className="ml-auto text-[11px] font-mono text-teal-700 flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-teal-500 animate-ping" />
                <span>Run: {activeRunId.slice(0, 14)}...</span>
              </span>
            )}
          </div>
        </Card>

        {/* Right Box: Simulation Settings */}
        <Card className="p-5 border-slate-200 bg-white shadow-sm flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                <Settings className="h-4 w-4 text-slate-600" />
                <span>Simulation Settings</span>
              </label>
              <span className="text-[11px] text-slate-500 font-mono">
                Parameters
              </span>
            </div>

            {/* Policy Toggle */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-700 flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5 text-teal-600" />
                <span>Policy:</span>
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  disabled={isRunning || isStarting}
                  onClick={() => onConfigChange({ ...config, policy: "PPO" })}
                  className={`py-2 px-3 rounded-lg border text-xs font-mono font-medium transition-all ${
                    config.policy === "PPO"
                      ? "bg-teal-50 border-teal-500 text-teal-800 shadow-sm"
                      : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  PPO
                </button>
                <button
                  type="button"
                  disabled={isRunning || isStarting}
                  onClick={() => onConfigChange({ ...config, policy: "PPO+ATGP" })}
                  className={`py-2 px-3 rounded-lg border text-xs font-mono font-medium transition-all ${
                    config.policy === "PPO+ATGP"
                      ? "bg-teal-50 border-teal-500 text-teal-800 shadow-sm"
                      : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
                  }`}
                >
                  PPO+ATGP
                </button>
              </div>
            </div>

            {/* Target Selection */}
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-700 flex items-center gap-1.5">
                <Target className="h-3.5 w-3.5 text-amber-600" />
                <span>Target:</span>
              </label>
              <select
                value={config.target}
                disabled={isRunning || isStarting}
                onChange={(e) => onConfigChange({ ...config, target: e.target.value })}
                className="w-full px-3 py-2 rounded-lg bg-slate-50 border border-slate-300 text-xs text-slate-900 focus:outline-none focus:border-teal-500"
              >
                {TARGET_HOSTS.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            {/* Steps & Seed */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700 flex items-center gap-1">
                    <Hash className="h-3.5 w-3.5 text-slate-500" />
                    <span>Steps</span>
                  </span>
                  <span className="font-mono text-teal-700 font-bold">
                    {config.maxSteps}
                  </span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={60}
                  step={5}
                  value={config.maxSteps}
                  disabled={isRunning || isStarting}
                  onChange={(e) =>
                    onConfigChange({ ...config, maxSteps: Number(e.target.value) })
                  }
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-slate-700">Seed</span>
                  <span className="font-mono text-slate-400 text-[10px]">Reproducible</span>
                </div>
                <input
                  type="number"
                  value={config.seed}
                  disabled={isRunning || isStarting}
                  onChange={(e) =>
                    onConfigChange({ ...config, seed: Number(e.target.value) })
                  }
                  className="w-full px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-300 text-xs font-mono text-slate-900 focus:outline-none focus:border-teal-500"
                />
              </div>
            </div>
          </div>

          <div className="pt-2 text-[11px] text-slate-400 font-mono border-t border-slate-100 flex items-center justify-between">
            <span>Isolated lab environment</span>
            <span>Safe research execution</span>
          </div>
        </Card>
      </div>
    </div>
  );
}
