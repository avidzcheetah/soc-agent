"use client";

import React, { useState, useEffect } from "react";
import {
  Play,
  Square,
  RotateCcw,
  Sliders,
  ShieldAlert,
  Cpu,
  Target,
  Hash,
  Info,
  Loader2,
  CheckCircle2,
  Activity,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { api } from "@/lib/api";
import { ScenarioItem } from "@/types";

export interface SimulationConfig {
  scenario: string;
  policy: "PPO" | "PPO+ATGP";
  target: string;
  maxSteps: number;
  seed: number;
}

const DEFAULT_SCENARIOS: (ScenarioItem & { disabled?: boolean; disabled_reason?: string })[] = [
  {
    id: "scenario-ransomware",
    alias: "ransomware_outbreak",
    name: "Ransomware Outbreak & Host Containment",
    description: "Sequential incident progression featuring active file encryption and lateral SMB propagation. Evaluates host isolation (Action 7) and process termination (Action 8) response actions against benchmark CISA playbooks.",
    simulation_mode: "Historical Replay",
    validation_status: "Validated (Historical Replay)",
    expected_actions: ["isolate_host (Class 7)", "kill_process (Class 8)"],
    measurable_metrics: "Contextual bandit reward (+1.0 / -1.0) and CISA playbook action alignment rate.",
    supported_policies: ["PPO (Ground-Truth Replay Mode)", "PPO+ATGP (Ground-Truth Replay Mode)"],
    target_default: "Subnet 10.0.1.0/24 (Corporate Workstations)",
    tactics: ["Initial Access", "Lateral Movement", "Impact"],
    supported_steps: { min: 5, max: 64, default: 20 },
    initial_state: "Corporate Workstation Subnet (10.0.1.0/24) - Uncontained",
  },
  {
    id: "scenario-c2",
    alias: "c2_beaconing",
    name: "Cobalt Strike C2 Beaconing & Network Infiltration",
    description: "Sequential incident progression tracking external adversary ingress probing, periodic outbound HTTPS beaconing, and egress communication. Evaluates perimeter IP blocking (Action 3) and DNS sinkholing (Action 5).",
    simulation_mode: "Historical Replay",
    validation_status: "Validated (Historical Replay)",
    expected_actions: ["block_source_ip (Class 3)", "block_dest_ip (Class 4)", "dns_sinkhole (Class 5)"],
    measurable_metrics: "Perimeter firewall deny-rule and egress block alignment with CISA playbooks.",
    supported_policies: ["PPO (Ground-Truth Replay Mode)", "PPO+ATGP (Ground-Truth Replay Mode)"],
    target_default: "Subnet 10.0.2.0/24 (DMZ Server Infrastructure)",
    tactics: ["Command and Control", "Exfiltration"],
    supported_steps: { min: 5, max: 50, default: 16 },
    initial_state: "DMZ Server Subnet (10.0.2.0/24) - Active Outbound Beaconing",
  },
  {
    id: "scenario-vulnerability",
    alias: "web_exploit",
    name: "Public CVE Exploitation & Vulnerability Eradication",
    description: "Sequential incident progression tracking remote code execution against unpatched public-facing services and backdoor persistence. Evaluates vulnerability patching (Action 16) and persistence removal (Action 13).",
    simulation_mode: "Historical Replay",
    validation_status: "Validated (Historical Replay)",
    expected_actions: ["patch_vulnerability (Class 16)", "remove_persistence (Class 13)"],
    measurable_metrics: "Eradication action match rate on known exploited vulnerabilities (+1.0 / -1.0).",
    supported_policies: ["PPO (Ground-Truth Replay Mode)", "PPO+ATGP (Ground-Truth Replay Mode)"],
    target_default: "Subnet 10.0.2.0/24 (DMZ Ingress Web Servers)",
    tactics: ["Initial Access", "Persistence", "Privilege Escalation"],
    supported_steps: { min: 5, max: 64, default: 20 },
    initial_state: "DMZ Ingress Cluster (10.0.2.0/24) - Unpatched Public CVE Detected",
  },
  {
    id: "scenario-caldera-live",
    name: "MITRE Caldera Multi-Host Operation (Phase 3 Planned)",
    description: "Multi-stage automated adversary emulating APT29 across real Linux/Windows virtual machines.",
    simulation_mode: "Sequential Attack Simulation",
    validation_status: "Disabled (Phase 3 Planned)",
    expected_actions: ["isolate_host", "remediate_network"],
    measurable_metrics: "Enterprise host infection graph, active breach status, and live time-to-containment.",
    supported_policies: ["PPO (Live Inference)"],
    target_default: "Enterprise Testbed (Caldera Daemon)",
    tactics: ["Initial Access", "Persistence", "Command and Control", "Lateral Movement"],
    disabled: true,
    disabled_reason: "Requires live MITRE Caldera daemon container and active host telemetry agents (commented out in Phase 2).",
  },
];

const TARGETS = [
  {
    id: "target-workstations",
    name: "Subnet 10.0.1.0/24 (Corporate Workstations)",
    description: "End-user enterprise machines monitored via software contextual bandit",
  },
  {
    id: "target-dmz",
    name: "Subnet 10.0.2.0/24 (DMZ Server Infrastructure)",
    description: "Public-facing ingress web servers and reverse proxies",
  },
  {
    id: "target-dc",
    name: "Host AD-DC-01 (Active Directory Domain Controller)",
    description: "Critical identity management host holding Kerberos KDC & NTDS.dit",
  },
];

interface Props {
  config: SimulationConfig;
  onConfigChange: (config: SimulationConfig) => void;
  onStart: () => void;
  onStop: () => void;
  isStarting?: boolean;
  isRunning?: boolean;
  activeRunId?: string | null;
}

export function SimulationConfigForm({
  config,
  onConfigChange,
  onStart,
  onStop,
  isStarting = false,
  isRunning = false,
  activeRunId = null,
}: Props) {
  const [scenariosList, setScenariosList] = useState(DEFAULT_SCENARIOS);

  useEffect(() => {
    let mounted = true;
    api
      .getScenarios()
      .then((res) => {
        if (mounted && res?.scenarios && res.scenarios.length > 0) {
          const apiScenarios = res.scenarios as ScenarioItem[];
          setScenariosList([
            ...apiScenarios,
            DEFAULT_SCENARIOS[3], // Disabled Phase 3 Planned option
          ]);
        }
      })
      .catch(() => {
        // Fall back to DEFAULT_SCENARIOS
      });
    return () => {
      mounted = false;
    };
  }, []);

  const activeScenario =
    scenariosList.find(
      (s) => s.id === config.scenario || s.alias === config.scenario
    ) || scenariosList[0];

  const handleReset = () => {
    onConfigChange({
      scenario: DEFAULT_SCENARIOS[0].id,
      policy: "PPO",
      target: TARGETS[0].name,
      maxSteps: 20,
      seed: 42,
    });
  };

  const handleFieldChange = (updates: Partial<SimulationConfig>) => {
    onConfigChange({
      ...config,
      ...updates,
    });
  };

  return (
    <Card className="p-5 border-slate-800/80 bg-slate-900/60 space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div>
          <h2 className="text-xs font-semibold text-slate-100 uppercase tracking-wider flex items-center gap-2">
            <Sliders className="h-4 w-4 text-teal-400" />
            <span>Simulation Parameters &amp; Policy Control</span>
          </h2>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Configure scenario inputs, target environment, policy algorithm, and step bounds.
          </p>
        </div>
        <div className="flex items-center gap-2 self-start sm:self-auto">
          {isRunning ? (
            <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-semibold text-teal-300 bg-teal-500/15 border border-teal-500/40 flex items-center gap-1.5 animate-pulse">
              <Activity className="h-3 w-3" />
              <span>Running: {activeRunId}</span>
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded text-[10px] font-mono text-emerald-300 bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-1">
              <CheckCircle2 className="h-3 w-3" />
              <span>Backend Ready (Historical Benchmark Replay)</span>
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
        {/* Validated Attack Scenario Selector & Metadata Card */}
        <div className="space-y-2 lg:col-span-2">
          <div className="flex items-center justify-between">
            <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
              <ShieldAlert className="h-3.5 w-3.5 text-rose-400" />
              <span>Attack Simulation Scenario</span>
            </label>
            {activeScenario && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/30">
                Mode: {activeScenario.simulation_mode || "Historical Replay"}
              </span>
            )}
          </div>
          <select
            value={config.scenario}
            disabled={isRunning || isStarting}
            onChange={(e) => {
              const selectedId = e.target.value;
              const found = scenariosList.find((s) => s.id === selectedId);
              if (found && found.disabled) return;
              handleFieldChange({ scenario: selectedId });
            }}
            className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 focus:outline-none focus:border-teal-500 font-sans disabled:opacity-60"
          >
            {scenariosList.map((s) => (
              <option key={s.id} value={s.id} disabled={s.disabled}>
                {s.name} {s.disabled ? "— [Disabled: Planned]" : "— [Validated Replay]"}
              </option>
            ))}
          </select>

          {/* Scenario Details Panel */}
          {activeScenario && (
            <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800/90 space-y-2 text-[11px]">
              <p className="text-slate-300 leading-relaxed font-sans">
                {activeScenario.description}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1 border-t border-slate-800/60 font-mono text-[10px]">
                <div>
                  <span className="text-slate-500">Validation Status: </span>
                  <span className={activeScenario.disabled ? "text-amber-400" : "text-emerald-400 font-semibold"}>
                    {activeScenario.validation_status || "Validated (Historical Replay)"}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500">Candidate Actions: </span>
                  <span className="text-teal-300">
                    {(activeScenario.expected_actions || []).join(", ") || "Canonical CISA Playbook"}
                  </span>
                </div>
                <div className="sm:col-span-2">
                  <span className="text-slate-500">Measurable Metrics: </span>
                  <span className="text-slate-300">
                    {activeScenario.measurable_metrics || "Contextual bandit reward (+1.0 / -1.0) and CISA playbook action alignment"}
                  </span>
                </div>
                {activeScenario.disabled && activeScenario.disabled_reason && (
                  <div className="sm:col-span-2 p-1.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-300 font-sans">
                    <strong>Notice:</strong> {activeScenario.disabled_reason}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Policy Selector */}
        <div className="space-y-1.5">
          <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
            <Cpu className="h-3.5 w-3.5 text-teal-400" />
            <span>Policy Algorithm</span>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              disabled={isRunning || isStarting}
              onClick={() => handleFieldChange({ policy: "PPO" })}
              className={`py-2 px-3 rounded-lg border text-xs font-mono font-medium transition-colors ${
                config.policy === "PPO"
                  ? "bg-teal-500/15 border-teal-500/50 text-teal-300 shadow-sm"
                  : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
              } disabled:opacity-60`}
            >
              PPO (Standard)
            </button>
            <button
              type="button"
              disabled={isRunning || isStarting}
              onClick={() => handleFieldChange({ policy: "PPO+ATGP" })}
              className={`py-2 px-3 rounded-lg border text-xs font-mono font-medium transition-colors ${
                config.policy === "PPO+ATGP"
                  ? "bg-teal-500/15 border-teal-500/50 text-teal-300 shadow-sm"
                  : "bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200"
              } disabled:opacity-60`}
            >
              PPO + ATGP
            </button>
          </div>
          <p className="text-[10px] text-slate-400 font-mono">
            {config.policy === "PPO"
              ? "Standard PPO with GAE (gamma=0.99, lambda=0.95)"
              : "Adaptive Teacher-Guided Preservation policy for rare classes"}
          </p>
        </div>

        {/* Target Environment */}
        <div className="space-y-1.5 lg:col-span-2">
          <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
            <Target className="h-3.5 w-3.5 text-amber-400" />
            <span>Simulated Target Infrastructure</span>
          </label>
          <select
            value={config.target}
            disabled={isRunning || isStarting}
            onChange={(e) => handleFieldChange({ target: e.target.value })}
            className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 focus:outline-none focus:border-teal-500 font-sans disabled:opacity-60"
          >
            {TARGETS.map((t) => (
              <option key={t.id} value={t.name}>
                {t.name}
              </option>
            ))}
          </select>
          <p className="text-[10px] text-slate-400 font-mono">
            {TARGETS.find((t) => t.name === config.target || t.id === config.target)?.description ||
              "Subnet monitored via software contextual bandit"}
          </p>
        </div>

        {/* Max Steps & Seed */}
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
              <Hash className="h-3.5 w-3.5 text-slate-400" />
              <span>Max Steps ({config.maxSteps})</span>
            </label>
            <input
              type="range"
              min={10}
              max={100}
              step={2}
              value={config.maxSteps}
              disabled={isRunning || isStarting}
              onChange={(e) => handleFieldChange({ maxSteps: Number(e.target.value) })}
              className="w-full accent-teal-400 cursor-pointer disabled:opacity-60"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>10</span>
              <span>{config.maxSteps} steps</span>
              <span>100</span>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-slate-300">
              <span>Random Seed</span>
            </label>
            <input
              type="number"
              value={config.seed}
              disabled={isRunning || isStarting}
              onChange={(e) => handleFieldChange({ seed: Number(e.target.value) })}
              className="w-full px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 font-mono focus:outline-none focus:border-teal-500 disabled:opacity-60"
            />
            <span className="text-[10px] text-slate-400 font-mono">Ensures deterministic replay</span>
          </div>
        </div>
      </div>

      {/* Action Controls & Execution State Status */}
      <div className="pt-3 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-teal-500/5 border border-teal-500/20 text-[11px] text-teal-300/90 max-w-xl">
          <Info className="h-4 w-4 shrink-0 text-teal-400" />
          <span>
            {isRunning ? (
              <span>
                <strong>Historical Benchmark Replay Active:</strong> Stepping sequentially through real benchmark incidents from test.csv. Streaming live SSE events.
              </span>
            ) : (
              <span>
                <strong>Mode: Historical Benchmark Incident Replay:</strong> Replaying real incident records with CISA actions and bandit rewards (+1.0 / -1.0). Checkpoints are uncommitted; synthetic probabilities and fake containment have been removed.
              </span>
            )}
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={handleReset}
            disabled={isRunning || isStarting}
            className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-slate-100 text-xs font-medium border border-slate-700 flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset</span>
          </button>

          <button
            type="button"
            onClick={onStop}
            disabled={!isRunning}
            className={`px-3 py-2 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-colors ${
              isRunning
                ? "bg-rose-500/15 border-rose-500/50 text-rose-300 hover:bg-rose-500/25 cursor-pointer"
                : "bg-slate-800 text-slate-500 border-slate-700/60 cursor-not-allowed opacity-50"
            }`}
            title="Stop Simulation"
          >
            <Square className="h-3.5 w-3.5" />
            <span>Stop</span>
          </button>

          <button
            type="button"
            onClick={onStart}
            disabled={isRunning || isStarting}
            className={`px-4 py-2 rounded-lg text-xs font-semibold border flex items-center gap-1.5 transition-colors ${
              !isRunning && !isStarting
                ? "bg-gradient-to-r from-teal-500 to-teal-600 hover:from-teal-400 hover:to-teal-500 text-slate-950 border-teal-400 cursor-pointer shadow-md shadow-teal-500/10"
                : "bg-slate-800 text-slate-500 border-slate-700/60 cursor-not-allowed opacity-50"
            }`}
            title="Start Simulation"
          >
            {isStarting ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                <span>Initializing...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Start Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>
    </Card>
  );
}
