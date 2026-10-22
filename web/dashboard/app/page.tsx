"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  ShieldCheck,
  Award,
  Layers,
  ArrowRight,
  ExternalLink,
  Info,
  Sliders,
  Cpu,
  BrainCircuit,
  Activity,
  AlertCircle,
  Network,
} from "lucide-react";
import { SystemStatusCards } from "@/components/simulation/SystemStatusCards";
import {
  SimulationConfigForm,
  SimulationConfig,
} from "@/components/simulation/SimulationConfigForm";
import { ArchitectureFlow } from "@/components/simulation/ArchitectureFlow";
import { EventTimeline } from "@/components/simulation/EventTimeline";
import { CurrentDecisionPanel } from "@/components/simulation/CurrentDecisionPanel";
import { ResultsSummaryCards } from "@/components/simulation/ResultsSummaryCards";
import { api } from "@/lib/api";
import { SimulationEvent, SimulationResults } from "@/types";

export default function SimulationWorkspacePage() {
  const [config, setConfig] = useState<SimulationConfig>({
    scenario: "scenario-ransomware",
    policy: "PPO",
    target: "Subnet 10.0.1.0/24 (Corporate Workstations)",
    maxSteps: 20,
    seed: 42,
  });

  const [isStarting, setIsStarting] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);

  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [currentDecision, setCurrentDecision] = useState<any | null>(null);
  const [currentStep, setCurrentStep] = useState<number | null>(null);
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string>("4");
  const [isStale, setIsStale] = useState<boolean>(false);
  const [results, setResults] = useState<SimulationResults | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const lastSeenSeqRef = useRef<number>(-1);
  const staleTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Helper to restart the 5-second stale connection watchdog
  const resetStaleWatchdog = useCallback(() => {
    setIsStale(false);
    if (staleTimerRef.current) {
      clearTimeout(staleTimerRef.current);
    }
    staleTimerRef.current = setTimeout(() => {
      // If 5 seconds elapse with no event while simulation is running, flag stale
      setIsStale(true);
    }, 5000);
  }, []);

  const clearStaleWatchdog = useCallback(() => {
    if (staleTimerRef.current) {
      clearTimeout(staleTimerRef.current);
      staleTimerRef.current = null;
    }
    setIsStale(false);
  }, []);

  // Clean up SSE connection and timer on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      clearStaleWatchdog();
    };
  }, [clearStaleWatchdog]);

  const handleStart = async () => {
    setIsStarting(true);
    setEvents([]);
    setCurrentDecision(null);
    setCurrentStep(null);
    setResults(null);
    setActiveStage(1);
    setSelectedNodeId("1");
    lastSeenSeqRef.current = -1;
    clearStaleWatchdog();

    try {
      const res = await api.startSimulation({
        scenario: config.scenario,
        policy: config.policy,
        target: config.target,
        step_limit: config.maxSteps,
        seed: config.seed,
      });

      const runId = res.run_id;
      setActiveRunId(runId);
      activeRunIdRef.current = runId;
      setIsRunning(true);
      setIsStarting(false);
      resetStaleWatchdog();

      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const sseUrl = api.getSimulationEventStreamUrl(runId);
      const es = new EventSource(sseUrl);
      eventSourceRef.current = es;

      const handleSseMessage = (evtType: string, e: MessageEvent) => {
        try {
          const parsed = JSON.parse(e.data);
          const eventRunId = parsed.run_id || runId;

          // Run ID Guard: Ignore events from previous or aborted runs
          if (activeRunIdRef.current && eventRunId !== activeRunIdRef.current) {
            return;
          }

          const seq = typeof parsed.seq === "number" ? parsed.seq : Date.now();

          // Sequence monotonic guard: reject duplicate or out-of-order events
          if (seq <= lastSeenSeqRef.current) {
            return;
          }
          lastSeenSeqRef.current = seq;

          // Reset the stale connection watchdog since a genuine event was received
          resetStaleWatchdog();

          const simEvt: SimulationEvent = {
            run_id: eventRunId,
            seq: seq,
            timestamp: parsed.timestamp || new Date().toISOString(),
            event_type: evtType as any,
            data: parsed.data || parsed,
          };

          setEvents((prev) => [...prev, simEvt]);

          const d = simEvt.data;
          if (evtType === "simulation_started") {
            setActiveStage(1);
          } else if (evtType === "attack_stage_updated") {
            setActiveStage(1);
            if (d.step !== undefined) setCurrentStep(d.step);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              target: d.target,
              stage: d.stage,
              tactics: d.tactics,
            }));
          } else if (evtType === "incident_observed") {
            setActiveStage(3);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              incident_text: d.text,
              prepared_text: d.prepared_text,
              cisa_phase: d.cisa_phase,
              ground_truth_action: d.ground_truth_action,
              ground_truth_name: d.ground_truth_name,
            }));
          } else if (evtType === "embedding_generated") {
            setActiveStage(4);
          } else if (evtType === "action_selected") {
            setActiveStage(5);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              chosen_action: d.chosen_action,
              chosen_action_name: d.chosen_action_name,
              chosen_action_phase: d.chosen_action_phase,
              confidence: d.confidence,
              critic_value: d.critic_value,
            }));
          } else if (evtType === "simulated_effect_applied") {
            setActiveStage(6);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              implementation: d.implementation,
              threat_match: d.threat_match,
            }));
          } else if (evtType === "reward_calculated") {
            setActiveStage(7);
            setCurrentDecision((prev: any) => ({
              ...(prev || {}),
              reward: d.step_reward,
              cumulative_reward: d.cumulative_reward,
              is_correct: d.is_correct,
            }));
          } else if (evtType === "state_updated") {
            setActiveStage(8);
            if (d.step !== undefined) setCurrentStep(d.step);
          } else if (
            evtType === "simulation_completed" ||
            evtType === "simulation_stopped"
          ) {
            setIsRunning(false);
            setActiveStage(null);
            clearStaleWatchdog();
            setResults(d as SimulationResults);
            es.close();
          }
        } catch (err) {
          console.error("SSE parse error", err);
        }
      };

      const eventTypes = [
        "simulation_started",
        "attack_stage_updated",
        "incident_observed",
        "embedding_generated",
        "action_selected",
        "simulated_effect_applied",
        "reward_calculated",
        "state_updated",
        "simulation_completed",
        "simulation_stopped",
      ];

      eventTypes.forEach((t) => {
        es.addEventListener(t, (e: MessageEvent) => handleSseMessage(t, e));
      });

      es.onerror = () => {
        api
          .getSimulationDetail(runId)
          .then((detail) => {
            const s = (detail.status || "").toLowerCase();
            if (s === "completed" || s === "stopped" || s === "failed") {
              setIsRunning(false);
              setActiveStage(null);
              clearStaleWatchdog();
              if (detail.results) setResults(detail.results as SimulationResults);
              es.close();
            }
          })
          .catch(() => {});
      };
    } catch (err) {
      setIsStarting(false);
      setIsRunning(false);
      setActiveStage(null);
      clearStaleWatchdog();
      console.error("Failed to start simulation", err);
    }
  };

  const handleStop = async () => {
    if (!activeRunId) return;
    try {
      await api.stopSimulation(activeRunId);
      clearStaleWatchdog();
    } catch (err) {
      console.error("Failed to stop simulation", err);
    }
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Workspace Header */}
      <div className="rounded-xl border border-slate-800 bg-gradient-to-r from-slate-900/90 via-[#0a1122]/80 to-slate-900/90 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-teal-500/10 text-teal-400 border border-teal-500/20">
                Department of Computer Engineering, University of Jaffna
              </span>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono text-emerald-300 bg-emerald-500/10 border border-emerald-500/30">
                Step 5: Live SOC Architecture Visualization Active
              </span>
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
              SOC Agent Simulation Control Panel
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 max-w-3xl leading-relaxed">
              Research control workspace for{" "}
              <strong className="text-slate-200">
                Optimizing Cybersecurity Incident Response Actions Using a
                Transformer-Based Agent
              </strong>
              . Demonstrates incident preprocessing, SecBERT 768-dim state
              representations, and PPO / PPO+ATGP policy decision-making in an
              authentic Gymnasium contextual bandit environment.
            </p>
          </div>

          <div className="flex sm:flex-col items-end gap-2 shrink-0">
            <Link
              href="/architecture"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
            >
              <Layers className="h-3.5 w-3.5 text-teal-400" />
              <span>Full Architecture</span>
            </Link>
            <Link
              href="/results"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
            >
              <Award className="h-3.5 w-3.5 text-amber-400" />
              <span>Research Results</span>
            </Link>
          </div>
        </div>
      </div>

      {/* A. System Status Cards */}
      <section>
        <SystemStatusCards />
      </section>

      {/* B. Simulation Configuration & Start/Stop Controls */}
      <section>
        <SimulationConfigForm
          config={config}
          onConfigChange={setConfig}
          onStart={handleStart}
          onStop={handleStop}
          isStarting={isStarting}
          isRunning={isRunning}
          activeRunId={activeRunId}
        />
      </section>

      {/* C. System Architecture Visualization (React Flow with 8 Stages and Edge Transitions) */}
      <section>
        <ArchitectureFlow
          activeStage={activeStage}
          activeRunId={activeRunId}
          currentStep={currentStep}
          maxSteps={config.maxSteps}
          selectedScenario={config.scenario}
          selectedPolicy={config.policy}
          simulationMode="Historical Replay"
          isStale={isStale}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          liveStepData={currentDecision}
        />
      </section>

      {/* D & E. Event Timeline and Current Decision Panel */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <EventTimeline
          events={events}
          isRunning={isRunning}
          activeRunId={activeRunId}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
        />
        <CurrentDecisionPanel
          selectedPolicy={config.policy}
          currentDecision={currentDecision}
          currentStep={currentStep}
          isRunning={isRunning}
        />
      </section>

      {/* F. Results Summary */}
      <section>
        <ResultsSummaryCards results={results} isRunning={isRunning} />
      </section>
    </div>
  );
}
